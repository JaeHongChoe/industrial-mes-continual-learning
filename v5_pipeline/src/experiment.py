from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .candidates import CandidatePrediction, ScoreRow
from .data_contract import (
    ACTION_LABELS,
    DataContractError,
    DatasetBundle,
    FeatureRow,
    IMMUTABLE_AUDIT_START,
    IMMUTABLE_LOCKBOX_MONTHS,
    IMMUTABLE_SELECTOR_MONTHS,
    prepare_dataset,
)
from .freeze import (
    AuditLabelVault,
    FreezeCoordinator,
    FreezeReceipt,
    LockboxLabelVault,
    PolicySpec,
)
from .metrics import GroupingBasis, Metrics, OutcomeRow, REVIEW_LABELS, compute_metrics, grouped_metrics
from .select import CandidateDecision, LockboxResult, SelectorUniverse
from .thresholds import ThresholdFit, apply_threshold_fit, validate_threshold_fit


SELECTOR_MODEL_TRAIN_END = "2025-07-31"
SELECTOR_PROBABILITY_CALIBRATION_MONTH = "2025-08"
SELECTOR_THRESHOLD_FIT_MONTH = "2025-09"
FINAL_REFIT_TRAIN_END = "2025-09-30"
FINAL_REFIT_PROBABILITY_CALIBRATION_MONTH = "2025-10"
FINAL_REFIT_THRESHOLD_FIT_MONTH = "2025-11"
AUDIT_STATUS = "retrospective_audit_only"
WARM_COLD_BASIS = "training_and_calibration_prior_history"


class ExperimentOrderError(DataContractError):
    """Raised when a label vault or audit step is reached out of order."""


@dataclass(frozen=True, slots=True)
class ChronologyPartitions:
    train: tuple[FeatureRow, ...]
    probability_calibration: tuple[FeatureRow, ...]
    threshold_fit: tuple[FeatureRow, ...]
    selector: tuple[FeatureRow, ...]
    train_labels: Mapping[str, str]
    probability_calibration_labels: Mapping[str, str]
    threshold_fit_labels: Mapping[str, str]
    selector_labels: Mapping[str, str]

    @property
    def warm_product_keys(self) -> frozenset[str]:
        return frozenset(
            row.product_key
            for row in (*self.train, *self.probability_calibration, *self.threshold_fit)
        )


@dataclass(frozen=True, slots=True)
class FinalRefitPartitions:
    train: tuple[FeatureRow, ...]
    probability_calibration: tuple[FeatureRow, ...]
    threshold_fit: tuple[FeatureRow, ...]
    lockbox_scoring: tuple[FeatureRow, ...]
    audit_scoring: tuple[FeatureRow, ...]
    train_labels: Mapping[str, str]
    probability_calibration_labels: Mapping[str, str]
    threshold_fit_labels: Mapping[str, str]
    frozen_policy_sha256: str
    frozen_candidate_registry_sha256: str
    frozen_company_assignments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class BaselineManifestReceipt:
    path: Path
    baseline_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class MetricMaterialization:
    evaluation_rows: int
    non_action_rows: int
    global_metrics: Metrics
    company_metrics: Mapping[str, Metrics]
    month_metrics: Mapping[str, Metrics]
    warm_cold_metrics: Mapping[str, Metrics]
    artifact_bytes: bytes
    sha256: str
    final_policy_sha256: str
    lockbox_assignment_sha256: str


class LeakageSafeExperiment:
    """Small orchestration layer around the immutable experiment modules.

    Candidate fitting and scoring remain in ``src.candidates``. This class only
    binds their outputs to the prepared row universe and enforces the order in
    which retrospective labels may be opened.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        run_dir: Path,
        bundle: DatasetBundle,
        chronology: ChronologyPartitions,
        selector_baseline: BaselineManifestReceipt,
        selector_universe: SelectorUniverse,
    ) -> None:
        self.config_path = config_path
        self.run_dir = run_dir
        self.bundle = bundle
        self.chronology = chronology
        self.selector_baseline = selector_baseline
        self.selector_universe = selector_universe
        self.state_path = run_dir / "freeze_state.json"
        self._receipt: FreezeReceipt | None = None
        self._lockbox_predictions: Mapping[str, str] | None = None
        self._lockbox_prediction_sha256: str | None = None
        self._lockbox_labels: Mapping[str, str] | None = None
        self._lockbox_result: LockboxResult | None = None
        self._lockbox_assignment_sha256: str | None = None
        self._final_policy_sha256: str | None = None
        self._audit_predictions: Mapping[str, str] | None = None
        self._audit_prediction_sha256: str | None = None
        self._report: MetricMaterialization | None = None

    def validate_selector_scores(
        self,
        score_table: Mapping[str, Iterable[ScoreRow]],
        threshold_fits_by_candidate: Mapping[str, ThresholdFit],
    ) -> Mapping[str, tuple[CandidatePrediction, ...]]:
        return validate_candidate_score_table(
            score_table,
            expected_rows=self.chronology.selector,
            threshold_fits_by_candidate=threshold_fits_by_candidate,
            source_dataset_id=self.bundle.metadata.source_dataset_id,
            history_sha256=self.bundle.metadata.source_hashes["historyRows"],
        )

    def selector_decisions(
        self,
        predictions_by_candidate: Mapping[str, Sequence[CandidatePrediction]],
    ) -> tuple[CandidateDecision, ...]:
        expected_ids = tuple(row.row_id for row in self.chronology.selector)
        expected_set = set(expected_ids)
        if not predictions_by_candidate:
            raise ValueError("selector predictions cannot be empty")
        decisions: list[CandidateDecision] = []
        for candidate_id in sorted(predictions_by_candidate):
            predictions = tuple(predictions_by_candidate[candidate_id])
            by_id = {item.row_id: item for item in predictions}
            if len(by_id) != len(predictions):
                raise ValueError(f"duplicate selector prediction row for candidate: {candidate_id}")
            if set(by_id) != expected_set:
                raise ValueError(
                    f"candidate predictions do not match the fixed selector row universe: {candidate_id}"
                )
            for row_id in expected_ids:
                universe_row = self.selector_universe.rows[row_id]
                prediction = by_id[row_id]
                decisions.append(
                    CandidateDecision(
                        candidate_id=candidate_id,
                        row_id=row_id,
                        episode_id=universe_row.episode_id,
                        company=universe_row.company,
                        month=universe_row.month,
                        actual=universe_row.actual,
                        predicted=prediction.label,
                    )
                )
        return tuple(decisions)

    def final_refit_partitions(
        self,
        receipt: FreezeReceipt | None,
    ) -> FinalRefitPartitions:
        self._require_frozen_receipt(receipt)
        if self._lockbox_labels is not None:
            raise ExperimentOrderError("final-refit partitions must be fixed before lockbox labels are opened")
        assert receipt is not None
        return split_final_refit_chronology(self.bundle, receipt)

    def freeze_policy(self, spec: PolicySpec) -> FreezeReceipt:
        if self._receipt is not None:
            raise ExperimentOrderError("policy is already frozen")
        receipt = FreezeCoordinator(self.config_path, self.bundle, self.state_path).freeze(spec)
        self._receipt = receipt
        return receipt

    def seal_lockbox_predictions(self, predictions: Mapping[str, str]) -> str:
        if self._receipt is None:
            raise ExperimentOrderError("policy freeze is required before lockbox predictions")
        if self._lockbox_labels is not None:
            raise ExperimentOrderError("lockbox predictions cannot be sealed after labels are opened")
        expected_ids = {row.row_id for row in self.bundle.lockbox}
        observed_ids = set(predictions)
        if observed_ids != expected_ids:
            raise ValueError(
                "lockbox predictions must match the exact prepared lockbox row universe: "
                f"expected={len(expected_ids)} observed={len(observed_ids)}"
            )
        normalized = _normalize_predictions(predictions, expected_ids, phase="lockbox")
        payload = {
            "schemaVersion": 1,
            "kind": "sealed_lockbox_predictions",
            "sourceDatasetId": self.bundle.metadata.source_dataset_id,
            "historySha256": self.bundle.metadata.source_hashes["historyRows"],
            "basePolicySha256": self._receipt.sha256,
            "lockboxRowUniverseSha256": _feature_row_universe_sha256(self.bundle.lockbox),
            "expectedRowCount": len(expected_ids),
            "predictions": normalized,
        }
        prediction_sha256 = hashlib.sha256(_stable_json_bytes(payload)).hexdigest()
        if self._lockbox_predictions is not None:
            if prediction_sha256 != self._lockbox_prediction_sha256:
                raise ExperimentOrderError("lockbox predictions are already sealed with different bytes")
            return prediction_sha256
        self._lockbox_predictions = MappingProxyType(normalized)
        self._lockbox_prediction_sha256 = prediction_sha256
        return prediction_sha256

    def open_lockbox(self, receipt: FreezeReceipt | None) -> Mapping[str, str]:
        self._require_frozen_receipt(receipt)
        if self._lockbox_predictions is None or self._lockbox_prediction_sha256 is None:
            raise ExperimentOrderError("sealed lockbox predictions are required before opening labels")
        if self._lockbox_labels is not None:
            raise ExperimentOrderError("lockbox labels were already opened")
        assert receipt is not None
        labels = LockboxLabelVault(self.config_path, self.bundle, self.state_path).open(receipt)
        self._lockbox_labels = labels
        return labels

    def bind_lockbox_result(self, result: LockboxResult) -> str:
        if self._receipt is None:
            raise ExperimentOrderError("policy freeze is required before binding a lockbox result")
        if self._lockbox_predictions is None or self._lockbox_prediction_sha256 is None:
            raise ExperimentOrderError("sealed lockbox predictions are required before binding a result")
        if self._lockbox_labels is None:
            raise ExperimentOrderError("lockbox labels must be opened before binding a result")
        if self._final_policy_sha256 is not None:
            raise ExperimentOrderError("lockbox result is already bound to the final policy")
        if not isinstance(result, LockboxResult):
            raise TypeError("lockbox result must be a LockboxResult")

        candidate_registry_sha256, frozen_assignments = _validate_final_refit_freeze(
            self.bundle,
            self._receipt,
        )
        result_assignments = {str(key): str(value) for key, value in result.assignments.items()}
        result_rejections = {
            str(key): tuple(str(reason) for reason in reasons)
            for key, reasons in result.rejection_reasons.items()
        }
        if set(result_assignments) != set(frozen_assignments) or set(result_rejections) != set(
            frozen_assignments
        ):
            raise ValueError("lockbox result assignments must cover the exact frozen company universe")
        for company, selected_candidate in frozen_assignments.items():
            final_candidate = result_assignments[company]
            if final_candidate not in {selected_candidate, "review_all"}:
                raise ValueError("lockbox result may not promote or swap a frozen company assignment")
            if final_candidate == "review_all" and selected_candidate != "review_all" and not result_rejections[
                company
            ]:
                raise ValueError("lockbox rejection to review_all requires a recorded reason")
        if not _is_sha256(result.lockbox_universe_sha256) or not _is_sha256(result.assignment_sha256):
            raise ValueError("lockbox result requires lowercase SHA-256 bindings")

        policy_payload = json.loads(self._receipt.policy_bytes)
        model_params = policy_payload["modelParams"]
        expected_assignment_mode = model_params.get("assignmentMode")
        if expected_assignment_mode is not None and result.assignment_mode != expected_assignment_mode:
            raise ValueError("lockbox result assignment mode differs from the frozen policy")
        assignment_payload = {
            "schemaVersion": 1,
            "registrySha256": candidate_registry_sha256,
            "lockboxUniverseSha256": result.lockbox_universe_sha256,
            "assignmentMode": result.assignment_mode,
            "assignments": dict(sorted(result_assignments.items())),
            "rejections": {
                company: list(result_rejections[company]) for company in sorted(result_rejections)
            },
        }
        expected_assignment_sha256 = hashlib.sha256(
            _stable_json_bytes(assignment_payload)
        ).hexdigest()
        if expected_assignment_sha256 != result.assignment_sha256:
            raise ValueError("lockbox result assignment SHA does not match its contents")

        final_policy_payload = {
            "schemaVersion": 1,
            "kind": "final_policy_binding",
            "basePolicySha256": self._receipt.sha256,
            "lockboxPredictionSha256": self._lockbox_prediction_sha256,
            "lockboxAssignmentSha256": result.assignment_sha256,
        }
        final_policy_sha256 = hashlib.sha256(_stable_json_bytes(final_policy_payload)).hexdigest()
        self._lockbox_result = result
        self._lockbox_assignment_sha256 = result.assignment_sha256
        self._final_policy_sha256 = final_policy_sha256
        return final_policy_sha256

    def seal_audit_predictions(
        self,
        predictions: Mapping[str, str],
        *,
        final_policy_sha256: str,
    ) -> str:
        if self._receipt is None:
            raise ExperimentOrderError("policy freeze is required before audit predictions")
        if self._lockbox_labels is None:
            raise ExperimentOrderError("lockbox must be opened before final audit predictions are sealed")
        if (
            self._final_policy_sha256 is None
            or self._lockbox_prediction_sha256 is None
            or self._lockbox_assignment_sha256 is None
        ):
            raise ExperimentOrderError("final policy must bind the reject-only lockbox result first")
        if final_policy_sha256 != self._final_policy_sha256:
            raise ExperimentOrderError("audit predictions require the exact composite final-policy hash")
        expected_ids = {row.row_id for row in self.bundle.audit}
        observed_ids = set(predictions)
        if observed_ids != expected_ids:
            raise ValueError(
                "audit predictions must match the exact prepared audit row universe: "
                f"expected={len(expected_ids)} observed={len(observed_ids)}"
            )
        normalized = _normalize_predictions(predictions, expected_ids, phase="audit")
        payload = {
            "schemaVersion": 2,
            "kind": "sealed_audit_predictions",
            "sourceDatasetId": self.bundle.metadata.source_dataset_id,
            "policySha256": self._final_policy_sha256,
            "basePolicySha256": self._receipt.sha256,
            "finalPolicySha256": self._final_policy_sha256,
            "lockboxPredictionSha256": self._lockbox_prediction_sha256,
            "lockboxAssignmentSha256": self._lockbox_assignment_sha256,
            "expectedRowCount": len(expected_ids),
            "predictions": normalized,
        }
        prediction_sha256 = hashlib.sha256(_stable_json_bytes(payload)).hexdigest()
        if self._audit_predictions is not None:
            if prediction_sha256 != self._audit_prediction_sha256:
                raise ExperimentOrderError("audit predictions are already sealed with different bytes")
            return prediction_sha256
        self._audit_predictions = MappingProxyType(normalized)
        self._audit_prediction_sha256 = prediction_sha256
        return prediction_sha256

    def open_audit_and_materialize(self, receipt: FreezeReceipt | None) -> MetricMaterialization:
        self._require_frozen_receipt(receipt)
        if self._lockbox_labels is None:
            raise ExperimentOrderError("lockbox must be opened before audit labels")
        if self._final_policy_sha256 is None or self._lockbox_assignment_sha256 is None:
            raise ExperimentOrderError("final policy must bind the reject-only lockbox result")
        if self._audit_predictions is None or self._audit_prediction_sha256 is None:
            raise ExperimentOrderError("audit predictions must be sealed before audit labels")
        if self._report is not None:
            raise ExperimentOrderError("audit labels were already opened and materialized")
        assert receipt is not None
        audit_labels = AuditLabelVault(self.config_path, self.bundle, self.state_path).open(receipt)
        report = _materialize_audit_metrics(
            self.bundle,
            self.chronology,
            audit_labels,
            self._audit_predictions,
            base_policy_sha256=receipt.sha256,
            final_policy_sha256=self._final_policy_sha256,
            lockbox_prediction_sha256=self._lockbox_prediction_sha256,
            lockbox_assignment_sha256=self._lockbox_assignment_sha256,
            prediction_sha256=self._audit_prediction_sha256,
        )
        self._report = report
        return report

    def _require_frozen_receipt(self, receipt: FreezeReceipt | None) -> None:
        if self._receipt is None:
            raise ExperimentOrderError("policy freeze is required before opening label vaults")
        if receipt is None or (
            receipt.sha256 != self._receipt.sha256
            or receipt.binding_sha256 != self._receipt.binding_sha256
            or receipt.policy_bytes != self._receipt.policy_bytes
        ):
            raise ExperimentOrderError("receipt does not match this experiment's frozen policy")


def prepare_experiment(config_path: str | Path, run_dir: str | Path) -> LeakageSafeExperiment:
    resolved_config = Path(config_path).resolve()
    resolved_run_dir = Path(run_dir).resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    bundle = prepare_dataset(resolved_config)
    chronology = split_chronology(bundle)
    selector_baseline = write_all_review_selector_baseline(
        bundle,
        resolved_run_dir / "selector_all_review_baseline.json",
    )
    selector_universe = SelectorUniverse.from_prepared_bundle(
        bundle,
        phase="selector",
        baseline_manifest_path=selector_baseline.path,
    )
    return LeakageSafeExperiment(
        config_path=resolved_config,
        run_dir=resolved_run_dir,
        bundle=bundle,
        chronology=chronology,
        selector_baseline=selector_baseline,
        selector_universe=selector_universe,
    )


def split_chronology(bundle: DatasetBundle) -> ChronologyPartitions:
    train = tuple(row for row in bundle.history if row.date <= SELECTOR_MODEL_TRAIN_END)
    probability_calibration = tuple(
        row for row in bundle.history if row.month == SELECTOR_PROBABILITY_CALIBRATION_MONTH
    )
    threshold_fit = tuple(
        row for row in bundle.history if row.month == SELECTOR_THRESHOLD_FIT_MONTH
    )
    selector = tuple(bundle.selector)
    _require_exact_partition(
        (*bundle.history, *bundle.selector),
        {
            "train": train,
            "probability_calibration": probability_calibration,
            "threshold_fit": threshold_fit,
            "selector": selector,
        },
        chronology_name="selector chronology",
    )
    if not train or not probability_calibration or not threshold_fit or not selector:
        raise DataContractError("all selector chronology partitions must be non-empty")
    selector_months = {row.month for row in selector}
    if selector_months != set(IMMUTABLE_SELECTOR_MONTHS):
        raise DataContractError("selector chronology must contain both 2025-10 and 2025-11")

    history_labels = bundle.development_labels("history")
    selector_labels = bundle.development_labels("selector")
    _require_exact_labels(history_labels, bundle.history, partition_name="history")
    _require_exact_labels(selector_labels, selector, partition_name="selector")
    return ChronologyPartitions(
        train=train,
        probability_calibration=probability_calibration,
        threshold_fit=threshold_fit,
        selector=selector,
        train_labels=_labels_for(train, history_labels),
        probability_calibration_labels=_labels_for(probability_calibration, history_labels),
        threshold_fit_labels=_labels_for(threshold_fit, history_labels),
        selector_labels=_labels_for(selector, selector_labels),
    )


def split_final_refit_chronology(
    bundle: DatasetBundle,
    receipt: FreezeReceipt,
) -> FinalRefitPartitions:
    candidate_registry_sha256, company_assignments = _validate_final_refit_freeze(
        bundle,
        receipt,
    )
    train = tuple(row for row in bundle.history if row.date <= FINAL_REFIT_TRAIN_END)
    probability_calibration = tuple(
        row
        for row in bundle.selector
        if row.month == FINAL_REFIT_PROBABILITY_CALIBRATION_MONTH
    )
    threshold_fit = tuple(
        row for row in bundle.selector if row.month == FINAL_REFIT_THRESHOLD_FIT_MONTH
    )
    lockbox_scoring = tuple(bundle.lockbox)
    audit_scoring = tuple(bundle.audit)
    _require_exact_partition(
        tuple(row for split in bundle.all_feature_splits() for row in split),
        {
            "train": train,
            "probability_calibration": probability_calibration,
            "threshold_fit": threshold_fit,
            "lockbox_scoring": lockbox_scoring,
            "audit_scoring": audit_scoring,
        },
        chronology_name="final-refit chronology",
    )
    if any(
        not partition
        for partition in (
            train,
            probability_calibration,
            threshold_fit,
            lockbox_scoring,
            audit_scoring,
        )
    ):
        raise DataContractError("all final-refit chronology partitions must be non-empty")
    if {row.month for row in lockbox_scoring} != set(IMMUTABLE_LOCKBOX_MONTHS):
        raise DataContractError("final-refit lockbox scoring must contain only 2025-12")
    if any(row.date < IMMUTABLE_AUDIT_START for row in audit_scoring):
        raise DataContractError("final-refit audit scoring must begin on or after 2026-01-01")

    history_labels = bundle.development_labels("history")
    selector_labels = bundle.development_labels("selector")
    _require_exact_labels(history_labels, train, partition_name="final-refit train")
    _require_exact_labels(
        selector_labels,
        (*probability_calibration, *threshold_fit),
        partition_name="final-refit calibration",
    )
    return FinalRefitPartitions(
        train=train,
        probability_calibration=probability_calibration,
        threshold_fit=threshold_fit,
        lockbox_scoring=lockbox_scoring,
        audit_scoring=audit_scoring,
        train_labels=_labels_for(train, history_labels),
        probability_calibration_labels=_labels_for(probability_calibration, selector_labels),
        threshold_fit_labels=_labels_for(threshold_fit, selector_labels),
        frozen_policy_sha256=receipt.sha256,
        frozen_candidate_registry_sha256=candidate_registry_sha256,
        frozen_company_assignments=company_assignments,
    )


def write_all_review_selector_baseline(
    bundle: DatasetBundle,
    output_path: str | Path,
) -> BaselineManifestReceipt:
    if not bundle.selector:
        raise DataContractError("selector rows cannot be empty")
    payload = {
        "schemaVersion": 1,
        "kind": "baseline_prediction_manifest",
        "phase": "selector",
        "sourceDatasetId": bundle.metadata.source_dataset_id,
        "historySha256": bundle.metadata.source_hashes["historyRows"],
        "expectedRowCount": len(bundle.selector),
        "predictions": {row.row_id: "REVIEW" for row in sorted(bundle.selector, key=lambda item: item.row_id)},
    }
    baseline_bytes = _stable_json_bytes(payload)
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(baseline_bytes)
    temporary.replace(path)
    return BaselineManifestReceipt(
        path=path,
        baseline_bytes=baseline_bytes,
        sha256=hashlib.sha256(baseline_bytes).hexdigest(),
    )


def validate_candidate_score_table(
    score_table: Mapping[str, Iterable[ScoreRow]],
    *,
    expected_rows: Sequence[FeatureRow],
    threshold_fits_by_candidate: Mapping[str, ThresholdFit],
    source_dataset_id: str,
    history_sha256: str,
) -> Mapping[str, tuple[CandidatePrediction, ...]]:
    if not score_table or set(score_table) != set(threshold_fits_by_candidate):
        raise ValueError("score table and threshold-fit candidates must match exactly")
    expected_ids = tuple(row.row_id for row in expected_rows)
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected selector rows must be non-empty and unique")
    expected_set = set(expected_ids)
    row_order = {row_id: index for index, row_id in enumerate(expected_ids)}
    output: dict[str, tuple[CandidatePrediction, ...]] = {}
    for candidate_id in sorted(score_table):
        fitted = threshold_fits_by_candidate[candidate_id]
        if not isinstance(fitted, ThresholdFit):
            raise TypeError("selector threshold receipt must be a ThresholdFit")
        validate_threshold_fit(fitted)
        if fitted.source_dataset_id != source_dataset_id:
            raise ValueError(f"threshold fit receipt source does not match selector run: {candidate_id}")
        if fitted.history_sha256 != history_sha256:
            raise ValueError(f"threshold fit receipt history does not match selector run: {candidate_id}")
        fitted_provenance = fitted.calibrated_model_provenance
        fitted_model_sha256 = str(fitted_provenance.get("modelSha256") or "")
        fitted_schema_sha256 = str(fitted_provenance.get("featureSchemaSha256") or "")
        if fitted_schema_sha256 != fitted.feature_schema_sha256:
            raise ValueError(f"threshold fit receipt schema binding is inconsistent: {candidate_id}")
        rows = tuple(score_table[candidate_id])
        row_ids = [row.row_id for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"duplicate score row for candidate: {candidate_id}")
        if set(row_ids) != expected_set:
            raise ValueError(
                f"candidate score table does not match the fixed selector row universe: {candidate_id}"
            )
        ordered = tuple(sorted(rows, key=lambda row: row_order[row.row_id]))
        for row in ordered:
            provenance = row.evidence.get("provenance")
            if row.family != fitted.score_family or not isinstance(provenance, Mapping):
                raise ValueError(f"selector score family/provenance binding mismatch: {candidate_id}")
            if str(provenance.get("modelSha256") or "") != fitted_model_sha256:
                raise ValueError(f"selector score model binding mismatch: {candidate_id}")
            if str(provenance.get("featureSchemaSha256") or "") != fitted_schema_sha256:
                raise ValueError(f"selector score schema binding mismatch: {candidate_id}")
            if _plain_json_value(provenance) != _plain_json_value(fitted_provenance):
                raise ValueError(f"selector score model provenance binding mismatch: {candidate_id}")
        output[candidate_id] = apply_threshold_fit(ordered, fitted)
    return MappingProxyType(output)


def _require_exact_partition(
    expected_rows: Sequence[FeatureRow],
    partitions: Mapping[str, Sequence[FeatureRow]],
    *,
    chronology_name: str,
) -> None:
    expected_ids = [row.row_id for row in expected_rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise DataContractError(f"{chronology_name} source identities are not disjoint")
    expected_set = set(expected_ids)
    observed: set[str] = set()
    for partition_name, rows in partitions.items():
        row_ids = [row.row_id for row in rows]
        row_set = set(row_ids)
        if len(row_ids) != len(row_set) or observed & row_set:
            raise DataContractError(
                f"{chronology_name} partitions are not disjoint at {partition_name}"
            )
        observed.update(row_set)
    if observed != expected_set:
        raise DataContractError(
            f"{chronology_name} partition coverage mismatch: "
            f"missing={len(expected_set - observed)} extra={len(observed - expected_set)}"
        )


def _require_exact_labels(
    labels: Mapping[str, str],
    rows: Sequence[FeatureRow],
    *,
    partition_name: str,
) -> None:
    expected_ids = {row.row_id for row in rows}
    if set(labels) != expected_ids:
        raise DataContractError(f"{partition_name} labels do not match the exact row coverage")


def _labels_for(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
) -> Mapping[str, str]:
    return MappingProxyType({row.row_id: labels[row.row_id] for row in rows})


def _validate_final_refit_freeze(
    bundle: DatasetBundle,
    receipt: FreezeReceipt,
) -> tuple[str, Mapping[str, str]]:
    if not isinstance(receipt, FreezeReceipt):
        raise ExperimentOrderError("a bound frozen policy receipt is required before final refit")
    if hashlib.sha256(receipt.policy_bytes).hexdigest() != receipt.sha256:
        raise ExperimentOrderError("final-refit policy receipt bytes do not match their digest")
    try:
        payload = json.loads(receipt.policy_bytes)
    except (TypeError, json.JSONDecodeError) as error:
        raise ExperimentOrderError("final-refit policy receipt is not valid JSON") from error
    if (
        payload.get("sourceDatasetId") != bundle.metadata.source_dataset_id
        or payload.get("historySha256") != bundle.metadata.source_hashes["historyRows"]
        or tuple(payload.get("selectionMonths") or ()) != tuple(IMMUTABLE_SELECTOR_MONTHS)
        or payload.get("auditInputsIncluded") is not False
    ):
        raise ExperimentOrderError("final-refit policy receipt does not match the prepared selector run")
    selector_ids_sha256 = hashlib.sha256(
        "\n".join(row.row_id for row in bundle.selector).encode("utf-8")
    ).hexdigest()
    binding_payload = {
        "sourceDatasetId": bundle.metadata.source_dataset_id,
        "historySha256": bundle.metadata.source_hashes["historyRows"],
        "selectionMonths": list(IMMUTABLE_SELECTOR_MONTHS),
        "selectorRowIdsSha256": selector_ids_sha256,
        "policySha256": receipt.sha256,
    }
    if hashlib.sha256(_stable_json_bytes(binding_payload)).hexdigest() != receipt.binding_sha256:
        raise ExperimentOrderError("final-refit policy receipt is not bound to the selector row universe")

    model_params = payload.get("modelParams")
    company_assignments = payload.get("companyAssignments")
    if not isinstance(model_params, Mapping) or not isinstance(company_assignments, Mapping):
        raise ExperimentOrderError("candidate definitions and assignments must be frozen before final refit")
    candidate_registry_sha256 = str(model_params.get("candidateRegistrySha256") or "")
    if not _is_sha256(candidate_registry_sha256) or not company_assignments:
        raise ExperimentOrderError("candidate definitions and assignments must be frozen before final refit")
    normalized_assignments = {
        str(company): str(candidate_id)
        for company, candidate_id in sorted(company_assignments.items())
        if str(company).strip() and str(candidate_id).strip()
    }
    if len(normalized_assignments) != len(company_assignments):
        raise ExperimentOrderError("frozen final-refit assignments contain an empty company or candidate")
    return candidate_registry_sha256, MappingProxyType(normalized_assignments)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _normalize_predictions(
    predictions: Mapping[str, str],
    expected_ids: set[str],
    *,
    phase: str,
) -> dict[str, str]:
    allowed_predictions = set(ACTION_LABELS) | set(REVIEW_LABELS)
    normalized = {row_id: str(predictions[row_id]) for row_id in sorted(expected_ids)}
    invalid = sorted({label for label in normalized.values() if label not in allowed_predictions})
    if invalid:
        raise ValueError(f"{phase} predictions contain unknown labels: {invalid}")
    return normalized


def _feature_row_universe_sha256(rows: Sequence[FeatureRow]) -> str:
    payload = {
        "schemaVersion": 1,
        "rows": [
            {
                "rowId": row.row_id,
                "sourceRow": row.source_row,
                "date": row.date,
                "month": row.month,
                "customerKey": row.customer_key,
                "episodeId": row.episode_id,
            }
            for row in sorted(rows, key=lambda item: item.row_id)
        ],
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _materialize_audit_metrics(
    bundle: DatasetBundle,
    chronology: ChronologyPartitions,
    actual_labels: Mapping[str, str],
    predictions: Mapping[str, str],
    *,
    base_policy_sha256: str,
    final_policy_sha256: str,
    lockbox_prediction_sha256: str,
    lockbox_assignment_sha256: str,
    prediction_sha256: str,
) -> MetricMaterialization:
    expected_ids = {row.row_id for row in bundle.audit}
    if set(actual_labels) != expected_ids or set(predictions) != expected_ids:
        raise DataContractError("audit labels and predictions must cover the exact prepared audit universe")
    action_ids = {row_id for row_id, actual in actual_labels.items() if actual in ACTION_LABELS}
    non_action_rows = len(actual_labels) - len(action_ids)
    if len(action_ids) != bundle.metadata.expected_audit_action_rows:
        raise DataContractError("audit action denominator differs from prepared metadata")
    if non_action_rows != bundle.metadata.expected_audit_non_action_rows:
        raise DataContractError("audit non-action denominator differs from prepared metadata")

    warm_products = chronology.warm_product_keys
    outcomes = tuple(
        OutcomeRow(
            row_id=row.row_id,
            customer_key=row.customer_key,
            raw_customer=row.raw_customer,
            legacy_customer_key=row.legacy_customer_key,
            product_key=row.product_key,
            month=row.month,
            episode_id=row.episode_id,
            actual=actual_labels[row.row_id],
            predicted=predictions[row.row_id],
            is_cold=row.product_key not in warm_products,
        )
        for row in bundle.audit
        if row.row_id in action_ids
    )
    global_metrics = compute_metrics(outcomes)
    empty_metrics = compute_metrics(())
    observed_company_metrics = grouped_metrics(outcomes, GroupingBasis.ACTUAL_CUSTOMER)
    company_metrics = {
        company: observed_company_metrics.get(company, empty_metrics)
        for company in sorted({row.customer_key for row in bundle.audit})
    }
    observed_month_metrics = grouped_metrics(outcomes, GroupingBasis.MONTH)
    month_metrics = {
        month: observed_month_metrics.get(month, empty_metrics)
        for month in sorted({row.month for row in bundle.audit})
    }
    observed_warm_cold_metrics = grouped_metrics(outcomes, GroupingBasis.WARM_COLD)
    warm_cold_metrics = {
        basis: observed_warm_cold_metrics.get(basis, empty_metrics)
        for basis in ("cold", "warm")
    }
    payload = {
        "schemaVersion": 2,
        "kind": "retrospective_audit_metrics",
        "evaluationStatus": AUDIT_STATUS,
        "sourceDatasetId": bundle.metadata.source_dataset_id,
        "policySha256": final_policy_sha256,
        "basePolicySha256": base_policy_sha256,
        "finalPolicySha256": final_policy_sha256,
        "lockboxPredictionSha256": lockbox_prediction_sha256,
        "lockboxAssignmentSha256": lockbox_assignment_sha256,
        "predictionSha256": prediction_sha256,
        "auditRawRows": len(bundle.audit),
        "auditActionRows": len(action_ids),
        "auditNonActionRows": non_action_rows,
        "warmColdBasis": WARM_COLD_BASIS,
        "global": _metrics_payload(global_metrics),
        "company": {key: _metrics_payload(value) for key, value in company_metrics.items()},
        "month": {key: _metrics_payload(value) for key, value in month_metrics.items()},
        "warmCold": {key: _metrics_payload(value) for key, value in warm_cold_metrics.items()},
    }
    artifact_bytes = _stable_json_bytes(payload)
    return MetricMaterialization(
        evaluation_rows=len(action_ids),
        non_action_rows=non_action_rows,
        global_metrics=global_metrics,
        company_metrics=MappingProxyType(company_metrics),
        month_metrics=MappingProxyType(month_metrics),
        warm_cold_metrics=MappingProxyType(warm_cold_metrics),
        artifact_bytes=artifact_bytes,
        sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        final_policy_sha256=final_policy_sha256,
        lockbox_assignment_sha256=lockbox_assignment_sha256,
    )


def _metrics_payload(metrics: Metrics) -> Mapping[str, Any]:
    return {
        field.name: _plain_json_value(getattr(metrics, field.name))
        for field in fields(Metrics)
    }


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(nested) for key, nested in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain_json_value(item) for item in value]
    return value


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

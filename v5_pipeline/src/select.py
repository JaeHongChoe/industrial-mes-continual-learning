from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .candidates import CandidatePrediction, ScoreRow
from .data_contract import (
    ACTION_LABELS,
    DatasetBundle,
    IMMUTABLE_LOCKBOX_MONTHS,
    IMMUTABLE_SELECTOR_MONTHS,
)
from .thresholds import (
    ThresholdFit,
    apply_threshold_policy,
    threshold_fit_receipt,
    validate_threshold_fit,
)
from .freeze import PolicySpec
from .risk import (
    EpisodeDecision,
    FrozenHypothesisRegistry,
    evaluate_selector_risk_gates,
)


FALLBACK_POLICY = "review_all"
BASELINE_POLICY = "loop93_baseline"
PROCESS_LABEL = "PROCESS"
RISK_ENDPOINTS = ("ship_to_process", "repair_to_process")
_TEST_UNBOUND_REGISTRY_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class CandidateDefinition:
    candidate_id: str
    family: str
    feature_schema_sha256: str
    feature_names: tuple[str, ...]
    thresholds: Mapping[str, float]
    parameters: Mapping[str, Any]
    calibration_method: str
    warm_cold_policy: str
    risk_budget: float
    union_policy: str
    veto_policy: str
    fallback_policy: str
    fitted_artifact_sha256: str
    complexity: int = 1
    threshold_enabled: Mapping[str, bool] | None = None
    threshold_fit: ThresholdFit | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.candidate_id):
            raise ValueError(f"invalid candidate_id: {self.candidate_id}")
        if not self.family:
            raise ValueError("candidate family must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.feature_schema_sha256):
            raise ValueError("feature_schema_sha256 must be a lowercase SHA-256")
        if (
            not isinstance(self.feature_names, tuple)
            or not self.feature_names
            or any(not isinstance(name, str) or not name.strip() for name in self.feature_names)
            or len(self.feature_names) != len(set(self.feature_names))
        ):
            raise ValueError("feature_names must be a non-empty tuple of unique names")
        if set(self.thresholds) != set(ACTION_LABELS):
            raise ValueError("candidate thresholds must contain exactly the three actions")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.thresholds.values()):
            raise ValueError("candidate thresholds must be finite probabilities")
        threshold_enabled = self.threshold_enabled
        if threshold_enabled is None:
            threshold_enabled = {action: True for action in ACTION_LABELS}
        if set(threshold_enabled) != set(ACTION_LABELS) or any(
            type(value) is not bool for value in threshold_enabled.values()
        ):
            raise ValueError("candidate threshold_enabled must contain exactly three booleans")
        if self.complexity < 0:
            raise ValueError("candidate complexity must be non-negative")
        for field_name in (
            "calibration_method",
            "warm_cold_policy",
            "union_policy",
            "veto_policy",
            "fallback_policy",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"candidate {field_name} must not be empty")
        if not 0 < self.risk_budget < 1:
            raise ValueError("candidate risk_budget must be between zero and one")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fitted_artifact_sha256):
            raise ValueError("fitted_artifact_sha256 must be a lowercase SHA-256")
        if self.threshold_fit is not None:
            validate_threshold_fit(self.threshold_fit)
            calibration = self.threshold_fit.calibrated_model_provenance["calibration"]
            if (
                dict(self.thresholds) != dict(self.threshold_fit.thresholds)
                or dict(threshold_enabled) != dict(self.threshold_fit.enabled)
                or self.family != self.threshold_fit.score_family
                or self.feature_schema_sha256 != self.threshold_fit.feature_schema_sha256
                or self.fitted_artifact_sha256 != self.threshold_fit.fitted_artifact_sha256
                or self.calibration_method != calibration["method"]
            ):
                raise ValueError("candidate definition does not match its threshold fit receipt")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(sorted(self.thresholds.items()))))
        object.__setattr__(
            self,
            "threshold_enabled",
            MappingProxyType(dict(sorted(threshold_enabled.items()))),
        )
        object.__setattr__(self, "parameters", MappingProxyType(dict(sorted(self.parameters.items()))))


@dataclass(frozen=True, slots=True)
class SelectorUniverseRow:
    row_id: str
    episode_id: str
    company: str
    month: str
    actual: str
    baseline_predicted: str


@dataclass(frozen=True, slots=True)
class SelectorUniverse:
    rows: Mapping[str, SelectorUniverseRow]
    universe_bytes: bytes
    sha256: str
    source_dataset_id: str = ""
    history_sha256: str = ""
    phase: str = ""
    expected_row_count: int = 0
    baseline_prediction_source_sha256: str = ""
    source_bound: bool = False

    @classmethod
    def freeze(cls, rows: Iterable[SelectorUniverseRow]) -> "SelectorUniverse":
        materialized = tuple(rows)
        if not materialized:
            raise ValueError("selector universe cannot be empty")
        by_id: dict[str, SelectorUniverseRow] = {}
        for row in materialized:
            if not row.row_id or not row.episode_id or not row.company or not row.month:
                raise ValueError("selector universe rows require identity, episode, company, and month")
            if row.actual not in ACTION_LABELS:
                raise ValueError(f"unknown selector universe actual label: {row.actual}")
            if row.baseline_predicted not in ACTION_LABELS and row.baseline_predicted not in {
                "", "REVIEW", "REVIEW_NEEDED", "REVIEW_REQUIRED"
            }:
                raise ValueError(f"unknown selector universe baseline label: {row.baseline_predicted}")
            if row.row_id in by_id:
                raise ValueError(f"duplicate selector universe row: {row.row_id}")
            by_id[row.row_id] = row
        payload = {
            "schemaVersion": 1,
            "rows": [
                {
                    "rowId": row.row_id,
                    "episodeId": row.episode_id,
                    "company": row.company,
                    "month": row.month,
                    "actual": row.actual,
                    "baselinePredicted": row.baseline_predicted,
                }
                for row in sorted(materialized, key=lambda value: value.row_id)
            ],
        }
        universe_bytes = _stable_json_bytes(payload)
        return cls(
            rows=MappingProxyType(dict(sorted(by_id.items()))),
            universe_bytes=universe_bytes,
            sha256=hashlib.sha256(universe_bytes).hexdigest(),
        )

    @classmethod
    def from_prepared_bundle(
        cls,
        bundle: DatasetBundle,
        *,
        phase: str,
        baseline_manifest_path: str | Path,
        lockbox_labels: Mapping[str, str] | None = None,
    ) -> "SelectorUniverse":
        if not isinstance(bundle, DatasetBundle):
            raise TypeError("bundle must be a prepared DatasetBundle")
        if phase == "selector":
            prepared_rows = bundle.selector
            labels = bundle.development_labels("selector")
            allowed_months = set(IMMUTABLE_SELECTOR_MONTHS)
        elif phase == "lockbox":
            prepared_rows = bundle.lockbox
            if lockbox_labels is None:
                raise ValueError("lockbox labels must be opened before freezing its universe")
            labels = lockbox_labels
            allowed_months = set(IMMUTABLE_LOCKBOX_MONTHS)
        else:
            raise ValueError("phase must be selector or lockbox")
        if not prepared_rows:
            raise ValueError(f"prepared {phase} rows cannot be empty")
        manifest_path = Path(baseline_manifest_path).resolve()
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("baseline prediction manifest is not readable JSON") from error
        if not isinstance(manifest, Mapping):
            raise ValueError("baseline prediction manifest must be an object")
        predictions = manifest.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError("baseline prediction manifest requires predictions")
        expected_count = int(manifest.get("expectedRowCount") or -1)
        if (
            manifest.get("schemaVersion") != 1
            or manifest.get("kind") != "baseline_prediction_manifest"
            or manifest.get("phase") != phase
            or manifest.get("sourceDatasetId") != bundle.metadata.source_dataset_id
            or manifest.get("historySha256") != bundle.metadata.source_hashes["historyRows"]
        ):
            raise ValueError("baseline prediction manifest does not match the prepared source")
        prepared_ids = {row.row_id for row in prepared_rows}
        if expected_count != len(prepared_rows) or set(predictions) != prepared_ids:
            raise ValueError("baseline prediction manifest does not match prepared selector rows")
        if set(labels) != prepared_ids:
            raise ValueError(f"prepared {phase} labels do not match prepared rows")
        universe_rows = []
        for row in prepared_rows:
            actual = labels[row.row_id]
            baseline = predictions[row.row_id]
            if row.month not in allowed_months or actual not in ACTION_LABELS:
                raise ValueError(f"prepared {phase} row violates immutable phase contract")
            universe_rows.append(
                SelectorUniverseRow(
                    row_id=row.row_id,
                    episode_id=row.episode_id,
                    company=row.customer_key,
                    month=row.month,
                    actual=actual,
                    baseline_predicted=str(baseline),
                )
            )
        frozen = cls.freeze(universe_rows)
        return cls(
            rows=frozen.rows,
            universe_bytes=frozen.universe_bytes,
            sha256=frozen.sha256,
            source_dataset_id=bundle.metadata.source_dataset_id,
            history_sha256=bundle.metadata.source_hashes["historyRows"],
            phase=phase,
            expected_row_count=expected_count,
            baseline_prediction_source_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            source_bound=True,
        )


@dataclass(frozen=True, slots=True)
class CandidateRegistry:
    definitions: Mapping[str, CandidateDefinition]
    expected_companies: frozenset[str]
    selector_months: tuple[str, ...]
    selector_universe_sha256: str
    selection_mode: str
    fallback_policy: str
    source_dataset_id: str
    history_sha256: str
    verified_artifact_receipts: Mapping[str, Mapping[str, str]]
    registry_bytes: bytes
    sha256: str
    risk_registry: FrozenHypothesisRegistry

    @classmethod
    def freeze_unbound_for_tests(
        cls,
        definitions: Iterable[CandidateDefinition],
        *,
        expected_companies: frozenset[str],
        selector_months: tuple[str, ...],
        selector_universe: SelectorUniverse,
        selection_mode: str,
        artifact_manifests: Mapping[str, Mapping[str, str | Path]] | None = None,
    ) -> "CandidateRegistry":
        """Compatibility constructor restricted to synthetic unit-test universes."""

        return cls.freeze(
            definitions,
            expected_companies=expected_companies,
            selector_months=selector_months,
            selector_universe=selector_universe,
            selection_mode=selection_mode,
            artifact_manifests=artifact_manifests,
            _test_unbound_token=_TEST_UNBOUND_REGISTRY_SENTINEL,
        )

    @classmethod
    def freeze(
        cls,
        definitions: Iterable[CandidateDefinition],
        *,
        expected_companies: frozenset[str],
        selector_months: tuple[str, ...],
        selector_universe: SelectorUniverse,
        selection_mode: str,
        artifact_manifests: Mapping[str, Mapping[str, str | Path]] | None = None,
        _test_unbound_token: object | None = None,
    ) -> "CandidateRegistry":
        materialized = tuple(definitions)
        if not materialized:
            raise ValueError("candidate registry cannot be empty")
        by_id = {definition.candidate_id: definition for definition in materialized}
        if len(by_id) != len(materialized):
            raise ValueError("candidate IDs must be unique")
        if not isinstance(expected_companies, frozenset) or not expected_companies:
            raise ValueError("expected_companies must be a non-empty frozenset")
        if tuple(selector_months) != tuple(IMMUTABLE_SELECTOR_MONTHS):
            raise ValueError("selector months violate immutable chronology")
        if not isinstance(selector_universe, SelectorUniverse):
            raise TypeError("selector_universe must be a frozen SelectorUniverse")
        if (
            not selector_universe.source_bound
            and _test_unbound_token is not _TEST_UNBOUND_REGISTRY_SENTINEL
        ):
            raise ValueError(
                "production candidate registries require a source-bound prepared selector universe"
            )
        if selection_mode not in {"standalone_clean", "monotone_overlay"}:
            raise ValueError("selection_mode must be standalone_clean or monotone_overlay")
        expected_fallback = (
            FALLBACK_POLICY if selection_mode == "standalone_clean" else BASELINE_POLICY
        )
        if any(item.fallback_policy != expected_fallback for item in materialized):
            raise ValueError(
                f"{selection_mode} candidates must use fallback_policy={expected_fallback}"
            )
        if any(row.company not in expected_companies for row in selector_universe.rows.values()):
            raise ValueError("selector universe contains a company outside expected_companies")
        if any(row.month not in selector_months for row in selector_universe.rows.values()):
            raise ValueError("selector universe contains a row outside selector months")
        if selection_mode == "standalone_clean" and any(
            row.baseline_predicted in ACTION_LABELS for row in selector_universe.rows.values()
        ):
            raise ValueError("standalone_clean universe must use review-only baseline state")
        _verify_threshold_fits(materialized, selector_universe)
        verified_artifact_receipts = _verify_artifact_manifests(
            materialized,
            artifact_manifests,
            required=selector_universe.source_bound,
        )
        hypothesis_ids = _hypothesis_ids(by_id, expected_companies, tuple(selector_months))
        payload = {
            "schemaVersion": 1,
            "definitions": [
                _candidate_definition_payload(item)
                for item in sorted(materialized, key=lambda value: value.candidate_id)
            ],
            "expectedCompanies": sorted(expected_companies),
            "selectorMonths": list(selector_months),
            "selectorUniverseSha256": selector_universe.sha256,
            "selectorUniverse": {
                "sourceDatasetId": selector_universe.source_dataset_id,
                "historySha256": selector_universe.history_sha256,
                "phase": selector_universe.phase,
                "expectedRowCount": selector_universe.expected_row_count,
                "baselinePredictionSourceSha256": selector_universe.baseline_prediction_source_sha256,
                "sourceBound": selector_universe.source_bound,
            },
            "selectionMode": selection_mode,
            "fallbackPolicy": expected_fallback,
            "riskEndpoints": list(RISK_ENDPOINTS),
            "hypothesisIds": list(hypothesis_ids),
            "mAll": len(hypothesis_ids),
            "verifiedArtifactReceipts": {
                candidate_id: dict(receipt)
                for candidate_id, receipt in sorted(verified_artifact_receipts.items())
            },
        }
        registry_bytes = _stable_json_bytes(payload)
        digest = hashlib.sha256(registry_bytes).hexdigest()
        return cls(
            definitions=MappingProxyType(dict(sorted(by_id.items()))),
            expected_companies=expected_companies,
            selector_months=tuple(selector_months),
            selector_universe_sha256=selector_universe.sha256,
            selection_mode=selection_mode,
            fallback_policy=expected_fallback,
            source_dataset_id=selector_universe.source_dataset_id,
            history_sha256=selector_universe.history_sha256,
            verified_artifact_receipts=MappingProxyType(
                {candidate_id: MappingProxyType(dict(receipt)) for candidate_id, receipt in verified_artifact_receipts.items()}
            ),
            registry_bytes=registry_bytes,
            sha256=digest,
            risk_registry=FrozenHypothesisRegistry(
                registry_sha256=digest,
                hypothesis_ids=hypothesis_ids,
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    candidate_id: str
    row_id: str
    episode_id: str
    company: str
    month: str
    actual: str
    predicted: str


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    risk_budget: float = 0.005
    minimum_rows_per_month: int = 30
    accuracy_floor: float = 0.98
    max_accuracy_drop: float = 0.001
    alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class SelectionResult:
    registry_sha256: str
    candidate_decisions_sha256: str
    strict_portfolio_sha256: str
    shadow_portfolio_sha256: str
    strict_assignments: Mapping[str, str]
    shadow_assignments: Mapping[str, str]
    status_by_company: Mapping[str, str]
    rejections: Mapping[str, Mapping[str, tuple[str, ...]]]
    scores: Mapping[str, Mapping[str, tuple[float, int, int, int]]]
    selector_alpha: float
    accuracy_floor: float
    max_accuracy_drop: float
    effective_risk_budget_by_candidate: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LockboxResult:
    assignment_mode: str
    assignments: Mapping[str, str]
    rejection_reasons: Mapping[str, tuple[str, ...]]
    lockbox_universe_sha256: str
    assignment_sha256: str


def select_company_policies(
    registry: CandidateRegistry,
    decisions: Iterable[CandidateDecision | Mapping[str, Any]],
    config: SelectorConfig,
    *,
    selector_universe: SelectorUniverse,
) -> SelectionResult:
    _validate_registry_integrity(registry)
    _validate_selector_config(config)
    if selector_universe.sha256 != registry.selector_universe_sha256:
        raise ValueError("selector universe does not match the frozen registry")
    materialized = tuple(_coerce_decision(item) for item in decisions)
    _validate_candidate_decisions(registry, materialized, allowed_months=set(registry.selector_months))
    _validate_fixed_row_universe(registry, materialized, selector_universe)
    by_candidate = {
        candidate_id: tuple(item for item in materialized if item.candidate_id == candidate_id)
        for candidate_id in registry.definitions
    }
    risk_by_candidate = {}
    for candidate_id, candidate_rows in by_candidate.items():
        risk_by_candidate[candidate_id] = evaluate_selector_risk_gates(
            (
                EpisodeDecision(
                    row_id=item.row_id,
                    episode_id=item.episode_id,
                    company=item.company,
                    month=item.month,
                    actual=item.actual,
                    predicted=item.predicted,
                    is_added=(
                        item.predicted == PROCESS_LABEL
                        and (
                            registry.selection_mode == "standalone_clean"
                            or selector_universe.rows[item.row_id].baseline_predicted
                            not in ACTION_LABELS
                        )
                    ),
                )
                for item in candidate_rows
            ),
            budget=min(config.risk_budget, registry.definitions[candidate_id].risk_budget),
            registry=registry.risk_registry,
            expected_companies=registry.expected_companies,
            alpha=config.alpha,
            required_months=registry.selector_months,
        )

    strict_assignments: dict[str, str] = {}
    shadow_assignments: dict[str, str] = {}
    statuses: dict[str, str] = {}
    all_rejections: dict[str, Mapping[str, tuple[str, ...]]] = {}
    all_scores: dict[str, Mapping[str, tuple[float, int, int, int]]] = {}
    for company in sorted(registry.expected_companies):
        strict_candidates: list[tuple[tuple[float, int, int, int, str], str]] = []
        shadow_candidates: list[tuple[tuple[float, int, int, int, str], str]] = []
        company_rejections: dict[str, tuple[str, ...]] = {}
        company_scores: dict[str, tuple[float, int, int, int]] = {}
        company_has_any_rows = False
        for candidate_id, definition in registry.definitions.items():
            candidate_rows = [item for item in by_candidate[candidate_id] if item.company == company]
            company_has_any_rows = company_has_any_rows or bool(candidate_rows)
            empirical_reasons: list[str] = []
            month_recoveries: list[float] = []
            total_correct_process = 0
            total_auto = 0
            total_correct_added_auto = 0
            total_rows = 0
            for month in registry.selector_months:
                month_rows = [item for item in candidate_rows if item.month == month]
                total_rows += len(month_rows)
                if len(month_rows) < config.minimum_rows_per_month:
                    empirical_reasons.append(f"{month}:minimum_rows")
                    month_recoveries.append(0.0)
                    continue
                if registry.selection_mode == "monotone_overlay":
                    for item in month_rows:
                        universe_row = selector_universe.rows[item.row_id]
                        if (
                            universe_row.baseline_predicted in ACTION_LABELS
                            and item.predicted != universe_row.baseline_predicted
                        ):
                            empirical_reasons.append(f"{month}:baseline_auto_changed")
                automatic = [item for item in month_rows if item.predicted in ACTION_LABELS]
                correct = sum(item.actual == item.predicted for item in automatic)
                accuracy = correct / len(automatic) if automatic else 0.0
                if accuracy < config.accuracy_floor:
                    empirical_reasons.append(f"{month}:accuracy_below_floor")
                if registry.selection_mode == "monotone_overlay":
                    baseline_rows = [
                        selector_universe.rows[item.row_id]
                        for item in month_rows
                        if selector_universe.rows[item.row_id].baseline_predicted in ACTION_LABELS
                    ]
                    baseline_accuracy = (
                        sum(row.actual == row.baseline_predicted for row in baseline_rows)
                        / len(baseline_rows)
                        if baseline_rows
                        else 0.0
                    )
                    if baseline_rows and accuracy < baseline_accuracy - config.max_accuracy_drop:
                        empirical_reasons.append(f"{month}:accuracy_noninferiority")
                actual_process = sum(item.actual == PROCESS_LABEL for item in month_rows)
                correct_process = sum(
                    item.actual == PROCESS_LABEL
                    and _is_added_process(item, registry, selector_universe)
                    for item in month_rows
                )
                recovery = correct_process / actual_process if actual_process else 0.0
                if correct_process == 0:
                    empirical_reasons.append(f"{month}:no_correct_process_recovery")
                month_recoveries.append(recovery)
                total_correct_process += correct_process
                total_auto += len(automatic)
                for item in automatic:
                    universe_row = selector_universe.rows[item.row_id]
                    if (
                        universe_row.baseline_predicted not in ACTION_LABELS
                        and item.actual == item.predicted
                    ):
                        total_correct_added_auto += 1
            risk_gate = risk_by_candidate[candidate_id][company]
            for month, gate in risk_gate.month_gates.items():
                if gate.summary.ship_to_process_episodes:
                    empirical_reasons.append(f"{month}:ship_to_process")
                if gate.summary.repair_to_process_episodes:
                    empirical_reasons.append(f"{month}:repair_to_process")
            empirical_reasons = sorted(set(empirical_reasons))
            strict_reasons = list(empirical_reasons)
            if not risk_gate.passed:
                strict_reasons.extend(
                    reason
                    for reason in risk_gate.rejection_reasons
                    if reason not in strict_reasons
                )
            score = (
                min(month_recoveries) if month_recoveries else 0.0,
                total_correct_process,
                total_correct_added_auto,
                -definition.complexity,
                candidate_id,
            )
            company_scores[candidate_id] = score[:4]
            company_rejections[candidate_id] = tuple(sorted(set(strict_reasons)))
            if not empirical_reasons:
                shadow_candidates.append((score, candidate_id))
            if not strict_reasons:
                strict_candidates.append((score, candidate_id))

        strict_id = max(strict_candidates)[1] if strict_candidates else registry.fallback_policy
        shadow_id = max(shadow_candidates)[1] if shadow_candidates else registry.fallback_policy
        strict_assignments[company] = strict_id
        shadow_assignments[company] = shadow_id
        if not company_has_any_rows:
            statuses[company] = "fallback_missing_company"
        elif strict_id != registry.fallback_policy:
            statuses[company] = "retrospective_strict_requires_future_shadow"
        elif shadow_id != registry.fallback_policy:
            statuses[company] = "retrospective_shadow_insufficient_bound"
        else:
            statuses[company] = "fallback_rejected"
        all_rejections[company] = MappingProxyType(company_rejections)
        all_scores[company] = MappingProxyType(company_scores)

    candidate_decisions_sha256 = _candidate_decisions_sha256(materialized)
    strict_portfolio_sha256 = _portfolio_sha256(
        registry.sha256,
        strict_assignments,
        materialized,
        registry.fallback_policy,
    )
    shadow_portfolio_sha256 = _portfolio_sha256(
        registry.sha256,
        shadow_assignments,
        materialized,
        registry.fallback_policy,
    )
    return SelectionResult(
        registry_sha256=registry.sha256,
        candidate_decisions_sha256=candidate_decisions_sha256,
        strict_portfolio_sha256=strict_portfolio_sha256,
        shadow_portfolio_sha256=shadow_portfolio_sha256,
        strict_assignments=MappingProxyType(strict_assignments),
        shadow_assignments=MappingProxyType(shadow_assignments),
        status_by_company=MappingProxyType(statuses),
        rejections=MappingProxyType(all_rejections),
        scores=MappingProxyType(all_scores),
        selector_alpha=config.alpha,
        accuracy_floor=config.accuracy_floor,
        max_accuracy_drop=config.max_accuracy_drop,
        effective_risk_budget_by_candidate=MappingProxyType(
            {
                candidate_id: min(config.risk_budget, definition.risk_budget)
                for candidate_id, definition in registry.definitions.items()
            }
        ),
    )


def apply_reject_only_lockbox(
    selection: SelectionResult,
    registry: CandidateRegistry,
    decisions: Iterable[CandidateDecision | Mapping[str, Any]],
    *,
    risk_budget: float | None = None,
    accuracy_floor: float | None = None,
    assignment_mode: str,
    lockbox_universe: SelectorUniverse,
) -> LockboxResult:
    _validate_registry_integrity(registry)
    if selection.registry_sha256 != registry.sha256:
        raise ValueError("selection and registry hashes do not match")
    if assignment_mode == "strict":
        source_assignments = selection.strict_assignments
    elif assignment_mode == "shadow":
        source_assignments = selection.shadow_assignments
    else:
        raise ValueError("assignment_mode must be strict or shadow")
    if not isinstance(lockbox_universe, SelectorUniverse):
        raise TypeError("lockbox_universe must be a frozen SelectorUniverse")
    if registry.source_dataset_id and (
        not lockbox_universe.source_bound
        or lockbox_universe.phase != "lockbox"
        or lockbox_universe.source_dataset_id != registry.source_dataset_id
        or lockbox_universe.history_sha256 != registry.history_sha256
    ):
        raise ValueError("lockbox universe does not match the frozen prepared source")
    if any(row.month != "2025-12" for row in lockbox_universe.rows.values()):
        raise ValueError("lockbox universe must contain only 2025-12 rows")
    unexpected_rows = [
        row
        for row in lockbox_universe.rows.values()
        if row.company not in registry.expected_companies
    ]
    if unexpected_rows and (
        registry.selection_mode != "standalone_clean"
        or registry.fallback_policy != FALLBACK_POLICY
        or any(row.baseline_predicted in ACTION_LABELS for row in unexpected_rows)
    ):
        raise ValueError("lockbox universe contains an unexpected company")
    materialized = tuple(_coerce_decision(item) for item in decisions)
    _validate_candidate_decisions(registry, materialized, allowed_months={"2025-12"})
    assignments: dict[str, str] = {}
    rejection_reasons: dict[str, tuple[str, ...]] = {}
    selected_owners: dict[str, tuple[str, str]] = {}
    for item in materialized:
        if source_assignments.get(item.company) != item.candidate_id:
            continue
        owner = (item.company, item.month)
        previous_owner = selected_owners.setdefault(item.episode_id, owner)
        if previous_owner != owner:
            raise ValueError(
                "selected lockbox episode ID is shared across companies: "
                f"{item.episode_id} belongs to {previous_owner} and {owner}"
            )
    adjusted_alpha = selection.selector_alpha / registry.risk_registry.hypothesis_count
    for company in sorted(registry.expected_companies):
        selected = source_assignments[company]
        if selected == registry.fallback_policy:
            assignments[company] = registry.fallback_policy
            rejection_reasons[company] = ("selector_fallback",)
            continue
        expected_rows = {
            row_id: (row.episode_id, row.company, row.month, row.actual)
            for row_id, row in lockbox_universe.rows.items()
            if row.company == company
        }
        current_rows = {
            item.row_id: (item.episode_id, item.company, item.month, item.actual)
            for item in materialized
            if item.company == company and item.candidate_id == selected
        }
        if current_rows != expected_rows:
            raise ValueError(
                "selected candidate predictions do not match the authoritative lockbox universe: "
                f"company={company} candidate={selected} "
                f"expected_rows={len(expected_rows)} rows={len(current_rows)}"
            )
        rows = [
            item
            for item in materialized
            if item.company == company and item.candidate_id == selected and item.month == "2025-12"
        ]
        reasons: list[str] = []
        if not rows:
            reasons.append("missing_lockbox_rows")
        automatic = [item for item in rows if item.predicted in ACTION_LABELS]
        accuracy = (
            sum(item.actual == item.predicted for item in automatic) / len(automatic)
            if automatic
            else 0.0
        )
        if accuracy < selection.accuracy_floor:
            reasons.append("lockbox_accuracy_below_floor")
        if registry.selection_mode == "monotone_overlay":
            baseline_rows = []
            for item in rows:
                baseline = lockbox_universe.rows[item.row_id]
                if baseline.baseline_predicted in ACTION_LABELS:
                    baseline_rows.append(baseline)
                    if item.predicted != baseline.baseline_predicted:
                        reasons.append("baseline_auto_changed")
            baseline_accuracy = (
                sum(row.actual == row.baseline_predicted for row in baseline_rows) / len(baseline_rows)
                if baseline_rows
                else 0.0
            )
            if baseline_rows and accuracy < baseline_accuracy - selection.max_accuracy_drop:
                reasons.append("lockbox_accuracy_noninferiority")
        added_process_rows = [
            item for item in rows if _is_added_process(item, registry, lockbox_universe)
        ]
        ship_risk = sum(item.actual == "RETURN" for item in added_process_rows)
        repair_risk = sum(item.actual == "REPAIR" for item in added_process_rows)
        if ship_risk:
            reasons.append("lockbox_ship_to_process")
        if repair_risk:
            reasons.append("lockbox_repair_to_process")
        process_episodes = {
            item.episode_id for item in added_process_rows
        }
        unsafe_episodes = {
            item.episode_id for item in added_process_rows if item.actual in {"RETURN", "REPAIR"}
        }
        if process_episodes:
            from .risk import U_CP

            if (
                U_CP(len(unsafe_episodes), len(process_episodes), adjusted_alpha)
                > selection.effective_risk_budget_by_candidate[selected]
            ):
                reasons.append("lockbox_ucb_exceeds_budget")
        else:
            reasons.append("lockbox_no_process_support")
        assignments[company] = registry.fallback_policy if reasons else selected
        rejection_reasons[company] = tuple(sorted(set(reasons)))
    assignment_sha256 = hashlib.sha256(
        _stable_json_bytes(
            {
                "schemaVersion": 1,
                "registrySha256": registry.sha256,
                "lockboxUniverseSha256": lockbox_universe.sha256,
                "assignmentMode": assignment_mode,
                "assignments": dict(sorted(assignments.items())),
                "rejections": {company: list(rejection_reasons[company]) for company in sorted(rejection_reasons)},
            }
        )
    ).hexdigest()
    return LockboxResult(
        assignment_mode=assignment_mode,
        assignments=MappingProxyType(assignments),
        rejection_reasons=MappingProxyType(rejection_reasons),
        lockbox_universe_sha256=lockbox_universe.sha256,
        assignment_sha256=assignment_sha256,
    )


def build_policy_spec(
    selection: SelectionResult,
    registry: CandidateRegistry,
    *,
    source_dataset_id: str,
    history_sha256: str,
    assignment_mode: str = "strict",
) -> PolicySpec:
    _validate_registry_integrity(registry)
    if selection.registry_sha256 != registry.sha256:
        raise ValueError("selection and registry hashes do not match")
    if registry.source_dataset_id and (
        source_dataset_id != registry.source_dataset_id or history_sha256 != registry.history_sha256
    ):
        raise ValueError("policy source binding does not match the source-bound registry")
    if assignment_mode == "strict":
        assignments = selection.strict_assignments
    elif assignment_mode == "shadow":
        assignments = selection.shadow_assignments
    else:
        raise ValueError("assignment_mode must be strict or shadow")
    selected_ids = sorted(set(assignments.values()) - {registry.fallback_policy})
    selected_feature_names = tuple(
        dict.fromkeys(
            feature_name
            for candidate_id in selected_ids
            for feature_name in registry.definitions[candidate_id].feature_names
        )
    )
    portfolio_sha256 = (
        selection.strict_portfolio_sha256
        if assignment_mode == "strict"
        else selection.shadow_portfolio_sha256
    )
    thresholds = {
        f"{candidate_id}:{action}": registry.definitions[candidate_id].thresholds[action]
        for candidate_id in selected_ids
        for action in sorted(ACTION_LABELS)
    }
    return PolicySpec(
        policy_name=f"clean-selector-{assignment_mode}",
        source_dataset_id=source_dataset_id,
        history_sha256=history_sha256,
        selection_months=registry.selector_months,
        feature_names=selected_feature_names,
        model_params={
            "candidateRegistrySha256": registry.sha256,
            "candidateDecisionsSha256": selection.candidate_decisions_sha256,
            "portfolioSha256": portfolio_sha256,
            "assignmentMode": assignment_mode,
            "promotionStatus": "requires_future_shadow",
            "selectorUniverseSha256": registry.selector_universe_sha256,
            "selectionMode": registry.selection_mode,
            "selectedFittedArtifacts": {
                candidate_id: registry.definitions[candidate_id].fitted_artifact_sha256
                for candidate_id in selected_ids
            },
            "selectedFeatureSchemas": {
                candidate_id: registry.definitions[candidate_id].feature_schema_sha256
                for candidate_id in selected_ids
            },
            "selectedThresholdEnabled": {
                candidate_id: dict(registry.definitions[candidate_id].threshold_enabled)
                for candidate_id in selected_ids
            },
            "selectedThresholdFitReceipts": {
                candidate_id: dict(
                    threshold_fit_receipt(registry.definitions[candidate_id].threshold_fit)
                )
                for candidate_id in selected_ids
                if registry.definitions[candidate_id].threshold_fit is not None
            },
            "selectedVerifiedArtifactReceipts": {
                candidate_id: dict(registry.verified_artifact_receipts[candidate_id])
                for candidate_id in selected_ids
                if candidate_id in registry.verified_artifact_receipts
            },
        },
        thresholds=thresholds,
        company_assignments=dict(assignments),
    )


def apply_candidate_thresholds(
    score_rows: Iterable[ScoreRow],
    definition: CandidateDefinition,
) -> tuple[CandidatePrediction, ...]:
    fitted = definition.threshold_fit
    return apply_threshold_policy(
        score_rows,
        thresholds=definition.thresholds,
        enabled=definition.threshold_enabled,
        policy=fitted.policy if fitted is not None else "candidate_definition",
        threshold_fit_sha256=fitted.sha256 if fitted is not None else None,
    )


def _candidate_definition_payload(definition: CandidateDefinition) -> dict[str, Any]:
    return {
        "candidateId": definition.candidate_id,
        "family": definition.family,
        "featureSchemaSha256": definition.feature_schema_sha256,
        "featureNames": list(definition.feature_names),
        "thresholds": dict(definition.thresholds),
        "thresholdEnabled": dict(definition.threshold_enabled),
        "thresholdFitReceipt": (
            dict(threshold_fit_receipt(definition.threshold_fit))
            if definition.threshold_fit is not None
            else None
        ),
        "parameters": dict(definition.parameters),
        "calibrationMethod": definition.calibration_method,
        "warmColdPolicy": definition.warm_cold_policy,
        "riskBudget": definition.risk_budget,
        "unionPolicy": definition.union_policy,
        "vetoPolicy": definition.veto_policy,
        "fallbackPolicy": definition.fallback_policy,
        "fittedArtifactSha256": definition.fitted_artifact_sha256,
        "complexity": definition.complexity,
    }


def _coerce_decision(value: CandidateDecision | Mapping[str, Any]) -> CandidateDecision:
    if isinstance(value, CandidateDecision):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("candidate decisions must be mappings or CandidateDecision values")
    return CandidateDecision(**value)


def _validate_candidate_decisions(
    registry: CandidateRegistry,
    decisions: tuple[CandidateDecision, ...],
    *,
    allowed_months: set[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    for item in decisions:
        if item.candidate_id not in registry.definitions:
            raise ValueError(f"decision references unknown candidate: {item.candidate_id}")
        if item.company not in registry.expected_companies:
            raise ValueError(f"decision references unexpected company: {item.company}")
        if item.month not in allowed_months:
            raise ValueError(f"decision month is outside the allowed phase: {item.month}")
        if item.actual not in ACTION_LABELS:
            raise ValueError(f"unknown actual action label: {item.actual}")
        if item.predicted not in ACTION_LABELS and item.predicted not in {"", "REVIEW", "REVIEW_NEEDED", "REVIEW_REQUIRED"}:
            raise ValueError(f"unknown predicted label: {item.predicted}")
        identity = (item.candidate_id, item.row_id)
        if identity in seen:
            raise ValueError(f"duplicate candidate row decision: {identity}")
        seen.add(identity)


def _validate_fixed_row_universe(
    registry: CandidateRegistry,
    decisions: tuple[CandidateDecision, ...],
    selector_universe: SelectorUniverse,
) -> None:
    expected = {
        row_id: (row.episode_id, row.company, row.month, row.actual)
        for row_id, row in selector_universe.rows.items()
    }
    for candidate_id in registry.definitions:
        current = {
            item.row_id: (item.episode_id, item.company, item.month, item.actual)
            for item in decisions
            if item.candidate_id == candidate_id
        }
        if current != expected:
            raise ValueError(
                "candidate predictions do not match the authoritative fixed row universe: "
                f"candidate={candidate_id} expected_rows={len(expected)} rows={len(current)}"
            )


def _validate_selector_config(config: SelectorConfig) -> None:
    if not 0 < config.risk_budget < 1:
        raise ValueError("risk_budget must be between zero and one")
    if config.minimum_rows_per_month < 1:
        raise ValueError("minimum_rows_per_month must be positive")
    if not 0 <= config.accuracy_floor <= 1:
        raise ValueError("accuracy_floor must be between zero and one")
    if not 0 <= config.max_accuracy_drop <= 1:
        raise ValueError("max_accuracy_drop must be between zero and one")
    if not 0 < config.alpha < 1:
        raise ValueError("alpha must be between zero and one")


def _verify_artifact_manifests(
    definitions: tuple[CandidateDefinition, ...],
    manifests: Mapping[str, Mapping[str, str | Path]] | None,
    *,
    required: bool,
) -> Mapping[str, Mapping[str, str]]:
    if manifests is None:
        if required:
            raise ValueError("source-bound registry requires verified artifact manifests")
        return MappingProxyType({})
    if set(manifests) != {definition.candidate_id for definition in definitions}:
        raise ValueError("artifact manifests must cover every candidate exactly once")
    receipts: dict[str, Mapping[str, str]] = {}
    for definition in definitions:
        entry = manifests[definition.candidate_id]
        if not isinstance(entry, Mapping) or set(entry) != {"feature_schema", "fitted_artifact"}:
            raise ValueError("artifact manifests require feature_schema and fitted_artifact paths")
        schema_path = Path(entry["feature_schema"]).resolve()
        fitted_manifest_path = Path(entry["fitted_artifact"]).resolve()
        try:
            schema_bytes = schema_path.read_bytes()
            schema_manifest = json.loads(schema_bytes)
            fitted_manifest_bytes = fitted_manifest_path.read_bytes()
            fitted_manifest = json.loads(fitted_manifest_bytes)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("artifact manifests must be readable JSON") from error
        schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()
        if (
            not isinstance(schema_manifest, Mapping)
            or schema_manifest.get("schemaVersion") != 1
            or schema_manifest.get("kind") != "feature_schema"
            or tuple(schema_manifest.get("featureNames") or ()) != definition.feature_names
            or schema_sha256 != definition.feature_schema_sha256
        ):
            raise ValueError("feature schema manifest does not verify the candidate claim")
        if not isinstance(fitted_manifest, Mapping):
            raise ValueError("fitted artifact manifest must be an object")
        artifact_file = fitted_manifest.get("artifactFile")
        artifact_sha256 = fitted_manifest.get("artifactSha256")
        if (
            fitted_manifest.get("schemaVersion") != 1
            or fitted_manifest.get("kind") != "fitted_artifact"
            or fitted_manifest.get("featureSchemaSha256") != definition.feature_schema_sha256
            or not isinstance(artifact_file, str)
            or artifact_sha256 != definition.fitted_artifact_sha256
        ):
            raise ValueError("fitted artifact manifest does not verify the candidate claim")
        artifact_path = (fitted_manifest_path.parent / artifact_file).resolve()
        try:
            observed_artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError("fitted artifact bytes are unavailable") from error
        if observed_artifact_sha256 != artifact_sha256:
            raise ValueError("fitted artifact bytes do not match the persisted manifest")
        receipts[definition.candidate_id] = MappingProxyType(
            {
                "featureSchemaSha256": schema_sha256,
                "featureSchemaManifestSha256": hashlib.sha256(schema_bytes).hexdigest(),
                "fittedArtifactSha256": observed_artifact_sha256,
                "fittedArtifactManifestSha256": hashlib.sha256(fitted_manifest_bytes).hexdigest(),
            }
        )
    return MappingProxyType(receipts)


def _verify_threshold_fits(
    definitions: tuple[CandidateDefinition, ...],
    selector_universe: SelectorUniverse,
) -> None:
    for definition in definitions:
        fitted = definition.threshold_fit
        if fitted is None:
            continue
        validate_threshold_fit(fitted)
        if selector_universe.source_bound and (
            fitted.source_dataset_id != selector_universe.source_dataset_id
            or fitted.history_sha256 != selector_universe.history_sha256
        ):
            raise ValueError("threshold fit receipt source does not match the selector universe")


def _validate_registry_integrity(registry: CandidateRegistry) -> None:
    if hashlib.sha256(registry.registry_bytes).hexdigest() != registry.sha256:
        raise ValueError("registry bytes do not match the frozen registry digest")
    expected_hypothesis_ids = _hypothesis_ids(
        registry.definitions,
        registry.expected_companies,
        registry.selector_months,
    )
    if (
        registry.risk_registry.registry_sha256 != registry.sha256
        or registry.risk_registry.hypothesis_ids != expected_hypothesis_ids
    ):
        raise ValueError("stale hypothesis family metadata does not match the frozen registry")
    try:
        payload = json.loads(registry.registry_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("registry bytes are not valid JSON") from error
    expected_definitions = [
        _candidate_definition_payload(definition)
        for definition in sorted(
            registry.definitions.values(),
            key=lambda value: value.candidate_id,
        )
    ]
    if payload.get("definitions") != expected_definitions:
        raise ValueError("registry definition bytes do not match the in-memory threshold contract")
    if (
        payload.get("mAll") != len(expected_hypothesis_ids)
        or tuple(payload.get("hypothesisIds") or ()) != expected_hypothesis_ids
    ):
        raise ValueError("stale hypothesis family bytes do not match the frozen registry")


def _hypothesis_ids(
    definitions: Mapping[str, CandidateDefinition],
    expected_companies: frozenset[str],
    selector_months: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        f"candidate={candidate_id}|company={company}|month={month}|endpoint={endpoint}"
        for candidate_id in sorted(definitions)
        for company in sorted(expected_companies)
        for month in selector_months
        for endpoint in RISK_ENDPOINTS
    )


def _candidate_decisions_sha256(decisions: tuple[CandidateDecision, ...]) -> str:
    payload = {
        "schemaVersion": 1,
        "decisions": [
            {
                "candidateId": item.candidate_id,
                "rowId": item.row_id,
                "episodeId": item.episode_id,
                "company": item.company,
                "month": item.month,
                "actual": item.actual,
                "predicted": item.predicted,
            }
            for item in sorted(decisions, key=lambda value: (value.candidate_id, value.row_id))
        ],
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _portfolio_sha256(
    registry_sha256: str,
    assignments: Mapping[str, str],
    decisions: tuple[CandidateDecision, ...],
    fallback_policy: str,
) -> str:
    selected_rows = [
        item
        for item in decisions
        if assignments.get(item.company) == item.candidate_id
        and item.candidate_id != fallback_policy
    ]
    payload = {
        "schemaVersion": 1,
        "registrySha256": registry_sha256,
        "assignments": dict(sorted(assignments.items())),
        "selectedDecisionsSha256": _candidate_decisions_sha256(tuple(selected_rows)),
    }
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _is_added_process(
    item: CandidateDecision,
    registry: CandidateRegistry,
    universe: SelectorUniverse,
) -> bool:
    if item.predicted != PROCESS_LABEL:
        return False
    if registry.selection_mode == "standalone_clean":
        return True
    return universe.rows[item.row_id].baseline_predicted not in ACTION_LABELS


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )

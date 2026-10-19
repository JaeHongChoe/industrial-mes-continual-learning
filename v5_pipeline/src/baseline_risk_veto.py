from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional model family
    LGBMClassifier = None  # type: ignore[assignment,misc]

from .data_contract import ACTION_LABELS, FeatureRow
from .features import FeatureSpec, RawFeatureEncoder
from .temporal_features import TemporalFeatureMatrix
from .v4_domain_features import V4DomainFeatureSpec, V4DomainFeatureTransformer
from .v4_hybrid_candidate import (
    _calibrated_probability,
    _fit_calibrator,
    _require_sha256,
    _row_ids_sha256,
    _sha256_json,
    _validated_date_token,
    _validated_month_token,
    _validated_weights,
)


PROCESS_LABEL = "PROCESS"
SHIP_LABEL = "RETURN"
REPAIR_LABEL = "REPAIR"


@dataclass(frozen=True, slots=True)
class DangerVetoConfig:
    model_family: str = "lightgbm"
    random_state: int = 17
    privileged_training: bool = False
    hard_negative_weight: float = 5.0
    training_end: str = "2025-11-30"
    calibration_month: str = "2025-12"
    inference_channels: tuple[str, ...] = ("raw", "domain", "temporal")
    training_population: str = "baseline_process"

    def __post_init__(self) -> None:
        if self.model_family not in {"linear", "lightgbm"}:
            raise ValueError("model_family must be linear or lightgbm")
        if self.model_family == "lightgbm" and LGBMClassifier is None:
            raise RuntimeError("LightGBM is unavailable")
        if self.training_population not in {"baseline_process", "all_actions"}:
            raise ValueError(
                "training_population must be baseline_process or all_actions"
            )
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")
        if (
            isinstance(self.hard_negative_weight, bool)
            or not isinstance(self.hard_negative_weight, (int, float))
            or not math.isfinite(float(self.hard_negative_weight))
            or float(self.hard_negative_weight) < 1.0
        ):
            raise ValueError("hard_negative_weight must be finite and at least one")
        _validated_date_token(self.training_end, "training_end")
        _validated_month_token(self.calibration_month, "calibration_month")
        if not self.inference_channels or any(
            "HOLD" in channel or "wait" in channel.lower()
            for channel in self.inference_channels
        ):
            raise ValueError("inference channels must be non-empty and wait-free")


@dataclass(frozen=True, slots=True)
class DangerScoreRow:
    row_id: str
    month: str
    shipment_probability: float
    repair_probability: float

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("danger score row_id must not be empty")
        for name in ("shipment_probability", "repair_probability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class DangerTrainingRow:
    row: FeatureRow
    ship_target: bool
    repair_target: bool


def make_paired_configs(
    *,
    seed: int,
    model_family: str,
) -> tuple[DangerVetoConfig, DangerVetoConfig]:
    waitless = DangerVetoConfig(
        model_family=model_family,
        random_state=seed,
        privileged_training=False,
    )
    return waitless, replace(waitless, privileged_training=True)


def build_danger_training_rows(
    rows: Iterable[FeatureRow],
    labels: Mapping[str, str],
    baseline_predictions: Mapping[str, str],
    *,
    population: str = "baseline_process",
) -> tuple[DangerTrainingRow, ...]:
    if population not in {"baseline_process", "all_actions"}:
        raise ValueError("population must be baseline_process or all_actions")
    materialized = tuple(rows)
    seen: set[str] = set()
    output: list[DangerTrainingRow] = []
    for row in materialized:
        if row.row_id in seen:
            raise ValueError(f"duplicate danger training row: {row.row_id}")
        seen.add(row.row_id)
        actual = labels.get(row.row_id)
        if actual not in ACTION_LABELS:
            raise ValueError(f"missing danger training action label: {row.row_id}")
        if row.row_id not in baseline_predictions:
            raise ValueError(f"missing danger training baseline prediction: {row.row_id}")
        if (
            population == "baseline_process"
            and baseline_predictions[row.row_id] != PROCESS_LABEL
        ):
            continue
        output.append(
            DangerTrainingRow(
                row=row,
                ship_target=actual == SHIP_LABEL,
                repair_target=actual == REPAIR_LABEL,
            )
        )
    if not output:
        raise ValueError("danger training population is empty")
    return tuple(output)


def apply_danger_veto(
    baseline_predictions: Mapping[str, str],
    scores: Sequence[DangerScoreRow],
    *,
    shipment_threshold: float,
    repair_threshold: float,
) -> dict[str, str]:
    for name, value in (
        ("shipment_threshold", shipment_threshold),
        ("repair_threshold", repair_threshold),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be between zero and one")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be between zero and one")
    by_id: dict[str, DangerScoreRow] = {}
    for score in scores:
        if score.row_id in by_id:
            raise ValueError(f"duplicate danger score row: {score.row_id}")
        by_id[score.row_id] = score
    required_ids = {
        row_id for row_id, prediction in baseline_predictions.items() if prediction == PROCESS_LABEL
    }
    if not required_ids <= set(by_id):
        raise ValueError(
            "danger scores do not cover every baseline process approval: "
            f"missing={len(required_ids - set(by_id))}"
        )

    output = {str(row_id): str(value) for row_id, value in baseline_predictions.items()}
    for row_id in required_ids:
        score = by_id[row_id]
        if (
            score.shipment_probability >= float(shipment_threshold)
            or score.repair_probability >= float(repair_threshold)
        ):
            output[row_id] = ""
    return output


class BaselineRiskVeto:
    """Two calibrated danger heads applied only to V4 process approvals.

    The default training population is also restricted to baseline process
    approvals. Experiments may train the heads on every action row to reduce
    dependence on the upstream baseline lineage, while calibration and
    inference remain restricted to the actual V4 process-approval population.
    """

    def __init__(
        self,
        config: DangerVetoConfig | None = None,
        *,
        feature_spec: FeatureSpec | None = None,
        domain_spec: V4DomainFeatureSpec | None = None,
        validation_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or DangerVetoConfig()
        self.feature_spec = feature_spec or FeatureSpec(include_customer_key=True)
        self.domain_spec = domain_spec or V4DomainFeatureSpec()
        self.validation_config = dict(validation_config or {})
        self._raw_encoder: RawFeatureEncoder | None = None
        self._score_domain: V4DomainFeatureTransformer | None = None
        self._domain_scaler: MaxAbsScaler | None = None
        self._temporal_scaler: MaxAbsScaler | None = None
        self._temporal_schema_sha256 = ""
        self._heads: dict[str, Any] = {}
        self._calibrators: dict[str, Any] = {}
        self._feature_names: tuple[str, ...] = ()
        self._fit_receipt: dict[str, Any] | None = None
        self._calibration_cutoff = ""

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return self._feature_names

    @property
    def fit_receipt(self) -> Mapping[str, Any]:
        self._require_fitted()
        assert self._fit_receipt is not None
        return MappingProxyType(dict(self._fit_receipt))

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        baseline_predictions: Mapping[str, str],
        *,
        calibration_rows: Iterable[FeatureRow],
        calibration_labels: Mapping[str, str],
        calibration_baseline: Mapping[str, str],
        training_temporal: TemporalFeatureMatrix | None,
        calibration_temporal: TemporalFeatureMatrix | None,
        source_dataset_id: str,
        history_sha256: str,
        sample_weights: Sequence[float] | np.ndarray | None = None,
    ) -> "BaselineRiskVeto":
        training = tuple(rows)
        calibration = tuple(calibration_rows)
        _validate_partition(
            training,
            labels,
            latest=self.config.training_end,
            name="danger training",
        )
        _validate_partition(
            calibration,
            calibration_labels,
            required_month=self.config.calibration_month,
            name="danger calibration",
        )
        if {row.row_id for row in training} & {row.row_id for row in calibration}:
            raise ValueError("danger training and calibration identities overlap")
        if not str(source_dataset_id).strip():
            raise ValueError("source_dataset_id must not be empty")
        _require_sha256(history_sha256, "history_sha256")
        if self.config.privileged_training and sample_weights is None:
            raise ValueError("privileged danger training requires sample weights")
        if not self.config.privileged_training and sample_weights is not None:
            raise ValueError("waitless danger training must not receive privileged sample weights")

        training_examples = build_danger_training_rows(
            training,
            labels,
            baseline_predictions,
            population=self.config.training_population,
        )
        calibration_examples = build_danger_training_rows(
            calibration,
            calibration_labels,
            calibration_baseline,
            population="baseline_process",
        )
        approved_training = tuple(item.row for item in training_examples)
        approved_calibration = tuple(item.row for item in calibration_examples)
        training_index = {row.row_id: index for index, row in enumerate(training)}
        base_weights = _validated_weights(sample_weights, len(training))
        approved_weights = np.asarray(
            [base_weights[training_index[row.row_id]] for row in approved_training],
            dtype=np.float64,
        )

        raw_encoder = RawFeatureEncoder(
            self.feature_spec,
            validation_config=self.validation_config,
        )
        raw_training = raw_encoder.fit_transform(approved_training)
        domain_oof = V4DomainFeatureTransformer(self.domain_spec).fit_transform_oof(
            training,
            labels,
        )
        approved_indices = np.asarray(
            [training_index[row.row_id] for row in approved_training],
            dtype=np.int64,
        )
        domain_training = domain_oof[approved_indices]
        domain_scaler = MaxAbsScaler().fit(domain_training)
        domain_training = domain_scaler.transform(domain_training)

        temporal_training, temporal_scaler, temporal_names, temporal_sha = _fit_temporal_block(
            training_temporal,
            tuple(row.row_id for row in approved_training),
        )
        training_matrix = sparse.hstack(
            (raw_training, domain_training, temporal_training),
            format="csr",
        )

        train_only_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
            training,
            labels,
            cutoff=max(row.date for row in training),
        )
        temporal_calibration = _transform_temporal_block(
            calibration_temporal,
            tuple(row.row_id for row in approved_calibration),
            temporal_scaler,
            temporal_sha,
        )
        calibration_matrix = sparse.hstack(
            (
                raw_encoder.transform(approved_calibration),
                domain_scaler.transform(
                    train_only_domain.transform_with_history(approved_calibration)
                ),
                temporal_calibration,
            ),
            format="csr",
        )

        heads: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        target_map = {
            SHIP_LABEL: np.asarray(
                [int(item.ship_target) for item in training_examples], dtype=np.int8
            ),
            REPAIR_LABEL: np.asarray(
                [int(item.repair_target) for item in training_examples], dtype=np.int8
            ),
        }
        calibration_target_map = {
            SHIP_LABEL: np.asarray(
                [int(item.ship_target) for item in calibration_examples], dtype=np.int8
            ),
            REPAIR_LABEL: np.asarray(
                [int(item.repair_target) for item in calibration_examples], dtype=np.int8
            ),
        }
        for offset, label in enumerate((SHIP_LABEL, REPAIR_LABEL)):
            target = target_map[label]
            if len(set(target.tolist())) != 2:
                raise ValueError(f"danger head requires positive and negative rows: {label}")
            head_weights = approved_weights * np.where(
                target == 1,
                float(self.config.hard_negative_weight),
                1.0,
            )
            head = self._new_head(offset)
            head.fit(training_matrix, target, sample_weight=head_weights)
            calibrator = _fit_calibrator(
                _raw_margin(head, calibration_matrix),
                calibration_target_map[label],
                random_state=self.config.random_state + 100 + offset,
            )
            heads[label] = head
            calibrators[label] = calibrator

        combined_rows = (*training, *calibration)
        combined_labels = {**labels, **calibration_labels}
        calibration_cutoff = max(row.date for row in calibration)
        score_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
            combined_rows,
            combined_labels,
            cutoff=calibration_cutoff,
        )
        feature_names = (
            *raw_encoder.feature_names,
            *(f"domain::{name}" for name in score_domain.feature_names),
            *(f"temporal::{name}" for name in temporal_names),
        )
        receipt = {
            "schemaVersion": 1,
            "kind": "baseline_process_danger_veto_fit",
            "sourceDatasetId": source_dataset_id,
            "historySha256": history_sha256,
            "modelFamily": self.config.model_family,
            "randomState": self.config.random_state,
            "privilegedTraining": self.config.privileged_training,
            "inferenceWaitFeature": False,
            "inferenceChannels": list(self.config.inference_channels),
            "hardNegativeWeight": float(self.config.hard_negative_weight),
            "trainingPopulation": self.config.training_population,
            "trainingRowsSha256": _row_ids_sha256(training),
            "calibrationRowsSha256": _row_ids_sha256(calibration),
            "approvedTrainingRows": len(approved_training),
            "trainingPopulationRows": len(approved_training),
            "approvedCalibrationRows": len(approved_calibration),
            "shipDangerTrainingRows": int(np.count_nonzero(target_map[SHIP_LABEL])),
            "repairDangerTrainingRows": int(np.count_nonzero(target_map[REPAIR_LABEL])),
            "sampleWeightSha256": hashlib.sha256(
                approved_weights.astype("<f8").tobytes()
            ).hexdigest(),
            "rawFeatureSchemaSha256": raw_encoder.schema_sha256,
            "domainFeatureSchemaSha256": score_domain.schema_sha256,
            "temporalFeatureSchemaSha256": temporal_sha,
            "featureNamesSha256": _sha256_json(feature_names),
            "trainingEndInclusive": self.config.training_end,
            "calibrationMonth": self.config.calibration_month,
            "calibrationCutoff": calibration_cutoff,
        }
        receipt["receiptSha256"] = _sha256_json(receipt)

        self._raw_encoder = raw_encoder
        self._score_domain = score_domain
        self._domain_scaler = domain_scaler
        self._temporal_scaler = temporal_scaler
        self._temporal_schema_sha256 = temporal_sha
        self._heads = heads
        self._calibrators = calibrators
        self._feature_names = tuple(feature_names)
        self._fit_receipt = receipt
        self._calibration_cutoff = calibration_cutoff
        return self

    def score_rows(
        self,
        rows: Iterable[FeatureRow],
        *,
        temporal: TemporalFeatureMatrix | None,
        prior_history_rows: Iterable[FeatureRow] | None = None,
        prior_history_labels: Mapping[str, str] | None = None,
        prior_history_cutoff: str | None = None,
    ) -> tuple[DangerScoreRow, ...]:
        self._require_fitted()
        materialized = tuple(rows)
        if any(row.date <= self._calibration_cutoff for row in materialized):
            raise ValueError("danger score rows must be after probability calibration")
        prior_arguments = (
            prior_history_rows is not None,
            prior_history_labels is not None,
            prior_history_cutoff is not None,
        )
        if any(prior_arguments) and not all(prior_arguments):
            raise ValueError("prior history rows, labels, and cutoff must be supplied together")
        assert self._raw_encoder is not None
        assert self._score_domain is not None
        assert self._domain_scaler is not None
        score_domain = self._score_domain
        if all(prior_arguments):
            assert prior_history_rows is not None
            assert prior_history_labels is not None
            assert prior_history_cutoff is not None
            cutoff = _validated_date_token(prior_history_cutoff, "prior_history_cutoff")
            if materialized and cutoff >= min(row.date for row in materialized):
                raise ValueError("prior history cutoff must be strictly before danger score rows")
            score_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
                prior_history_rows,
                prior_history_labels,
                cutoff=cutoff,
            )
        temporal_matrix = _transform_temporal_block(
            temporal,
            tuple(row.row_id for row in materialized),
            self._temporal_scaler,
            self._temporal_schema_sha256,
        )
        matrix = sparse.hstack(
            (
                self._raw_encoder.transform(materialized),
                self._domain_scaler.transform(score_domain.transform_with_history(materialized)),
                temporal_matrix,
            ),
            format="csr",
        )
        shipment = _calibrated_probability(
            self._heads[SHIP_LABEL],
            self._calibrators[SHIP_LABEL],
            matrix,
        )
        repair = _calibrated_probability(
            self._heads[REPAIR_LABEL],
            self._calibrators[REPAIR_LABEL],
            matrix,
        )
        return tuple(
            DangerScoreRow(
                row_id=row.row_id,
                month=row.month,
                shipment_probability=float(shipment[index]),
                repair_probability=float(repair[index]),
            )
            for index, row in enumerate(materialized)
        )

    def _new_head(self, offset: int) -> Any:
        if self.config.model_family == "linear":
            return SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=1e-5,
                max_iter=1_500,
                tol=1e-4,
                random_state=self.config.random_state + offset,
            )
        assert LGBMClassifier is not None
        return LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.config.random_state + offset,
            n_jobs=-1,
            verbosity=-1,
        )

    def _require_fitted(self) -> None:
        if (
            self._raw_encoder is None
            or self._score_domain is None
            or self._domain_scaler is None
            or self._fit_receipt is None
        ):
            raise RuntimeError("BaselineRiskVeto must be fitted before use")


def _fit_temporal_block(
    temporal: TemporalFeatureMatrix | None,
    row_ids: Sequence[str],
) -> tuple[sparse.csr_matrix, MaxAbsScaler | None, tuple[str, ...], str]:
    if temporal is None:
        return sparse.csr_matrix((len(row_ids), 0)), None, (), ""
    values = temporal.select(row_ids)
    scaler = MaxAbsScaler().fit(values)
    transformed = sparse.csr_matrix(scaler.transform(values), dtype=np.float32)
    return transformed, scaler, temporal.feature_names, temporal.schema_sha256


def _transform_temporal_block(
    temporal: TemporalFeatureMatrix | None,
    row_ids: Sequence[str],
    scaler: MaxAbsScaler | None,
    expected_schema_sha256: str,
) -> sparse.csr_matrix:
    if scaler is None:
        if temporal is not None and temporal.values.shape[1]:
            raise ValueError("unexpected temporal features for a model fitted without them")
        return sparse.csr_matrix((len(row_ids), 0))
    if temporal is None or temporal.schema_sha256 != expected_schema_sha256:
        raise ValueError("temporal feature schema does not match the fitted danger model")
    return sparse.csr_matrix(scaler.transform(temporal.select(row_ids)), dtype=np.float32)


def _validate_partition(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    *,
    name: str,
    latest: str | None = None,
    required_month: str | None = None,
) -> None:
    if not rows:
        raise ValueError(f"{name} rows cannot be empty")
    seen: set[str] = set()
    for row in rows:
        if row.row_id in seen:
            raise ValueError(f"duplicate {name} row: {row.row_id}")
        seen.add(row.row_id)
        if labels.get(row.row_id) not in ACTION_LABELS:
            raise ValueError(f"missing {name} label: {row.row_id}")
        if latest is not None and row.date > latest:
            raise ValueError(f"{name} rows must be on or before {latest}")
        if required_month is not None and row.month != required_month:
            raise ValueError(f"{name} rows must be in {required_month}")


def _raw_margin(model: Any, matrix: sparse.csr_matrix) -> np.ndarray:
    if hasattr(model, "decision_function"):
        output = np.asarray(model.decision_function(matrix), dtype=np.float64)
        if output.ndim != 1:
            raise ValueError("danger decision margin must be one-dimensional")
        return output
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    positive_index = list(model.classes_).index(1)
    positive = np.clip(probabilities[:, positive_index], 1e-8, 1 - 1e-8)
    return np.log(positive / (1 - positive))

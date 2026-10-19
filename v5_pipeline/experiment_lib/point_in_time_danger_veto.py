from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

from .calibration_population import require_calibration_population


PROCESS_LABEL = "PROCESS"
SHIP_LABEL = "RETURN"
REPAIR_LABEL = "REPAIR"


class PointInTimeDangerVeto:
    """Fit the two danger heads from already sealed point-in-time features.

    The rolling runtime still supplies the proven raw-text encoder, model heads,
    calibration routine, and score row contract.  Label-dependent domain and
    temporal features are deliberately supplied by the caller so this class
    cannot silently rebuild them from occurrence-time labels.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        config: Any,
        feature_spec: Any | None = None,
        validation_config: Mapping[str, Any] | None = None,
        calibration_population: str = "baseline_process",
        calibration_start: str | None = None,
        calibration_end: str | None = None,
    ) -> None:
        self.runtime = runtime
        if (calibration_start is None) != (calibration_end is None):
            raise ValueError('explicit calibration requires both start and end')
        if calibration_start is not None:
            if not (date.fromisoformat(config.training_end) < date.fromisoformat(calibration_start)
                    <= date.fromisoformat(calibration_end)):
                raise ValueError('explicit calibration range must follow training')
        self.calibration_start = calibration_start
        self.calibration_end = calibration_end
        self.config = config
        self.feature_spec = feature_spec or runtime.FeatureSpec(
            include_customer_key=True
        )
        self.validation_config = dict(validation_config or {})
        self.calibration_population = require_calibration_population(
            calibration_population
        )
        self._raw_encoder: Any | None = None
        self._domain_scaler: MaxAbsScaler | None = None
        self._temporal_scaler: MaxAbsScaler | None = None
        self._domain_schema_sha256 = ""
        self._temporal_schema_sha256 = ""
        self._calibration_cutoff = ""
        self._heads: dict[str, Any] = {}
        self._calibrators: dict[str, Any] = {}
        self._fit_receipt: dict[str, Any] | None = None
        self._feature_names: tuple[str, ...] = ()

    @property
    def fit_receipt(self) -> Mapping[str, Any]:
        self._require_fitted()
        assert self._fit_receipt is not None
        return MappingProxyType(dict(self._fit_receipt))

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return self._feature_names

    def fit(
        self,
        rows: Iterable[Any],
        labels: Mapping[str, str],
        baseline_predictions: Mapping[str, str],
        *,
        calibration_rows: Iterable[Any],
        calibration_labels: Mapping[str, str],
        calibration_baseline: Mapping[str, str],
        training_temporal: Any | None,
        calibration_temporal: Any | None,
        training_domain: sparse.spmatrix | np.ndarray,
        calibration_domain: sparse.spmatrix | np.ndarray,
        domain_schema_sha256: str,
        source_dataset_id: str,
        history_sha256: str,
        label_availability_receipt: Mapping[str, Any],
        scoring_start: str,
        sample_weights: Sequence[float] | np.ndarray | None = None,
    ) -> PointInTimeDangerVeto:
        runtime = self.runtime
        training = tuple(rows)
        calibration = tuple(calibration_rows)
        runtime._validate_partition(
            training,
            labels,
            latest=self.config.training_end,
            name="point-in-time danger training",
        )
        if self.calibration_start is None:
            runtime._validate_partition(calibration, calibration_labels,
                required_month=self.config.calibration_month, name="point-in-time danger calibration")
        else:
            runtime._validate_partition(calibration, calibration_labels,
                latest=self.calibration_end, name="point-in-time danger calibration")
            if any(row.date < self.calibration_start for row in calibration):
                raise ValueError('calibration row before explicit start')
        if {row.row_id for row in training} & {row.row_id for row in calibration}:
            raise ValueError("danger training and calibration identities overlap")
        first_score = date.fromisoformat(scoring_start)
        knowledge_cutoff = date.fromisoformat(label_availability_receipt['cutoffExclusive'])
        if first_score < knowledge_cutoff or any(date.fromisoformat(row.date) >= first_score for row in calibration):
            raise ValueError('scoring_start precedes calibration knowledge cutoff')
        calibration_cutoff = (first_score - timedelta(days=1)).isoformat()
        if not str(source_dataset_id).strip():
            raise ValueError("source_dataset_id must not be empty")
        runtime._require_sha256(history_sha256, "history_sha256")
        runtime._require_sha256(domain_schema_sha256, "domain_schema_sha256")
        availability_sha256 = str(
            label_availability_receipt.get("receiptSha256", "")
        ) or _payload_sha256(label_availability_receipt)
        runtime._require_sha256(availability_sha256, "label_availability_receipt")
        if self.config.privileged_training and sample_weights is None:
            raise ValueError("privileged danger training requires sample weights")
        if not self.config.privileged_training and sample_weights is not None:
            raise ValueError(
                "waitless danger training must not receive privileged sample weights"
            )

        training_domain_matrix = _validated_matrix(
            training_domain,
            expected_rows=len(training),
            name="training_domain",
        )
        calibration_domain_matrix = _validated_matrix(
            calibration_domain,
            expected_rows=len(calibration),
            expected_columns=training_domain_matrix.shape[1],
            name="calibration_domain",
        )
        training_examples = runtime.build_danger_training_rows(
            training,
            labels,
            baseline_predictions,
            population=self.config.training_population,
        )
        calibration_examples = runtime.build_danger_training_rows(
            calibration,
            calibration_labels,
            calibration_baseline,
            population=self.calibration_population,
        )
        approved_training = tuple(item.row for item in training_examples)
        approved_calibration = tuple(item.row for item in calibration_examples)
        training_index = {row.row_id: index for index, row in enumerate(training)}
        calibration_index = {
            row.row_id: index for index, row in enumerate(calibration)
        }
        approved_training_indices = np.asarray(
            [training_index[row.row_id] for row in approved_training],
            dtype=np.int64,
        )
        approved_calibration_indices = np.asarray(
            [calibration_index[row.row_id] for row in approved_calibration],
            dtype=np.int64,
        )
        base_weights = runtime._validated_weights(sample_weights, len(training))
        approved_weights = np.asarray(
            [base_weights[index] for index in approved_training_indices],
            dtype=np.float64,
        )

        raw_encoder = runtime.RawFeatureEncoder(
            self.feature_spec,
            validation_config=self.validation_config,
        )
        raw_training = raw_encoder.fit_transform(approved_training)
        domain_scaler = MaxAbsScaler().fit(
            training_domain_matrix[approved_training_indices]
        )
        domain_training = domain_scaler.transform(
            training_domain_matrix[approved_training_indices]
        )
        temporal_training, temporal_scaler, temporal_names, temporal_sha256 = (
            runtime._fit_temporal_block(
                training_temporal,
                tuple(row.row_id for row in approved_training),
            )
        )
        training_matrix = sparse.hstack(
            (raw_training, domain_training, temporal_training),
            format="csr",
        )

        temporal_calibration = runtime._transform_temporal_block(
            calibration_temporal,
            tuple(row.row_id for row in approved_calibration),
            temporal_scaler,
            temporal_sha256,
        )
        calibration_matrix = sparse.hstack(
            (
                raw_encoder.transform(approved_calibration),
                domain_scaler.transform(
                    calibration_domain_matrix[approved_calibration_indices]
                ),
                temporal_calibration,
            ),
            format="csr",
        )

        target_map = {
            SHIP_LABEL: np.asarray(
                [int(item.ship_target) for item in training_examples],
                dtype=np.int8,
            ),
            REPAIR_LABEL: np.asarray(
                [int(item.repair_target) for item in training_examples],
                dtype=np.int8,
            ),
        }
        calibration_target_map = {
            SHIP_LABEL: np.asarray(
                [int(item.ship_target) for item in calibration_examples],
                dtype=np.int8,
            ),
            REPAIR_LABEL: np.asarray(
                [int(item.repair_target) for item in calibration_examples],
                dtype=np.int8,
            ),
        }
        heads: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        probe = runtime.BaselineRiskVeto(self.config)
        for offset, label in enumerate((SHIP_LABEL, REPAIR_LABEL)):
            target = target_map[label]
            if len(set(target.tolist())) != 2:
                raise ValueError(
                    f"danger head requires positive and negative rows: {label}"
                )
            head_weights = approved_weights * np.where(
                target == 1,
                float(self.config.hard_negative_weight),
                1.0,
            )
            head = probe._new_head(offset)
            head.fit(training_matrix, target, sample_weight=head_weights)
            calibrator = runtime._fit_calibrator(
                runtime._raw_margin(head, calibration_matrix),
                calibration_target_map[label],
                random_state=self.config.random_state + 100 + offset,
            )
            heads[label] = head
            calibrators[label] = calibrator

        feature_names = (
            *raw_encoder.feature_names,
            *(
                f"domain::feature_{index}"
                for index in range(training_domain_matrix.shape[1])
            ),
            *(f"temporal::{name}" for name in temporal_names),
        )
        receipt = {
            "schemaVersion": 1,
            "kind": "point_in_time_danger_veto_fit",
            "sourceDatasetId": source_dataset_id,
            "historySha256": history_sha256,
            "labelAvailabilityReceiptSha256": availability_sha256,
            "modelFamily": self.config.model_family,
            "randomState": self.config.random_state,
            "privilegedTraining": self.config.privileged_training,
            "inferenceWaitFeature": False,
            "inferenceChannels": list(self.config.inference_channels),
            "hardNegativeWeight": float(self.config.hard_negative_weight),
            "trainingPopulation": self.config.training_population,
            "calibrationPopulation": self.calibration_population,
            "trainingRowsSha256": runtime._row_ids_sha256(training),
            "trainingLabelsSha256": _labels_sha256(training, labels),
            "calibrationRowsSha256": runtime._row_ids_sha256(calibration),
            "calibrationLabelsSha256": _labels_sha256(
                calibration,
                calibration_labels,
            ),
            "approvedTrainingRows": len(approved_training),
            "approvedCalibrationRows": len(approved_calibration),
            "approvedCalibrationRowsSha256": runtime._row_ids_sha256(
                approved_calibration
            ),
            "calibrationBaselineSha256": runtime._sha256_json(
                tuple(
                    (row.row_id, str(calibration_baseline[row.row_id]))
                    for row in calibration
                )
            ),
            "shipDangerTrainingRows": int(
                np.count_nonzero(target_map[SHIP_LABEL])
            ),
            "repairDangerTrainingRows": int(
                np.count_nonzero(target_map[REPAIR_LABEL])
            ),
            "sampleWeightSha256": hashlib.sha256(
                approved_weights.astype("<f8").tobytes()
            ).hexdigest(),
            "rawFeatureSchemaSha256": raw_encoder.schema_sha256,
            "domainFeatureSchemaSha256": domain_schema_sha256,
            "domainTrainingValuesSha256": _matrix_sha256(training_domain_matrix),
            "domainCalibrationValuesSha256": _matrix_sha256(
                calibration_domain_matrix
            ),
            "temporalFeatureSchemaSha256": temporal_sha256,
            "featureNamesSha256": runtime._sha256_json(feature_names),
            "trainingEndInclusive": self.config.training_end,
            "calibrationMonth": self.config.calibration_month,
            "calibrationCutoff": calibration_cutoff,
        }
        if self.calibration_start is not None:
            receipt.update(calibrationStartInclusive=self.calibration_start,
                           calibrationEndInclusive=self.calibration_end,
                           calibrationMonth=None)
        receipt["receiptSha256"] = runtime._sha256_json(receipt)

        self._raw_encoder = raw_encoder
        self._domain_scaler = domain_scaler
        self._temporal_scaler = temporal_scaler
        self._domain_schema_sha256 = domain_schema_sha256
        self._temporal_schema_sha256 = temporal_sha256
        self._calibration_cutoff = calibration_cutoff
        self._heads = heads
        self._calibrators = calibrators
        self._fit_receipt = receipt
        self._feature_names = tuple(feature_names)
        return self

    def score_rows(
        self,
        rows: Iterable[Any],
        *,
        temporal: Any | None,
        domain: sparse.spmatrix | np.ndarray,
        domain_schema_sha256: str,
    ) -> tuple[Any, ...]:
        self._require_fitted()
        materialized = tuple(rows)
        if any(row.date <= self._calibration_cutoff for row in materialized):
            raise ValueError("danger score rows must be after probability calibration")
        if domain_schema_sha256 != self._domain_schema_sha256:
            raise ValueError("domain feature schema does not match the fitted danger model")
        domain_matrix = _validated_matrix(
            domain,
            expected_rows=len(materialized),
            name="domain",
        )
        assert self._raw_encoder is not None
        assert self._domain_scaler is not None
        temporal_matrix = self.runtime._transform_temporal_block(
            temporal,
            tuple(row.row_id for row in materialized),
            self._temporal_scaler,
            self._temporal_schema_sha256,
        )
        matrix = sparse.hstack(
            (
                self._raw_encoder.transform(materialized),
                self._domain_scaler.transform(domain_matrix),
                temporal_matrix,
            ),
            format="csr",
        )
        shipment = self.runtime._calibrated_probability(
            self._heads[SHIP_LABEL],
            self._calibrators[SHIP_LABEL],
            matrix,
        )
        repair = self.runtime._calibrated_probability(
            self._heads[REPAIR_LABEL],
            self._calibrators[REPAIR_LABEL],
            matrix,
        )
        return tuple(
            self.runtime.DangerScoreRow(
                row_id=row.row_id,
                month=row.month,
                shipment_probability=float(shipment[index]),
                repair_probability=float(repair[index]),
            )
            for index, row in enumerate(materialized)
        )

    def _require_fitted(self) -> None:
        if (
            self._raw_encoder is None
            or self._domain_scaler is None
            or self._fit_receipt is None
        ):
            raise RuntimeError("PointInTimeDangerVeto must be fitted before use")


def _validated_matrix(
    values: sparse.spmatrix | np.ndarray,
    *,
    expected_rows: int,
    name: str,
    expected_columns: int | None = None,
) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(values, dtype=np.float32)
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{name} row count mismatch: expected={expected_rows} "
            f"observed={matrix.shape[0]}"
        )
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{name} column count mismatch: expected={expected_columns} "
            f"observed={matrix.shape[1]}"
        )
    if matrix.data.size and not np.isfinite(matrix.data).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _matrix_sha256(matrix: sparse.csr_matrix) -> str:
    normalized = matrix.copy()
    normalized.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
    digest.update(normalized.indptr.astype("<i8", copy=False).tobytes())
    digest.update(normalized.indices.astype("<i8", copy=False).tobytes())
    digest.update(normalized.data.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _labels_sha256(rows: Sequence[Any], labels: Mapping[str, str]) -> str:
    payload = [(row.row_id, labels[row.row_id]) for row in rows]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _payload_sha256(payload: Any) -> str:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    return hashlib.sha256(
        json.dumps(
            plain(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

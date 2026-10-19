from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MaxAbsScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional family
    LGBMClassifier = None  # type: ignore[assignment,misc]

from .data_contract import ACTION_LABELS, FeatureRow
from .features import FeatureSpec, RawFeatureEncoder
from .v4_domain_features import V4DomainFeatureSpec, V4DomainFeatureTransformer


_LABELS = ("PROCESS", "RETURN", "REPAIR")


@dataclass(frozen=True, slots=True)
class V4HybridScoreRow:
    row_id: str
    month: str
    process_probability: float
    shipment_probability: float
    repair_probability: float
    gray_probability: float = 0.0


@dataclass(frozen=True, slots=True)
class HybridThresholdFit:
    process_threshold: float
    shipment_veto: float
    repair_veto: float
    shipment_auto_threshold: float
    repair_auto_threshold: float
    gray_veto: float
    selection_month: str

    def __post_init__(self) -> None:
        for name in (
            "process_threshold",
            "shipment_veto",
            "repair_veto",
            "shipment_auto_threshold",
            "repair_auto_threshold",
            "gray_veto",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be between zero and one")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")


class V4HybridCandidate:
    """Three independent heads over raw and V4-derived intake-time features."""

    def __init__(
        self,
        *,
        model_family: str = "linear",
        random_state: int = 17,
        training_end: str = "2025-07-31",
        calibration_month: str = "2025-08",
        score_batch_size: int = 10_000,
        feature_spec: FeatureSpec | None = None,
        domain_spec: V4DomainFeatureSpec | None = None,
        validation_config: Mapping[str, Any] | None = None,
    ) -> None:
        if model_family not in {"linear", "lightgbm"}:
            raise ValueError("model_family must be linear or lightgbm")
        if model_family == "lightgbm" and LGBMClassifier is None:
            raise RuntimeError("LightGBM is unavailable")
        self.model_family = model_family
        self.random_state = int(random_state)
        self.training_end = _validated_date_token(training_end, "training_end")
        self.calibration_month = _validated_month_token(
            calibration_month,
            "calibration_month",
        )
        if isinstance(score_batch_size, bool) or not isinstance(score_batch_size, int):
            raise ValueError("score_batch_size must be a positive integer")
        if score_batch_size < 1:
            raise ValueError("score_batch_size must be a positive integer")
        self.score_batch_size = score_batch_size
        self.feature_spec = feature_spec or FeatureSpec(include_customer_key=True)
        self.domain_spec = domain_spec or V4DomainFeatureSpec()
        self.validation_config = dict(validation_config or {})
        self._raw_encoder: RawFeatureEncoder | None = None
        self._score_domain: V4DomainFeatureTransformer | None = None
        self._domain_scaler: MaxAbsScaler | None = None
        self._heads: dict[str, Any] = {}
        self._calibrators: dict[str, Any] = {}
        self._feature_names: tuple[str, ...] = ()
        self._fit_receipt: dict[str, Any] | None = None
        self._calibration_cutoff: str | None = None

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
        *,
        calibration_rows: Iterable[FeatureRow],
        calibration_labels: Mapping[str, str],
        source_dataset_id: str,
        history_sha256: str,
        sample_weights: Sequence[float] | np.ndarray | None = None,
        gray_labels: Mapping[str, bool] | None = None,
        calibration_gray_labels: Mapping[str, bool] | None = None,
    ) -> "V4HybridCandidate":
        training = tuple(rows)
        calibration = tuple(calibration_rows)
        _validate_partition(training, labels, "training", latest=self.training_end)
        _validate_partition(
            calibration,
            calibration_labels,
            "calibration",
            required_month=self.calibration_month,
        )
        if not str(source_dataset_id).strip():
            raise ValueError("source_dataset_id must not be empty")
        _require_sha256(history_sha256, "history_sha256")
        training_ids = {row.row_id for row in training}
        calibration_ids = {row.row_id for row in calibration}
        if training_ids & calibration_ids:
            raise ValueError("training and calibration row identities overlap")
        weights = _validated_weights(sample_weights, len(training))

        raw_encoder = RawFeatureEncoder(
            self.feature_spec,
            validation_config=self.validation_config,
        )
        raw_training = raw_encoder.fit_transform(training)
        domain_training_encoder = V4DomainFeatureTransformer(self.domain_spec)
        domain_training = domain_training_encoder.fit_transform_oof(training, labels)
        domain_scaler = MaxAbsScaler().fit(domain_training)
        domain_training = domain_scaler.transform(domain_training)
        training_matrix = sparse.hstack((raw_training, domain_training), format="csr")

        train_only_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
            training,
            labels,
            cutoff=max(row.date for row in training),
        )
        calibration_matrix = sparse.hstack(
            (
                raw_encoder.transform(calibration),
                domain_scaler.transform(train_only_domain.transform_with_history(calibration)),
            ),
            format="csr",
        )
        heads: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        for offset, label in enumerate(_LABELS):
            target = np.asarray([int(labels[row.row_id] == label) for row in training], dtype=np.int8)
            calibration_target = np.asarray(
                [int(calibration_labels[row.row_id] == label) for row in calibration],
                dtype=np.int8,
            )
            if len(set(target.tolist())) != 2:
                raise ValueError(f"training head requires positive and negative rows: {label}")
            head = self._new_head(offset)
            head.fit(training_matrix, target, sample_weight=weights)
            raw_margin = _raw_margin(head, calibration_matrix)
            calibrator = _fit_calibrator(
                raw_margin,
                calibration_target,
                random_state=self.random_state + 100 + offset,
            )
            heads[label] = head
            calibrators[label] = calibrator

        gray_auxiliary = gray_labels is not None or calibration_gray_labels is not None
        if gray_auxiliary:
            if gray_labels is None or calibration_gray_labels is None:
                raise ValueError("gray auxiliary training requires both training and calibration labels")
            gray_target = _boolean_target(training, gray_labels, "gray training")
            gray_calibration_target = _boolean_target(
                calibration,
                calibration_gray_labels,
                "gray calibration",
            )
            if len(set(gray_target.tolist())) != 2:
                raise ValueError("gray training requires both gray and non-gray rows")
            gray_head = self._new_head(len(_LABELS))
            gray_head.fit(training_matrix, gray_target, sample_weight=weights)
            gray_raw = _raw_margin(gray_head, calibration_matrix)
            gray_calibrator = _fit_calibrator(
                gray_raw,
                gray_calibration_target,
                random_state=self.random_state + 200,
            )
            heads["__gray__"] = gray_head
            calibrators["__gray__"] = gray_calibrator

        combined_rows = (*training, *calibration)
        combined_labels = {**labels, **calibration_labels}
        score_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
            combined_rows,
            combined_labels,
            cutoff=max(row.date for row in calibration),
        )
        weight_sha = hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()
        feature_names = (
            *raw_encoder.feature_names,
            *(f"domain::{name}" for name in score_domain.feature_names),
        )
        receipt = {
            "schemaVersion": 1,
            "kind": "v4_hybrid_candidate_fit",
            "modelFamily": self.model_family,
            "randomState": self.random_state,
            "sourceDatasetId": source_dataset_id,
            "historySha256": history_sha256,
            "sampleWeightSha256": weight_sha,
            "trainingRowsSha256": _row_ids_sha256(training),
            "calibrationRowsSha256": _row_ids_sha256(calibration),
            "rawFeatureSchemaSha256": raw_encoder.schema_sha256,
            "domainFeatureSchemaSha256": score_domain.schema_sha256,
            "domainScaling": "maxabs_fit_on_training_oof",
            "featureNamesSha256": _sha256_json(feature_names),
            "calibrationCutoff": max(row.date for row in calibration),
            "trainingEndInclusive": self.training_end,
            "calibrationMonth": self.calibration_month,
            "scoreBatchSize": self.score_batch_size,
            "trainingOnlyWaitMetadata": True,
            "inferenceWaitFeature": False,
            "grayAuxiliaryHead": gray_auxiliary,
            "grayTrainingTargetSha256": _boolean_labels_sha256(gray_labels)
            if gray_labels is not None
            else None,
        }
        receipt["receiptSha256"] = _sha256_json(receipt)

        self._raw_encoder = raw_encoder
        self._score_domain = score_domain
        self._domain_scaler = domain_scaler
        self._heads = heads
        self._calibrators = calibrators
        self._feature_names = tuple(feature_names)
        self._fit_receipt = receipt
        self._calibration_cutoff = max(row.date for row in calibration)
        return self

    def score_rows(
        self,
        rows: Iterable[FeatureRow],
        *,
        prior_history_rows: Iterable[FeatureRow] | None = None,
        prior_history_labels: Mapping[str, str] | None = None,
        prior_history_cutoff: str | None = None,
    ) -> tuple[V4HybridScoreRow, ...]:
        self._require_fitted()
        materialized = tuple(rows)
        assert self._raw_encoder is not None
        assert self._score_domain is not None
        assert self._domain_scaler is not None
        assert self._calibration_cutoff is not None
        if any(row.date <= self._calibration_cutoff for row in materialized):
            raise ValueError("hybrid score rows must be after probability calibration")
        prior_arguments = (
            prior_history_rows is not None,
            prior_history_labels is not None,
            prior_history_cutoff is not None,
        )
        if any(prior_arguments) and not all(prior_arguments):
            raise ValueError("prior history rows, labels, and cutoff must be supplied together")
        score_domain = self._score_domain
        if all(prior_arguments):
            assert prior_history_rows is not None
            assert prior_history_labels is not None
            assert prior_history_cutoff is not None
            cutoff = _validated_date_token(prior_history_cutoff, "prior_history_cutoff")
            if materialized and cutoff >= min(row.date for row in materialized):
                raise ValueError("prior history cutoff must be strictly before scored rows")
            score_domain = V4DomainFeatureTransformer(self.domain_spec).fit_history(
                prior_history_rows,
                prior_history_labels,
                cutoff=cutoff,
            )
        output: list[V4HybridScoreRow] = []
        for start in range(0, len(materialized), self.score_batch_size):
            batch = materialized[start : start + self.score_batch_size]
            matrix = sparse.hstack(
                (
                    self._raw_encoder.transform(batch),
                    self._domain_scaler.transform(score_domain.transform_with_history(batch)),
                ),
                format="csr",
            )
            probabilities = {
                label: _calibrated_probability(
                    self._heads[label],
                    self._calibrators[label],
                    matrix,
                )
                for label in _LABELS
            }
            gray_probability = (
                _calibrated_probability(
                    self._heads["__gray__"],
                    self._calibrators["__gray__"],
                    matrix,
                )
                if "__gray__" in self._heads
                else np.zeros(len(batch), dtype=np.float64)
            )
            output.extend(
                V4HybridScoreRow(
                    row_id=row.row_id,
                    month=row.month,
                    process_probability=float(probabilities["PROCESS"][index]),
                    shipment_probability=float(probabilities["RETURN"][index]),
                    repair_probability=float(probabilities["REPAIR"][index]),
                    gray_probability=float(gray_probability[index]),
                )
                for index, row in enumerate(batch)
            )
        return tuple(output)

    def _new_head(self, offset: int):
        if self.model_family == "linear":
            return SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=1e-5,
                max_iter=1_500,
                tol=1e-4,
                random_state=self.random_state + offset,
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
            random_state=self.random_state + offset,
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
            raise RuntimeError("V4HybridCandidate must be fitted before use")


def apply_hybrid_thresholds(
    scores: Iterable[V4HybridScoreRow],
    fitted: HybridThresholdFit,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in scores:
        if row.row_id in output:
            raise ValueError(f"duplicate hybrid score row: {row.row_id}")
        if (
            row.shipment_probability >= fitted.shipment_auto_threshold
            and row.shipment_probability >= row.repair_probability
        ):
            label = "RETURN"
        elif row.repair_probability >= fitted.repair_auto_threshold:
            label = "REPAIR"
        elif (
            row.process_probability >= fitted.process_threshold
            and row.shipment_probability < fitted.shipment_veto
            and row.repair_probability < fitted.repair_veto
            and row.gray_probability <= fitted.gray_veto
        ):
            label = "PROCESS"
        else:
            label = ""
        output[row.row_id] = label
    return output


def fit_hybrid_thresholds(
    scores: Iterable[V4HybridScoreRow],
    labels: Mapping[str, str],
    *,
    process_thresholds: Sequence[float] = (0.8, 0.9, 0.95, 0.99),
    risk_vetoes: Sequence[float] = (0.02, 0.05, 0.1),
    ng_thresholds: Sequence[float] = (0.95, 0.99),
    gray_vetoes: Sequence[float] = (1.0,),
    selection_month: str = "2025-09",
    max_ship_to_process: int = 0,
    max_repair_to_process: int = 0,
    max_risk_to_process: int = 0,
    min_auto_accuracy: float = 0.0,
) -> HybridThresholdFit:
    materialized = tuple(scores)
    required_month = _validated_month_token(selection_month, "selection_month")
    if not materialized or any(row.month != required_month for row in materialized):
        if required_month == "2025-09":
            raise ValueError("hybrid threshold fitting requires September 2025 rows")
        raise ValueError(f"hybrid threshold fitting requires {required_month} rows")
    if any(labels.get(row.row_id) not in ACTION_LABELS for row in materialized):
        raise ValueError("hybrid threshold fitting requires an action label for every row")
    for name, value in (
        ("max_ship_to_process", max_ship_to_process),
        ("max_repair_to_process", max_repair_to_process),
        ("max_risk_to_process", max_risk_to_process),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(min_auto_accuracy, (int, float)) or isinstance(min_auto_accuracy, bool):
        raise ValueError("min_auto_accuracy must be between zero and one")
    if not math.isfinite(float(min_auto_accuracy)) or not 0 <= float(min_auto_accuracy) <= 1:
        raise ValueError("min_auto_accuracy must be between zero and one")
    process_probability = np.asarray(
        [row.process_probability for row in materialized],
        dtype=np.float64,
    )
    shipment_probability = np.asarray(
        [row.shipment_probability for row in materialized],
        dtype=np.float64,
    )
    repair_probability = np.asarray(
        [row.repair_probability for row in materialized],
        dtype=np.float64,
    )
    gray_probability = np.asarray(
        [row.gray_probability for row in materialized],
        dtype=np.float64,
    )
    actual = np.asarray([labels[row.row_id] for row in materialized], dtype=object)
    best: tuple[tuple[int, int, int, int, int], HybridThresholdFit] | None = None
    for process_threshold in process_thresholds:
        for shipment_veto in risk_vetoes:
            for repair_veto in risk_vetoes:
                for shipment_auto in ng_thresholds:
                    for repair_auto in ng_thresholds:
                        for gray_veto in gray_vetoes:
                            fitted = HybridThresholdFit(
                                process_threshold=float(process_threshold),
                                shipment_veto=float(shipment_veto),
                                repair_veto=float(repair_veto),
                                shipment_auto_threshold=float(shipment_auto),
                                repair_auto_threshold=float(repair_auto),
                                gray_veto=float(gray_veto),
                                selection_month=required_month,
                            )
                            ship_mask = (
                                (shipment_probability >= fitted.shipment_auto_threshold)
                                & (shipment_probability >= repair_probability)
                            )
                            repair_mask = (~ship_mask) & (
                                repair_probability >= fitted.repair_auto_threshold
                            )
                            process_mask = (
                                (~ship_mask)
                                & (~repair_mask)
                                & (process_probability >= fitted.process_threshold)
                                & (shipment_probability < fitted.shipment_veto)
                                & (repair_probability < fitted.repair_veto)
                                & (gray_probability <= fitted.gray_veto)
                            )
                            ship_to_process = int(np.count_nonzero((actual == "RETURN") & process_mask))
                            repair_to_process = int(np.count_nonzero((actual == "REPAIR") & process_mask))
                            risk_to_process = ship_to_process + repair_to_process
                            if (
                                ship_to_process > max_ship_to_process
                                or repair_to_process > max_repair_to_process
                                or risk_to_process > max_risk_to_process
                            ):
                                continue
                            auto = int(np.count_nonzero(ship_mask | repair_mask | process_mask))
                            correct_process = int(np.count_nonzero((actual == "PROCESS") & process_mask))
                            correct = int(
                                np.count_nonzero((actual == "PROCESS") & process_mask)
                                + np.count_nonzero((actual == "RETURN") & ship_mask)
                                + np.count_nonzero((actual == "REPAIR") & repair_mask)
                            )
                            if auto == 0 or correct / auto < float(min_auto_accuracy):
                                continue
                            process_to_ng = int(
                                np.count_nonzero((actual == "PROCESS") & (ship_mask | repair_mask))
                            )
                            score = (
                                correct_process,
                                correct,
                                auto,
                                -process_to_ng,
                                -risk_to_process,
                            )
                            if best is None or score > best[0]:
                                best = score, fitted
    if best is None:
        raise ValueError("no hybrid threshold combination satisfies the risk and accuracy budget")
    return best[1]


def _selection_score(
    predictions: Mapping[str, str],
    labels: Mapping[str, str],
) -> tuple[int, int, int, int, int]:
    risk_to_process = sum(
        labels[row_id] in {"RETURN", "REPAIR"} and predicted == "PROCESS"
        for row_id, predicted in predictions.items()
    )
    correct_process = sum(
        labels[row_id] == "PROCESS" and predicted == "PROCESS"
        for row_id, predicted in predictions.items()
    )
    correct = sum(labels[row_id] == predicted for row_id, predicted in predictions.items() if predicted)
    auto = sum(bool(value) for value in predictions.values())
    process_to_ng = sum(
        labels[row_id] == "PROCESS" and predicted in {"RETURN", "REPAIR"}
        for row_id, predicted in predictions.items()
    )
    return -risk_to_process, correct_process, correct, auto, -process_to_ng


def _validated_date_token(value: str, name: str) -> str:
    token = str(value)
    if len(token) != 10 or token[4] != "-" or token[7] != "-":
        raise ValueError(f"{name} must use YYYY-MM-DD")
    try:
        year, month, day = (int(part) for part in token.split("-"))
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM-DD") from error
    if year < 1 or not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return token


def _validated_month_token(value: str, name: str) -> str:
    token = str(value)
    if len(token) != 7 or token[4] != "-":
        raise ValueError(f"{name} must use YYYY-MM")
    try:
        year, month = (int(part) for part in token.split("-"))
    except ValueError as error:
        raise ValueError(f"{name} must use YYYY-MM") from error
    if year < 1 or not 1 <= month <= 12:
        raise ValueError(f"{name} must use YYYY-MM")
    return token


def _validate_partition(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    name: str,
    *,
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


def _validated_weights(
    weights: Sequence[float] | np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    if weights is None:
        return np.ones(row_count, dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != (row_count,):
        raise ValueError("sample_weights must align to training rows")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("sample_weights must be positive and finite")
    return array


def _boolean_target(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, bool],
    name: str,
) -> np.ndarray:
    missing = [row.row_id for row in rows if row.row_id not in labels]
    if missing:
        raise ValueError(f"missing {name} labels: {missing[:3]}")
    return np.asarray([int(bool(labels[row.row_id])) for row in rows], dtype=np.int8)


def _boolean_labels_sha256(labels: Mapping[str, bool]) -> str:
    return _sha256_json({row_id: bool(value) for row_id, value in sorted(labels.items())})


def _positive_probability(model: Any, matrix: sparse.csr_matrix) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    positive_index = list(model.classes_).index(1)
    return probabilities[:, positive_index]


def _raw_margin(model: Any, matrix: sparse.csr_matrix) -> np.ndarray:
    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(matrix), dtype=np.float64)
        if margin.ndim != 1:
            raise ValueError("binary decision margin must be one-dimensional")
        return margin
    return _logit(_positive_probability(model, matrix))


def _fit_calibrator(
    margin: np.ndarray,
    target: np.ndarray,
    *,
    random_state: int,
) -> Any:
    if len(set(target.tolist())) == 1:
        return _ConstantCalibrator(float(target[0]))
    values = np.asarray(margin, dtype=np.float64)
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    calibrator = LogisticRegression(max_iter=1_000, random_state=random_state)
    calibrator.fit(((values - center) / scale).reshape(-1, 1), target)
    return _ScaledLogisticCalibrator(center=center, scale=scale, model=calibrator)


def _calibrated_probability(
    head: Any,
    calibrator: Any,
    matrix: sparse.csr_matrix,
) -> np.ndarray:
    raw = _raw_margin(head, matrix)
    if isinstance(calibrator, _ConstantCalibrator):
        return np.full(len(raw), calibrator.value, dtype=np.float64)
    if isinstance(calibrator, _ScaledLogisticCalibrator):
        calibrated_input = ((raw - calibrator.center) / calibrator.scale).reshape(-1, 1)
        model = calibrator.model
    else:  # Backward compatibility for probe artifacts created before scaled calibration.
        calibrated_input = raw.reshape(-1, 1)
        model = calibrator
    output = model.predict_proba(calibrated_input)
    positive_index = list(model.classes_).index(1)
    return np.asarray(output[:, positive_index], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ConstantCalibrator:
    value: float


@dataclass(frozen=True, slots=True)
class _ScaledLogisticCalibrator:
    center: float
    scale: float
    model: LogisticRegression


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-8, 1 - 1e-8)
    return np.log(clipped / (1 - clipped))


def _row_ids_sha256(rows: Sequence[FeatureRow]) -> str:
    return _sha256_json(tuple(sorted(row.row_id for row in rows)))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")

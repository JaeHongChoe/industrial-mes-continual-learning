from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .candidates import ACTION_ORDER, CandidatePrediction, ScoreRow
from .data_contract import ACTION_LABELS, FeatureRow
from .features import FeatureSpec, RawFeatureEncoder


PROCESS_LABEL = "PROCESS"
REVIEW_LABEL = "REVIEW"
UNSAFE_LABELS = frozenset({"RETURN", "REPAIR"})
PROFILE_TARGETS = MappingProxyType(
    {
        "zero_unsafe": 1.0,
        "precision_0.995": 0.995,
        "precision_0.990": 0.990,
    }
)
CALIBRATION_METHOD = "binary_platt_sigmoid_v1"
FAMILY = "binary_process_linear"
OBJECTIVE = "process_vs_unsafe_ship_or_repair"
_TRAIN_END = "2025-07-31"
_CALIBRATION_MONTH = "2025-08"
_THRESHOLD_MONTH = "2025-09"
_SELECTOR_MONTHS = frozenset({"2025-10", "2025-11"})
_PRODUCT_IDENTITY_FIELDS = frozenset(
    {"product_key", "product_key_label", "process_code", "part_code", "part_name"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ProcessThresholdFit:
    profile: str
    target_precision: float
    threshold: float
    enabled: bool
    min_support: int
    automatic: int
    correct: int
    unsafe: int
    precision: float | None
    source_dataset_id: str
    history_sha256: str
    model_sha256: str
    feature_schema_sha256: str
    calibration_method: str
    calibration_cutoff: str
    threshold_start_date: str
    threshold_end_date: str
    row_universe_sha256: str
    score_rows_sha256: str
    sha256: str


class BinaryProcessCandidate:
    """Direct calibrated P(process) candidate over intake-time raw features."""

    family = FAMILY

    def __init__(
        self,
        *,
        customer_aware: bool = False,
        random_state: int = 17,
        max_iter: int = 1_500,
        score_batch_size: int = 10_000,
        max_score_rows: int = 500_000,
        validation_config: Mapping[str, Any] | None = None,
    ) -> None:
        if max_iter < 1 or score_batch_size < 1 or max_score_rows < 1:
            raise ValueError("model and scoring limits must be positive")
        self.customer_aware = bool(customer_aware)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.score_batch_size = int(score_batch_size)
        self.max_score_rows = int(max_score_rows)
        self.validation_config = dict(validation_config or {})
        self.feature_spec = FeatureSpec(
            include_customer_key=self.customer_aware,
            include_product_key=False,
        )
        _reject_product_identity(self.feature_spec)
        self._encoder: RawFeatureEncoder | None = None
        self._head: SGDClassifier | None = None
        self._calibrator: LogisticRegression | None = None
        self._training_ids: frozenset[str] = frozenset()
        self._calibration_ids: frozenset[str] = frozenset()
        self._training_cutoff: str | None = None
        self._calibration_cutoff: str | None = None
        self._calibration_row_sha256: str | None = None
        self._source_dataset_id: str | None = None
        self._history_sha256: str | None = None
        self._model_sha256: str | None = None
        self._feature_schema_sha256: str | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        assert self._encoder is not None
        return self._encoder.feature_names

    @property
    def source_dataset_id(self) -> str:
        self._require_fitted()
        assert self._source_dataset_id is not None
        return self._source_dataset_id

    @property
    def history_sha256(self) -> str:
        self._require_fitted()
        assert self._history_sha256 is not None
        return self._history_sha256

    @property
    def model_sha256(self) -> str:
        self._require_fitted()
        assert self._model_sha256 is not None
        return self._model_sha256

    @property
    def feature_schema_sha256(self) -> str:
        self._require_source_bound_artifact()
        assert self._feature_schema_sha256 is not None
        return self._feature_schema_sha256

    @property
    def calibration_cutoff(self) -> str:
        self._require_fitted()
        assert self._calibration_cutoff is not None
        return self._calibration_cutoff

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        calibration_rows: Iterable[FeatureRow],
        calibration_labels: Mapping[str, str],
        source_dataset_id: str,
        history_sha256: str,
    ) -> "BinaryProcessCandidate":
        if not str(source_dataset_id).strip():
            raise ValueError("source_dataset_id must not be empty")
        _require_sha256(history_sha256, "history_sha256")
        training = _validate_labeled_rows(
            rows,
            labels,
            partition="training",
            allowed=lambda row: row.date <= _TRAIN_END,
            expected="on or before 2025-07-31",
        )
        calibration = _validate_labeled_rows(
            calibration_rows,
            calibration_labels,
            partition="probability calibration",
            allowed=lambda row: (
                row.month == _CALIBRATION_MONTH
                and row.date[:7] == _CALIBRATION_MONTH
            ),
            expected="exactly 2025-08",
        )
        training_ids = frozenset(row.row_id for row in training)
        calibration_ids = frozenset(row.row_id for row in calibration)
        overlap = sorted(training_ids & calibration_ids)
        if overlap:
            raise ValueError(f"training/calibration identities overlap: {overlap[:3]}")
        training_cutoff = max(row.date for row in training)
        if any(row.date <= training_cutoff for row in calibration):
            raise ValueError("probability calibration must be strictly after training")
        train_target = _binary_target(training, labels)
        calibration_target = _binary_target(calibration, calibration_labels)
        if set(train_target) != {0, 1}:
            raise ValueError("binary training requires both process and unsafe labels")
        if set(calibration_target) != {0, 1}:
            raise ValueError("binary calibration requires both process and unsafe labels")

        encoder = RawFeatureEncoder(
            self.feature_spec,
            validation_config=self.validation_config,
        )
        training_matrix = encoder.fit_transform(training)
        head = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=self.random_state,
        )
        head.fit(training_matrix, np.asarray(train_target, dtype=np.int8))
        calibration_matrix = encoder.transform(calibration)
        raw_scores = np.asarray(head.decision_function(calibration_matrix), dtype=np.float64)
        calibrator = LogisticRegression(
            max_iter=1_000,
            random_state=self.random_state + 100,
        )
        calibrator.fit(raw_scores.reshape(-1, 1), np.asarray(calibration_target, dtype=np.int8))

        calibration_cutoff = max(row.date for row in calibration)
        calibration_row_sha256 = _labeled_rows_sha256(calibration, calibration_labels)
        model_sha256 = _sha256_pickle(
            {
                "objective": OBJECTIVE,
                "sourceDatasetId": source_dataset_id,
                "historySha256": history_sha256,
                "customerAware": self.customer_aware,
                "encoderManifest": encoder.encoder_manifest,
                "head": head,
                "calibrator": calibrator,
                "trainingIds": tuple(sorted(training_ids)),
                "calibrationIds": tuple(sorted(calibration_ids)),
                "trainingCutoff": training_cutoff,
                "calibrationCutoff": calibration_cutoff,
            }
        )

        self._encoder = encoder
        self._head = head
        self._calibrator = calibrator
        self._training_ids = training_ids
        self._calibration_ids = calibration_ids
        self._training_cutoff = training_cutoff
        self._calibration_cutoff = calibration_cutoff
        self._calibration_row_sha256 = calibration_row_sha256
        self._source_dataset_id = source_dataset_id
        self._history_sha256 = history_sha256
        self._model_sha256 = model_sha256
        self._feature_schema_sha256 = None
        return self

    def score_threshold_fit(self, rows: Iterable[FeatureRow]) -> tuple[ScoreRow, ...]:
        materialized = _validate_scoring_rows(
            rows,
            fitted_ids=self._fitted_ids,
            required_months=frozenset({_THRESHOLD_MONTH}),
            phase="threshold fit",
        )
        return self._score(materialized)

    def score_selector(self, rows: Iterable[FeatureRow]) -> tuple[ScoreRow, ...]:
        materialized = _validate_scoring_rows(
            rows,
            fitted_ids=self._fitted_ids,
            required_months=_SELECTOR_MONTHS,
            phase="selector",
        )
        return self._score(materialized)

    def score_postfit(self, rows: Iterable[FeatureRow]) -> tuple[ScoreRow, ...]:
        """Score any identity-safe rows strictly after probability calibration."""

        materialized = _validate_postfit_scoring_rows(
            rows,
            fitted_ids=self._fitted_ids,
            calibration_cutoff=self._calibration_cutoff,
        )
        return self._score(materialized)

    def persist_artifacts(
        self,
        output_dir: str | Path,
        *,
        stem: str,
    ) -> Mapping[str, str]:
        self._require_fitted()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            raise ValueError("artifact stem contains unsupported characters")
        assert self._encoder is not None
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        feature_schema_path = directory / f"{stem}.feature_schema.json"
        artifact_path = directory / f"{stem}.fitted.pkl"
        fitted_manifest_path = directory / f"{stem}.fitted_artifact.json"
        feature_schema_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "feature_schema",
                "featureNames": list(self._encoder.feature_names),
                "encoderSchemaSha256": self._encoder.schema_sha256,
                "sourceDatasetId": self.source_dataset_id,
                "historySha256": self.history_sha256,
                "objective": OBJECTIVE,
                "customerAware": self.customer_aware,
                "productIdentityIncluded": False,
            }
        )
        self._feature_schema_sha256 = hashlib.sha256(feature_schema_bytes).hexdigest()
        artifact_bytes = pickle.dumps(
            {
                "schemaVersion": 1,
                "kind": "binary_process_fitted_model",
                "model": self,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        fitted_manifest_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "fitted_artifact",
                "featureSchemaFile": feature_schema_path.name,
                "featureSchemaSha256": self.feature_schema_sha256,
                "artifactFile": artifact_path.name,
                "artifactSha256": artifact_sha256,
                "sourceDatasetId": self.source_dataset_id,
                "historySha256": self.history_sha256,
                "modelSha256": self.model_sha256,
                "objective": OBJECTIVE,
            }
        )
        _atomic_write(feature_schema_path, feature_schema_bytes)
        _atomic_write(artifact_path, artifact_bytes)
        _atomic_write(fitted_manifest_path, fitted_manifest_bytes)
        return MappingProxyType(
            {
                "feature_schema": str(feature_schema_path),
                "fitted_artifact": str(fitted_manifest_path),
            }
        )

    @property
    def _fitted_ids(self) -> frozenset[str]:
        self._require_fitted()
        return self._training_ids | self._calibration_ids

    def _score(self, rows: Sequence[FeatureRow]) -> tuple[ScoreRow, ...]:
        self._require_source_bound_artifact()
        if len(rows) > self.max_score_rows:
            raise ValueError(f"max_score_rows exceeded: {len(rows)} > {self.max_score_rows}")
        assert self._encoder is not None
        assert self._head is not None
        assert self._calibrator is not None
        output: list[ScoreRow] = []
        provenance = self._score_provenance()
        for start in range(0, len(rows), self.score_batch_size):
            batch = rows[start : start + self.score_batch_size]
            matrix = self._encoder.transform(batch)
            raw_scores = np.asarray(self._head.decision_function(matrix), dtype=np.float64)
            calibrated = self._calibrator.predict_proba(raw_scores.reshape(-1, 1))
            process_index = list(self._calibrator.classes_).index(1)
            for index, row in enumerate(batch):
                process_probability = float(calibrated[index, process_index])
                unsafe_probability = max(0.0, 1.0 - process_probability) / 2.0
                output.append(
                    ScoreRow(
                        row_id=row.row_id,
                        scores=MappingProxyType(
                            {
                                PROCESS_LABEL: process_probability,
                                "RETURN": unsafe_probability,
                                "REPAIR": unsafe_probability,
                            }
                        ),
                        family=self.family,
                        evidence=MappingProxyType(
                            {
                                "objective": OBJECTIVE,
                                "customerAware": self.customer_aware,
                                "productIdentityIncluded": False,
                                "provenance": provenance,
                            }
                        ),
                    )
                )
        return tuple(output)

    def _score_provenance(self) -> Mapping[str, Any]:
        assert self._calibration_row_sha256 is not None
        return _freeze_json(
            {
                "family": self.family,
                "objective": OBJECTIVE,
                "classOrder": ("unsafe", PROCESS_LABEL),
                "sourceDatasetId": self.source_dataset_id,
                "historySha256": self.history_sha256,
                "modelSha256": self.model_sha256,
                "featureSchemaSha256": self.feature_schema_sha256,
                "calibration": {
                    "state": "calibrated",
                    "method": CALIBRATION_METHOD,
                    "cutoff": self.calibration_cutoff,
                    "rowSha256": self._calibration_row_sha256,
                },
            }
        )

    def _require_fitted(self) -> None:
        if self._encoder is None or self._head is None or self._calibrator is None:
            raise RuntimeError("BinaryProcessCandidate must be fitted before use")

    def _require_source_bound_artifact(self) -> None:
        self._require_fitted()
        if self._feature_schema_sha256 is None:
            raise RuntimeError("persist source-bound artifacts before scoring")


def fit_process_threshold_profiles(
    score_rows: Iterable[ScoreRow],
    labels: Mapping[str, str],
    *,
    row_dates: Mapping[str, str],
    min_support: int = 10,
) -> Mapping[str, ProcessThresholdFit]:
    if min_support < 1:
        raise ValueError("min_support must be positive")
    rows = tuple(sorted(score_rows, key=lambda row: row.row_id))
    if not rows:
        raise ValueError("threshold score rows cannot be empty")
    row_ids = tuple(row.row_id for row in rows)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("threshold score rows contain duplicate identities")
    if set(labels) != set(row_ids) or set(row_dates) != set(row_ids):
        raise ValueError("threshold score, label, and date identities must match exactly")
    if any(label not in ACTION_LABELS for label in labels.values()):
        raise ValueError("threshold labels contain an unknown action")
    if {str(row_dates[row_id])[:7] for row_id in row_ids} != {_THRESHOLD_MONTH}:
        raise ValueError("threshold labels must come from exactly 2025-09")
    for row in rows:
        _validate_score_row(row)
    provenance = _common_score_provenance(rows)
    calibration = provenance["calibration"]
    if calibration["cutoff"][:7] != _CALIBRATION_MONTH:
        raise ValueError("threshold fitting requires an August calibration cutoff")
    start_date = min(str(row_dates[row_id]) for row_id in row_ids)
    end_date = max(str(row_dates[row_id]) for row_id in row_ids)
    row_universe_sha256 = _sha256_json(
        [{"rowId": row_id, "date": str(row_dates[row_id])} for row_id in sorted(row_ids)]
    )
    score_rows_sha256 = _sha256_json(
        [
            {
                "rowId": row.row_id,
                "date": str(row_dates[row.row_id]),
                "processProbability": float(row.scores[PROCESS_LABEL]),
                "actual": labels[row.row_id],
            }
            for row in rows
        ]
    )
    output: dict[str, ProcessThresholdFit] = {}
    for profile, target_precision in PROFILE_TARGETS.items():
        threshold, enabled, automatic, correct, unsafe, precision = _fit_profile(
            rows,
            labels,
            profile=profile,
            target_precision=target_precision,
            min_support=min_support,
        )
        fitted = ProcessThresholdFit(
            profile=profile,
            target_precision=target_precision,
            threshold=threshold,
            enabled=enabled,
            min_support=min_support,
            automatic=automatic,
            correct=correct,
            unsafe=unsafe,
            precision=precision,
            source_dataset_id=str(provenance["sourceDatasetId"]),
            history_sha256=str(provenance["historySha256"]),
            model_sha256=str(provenance["modelSha256"]),
            feature_schema_sha256=str(provenance["featureSchemaSha256"]),
            calibration_method=str(calibration["method"]),
            calibration_cutoff=str(calibration["cutoff"]),
            threshold_start_date=start_date,
            threshold_end_date=end_date,
            row_universe_sha256=row_universe_sha256,
            score_rows_sha256=score_rows_sha256,
            sha256="",
        )
        output[profile] = replace(
            fitted,
            sha256=_sha256_json(_threshold_fit_payload(fitted)),
        )
    return MappingProxyType(output)


def apply_process_threshold(
    score_rows: Iterable[ScoreRow],
    fitted: ProcessThresholdFit,
) -> tuple[CandidatePrediction, ...]:
    validate_process_threshold_fit(fitted)
    output: list[CandidatePrediction] = []
    for row in score_rows:
        _validate_score_row(row)
        provenance = _common_score_provenance((row,))
        expected = {
            "sourceDatasetId": fitted.source_dataset_id,
            "historySha256": fitted.history_sha256,
            "modelSha256": fitted.model_sha256,
            "featureSchemaSha256": fitted.feature_schema_sha256,
        }
        if any(str(provenance[key]) != value for key, value in expected.items()):
            raise ValueError("score provenance does not match the frozen process threshold")
        score = float(row.scores[PROCESS_LABEL])
        automatic = fitted.enabled and score >= fitted.threshold
        output.append(
            CandidatePrediction(
                row_id=row.row_id,
                label=PROCESS_LABEL if automatic else REVIEW_LABEL,
                score=score,
                family=row.family,
                evidence=MappingProxyType(
                    {
                        **dict(row.evidence),
                        "abstained": not automatic,
                        "winningClass": PROCESS_LABEL if automatic else REVIEW_LABEL,
                        "threshold": fitted.threshold,
                        "thresholdProfile": fitted.profile,
                        "thresholdFitSha256": fitted.sha256,
                        "objective": OBJECTIVE,
                    }
                ),
            )
        )
    return tuple(output)


def threshold_fit_receipt(fitted: ProcessThresholdFit) -> Mapping[str, Any]:
    validate_process_threshold_fit(fitted)
    return MappingProxyType({**_threshold_fit_payload(fitted), "sha256": fitted.sha256})


def validate_process_threshold_fit(fitted: ProcessThresholdFit) -> None:
    if not isinstance(fitted, ProcessThresholdFit):
        raise TypeError("process threshold fit must be ProcessThresholdFit")
    if fitted.profile not in PROFILE_TARGETS:
        raise ValueError("unknown process threshold profile")
    if fitted.target_precision != PROFILE_TARGETS[fitted.profile]:
        raise ValueError("process threshold target does not match its preregistered profile")
    if not 0 <= fitted.threshold <= 1 or fitted.min_support < 1:
        raise ValueError("process threshold receipt contains invalid limits")
    if fitted.enabled and fitted.automatic < fitted.min_support:
        raise ValueError("enabled process threshold has insufficient support")
    if not fitted.enabled and any((fitted.automatic, fitted.correct, fitted.unsafe)):
        raise ValueError("disabled process threshold must not claim automatic rows")
    if fitted.enabled:
        if fitted.precision is None or not math.isclose(
            fitted.precision,
            fitted.correct / fitted.automatic,
        ):
            raise ValueError("process threshold precision does not reconcile")
        if fitted.profile == "zero_unsafe" and fitted.unsafe != 0:
            raise ValueError("zero_unsafe threshold contains unsafe acceptances")
        if fitted.profile != "zero_unsafe" and fitted.precision < fitted.target_precision:
            raise ValueError("process threshold misses its precision target")
    for name, value in (
        ("history_sha256", fitted.history_sha256),
        ("model_sha256", fitted.model_sha256),
        ("feature_schema_sha256", fitted.feature_schema_sha256),
        ("row_universe_sha256", fitted.row_universe_sha256),
        ("score_rows_sha256", fitted.score_rows_sha256),
        ("sha256", fitted.sha256),
    ):
        _require_sha256(value, name)
    if fitted.calibration_method != CALIBRATION_METHOD:
        raise ValueError("process threshold calibration method is unsupported")
    if fitted.calibration_cutoff[:7] != _CALIBRATION_MONTH:
        raise ValueError("process threshold calibration cutoff must be in August")
    if (
        fitted.threshold_start_date[:7] != _THRESHOLD_MONTH
        or fitted.threshold_end_date[:7] != _THRESHOLD_MONTH
    ):
        raise ValueError("process threshold labels must be restricted to September")
    if _sha256_json(_threshold_fit_payload(fitted)) != fitted.sha256:
        raise ValueError("process threshold receipt digest mismatch")


def load_binary_process_artifact(
    fitted_manifest: str | Path,
) -> BinaryProcessCandidate:
    manifest_path = Path(fitted_manifest).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("binary fitted artifact manifest must be readable JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("binary fitted artifact manifest must be an object")
    artifact_file = manifest.get("artifactFile")
    schema_file = manifest.get("featureSchemaFile")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("kind") != "fitted_artifact"
        or manifest.get("objective") != OBJECTIVE
        or not isinstance(artifact_file, str)
        or Path(artifact_file).name != artifact_file
        or not isinstance(schema_file, str)
        or Path(schema_file).name != schema_file
    ):
        raise ValueError("unsupported binary fitted artifact manifest")
    artifact_path = manifest_path.parent / artifact_file
    schema_path = manifest_path.parent / schema_file
    try:
        artifact_bytes = artifact_path.read_bytes()
        schema_bytes = schema_path.read_bytes()
        schema = json.loads(schema_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("binary fitted artifact bytes are unavailable") from error
    if hashlib.sha256(artifact_bytes).hexdigest() != manifest.get("artifactSha256"):
        raise ValueError("binary fitted artifact digest mismatch")
    if hashlib.sha256(schema_bytes).hexdigest() != manifest.get("featureSchemaSha256"):
        raise ValueError("binary feature schema digest mismatch")
    try:
        payload = pickle.loads(artifact_bytes)
    except Exception as error:  # noqa: BLE001 - normalize trusted artifact decode failures.
        raise ValueError("binary fitted artifact cannot be decoded") from error
    model = payload.get("model") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != 1
        or payload.get("kind") != "binary_process_fitted_model"
        or not isinstance(model, BinaryProcessCandidate)
    ):
        raise ValueError("unsupported binary fitted artifact payload")
    if (
        manifest.get("sourceDatasetId") != model.source_dataset_id
        or manifest.get("historySha256") != model.history_sha256
        or manifest.get("modelSha256") != model.model_sha256
        or manifest.get("featureSchemaSha256") != model.feature_schema_sha256
        or not isinstance(schema, Mapping)
        or tuple(schema.get("featureNames") or ()) != model.feature_names
        or schema.get("sourceDatasetId") != model.source_dataset_id
        or schema.get("historySha256") != model.history_sha256
        or schema.get("productIdentityIncluded") is not False
    ):
        raise ValueError("binary fitted artifact source/model/schema binding mismatch")
    return model


def _fit_profile(
    rows: Sequence[ScoreRow],
    labels: Mapping[str, str],
    *,
    profile: str,
    target_precision: float,
    min_support: int,
) -> tuple[float, bool, int, int, int, float | None]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row.scores[PROCESS_LABEL]), row.row_id),
    )
    best: tuple[int, float, int, int, float] | None = None
    automatic = correct = unsafe = 0
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index].scores[PROCESS_LABEL])
        while index < len(ordered) and math.isclose(
            float(ordered[index].scores[PROCESS_LABEL]),
            threshold,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            row = ordered[index]
            automatic += 1
            if labels[row.row_id] == PROCESS_LABEL:
                correct += 1
            else:
                unsafe += 1
            index += 1
        precision = correct / automatic
        eligible = automatic >= min_support and (
            unsafe == 0 if profile == "zero_unsafe" else precision >= target_precision
        )
        if eligible and (best is None or automatic > best[0]):
            best = (automatic, threshold, correct, unsafe, precision)
    if best is None:
        return 1.0, False, 0, 0, 0, None
    support, threshold, correct, unsafe, precision = best
    return threshold, True, support, correct, unsafe, precision


def _validate_labeled_rows(
    rows: Iterable[FeatureRow],
    labels: Mapping[str, str],
    *,
    partition: str,
    allowed: Any,
    expected: str,
) -> tuple[FeatureRow, ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError(f"{partition} rows cannot be empty")
    row_ids = tuple(row.row_id for row in materialized)
    if len(row_ids) != len(set(row_ids)) or set(labels) != set(row_ids):
        raise ValueError(f"{partition} rows and labels must match exactly without duplicates")
    if any(label not in ACTION_LABELS for label in labels.values()):
        raise ValueError(f"{partition} labels contain an unknown action")
    if any(not allowed(row) for row in materialized):
        raise ValueError(f"{partition} labels must be from {expected}")
    return materialized


def _validate_scoring_rows(
    rows: Iterable[FeatureRow],
    *,
    fitted_ids: frozenset[str],
    required_months: frozenset[str],
    phase: str,
) -> tuple[FeatureRow, ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError(f"{phase} rows cannot be empty")
    row_ids = tuple(row.row_id for row in materialized)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"{phase} rows contain duplicate identities")
    overlap = sorted(set(row_ids) & fitted_ids)
    if overlap:
        raise ValueError(f"{phase} identities overlap fitted labels: {overlap[:3]}")
    observed_months = {row.month for row in materialized}
    if (
        observed_months != set(required_months)
        or any(row.date[:7] != row.month for row in materialized)
    ):
        expected = " and ".join(sorted(required_months))
        raise ValueError(f"{phase} rows must cover exactly {expected}")
    return materialized


def _validate_postfit_scoring_rows(
    rows: Iterable[FeatureRow],
    *,
    fitted_ids: frozenset[str],
    calibration_cutoff: str | None,
) -> tuple[FeatureRow, ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("postfit rows cannot be empty")
    row_ids = tuple(row.row_id for row in materialized)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("postfit rows contain duplicate identities")
    overlap = sorted(set(row_ids) & fitted_ids)
    if overlap:
        raise ValueError(f"postfit identities overlap fitted labels: {overlap[:3]}")
    if not calibration_cutoff:
        raise RuntimeError("postfit scoring requires a calibration cutoff")
    if any(
        row.date <= calibration_cutoff or row.date[:7] != row.month
        for row in materialized
    ):
        raise ValueError("postfit rows must be after calibration cutoff")
    return materialized


def _binary_target(rows: Sequence[FeatureRow], labels: Mapping[str, str]) -> list[int]:
    return [int(labels[row.row_id] == PROCESS_LABEL) for row in rows]


def _validate_score_row(row: ScoreRow) -> None:
    if set(row.scores) != set(ACTION_ORDER):
        raise ValueError("binary process score rows must expose the three-action adapter")
    values = [float(row.scores[action]) for action in ACTION_ORDER]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("binary process scores must be finite probabilities")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("binary process score adapter must sum to one")
    if row.family != FAMILY:
        raise ValueError("binary process score family mismatch")


def _common_score_provenance(rows: Sequence[ScoreRow]) -> Mapping[str, Any]:
    payloads: list[dict[str, Any]] = []
    for row in rows:
        provenance = row.evidence.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("binary process scores require source-bound provenance")
        plain = _plain_json(provenance)
        calibration = plain.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("state") != "calibrated"
            or calibration.get("method") != CALIBRATION_METHOD
        ):
            raise ValueError("binary process scores require calibrated provenance")
        for key in ("historySha256", "modelSha256", "featureSchemaSha256"):
            _require_sha256(str(plain.get(key) or ""), key)
        _require_sha256(str(calibration.get("rowSha256") or ""), "calibration row SHA-256")
        if not str(plain.get("sourceDatasetId") or "").strip():
            raise ValueError("binary process score provenance requires a source dataset")
        payloads.append(plain)
    digests = {_sha256_json(payload) for payload in payloads}
    if len(digests) != 1:
        raise ValueError("binary process score rows contain mixed provenance")
    return _freeze_json(payloads[0])


def _threshold_fit_payload(fitted: ProcessThresholdFit) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "binary_process_threshold_fit",
        "objective": OBJECTIVE,
        "profile": fitted.profile,
        "targetPrecision": fitted.target_precision,
        "threshold": fitted.threshold,
        "enabled": fitted.enabled,
        "minSupport": fitted.min_support,
        "metrics": {
            "automatic": fitted.automatic,
            "correct": fitted.correct,
            "unsafe": fitted.unsafe,
            "precision": fitted.precision,
        },
        "sourceDatasetId": fitted.source_dataset_id,
        "historySha256": fitted.history_sha256,
        "modelSha256": fitted.model_sha256,
        "featureSchemaSha256": fitted.feature_schema_sha256,
        "calibration": {
            "method": fitted.calibration_method,
            "cutoff": fitted.calibration_cutoff,
        },
        "thresholdDateInterval": {
            "start": fitted.threshold_start_date,
            "end": fitted.threshold_end_date,
        },
        "rowUniverseSha256": fitted.row_universe_sha256,
        "scoreRowsSha256": fitted.score_rows_sha256,
        "selectionSemantics": "direct_process_probability_threshold_else_review",
    }


def _labeled_rows_sha256(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
) -> str:
    return _sha256_json(
        [
            {"rowId": row.row_id, "date": row.date, "actual": labels[row.row_id]}
            for row in sorted(rows, key=lambda item: item.row_id)
        ]
    )


def _reject_product_identity(spec: FeatureSpec) -> None:
    requested = {
        *spec.text_fields,
        *spec.categorical_fields,
        *spec.numeric_fields,
    }
    direct = requested & _PRODUCT_IDENTITY_FIELDS
    if spec.include_product_key or direct:
        raise ValueError(f"product identity is forbidden: {sorted(direct)}")


def _require_sha256(value: str, name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(str(value)):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _sha256_pickle(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)

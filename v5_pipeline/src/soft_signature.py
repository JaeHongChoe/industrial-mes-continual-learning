from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .candidates import ACTION_ORDER, CandidatePrediction, ScoreRow
from .data_contract import ACTION_LABELS, FeatureRow


FAMILY = "soft_signature_lattice"
PROCESS_SAFE_PROFILE = "process_safe"
FULL_MULTICLASS_PROFILE = "full_multiclass"
FIT_MONTH = "2025-07"
CALIBRATION_MONTH = "2025-08"
THRESHOLD_MONTH = "2025-09"
SCORE_MONTHS = ("2025-10", "2025-11")

TEXT_FIELDS = ("type", "symptom", "detail")
COUNT_FIELDS = ("intake_count", "service_count", "treatment_count")
FEATURE_NAMES = (
    "customer_key",
    "product_key",
    *(f"raw:{name}" for name in TEXT_FIELDS),
    *(f"fixed_bin:{name}" for name in COUNT_FIELDS),
)
LATTICE_LEVELS = (
    "customer_product",
    "cross_customer_product",
    "customer_coarse_raw",
    "global",
)

# These values are fixed before selector scoring. They are not searched on August,
# September, or the October-November selector period.
MIN_LEVEL_SUPPORT = 5
PARENT_STRENGTH = 5.0
GLOBAL_STRENGTH = 3.0
CALIBRATOR_C = 1.0
CALIBRATOR_MAX_ITER = 1_000
CALIBRATOR_RANDOM_STATE = 17
THRESHOLD_MIN_SUPPORT = 10
FULL_MULTICLASS_PRECISION = 0.995
COUNT_BIN_EDGES = (0.0, 1.0, 2.0, 4.0, 9.0)
COUNT_BIN_LABELS = ("0", "1", "2", "3-4", "5-9", "10+")
MAX_COARSE_DETAIL_TOKENS = 4


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"unsupported soft signature JSON value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


_FEATURE_SCHEMA_PAYLOAD = {
    "schemaVersion": 1,
    "kind": "feature_schema",
    "family": FAMILY,
    "featureNames": list(FEATURE_NAMES),
    "intakeTimeOnly": True,
    "rawTextFields": list(TEXT_FIELDS),
    "rawCountFields": list(COUNT_FIELDS),
    "countBinEdges": list(COUNT_BIN_EDGES),
    "postAuditDerivedTags": [],
}
FEATURE_SCHEMA_BYTES = _canonical_json_bytes(_FEATURE_SCHEMA_PAYLOAD)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(FEATURE_SCHEMA_BYTES).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceBinding:
    source_dataset_id: str
    source_sha256: str
    feature_schema_sha256: str = FEATURE_SCHEMA_SHA256

    def __post_init__(self) -> None:
        if not self.source_dataset_id.strip():
            raise ValueError("source_dataset_id must not be empty")
        for name, value in (
            ("source_sha256", self.source_sha256),
            ("feature_schema_sha256", self.feature_schema_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256:
            raise ValueError("source binding feature schema does not match soft signature schema")

    def payload(self) -> dict[str, str]:
        return {
            "sourceDatasetId": self.source_dataset_id,
            "sourceSha256": self.source_sha256,
            "featureSchemaSha256": self.feature_schema_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.payload())


@dataclass(frozen=True, slots=True)
class ThresholdMetric:
    automatic: int
    correct: int
    unsafe: int
    precision: float | None


@dataclass(frozen=True, slots=True)
class SoftSignatureProfile:
    name: str
    thresholds: Mapping[str, float]
    enabled: Mapping[str, bool]
    metrics: Mapping[str, ThresholdMetric]
    probability_semantics: str
    calibration_method: str
    calibrator_sha256: str
    threshold_rows_sha256: str
    threshold_start_date: str
    threshold_end_date: str
    source_binding_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _LinearCalibrator:
    mode: str
    classes: tuple[int, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    method: str
    calibration_rows_sha256: str
    calibration_start_date: str
    calibration_end_date: str
    sha256: str

    def predict(self, raw_scores: Mapping[str, float]) -> dict[str, float]:
        raw = np.asarray([float(raw_scores[action]) for action in ACTION_ORDER])
        clipped = np.clip(raw, 1e-12, 1.0)
        if self.mode == "process_binary":
            process = float(clipped[0])
            x = np.asarray([math.log(process / max(1.0 - process, 1e-12))])
            logit = float(np.dot(np.asarray(self.coefficients[0]), x) + self.intercepts[0])
            calibrated_process = _sigmoid(logit)
            residual = 1.0 - calibrated_process
            risk_total = float(clipped[1] + clipped[2])
            if risk_total <= 0:
                ship_share = repair_share = 0.5
            else:
                ship_share = float(clipped[1] / risk_total)
                repair_share = float(clipped[2] / risk_total)
            return {
                "PROCESS": calibrated_process,
                "RETURN": residual * ship_share,
                "REPAIR": residual * repair_share,
            }
        if self.mode != "full_multiclass":
            raise ValueError(f"unsupported calibrator mode: {self.mode}")
        x = np.log(clipped)
        logits = np.asarray(self.coefficients) @ x + np.asarray(self.intercepts)
        probabilities = _softmax(logits)
        by_class = {class_index: float(value) for class_index, value in zip(self.classes, probabilities)}
        return {
            action: by_class.get(index, 0.0)
            for index, action in enumerate(ACTION_ORDER)
        }

    def payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "classes": list(self.classes),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "method": self.method,
            "calibrationRowsSha256": self.calibration_rows_sha256,
            "dateInterval": {
                "start": self.calibration_start_date,
                "end": self.calibration_end_date,
            },
        }


class SoftSignatureCandidate:
    """Leakage-safe empirical-Bayes lattice over frozen intake-time raw fields."""

    def __init__(self) -> None:
        self._counts: dict[str, dict[tuple[str, ...], Counter[str]]] = {
            level: {} for level in LATTICE_LEVELS
        }
        self._overall_global: Counter[str] = Counter()
        self._source_binding: SourceBinding | None = None
        self._fit_ids: frozenset[str] = frozenset()
        self._calibration_ids: frozenset[str] = frozenset()
        self._threshold_ids: frozenset[str] = frozenset()
        self._fit_start_date = ""
        self._fit_end_date = ""
        self._calibrators: dict[str, _LinearCalibrator] = {}
        self._profiles: dict[str, SoftSignatureProfile] = {}
        self._model_sha256 = ""

    @property
    def source_binding(self) -> SourceBinding | None:
        return self._source_binding

    @property
    def profiles(self) -> Mapping[str, SoftSignatureProfile]:
        return MappingProxyType(dict(self._profiles))

    @property
    def model_sha256(self) -> str:
        if not self._model_sha256:
            raise RuntimeError("SoftSignatureCandidate must be fitted before model digest access")
        return self._model_sha256

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        source_binding: SourceBinding,
    ) -> "SoftSignatureCandidate":
        if self._source_binding is not None:
            raise RuntimeError("SoftSignatureCandidate is already fitted")
        materialized = _validate_labelled_rows(
            rows,
            labels,
            expected_month=FIT_MONTH,
            chronology_message="soft signature fit is July-only",
            source_binding=source_binding,
        )
        mutable_counts: dict[str, defaultdict[tuple[str, ...], Counter[str]]] = {
            level: defaultdict(Counter) for level in LATTICE_LEVELS
        }
        overall: Counter[str] = Counter()
        for row in materialized:
            label = labels[row.row_id]
            overall[label] += 1
            signatures = _signatures(row)
            for level in LATTICE_LEVELS:
                mutable_counts[level][signatures[level]][label] += 1
        self._counts = {
            level: {key: Counter(value) for key, value in mutable_counts[level].items()}
            for level in LATTICE_LEVELS
        }
        self._overall_global = overall
        self._source_binding = source_binding
        self._fit_ids = frozenset(row.row_id for row in materialized)
        self._fit_start_date, self._fit_end_date = _date_interval(materialized)
        self._model_sha256 = self._compute_model_sha256()
        return self

    def calibrate(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        source_binding: SourceBinding,
    ) -> "SoftSignatureCandidate":
        self._require_fitted()
        self._require_source_binding(source_binding)
        materialized = tuple(rows)
        _reject_overlap(materialized, self._fit_ids, "calibration rows overlap July fit identities")
        materialized = _validate_labelled_rows(
            materialized,
            labels,
            expected_month=CALIBRATION_MONTH,
            chronology_message="soft signature calibration is August-only",
            source_binding=source_binding,
        )
        raw_scores = [self._empirical_score(row)[0] for row in materialized]
        self._calibrators = {
            PROCESS_SAFE_PROFILE: _fit_process_calibrator(materialized, raw_scores, labels),
            FULL_MULTICLASS_PROFILE: _fit_multiclass_calibrator(materialized, raw_scores, labels),
        }
        self._calibration_ids = frozenset(row.row_id for row in materialized)
        return self

    def fit_threshold_profiles(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        source_binding: SourceBinding,
        include_full_multiclass: bool = False,
    ) -> "SoftSignatureCandidate":
        self._require_calibrated()
        self._require_source_binding(source_binding)
        materialized = _validate_labelled_rows(
            rows,
            labels,
            expected_month=THRESHOLD_MONTH,
            chronology_message="soft signature threshold fit is September-only",
            source_binding=source_binding,
        )
        _reject_overlap(
            materialized,
            self._fit_ids | self._calibration_ids,
            "threshold rows overlap fit or calibration identities",
        )
        profiles = [PROCESS_SAFE_PROFILE]
        if include_full_multiclass:
            profiles.append(FULL_MULTICLASS_PROFILE)
        fitted: dict[str, SoftSignatureProfile] = {}
        for profile_name in profiles:
            scores = self._calibrated_scores(materialized, profile_name)
            fitted[profile_name] = _fit_profile(
                profile_name,
                scores,
                labels,
                source_binding=source_binding,
                calibrator=self._calibrators[profile_name],
                rows=materialized,
            )
        self._profiles = fitted
        self._threshold_ids = frozenset(row.row_id for row in materialized)
        return self

    def predict_empirical_scores(
        self,
        rows: Iterable[FeatureRow],
        *,
        source_binding: SourceBinding,
    ) -> tuple[ScoreRow, ...]:
        self._require_fitted()
        self._require_source_binding(source_binding)
        materialized = tuple(rows)
        _reject_overlap(materialized, self._fit_ids, "score rows overlap July fit identities")
        if not materialized or any(row.month not in {CALIBRATION_MONTH, THRESHOLD_MONTH, *SCORE_MONTHS} for row in materialized):
            raise ValueError("empirical score rows must be August through November 2025")
        _validate_score_identities(materialized, source_binding)
        output = []
        for row in materialized:
            scores, evidence = self._empirical_score(row)
            output.append(
                ScoreRow(
                    row_id=row.row_id,
                    scores=MappingProxyType(scores),
                    family=FAMILY,
                    evidence=MappingProxyType(
                        {
                            **evidence,
                            "scoreSemantics": "empirical_bayes_score_not_calibrated_probability",
                            "sourceBinding": MappingProxyType(source_binding.payload()),
                            "modelSha256": self._model_sha256,
                        }
                    ),
                )
            )
        return tuple(output)

    def predict_scores(
        self,
        rows: Iterable[FeatureRow],
        *,
        source_binding: SourceBinding,
        profile: str = PROCESS_SAFE_PROFILE,
    ) -> tuple[ScoreRow, ...]:
        self._require_profile(profile)
        self._require_source_binding(source_binding)
        materialized = tuple(rows)
        _reject_overlap(
            materialized,
            self._fit_ids | self._calibration_ids | self._threshold_ids,
            "score rows overlap fit, calibration, or threshold identities",
        )
        if not materialized or any(row.month not in SCORE_MONTHS for row in materialized):
            raise ValueError("soft signature scoring is October-November 2025 only")
        _validate_score_identities(materialized, source_binding)
        return self._calibrated_scores(materialized, profile)

    def predict(
        self,
        rows: Iterable[FeatureRow],
        *,
        source_binding: SourceBinding,
        profile: str = PROCESS_SAFE_PROFILE,
    ) -> tuple[CandidatePrediction, ...]:
        profile_fit = self._require_profile(profile)
        scores = self.predict_scores(rows, source_binding=source_binding, profile=profile)
        output = []
        for scored in scores:
            winner = max(ACTION_ORDER, key=lambda action: scored.scores[action])
            score = float(scored.scores[winner])
            threshold = float(profile_fit.thresholds[winner])
            automatic = bool(profile_fit.enabled[winner]) and score >= threshold
            output.append(
                CandidatePrediction(
                    row_id=scored.row_id,
                    label=winner if automatic else "REVIEW",
                    score=score,
                    family=FAMILY,
                    evidence=MappingProxyType(
                        {
                            **dict(scored.evidence),
                            "abstained": not automatic,
                            "winningClass": winner,
                            "threshold": threshold,
                            "classEnabled": bool(profile_fit.enabled[winner]),
                            "fallbackPolicy": "review",
                        }
                    ),
                )
            )
        return tuple(output)

    def persist_artifacts(
        self,
        output_dir: str | Path,
        *,
        stem: str,
    ) -> Mapping[str, str]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            raise ValueError("artifact stem contains unsupported characters")
        if not self._profiles or self._source_binding is None:
            raise RuntimeError("soft signature profiles must be fitted before persistence")
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        schema_path = directory / f"{stem}.feature_schema.json"
        artifact_path = directory / f"{stem}.fitted_model.json"
        manifest_path = directory / f"{stem}.fitted_artifact.json"
        artifact_bytes = _canonical_json_bytes(self._artifact_payload())
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        manifest_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "fitted_artifact",
                "family": FAMILY,
                "featureSchemaFile": schema_path.name,
                "featureSchemaSha256": FEATURE_SCHEMA_SHA256,
                "artifactFile": artifact_path.name,
                "artifactSha256": artifact_sha256,
                "sourceBinding": self._source_binding.payload(),
                "dateBinding": _date_binding_payload(),
                "modelSha256": self._model_sha256,
                "profileSha256": {
                    name: profile.sha256 for name, profile in sorted(self._profiles.items())
                },
            }
        )
        _atomic_write(schema_path, FEATURE_SCHEMA_BYTES)
        _atomic_write(artifact_path, artifact_bytes)
        _atomic_write(manifest_path, manifest_bytes)
        return MappingProxyType(
            {
                "feature_schema": str(schema_path),
                "fitted_artifact": str(manifest_path),
            }
        )

    def _empirical_score(self, row: FeatureRow) -> tuple[dict[str, float], dict[str, Any]]:
        signatures = _signatures(row)
        uniform = {action: 1.0 / len(ACTION_ORDER) for action in ACTION_ORDER}
        overall = _shrink(self._overall_global, uniform, GLOBAL_STRENGTH)
        global_counts = self._counts["global"].get(signatures["global"], Counter())
        global_probability = _shrink(global_counts, overall, PARENT_STRENGTH)
        cross_counts = self._counts["cross_customer_product"].get(
            signatures["cross_customer_product"], Counter()
        )
        cross_probability = _shrink(cross_counts, global_probability, PARENT_STRENGTH)
        customer_product_counts = self._counts["customer_product"].get(
            signatures["customer_product"], Counter()
        )
        customer_product_probability = _shrink(
            customer_product_counts,
            cross_probability,
            PARENT_STRENGTH,
        )
        coarse_counts = self._counts["customer_coarse_raw"].get(
            signatures["customer_coarse_raw"], Counter()
        )
        coarse_probability = _shrink(coarse_counts, global_probability, PARENT_STRENGTH)

        options = (
            ("customer_product", customer_product_counts, customer_product_probability),
            ("cross_customer_product", cross_counts, cross_probability),
            ("customer_coarse_raw", coarse_counts, coarse_probability),
        )
        selected_level = "global"
        selected_counts = global_counts if sum(global_counts.values()) >= MIN_LEVEL_SUPPORT else self._overall_global
        selected_probability = global_probability if sum(global_counts.values()) >= MIN_LEVEL_SUPPORT else overall
        for level, counts, probabilities in options:
            if sum(counts.values()) >= MIN_LEVEL_SUPPORT:
                selected_level = level
                selected_counts = counts
                selected_probability = probabilities
                break
        count_bins = {
            field: value for field, value in zip(COUNT_FIELDS, signatures["global"])
        }
        evidence = {
            "level": selected_level,
            "support": int(sum(selected_counts.values())),
            "classCounts": MappingProxyType(
                {action: int(selected_counts[action]) for action in ACTION_ORDER}
            ),
            "parentShrinkageStrength": PARENT_STRENGTH,
            "countBins": MappingProxyType(count_bins),
            "signatureSha256": _sha256_json(signatures[selected_level]),
            "rawFields": TEXT_FIELDS + COUNT_FIELDS,
            "postAuditDerivedTagsUsed": False,
        }
        return (
            {action: float(selected_probability[action]) for action in ACTION_ORDER},
            evidence,
        )

    def _calibrated_scores(
        self,
        rows: Sequence[FeatureRow],
        profile: str,
    ) -> tuple[ScoreRow, ...]:
        calibrator = self._calibrators[profile]
        profile_fit = self._profiles.get(profile)
        output = []
        for row in rows:
            raw_scores, raw_evidence = self._empirical_score(row)
            probabilities = calibrator.predict(raw_scores)
            probabilities = _normalise_probabilities(probabilities)
            provenance = self._provenance(profile, profile_fit)
            output.append(
                ScoreRow(
                    row_id=row.row_id,
                    scores=MappingProxyType(probabilities),
                    family=FAMILY,
                    evidence=MappingProxyType(
                        {
                            **raw_evidence,
                            "profile": profile,
                            "probabilitySemantics": (
                                "august_calibrated_process_probability_with_empirical_risk_split"
                                if profile == PROCESS_SAFE_PROFILE
                                else "august_calibrated_multiclass_probability"
                            ),
                            "provenance": provenance,
                        }
                    ),
                )
            )
        return tuple(output)

    def _provenance(
        self,
        profile: str,
        fitted_profile: SoftSignatureProfile | None,
    ) -> Mapping[str, Any]:
        binding = self._source_binding
        calibrator = self._calibrators[profile]
        if binding is None:
            raise RuntimeError("source binding is unavailable")
        threshold_payload: Mapping[str, Any]
        if fitted_profile is None:
            threshold_payload = MappingProxyType(
                {
                    "state": "pending",
                    "dateInterval": None,
                    "sha256": None,
                }
            )
        else:
            threshold_payload = MappingProxyType(
                {
                    "state": "fitted",
                    "dateInterval": MappingProxyType(
                        {
                            "start": fitted_profile.threshold_start_date,
                            "end": fitted_profile.threshold_end_date,
                        }
                    ),
                    "sha256": fitted_profile.sha256,
                }
            )
        return MappingProxyType(
            {
                "family": FAMILY,
                "modelSha256": self._model_sha256,
                "calibratorSha256": calibrator.sha256,
                "sourceDatasetId": binding.source_dataset_id,
                "sourceSha256": binding.source_sha256,
                "featureSchemaSha256": binding.feature_schema_sha256,
                "sourceBindingSha256": binding.sha256,
                "classOrder": ACTION_ORDER,
                "fit": MappingProxyType(
                    {
                        "dateInterval": MappingProxyType(
                            {"start": self._fit_start_date, "end": self._fit_end_date}
                        ),
                        "month": FIT_MONTH,
                    }
                ),
                "calibration": MappingProxyType(
                    {
                        "state": "calibrated",
                        "method": calibrator.method,
                        "dateInterval": MappingProxyType(
                            {
                                "start": calibrator.calibration_start_date,
                                "end": calibrator.calibration_end_date,
                            }
                        ),
                        "cutoff": calibrator.calibration_end_date,
                        "rowSha256": calibrator.calibration_rows_sha256,
                    }
                ),
                "thresholdFit": threshold_payload,
            }
        )

    def _artifact_payload(self) -> dict[str, Any]:
        if self._source_binding is None:
            raise RuntimeError("source binding is unavailable")
        return {
            "schemaVersion": 1,
            "kind": "soft_signature_fitted_model",
            "family": FAMILY,
            "sourceBinding": self._source_binding.payload(),
            "dateBinding": _date_binding_payload(),
            "fixedHyperparameters": _fixed_hyperparameter_payload(),
            "featureSchemaSha256": FEATURE_SCHEMA_SHA256,
            "counts": {
                level: _serialise_level(self._counts[level]) for level in LATTICE_LEVELS
            },
            "overallGlobal": _counter_payload(self._overall_global),
            "fitIds": sorted(self._fit_ids),
            "calibrationIds": sorted(self._calibration_ids),
            "thresholdIds": sorted(self._threshold_ids),
            "fitDateInterval": {
                "start": self._fit_start_date,
                "end": self._fit_end_date,
            },
            "calibrators": {
                name: calibrator.payload()
                for name, calibrator in sorted(self._calibrators.items())
            },
            "profiles": {
                name: _profile_payload(profile)
                for name, profile in sorted(self._profiles.items())
            },
            "modelSha256": self._model_sha256,
        }

    def _compute_model_sha256(self) -> str:
        if self._source_binding is None:
            raise RuntimeError("source binding is unavailable")
        return _sha256_json(
            {
                "family": FAMILY,
                "sourceBinding": self._source_binding.payload(),
                "fixedHyperparameters": _fixed_hyperparameter_payload(),
                "counts": {
                    level: _serialise_level(self._counts[level]) for level in LATTICE_LEVELS
                },
                "overallGlobal": _counter_payload(self._overall_global),
                "fitIds": sorted(self._fit_ids),
                "fitDateInterval": {
                    "start": self._fit_start_date,
                    "end": self._fit_end_date,
                },
            }
        )

    def _require_fitted(self) -> None:
        if self._source_binding is None or not self._fit_ids or not self._model_sha256:
            raise RuntimeError("SoftSignatureCandidate must be fitted first")

    def _require_calibrated(self) -> None:
        self._require_fitted()
        if set(self._calibrators) != {PROCESS_SAFE_PROFILE, FULL_MULTICLASS_PROFILE}:
            raise RuntimeError("SoftSignatureCandidate must be August-calibrated first")

    def _require_profile(self, profile: str) -> SoftSignatureProfile:
        self._require_calibrated()
        fitted = self._profiles.get(profile)
        if fitted is None:
            raise ValueError(f"soft signature profile is not fitted: {profile}")
        return fitted

    def _require_source_binding(self, source_binding: SourceBinding) -> None:
        if source_binding != self._source_binding:
            raise ValueError("source binding does not match fitted soft signature candidate")


def load_soft_signature_artifact(fitted_manifest: str | Path) -> SoftSignatureCandidate:
    manifest_path = Path(fitted_manifest).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("soft signature manifest must be readable JSON") from error
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schemaVersion") != 1
        or manifest.get("kind") != "fitted_artifact"
        or manifest.get("family") != FAMILY
    ):
        raise ValueError("unsupported soft signature manifest schema")
    schema_path = (manifest_path.parent / str(manifest.get("featureSchemaFile") or "")).resolve()
    artifact_path = (manifest_path.parent / str(manifest.get("artifactFile") or "")).resolve()
    try:
        schema_bytes = schema_path.read_bytes()
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("soft signature artifact files are unavailable") from error
    if (
        hashlib.sha256(schema_bytes).hexdigest() != manifest.get("featureSchemaSha256")
        or schema_bytes != FEATURE_SCHEMA_BYTES
    ):
        raise ValueError("soft signature feature schema digest mismatch")
    if hashlib.sha256(artifact_bytes).hexdigest() != manifest.get("artifactSha256"):
        raise ValueError("soft signature fitted artifact digest mismatch")
    try:
        payload = json.loads(artifact_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("soft signature fitted artifact is not valid JSON") from error
    candidate = _candidate_from_payload(payload)
    if (
        candidate.source_binding is None
        or candidate.source_binding.payload() != manifest.get("sourceBinding")
        or _date_binding_payload() != manifest.get("dateBinding")
        or candidate.model_sha256 != manifest.get("modelSha256")
        or {
            name: profile.sha256 for name, profile in sorted(candidate.profiles.items())
        }
        != manifest.get("profileSha256")
    ):
        raise ValueError("soft signature manifest binding does not match replay state")
    return candidate


def _candidate_from_payload(payload: Any) -> SoftSignatureCandidate:
    if (
        not isinstance(payload, Mapping)
        or payload.get("schemaVersion") != 1
        or payload.get("kind") != "soft_signature_fitted_model"
        or payload.get("family") != FAMILY
        or payload.get("featureSchemaSha256") != FEATURE_SCHEMA_SHA256
        or payload.get("dateBinding") != _date_binding_payload()
        or payload.get("fixedHyperparameters") != _fixed_hyperparameter_payload()
    ):
        raise ValueError("unsupported soft signature replay schema")
    source = payload.get("sourceBinding")
    if not isinstance(source, Mapping):
        raise ValueError("soft signature replay source binding is missing")
    binding = SourceBinding(
        source_dataset_id=str(source.get("sourceDatasetId") or ""),
        source_sha256=str(source.get("sourceSha256") or ""),
        feature_schema_sha256=str(source.get("featureSchemaSha256") or ""),
    )
    counts_payload = payload.get("counts")
    if not isinstance(counts_payload, Mapping) or set(counts_payload) != set(LATTICE_LEVELS):
        raise ValueError("soft signature replay lattice levels are invalid")
    candidate = SoftSignatureCandidate()
    candidate._source_binding = binding
    candidate._counts = {
        level: _restore_level(counts_payload[level]) for level in LATTICE_LEVELS
    }
    candidate._overall_global = _restore_counter(payload.get("overallGlobal"))
    candidate._fit_ids = _restore_ids(payload.get("fitIds"), "fit")
    candidate._calibration_ids = _restore_ids(payload.get("calibrationIds"), "calibration")
    candidate._threshold_ids = _restore_ids(payload.get("thresholdIds"), "threshold")
    if (
        candidate._fit_ids & candidate._calibration_ids
        or candidate._fit_ids & candidate._threshold_ids
        or candidate._calibration_ids & candidate._threshold_ids
    ):
        raise ValueError("soft signature replay identities overlap")
    fit_interval = payload.get("fitDateInterval")
    if not isinstance(fit_interval, Mapping):
        raise ValueError("soft signature fit date binding is missing")
    candidate._fit_start_date = str(fit_interval.get("start") or "")
    candidate._fit_end_date = str(fit_interval.get("end") or "")
    calibrators = payload.get("calibrators")
    if not isinstance(calibrators, Mapping) or set(calibrators) != {
        PROCESS_SAFE_PROFILE,
        FULL_MULTICLASS_PROFILE,
    }:
        raise ValueError("soft signature calibrator replay state is invalid")
    candidate._calibrators = {
        name: _restore_calibrator(calibrators[name]) for name in sorted(calibrators)
    }
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("soft signature profile replay state is invalid")
    candidate._profiles = {
        name: _restore_profile(name, value) for name, value in sorted(profiles.items())
    }
    candidate._model_sha256 = candidate._compute_model_sha256()
    if candidate._model_sha256 != payload.get("modelSha256"):
        raise ValueError("soft signature model digest does not match replay state")
    return candidate


def _fit_process_calibrator(
    rows: Sequence[FeatureRow],
    raw_scores: Sequence[Mapping[str, float]],
    labels: Mapping[str, str],
) -> _LinearCalibrator:
    x = np.asarray(
        [
            [
                math.log(
                    min(max(float(score["PROCESS"]), 1e-12), 1.0 - 1e-12)
                    / (1.0 - min(max(float(score["PROCESS"]), 1e-12), 1.0 - 1e-12))
                )
            ]
            for score in raw_scores
        ]
    )
    y = np.asarray([1 if labels[row.row_id] == "PROCESS" else 0 for row in rows])
    if set(y.tolist()) != {0, 1}:
        raise ValueError("August process calibration requires process and non-process labels")
    model = LogisticRegression(
        C=CALIBRATOR_C,
        solver="lbfgs",
        max_iter=CALIBRATOR_MAX_ITER,
        random_state=CALIBRATOR_RANDOM_STATE,
    ).fit(x, y)
    return _freeze_calibrator(
        mode="process_binary",
        method="august_binary_platt_lbfgs_v1",
        model=model,
        rows=rows,
        raw_scores=raw_scores,
        labels=labels,
    )


def _fit_multiclass_calibrator(
    rows: Sequence[FeatureRow],
    raw_scores: Sequence[Mapping[str, float]],
    labels: Mapping[str, str],
) -> _LinearCalibrator:
    x = np.asarray(
        [
            [math.log(max(float(score[action]), 1e-12)) for action in ACTION_ORDER]
            for score in raw_scores
        ]
    )
    y = np.asarray([ACTION_ORDER.index(labels[row.row_id]) for row in rows])
    if set(y.tolist()) != set(range(len(ACTION_ORDER))):
        raise ValueError("August multiclass calibration requires all three action labels")
    model = LogisticRegression(
        C=CALIBRATOR_C,
        solver="lbfgs",
        max_iter=CALIBRATOR_MAX_ITER,
        random_state=CALIBRATOR_RANDOM_STATE,
    ).fit(x, y)
    return _freeze_calibrator(
        mode="full_multiclass",
        method="august_multiclass_logit_lbfgs_v1",
        model=model,
        rows=rows,
        raw_scores=raw_scores,
        labels=labels,
    )


def _freeze_calibrator(
    *,
    mode: str,
    method: str,
    model: LogisticRegression,
    rows: Sequence[FeatureRow],
    raw_scores: Sequence[Mapping[str, float]],
    labels: Mapping[str, str],
) -> _LinearCalibrator:
    start, end = _date_interval(rows)
    row_sha = _sha256_json(
        [
            {
                "rowId": row.row_id,
                "date": row.date,
                "rawScores": {action: float(score[action]) for action in ACTION_ORDER},
                "actual": labels[row.row_id],
            }
            for row, score in sorted(zip(rows, raw_scores), key=lambda item: item[0].row_id)
        ]
    )
    provisional = _LinearCalibrator(
        mode=mode,
        classes=tuple(int(value) for value in model.classes_.tolist()),
        coefficients=tuple(
            tuple(float(value) for value in row) for row in model.coef_.tolist()
        ),
        intercepts=tuple(float(value) for value in model.intercept_.tolist()),
        method=method,
        calibration_rows_sha256=row_sha,
        calibration_start_date=start,
        calibration_end_date=end,
        sha256="",
    )
    return _LinearCalibrator(
        mode=provisional.mode,
        classes=provisional.classes,
        coefficients=provisional.coefficients,
        intercepts=provisional.intercepts,
        method=provisional.method,
        calibration_rows_sha256=provisional.calibration_rows_sha256,
        calibration_start_date=provisional.calibration_start_date,
        calibration_end_date=provisional.calibration_end_date,
        sha256=_sha256_json(provisional.payload()),
    )


def _fit_profile(
    name: str,
    score_rows: Sequence[ScoreRow],
    labels: Mapping[str, str],
    *,
    source_binding: SourceBinding,
    calibrator: _LinearCalibrator,
    rows: Sequence[FeatureRow],
) -> SoftSignatureProfile:
    thresholds: dict[str, float] = {}
    enabled: dict[str, bool] = {}
    metrics: dict[str, ThresholdMetric] = {}
    for action in ACTION_ORDER:
        if name == PROCESS_SAFE_PROFILE and action != "PROCESS":
            thresholds[action] = 1.0
            enabled[action] = False
            metrics[action] = ThresholdMetric(0, 0, 0, None)
            continue
        threshold, metric = _fit_action_threshold(score_rows, labels, action)
        thresholds[action] = threshold
        enabled[action] = metric.automatic > 0
        metrics[action] = metric
    start, end = _date_interval(rows)
    threshold_rows_sha = _sha256_json(
        [
            {
                "rowId": score.row_id,
                "date": next(row.date for row in rows if row.row_id == score.row_id),
                "scores": {action: float(score.scores[action]) for action in ACTION_ORDER},
                "actual": labels[score.row_id],
            }
            for score in sorted(score_rows, key=lambda item: item.row_id)
        ]
    )
    semantics = (
        "august_calibrated_process_probability_zero_unsafe_september_threshold"
        if name == PROCESS_SAFE_PROFILE
        else "august_calibrated_multiclass_probability_september_precision_thresholds"
    )
    payload = {
        "name": name,
        "thresholds": thresholds,
        "enabled": enabled,
        "metrics": _metrics_payload(metrics),
        "probabilitySemantics": semantics,
        "calibrationMethod": calibrator.method,
        "calibratorSha256": calibrator.sha256,
        "thresholdRowsSha256": threshold_rows_sha,
        "thresholdDateInterval": {"start": start, "end": end},
        "sourceBindingSha256": source_binding.sha256,
        "fallbackPolicy": "review",
    }
    return SoftSignatureProfile(
        name=name,
        thresholds=MappingProxyType(thresholds),
        enabled=MappingProxyType(enabled),
        metrics=MappingProxyType(metrics),
        probability_semantics=semantics,
        calibration_method=calibrator.method,
        calibrator_sha256=calibrator.sha256,
        threshold_rows_sha256=threshold_rows_sha,
        threshold_start_date=start,
        threshold_end_date=end,
        source_binding_sha256=source_binding.sha256,
        sha256=_sha256_json(payload),
    )


def _fit_action_threshold(
    score_rows: Sequence[ScoreRow],
    labels: Mapping[str, str],
    action: str,
) -> tuple[float, ThresholdMetric]:
    winning = [
        row
        for row in score_rows
        if max(ACTION_ORDER, key=lambda candidate: row.scores[candidate]) == action
    ]
    valid: list[tuple[tuple[int, int, float], float, ThresholdMetric]] = []
    for threshold in sorted({float(row.scores[action]) for row in winning}):
        selected = [row for row in winning if float(row.scores[action]) >= threshold]
        if len(selected) < THRESHOLD_MIN_SUPPORT:
            continue
        correct = sum(labels[row.row_id] == action for row in selected)
        unsafe = (
            sum(labels[row.row_id] in {"RETURN", "REPAIR"} for row in selected)
            if action == "PROCESS"
            else 0
        )
        precision = correct / len(selected)
        passes = unsafe == 0 if action == "PROCESS" else precision >= FULL_MULTICLASS_PRECISION
        if passes:
            metric = ThresholdMetric(len(selected), correct, unsafe, precision)
            valid.append(((len(selected), correct, -threshold), threshold, metric))
    if not valid:
        return 1.0, ThresholdMetric(0, 0, 0, None)
    _, threshold, metric = max(valid, key=lambda item: item[0])
    return threshold, metric


def _signatures(row: FeatureRow) -> dict[str, tuple[str, ...]]:
    count_bins = tuple(_count_bin(row.features.get(field)) for field in COUNT_FIELDS)
    coarse = (
        _coarse_category(row.features.get("type")),
        _coarse_category(row.features.get("symptom")),
        _coarse_detail(row.features.get("detail")),
    )
    customer = _normalise_identity(row.customer_key)
    product = _normalise_identity(row.product_key)
    return {
        "customer_product": (customer, product, *count_bins),
        "cross_customer_product": (product, *count_bins),
        "customer_coarse_raw": (customer, *coarse, *count_bins),
        "global": count_bins,
    }


def _count_bin(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "missing"
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(parsed) or parsed < 0:
        return "invalid"
    for edge, label in zip(COUNT_BIN_EDGES, COUNT_BIN_LABELS):
        if parsed <= edge:
            return label
    return COUNT_BIN_LABELS[-1]


def _coarse_category(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"\d+", "#", text)
    text = re.sub(r"[^\w#]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip() or "missing"


def _coarse_detail(value: Any) -> str:
    normalized = _coarse_category(value)
    return " ".join(normalized.split()[:MAX_COARSE_DETAIL_TOKENS])


def _normalise_identity(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")).strip().lower(),
    ) or "missing"


def _shrink(
    counts: Mapping[str, int],
    parent: Mapping[str, float],
    strength: float,
) -> dict[str, float]:
    support = float(sum(int(counts.get(action, 0)) for action in ACTION_ORDER))
    denominator = support + strength
    if denominator <= 0:
        return {action: float(parent[action]) for action in ACTION_ORDER}
    return {
        action: (float(counts.get(action, 0)) + strength * float(parent[action]))
        / denominator
        for action in ACTION_ORDER
    }


def _validate_labelled_rows(
    rows: Iterable[FeatureRow],
    labels: Mapping[str, str],
    *,
    expected_month: str,
    chronology_message: str,
    source_binding: SourceBinding,
) -> tuple[FeatureRow, ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("soft signature labelled rows must not be empty")
    row_ids = [row.row_id for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("soft signature labelled rows contain duplicate identities")
    if set(row_ids) != set(labels):
        raise ValueError("soft signature label identity mismatch")
    if any(label not in ACTION_LABELS for label in labels.values()):
        raise ValueError("soft signature labels contain an unknown action")
    if any(row.month != expected_month for row in materialized):
        raise ValueError(chronology_message)
    _validate_score_identities(materialized, source_binding)
    return tuple(sorted(materialized, key=lambda row: row.row_id))


def _validate_score_identities(
    rows: Sequence[FeatureRow],
    source_binding: SourceBinding,
) -> None:
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("soft signature score rows contain duplicate identities")
    expected_prefix = f"{source_binding.source_dataset_id}:"
    if any(not row.row_id.startswith(expected_prefix) for row in rows):
        raise ValueError("soft signature row identities do not match source binding")


def _reject_overlap(
    rows: Sequence[FeatureRow],
    protected_ids: frozenset[str],
    message: str,
) -> None:
    overlap = sorted({row.row_id for row in rows} & protected_ids)
    if overlap:
        raise ValueError(f"{message}: {overlap[:3]}")


def _date_interval(rows: Sequence[FeatureRow]) -> tuple[str, str]:
    dates = sorted(row.date for row in rows)
    return dates[0], dates[-1]


def _normalise_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    values = {action: max(float(probabilities[action]), 0.0) for action in ACTION_ORDER}
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("calibrated soft signature probabilities are invalid")
    normalised = {action: value / total for action, value in values.items()}
    residual = 1.0 - sum(normalised.values())
    normalised[ACTION_ORDER[-1]] += residual
    return normalised


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential)


def _fixed_hyperparameter_payload() -> dict[str, Any]:
    return {
        "minLevelSupport": MIN_LEVEL_SUPPORT,
        "parentStrength": PARENT_STRENGTH,
        "globalStrength": GLOBAL_STRENGTH,
        "countBinEdges": list(COUNT_BIN_EDGES),
        "countBinLabels": list(COUNT_BIN_LABELS),
        "maxCoarseDetailTokens": MAX_COARSE_DETAIL_TOKENS,
        "calibratorC": CALIBRATOR_C,
        "calibratorMaxIterations": CALIBRATOR_MAX_ITER,
        "calibratorRandomState": CALIBRATOR_RANDOM_STATE,
        "thresholdMinSupport": THRESHOLD_MIN_SUPPORT,
        "fullMulticlassPrecision": FULL_MULTICLASS_PRECISION,
        "hyperparameterDiscoveryCutoff": "pre-2025-07",
    }


def _date_binding_payload() -> dict[str, Any]:
    return {
        "fitMonth": FIT_MONTH,
        "probabilityCalibrationMonth": CALIBRATION_MONTH,
        "thresholdFitMonth": THRESHOLD_MONTH,
        "scoreMonths": list(SCORE_MONTHS),
    }


def _serialise_level(
    values: Mapping[tuple[str, ...], Counter[str]],
) -> list[dict[str, Any]]:
    return [
        {"key": list(key), "counts": _counter_payload(values[key])}
        for key in sorted(values)
    ]


def _counter_payload(counter: Mapping[str, int]) -> dict[str, int]:
    return {action: int(counter.get(action, 0)) for action in ACTION_ORDER}


def _restore_level(value: Any) -> dict[tuple[str, ...], Counter[str]]:
    if not isinstance(value, list):
        raise ValueError("soft signature replay level must be a list")
    restored: dict[tuple[str, ...], Counter[str]] = {}
    for item in value:
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), list):
            raise ValueError("soft signature replay level key is invalid")
        key = tuple(str(part) for part in item["key"])
        if key in restored:
            raise ValueError("soft signature replay level contains duplicate keys")
        restored[key] = _restore_counter(item.get("counts"))
    return restored


def _restore_counter(value: Any) -> Counter[str]:
    if not isinstance(value, Mapping) or set(value) != set(ACTION_ORDER):
        raise ValueError("soft signature replay class counts are invalid")
    output: Counter[str] = Counter()
    for action in ACTION_ORDER:
        count = value[action]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("soft signature replay class count is invalid")
        output[action] = count
    return output


def _restore_ids(value: Any, name: str) -> frozenset[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"soft signature replay {name} identities are invalid")
    return frozenset(value)


def _restore_calibrator(value: Any) -> _LinearCalibrator:
    if not isinstance(value, Mapping):
        raise ValueError("soft signature replay calibrator is invalid")
    interval = value.get("dateInterval")
    if not isinstance(interval, Mapping):
        raise ValueError("soft signature replay calibration date interval is invalid")
    provisional = _LinearCalibrator(
        mode=str(value.get("mode") or ""),
        classes=tuple(int(item) for item in value.get("classes") or ()),
        coefficients=tuple(
            tuple(float(item) for item in row)
            for row in value.get("coefficients") or ()
        ),
        intercepts=tuple(float(item) for item in value.get("intercepts") or ()),
        method=str(value.get("method") or ""),
        calibration_rows_sha256=str(value.get("calibrationRowsSha256") or ""),
        calibration_start_date=str(interval.get("start") or ""),
        calibration_end_date=str(interval.get("end") or ""),
        sha256="",
    )
    digest = _sha256_json(provisional.payload())
    return _LinearCalibrator(
        mode=provisional.mode,
        classes=provisional.classes,
        coefficients=provisional.coefficients,
        intercepts=provisional.intercepts,
        method=provisional.method,
        calibration_rows_sha256=provisional.calibration_rows_sha256,
        calibration_start_date=provisional.calibration_start_date,
        calibration_end_date=provisional.calibration_end_date,
        sha256=digest,
    )


def _profile_payload(profile: SoftSignatureProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "thresholds": dict(profile.thresholds),
        "enabled": dict(profile.enabled),
        "metrics": _metrics_payload(profile.metrics),
        "probabilitySemantics": profile.probability_semantics,
        "calibrationMethod": profile.calibration_method,
        "calibratorSha256": profile.calibrator_sha256,
        "thresholdRowsSha256": profile.threshold_rows_sha256,
        "thresholdDateInterval": {
            "start": profile.threshold_start_date,
            "end": profile.threshold_end_date,
        },
        "sourceBindingSha256": profile.source_binding_sha256,
        "fallbackPolicy": "review",
        "sha256": profile.sha256,
    }


def _restore_profile(name: str, value: Any) -> SoftSignatureProfile:
    if not isinstance(value, Mapping) or value.get("name") != name:
        raise ValueError("soft signature replay profile is invalid")
    thresholds = value.get("thresholds")
    enabled = value.get("enabled")
    metrics_payload = value.get("metrics")
    interval = value.get("thresholdDateInterval")
    if (
        not isinstance(thresholds, Mapping)
        or set(thresholds) != set(ACTION_ORDER)
        or not isinstance(enabled, Mapping)
        or set(enabled) != set(ACTION_ORDER)
        or not isinstance(metrics_payload, Mapping)
        or set(metrics_payload) != set(ACTION_ORDER)
        or not isinstance(interval, Mapping)
    ):
        raise ValueError("soft signature replay profile contract is invalid")
    metrics = {
        action: ThresholdMetric(
            automatic=int(metrics_payload[action]["automatic"]),
            correct=int(metrics_payload[action]["correct"]),
            unsafe=int(metrics_payload[action]["unsafe"]),
            precision=(
                None
                if metrics_payload[action]["precision"] is None
                else float(metrics_payload[action]["precision"])
            ),
        )
        for action in ACTION_ORDER
    }
    profile = SoftSignatureProfile(
        name=name,
        thresholds=MappingProxyType({action: float(thresholds[action]) for action in ACTION_ORDER}),
        enabled=MappingProxyType({action: bool(enabled[action]) for action in ACTION_ORDER}),
        metrics=MappingProxyType(metrics),
        probability_semantics=str(value.get("probabilitySemantics") or ""),
        calibration_method=str(value.get("calibrationMethod") or ""),
        calibrator_sha256=str(value.get("calibratorSha256") or ""),
        threshold_rows_sha256=str(value.get("thresholdRowsSha256") or ""),
        threshold_start_date=str(interval.get("start") or ""),
        threshold_end_date=str(interval.get("end") or ""),
        source_binding_sha256=str(value.get("sourceBindingSha256") or ""),
        sha256=str(value.get("sha256") or ""),
    )
    expected = {**_profile_payload(profile), "sha256": profile.sha256}
    expected.pop("sha256")
    if _sha256_json(expected) != profile.sha256:
        raise ValueError("soft signature replay profile digest mismatch")
    return profile


def _metrics_payload(metrics: Mapping[str, ThresholdMetric]) -> dict[str, Any]:
    return {
        action: {
            "automatic": metrics[action].automatic,
            "correct": metrics[action].correct,
            "unsafe": metrics[action].unsafe,
            "precision": metrics[action].precision,
        }
        for action in ACTION_ORDER
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)

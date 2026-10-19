from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import re
import resource
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier

try:  # LightGBM is an optional candidate family, not a module dependency.
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - exercised in environments without LightGBM.
    LGBMClassifier = None  # type: ignore[assignment,misc]

from .data_contract import ACTION_LABELS, FeatureRow
from .features import FeatureSpec, RawFeatureEncoder


ACTION_ORDER = ("PROCESS", "RETURN", "REPAIR")
_DEVELOPMENT_END = date(2025, 12, 1)
_CALIBRATION_METHOD = "multiclass_platt_ovr_simplex_v1"
_UNCALIBRATED_METHOD = "uncalibrated_ovr_simplex_v1"
_TRAINING_DIGEST_VERSION = "feature-row-label-source-v2"
_FITTED_ARTIFACT_SCHEMA_VERSION = 2
_TEMPORAL_ARTIFACT_SCHEMA_VERSION = 1
_PRODUCT_IDENTITY_FIELDS = frozenset(
    {"product_key", "product_key_label", "process_code", "part_code", "part_name"}
)


class LightGBMUnavailableError(RuntimeError):
    """Raised only when the optional LightGBM family is selected without LightGBM."""


class CandidateFitResourceLimitError(RuntimeError):
    """Raised before OVR fitting would exceed its conservative sparse allocation budget."""


@dataclass(frozen=True, slots=True)
class CandidateScoreBenchmark:
    row_count: int
    batch_count: int
    elapsed_seconds: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class CandidateFitPreflight:
    row_count: int
    estimated_nnz: int
    estimated_matrix_bytes: int
    estimated_peak_working_bytes: int
    max_output_nnz: int
    max_matrix_bytes: int
    max_peak_working_bytes: int


@dataclass(frozen=True, slots=True)
class ScoreRow:
    row_id: str
    scores: Mapping[str, float]
    family: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    row_id: str
    label: str
    score: float
    family: str
    evidence: Mapping[str, Any]


class TemporalSignatureCandidate:
    """Prior-only empirical scores over a frozen raw intake-field signature."""

    def __init__(
        self,
        *,
        min_support: int = 5,
        smoothing: float = 2.0,
        customer_identity_basis: str = "customer_key",
    ) -> None:
        if min_support < 1:
            raise ValueError("min_support must be positive")
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        if customer_identity_basis not in {
            "customer_key",
            "raw_customer",
            "legacy_customer_key",
        }:
            raise ValueError(
                "customer_identity_basis must be customer_key, raw_customer, or legacy_customer_key"
            )
        self.min_support = min_support
        self.smoothing = float(smoothing)
        self.customer_identity_basis = customer_identity_basis
        self._counts: dict[str, dict[tuple[Any, ...], Counter[str]]] = {}
        self._global = Counter()
        self._training_ids: frozenset[str] = frozenset()
        self._cutoff: str | None = None
        self._provenance: Mapping[str, Any] | None = None

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        cutoff: str | None = None,
    ) -> "TemporalSignatureCandidate":
        materialized = _validate_training_rows(rows, labels)
        resolved_cutoff = _resolve_cutoff(
            materialized, cutoff, strict=cutoff is not None
        )
        levels = ("product", "customer_transfer", "global_transfer")
        counts: dict[str, dict[tuple[Any, ...], Counter[str]]] = {
            level: defaultdict(Counter) for level in levels
        }
        global_counts: Counter[str] = Counter()
        for row in materialized:
            label = labels[row.row_id]
            global_counts[label] += 1
            for level in levels:
                counts[level][_signature(row, level, self.customer_identity_basis)][
                    label
                ] += 1
        self._counts = {level: dict(values) for level, values in counts.items()}
        self._global = global_counts
        self._training_ids = frozenset(row.row_id for row in materialized)
        self._cutoff = resolved_cutoff
        training_sha = _training_rows_sha256(materialized, labels)
        model_sha = _temporal_model_sha256(
            self._counts,
            global_counts,
            cutoff=resolved_cutoff,
            min_support=self.min_support,
            smoothing=self.smoothing,
            customer_identity_basis=self.customer_identity_basis,
        )
        self._provenance = _freeze(
            {
                "family": "temporal_signature",
                "trainingRowSha256": training_sha,
                "trainingDigestVersion": _TRAINING_DIGEST_VERSION,
                "modelSha256": model_sha,
                "cutoff": resolved_cutoff,
                "training": {
                    "rowIds": tuple(sorted(self._training_ids)),
                    "rowSha256": training_sha,
                    "cutoff": resolved_cutoff,
                    "digestVersion": _TRAINING_DIGEST_VERSION,
                },
                "classOrder": ACTION_ORDER,
                "hyperparameters": {
                    "minSupport": self.min_support,
                    "smoothing": self.smoothing,
                },
                "customerIdentityBasis": self.customer_identity_basis,
                "dependencyVersions": _dependency_versions(),
            }
        )
        return self

    def predict_scores(self, rows: Iterable[FeatureRow]) -> tuple[ScoreRow, ...]:
        if self._cutoff is None or self._provenance is None:
            raise RuntimeError(
                "TemporalSignatureCandidate must be fitted before prediction"
            )
        materialized = tuple(rows)
        overlap = sorted({row.row_id for row in materialized} & self._training_ids)
        if overlap:
            raise ValueError(f"score rows overlap training identities: {overlap[:3]}")
        for row in materialized:
            if _row_date(row) <= self._cutoff:
                raise ValueError("temporal signature score rows must be after cutoff")
        global_total = sum(self._global.values())
        global_probabilities = {
            label: self._global[label] / global_total
            if global_total
            else 1 / len(ACTION_ORDER)
            for label in ACTION_ORDER
        }
        output: list[ScoreRow] = []
        for row in materialized:
            selected_level = "global"
            selected_counts: Counter[str] | None = None
            support = global_total
            signature: tuple[Any, ...] | None = None
            for level in ("product", "customer_transfer", "global_transfer"):
                candidate_signature = _signature(
                    row, level, self.customer_identity_basis
                )
                candidate = self._counts[level].get(candidate_signature)
                candidate_support = sum(candidate.values()) if candidate else 0
                if candidate_support >= self.min_support:
                    selected_level = level
                    selected_counts = candidate
                    support = candidate_support
                    signature = candidate_signature
                    break
            if selected_counts is None:
                probabilities = global_probabilities
            else:
                denominator = support + self.smoothing
                probabilities = {
                    label: (
                        selected_counts[label]
                        + self.smoothing * global_probabilities[label]
                    )
                    / denominator
                    for label in ACTION_ORDER
                }
            evidence = MappingProxyType({
                "level": selected_level,
                "support": support,
                "classCounts": MappingProxyType(dict(selected_counts or self._global)),
                "signatureSha256": _sha256_json(repr(signature))
                if signature is not None
                else None,
                "customerIdentityBasis": self.customer_identity_basis,
                "rawFields": (
                    "customer",
                    "product_key_label",
                    "type",
                    "symptom",
                    "detail",
                    "intake_count",
                    "service_count",
                    "treatment_count",
                ),
                # Provenance is frozen once at fit/load time. Reusing the same
                # object avoids recursively copying the complete training-ID
                # contract for every scored row.
                "provenance": self._provenance,
            })
            output.append(
                ScoreRow(
                    row_id=row.row_id,
                    scores=MappingProxyType(
                        {label: float(probabilities[label]) for label in ACTION_ORDER}
                    ),
                    family="temporal_signature",
                    evidence=evidence,
                )
            )
        return tuple(output)

    def persist_artifacts(
        self, output_dir: str | Path, *, stem: str
    ) -> Mapping[str, str]:
        """Persist registry-compatible manifests and a versioned temporal fitted payload."""
        if self._cutoff is None or self._provenance is None:
            raise RuntimeError(
                "TemporalSignatureCandidate must be fitted before artifact persistence"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            raise ValueError(
                "artifact stem must contain only letters, digits, dot, underscore, or hyphen"
            )
        directory = Path(output_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        feature_schema_path = directory / f"{stem}.feature_schema.json"
        artifact_path = directory / f"{stem}.fitted.pkl"
        fitted_manifest_path = directory / f"{stem}.fitted_artifact.json"
        feature_names = _temporal_feature_names(self.customer_identity_basis)
        feature_schema_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "feature_schema",
                "featureNames": list(feature_names),
                "encoderSchemaSha256": _sha256_json(
                    {"family": "temporal_signature", "featureNames": feature_names}
                ),
            }
        )
        artifact_bytes = pickle.dumps(
            {
                "schemaVersion": _TEMPORAL_ARTIFACT_SCHEMA_VERSION,
                "kind": "temporal_signature_fitted_model",
                "family": "temporal_signature",
                "constructor": {
                    "minSupport": self.min_support,
                    "smoothing": self.smoothing,
                    "customerIdentityBasis": self.customer_identity_basis,
                },
                "counts": self._counts,
                "global": self._global,
                "trainingIds": tuple(sorted(self._training_ids)),
                "cutoff": self._cutoff,
                "provenance": _plain(self._provenance),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        feature_schema_sha256 = hashlib.sha256(feature_schema_bytes).hexdigest()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        fitted_manifest_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "fitted_artifact",
                "featureSchemaSha256": feature_schema_sha256,
                "artifactFile": artifact_path.name,
                "artifactSha256": artifact_sha256,
            }
        )
        feature_schema_path.write_bytes(feature_schema_bytes)
        artifact_path.write_bytes(artifact_bytes)
        fitted_manifest_path.write_bytes(fitted_manifest_bytes)
        return MappingProxyType(
            {
                "feature_schema": str(feature_schema_path),
                "fitted_artifact": str(fitted_manifest_path),
            }
        )


class _OVRCandidate:
    family = "ovr"

    def __init__(
        self,
        *,
        spec: FeatureSpec | None,
        validation_config: Mapping[str, Any] | None,
        random_state: int,
        score_batch_size: int = 10_000,
        max_score_rows: int = 500_000,
        product_independent: bool | None = None,
    ) -> None:
        self.spec = spec or FeatureSpec()
        self.validation_config = validation_config
        self.random_state = random_state
        if score_batch_size < 1 or max_score_rows < 1:
            raise ValueError("score_batch_size and max_score_rows must be positive")
        inferred_product_independence = not self.spec.include_product_key
        if (
            product_independent is not None
            and product_independent != inferred_product_independence
        ):
            raise ValueError(
                "product_independent must match FeatureSpec.include_product_key"
            )
        direct_product_channels = _direct_product_identity_channels(self.spec)
        if inferred_product_independence and direct_product_channels:
            raise ValueError(
                "product identity channels are forbidden in product-independent mode: "
                f"{sorted(direct_product_channels)}"
            )
        self.product_independent = inferred_product_independence
        self.score_batch_size = score_batch_size
        self.max_score_rows = max_score_rows
        self._encoder: RawFeatureEncoder | None = None
        self._heads: dict[str, Any] = {}
        self._calibrator: LogisticRegression | None = None
        self._training_ids: frozenset[str] = frozenset()
        self._training_cutoff: str | None = None
        self._warm_reference_products: frozenset[str] = frozenset()
        self._calibration_ids: frozenset[str] = frozenset()
        self._calibration_cutoff: str | None = None
        self._calibration_score_rows: tuple[ScoreRow, ...] = ()
        self._provenance: Mapping[str, Any] | None = None
        self._last_score_benchmark: CandidateScoreBenchmark | None = None
        self._last_fit_preflight: CandidateFitPreflight | None = None

    @property
    def last_score_benchmark(self) -> CandidateScoreBenchmark | None:
        return self._last_score_benchmark

    @property
    def last_fit_preflight(self) -> CandidateFitPreflight | None:
        return self._last_fit_preflight

    def _fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        calibration_rows: Iterable[FeatureRow] | None,
        calibration_labels: Mapping[str, str] | None,
    ) -> None:
        materialized = _validate_training_rows(rows, labels)
        self._preflight_fit_resources(materialized)
        training_cutoff = max(_row_date(row) for row in materialized)
        training_ids = frozenset(row.row_id for row in materialized)
        encoder = RawFeatureEncoder(self.spec, validation_config=self.validation_config)
        matrix = encoder.fit_transform(materialized)
        heads = {
            action: _fit_binary_head(
                matrix,
                np.asarray([int(labels[row.row_id] == action) for row in materialized]),
                self._head_factory(index),
            )
            for index, action in enumerate(ACTION_ORDER)
        }
        calibrator: LogisticRegression | None = None
        calibration: tuple[FeatureRow, ...] = ()
        calibration_label_map: Mapping[str, str] = {}
        calibration_state = "uncalibrated"
        calibration_cutoff = training_cutoff
        calibration_probabilities: np.ndarray | None = None
        if calibration_rows is not None or calibration_labels is not None:
            if calibration_rows is None or calibration_labels is None:
                raise ValueError(
                    "calibration rows and labels must be provided together"
                )
            calibration = _validate_training_rows(calibration_rows, calibration_labels)
            calibration_label_map = calibration_labels
            train_ids = {row.row_id for row in materialized}
            overlap = sorted(train_ids & {row.row_id for row in calibration})
            if overlap:
                raise ValueError(
                    f"calibration identities must be disjoint from training identities: {overlap[:3]}"
                )
            if any(_row_date(row) <= training_cutoff for row in calibration):
                raise ValueError(
                    "calibration rows must be strictly later than the training cutoff"
                )
            observed_classes = {calibration_labels[row.row_id] for row in calibration}
            if observed_classes != set(ACTION_ORDER):
                raise ValueError("calibration rows must contain all action classes")
            calibration_matrix = encoder.transform(calibration)
            raw_simplex = _ovr_simplex(heads, calibration_matrix)
            target = np.asarray(
                [
                    ACTION_ORDER.index(calibration_labels[row.row_id])
                    for row in calibration
                ]
            )
            calibrator = LogisticRegression(
                max_iter=1000, random_state=self.random_state + 100
            )
            calibrator.fit(raw_simplex, target)
            calibration_probabilities = _calibrated_simplex(calibrator, raw_simplex)
            calibration_state = "calibrated"
            calibration_cutoff = max(_row_date(row) for row in calibration)
        calibration_ids = frozenset(row.row_id for row in calibration)
        resolved_calibration_cutoff = (
            calibration_cutoff if calibrator is not None else None
        )
        warm_reference_rows = (*materialized, *calibration)
        warm_reference_products = frozenset(
            row.product_key for row in warm_reference_rows
        )
        warm_reference = {
            "basis": "training_and_calibration_prior_history",
            "cutoff": calibration_cutoff,
            "rowCount": len(warm_reference_rows),
            "rowSha256": _row_universe_sha256(warm_reference_rows),
            "productUniverseSha256": _sha256_json(sorted(warm_reference_products)),
        }
        training_sha256 = _training_rows_sha256(materialized, labels)
        calibration_sha256 = (
            _training_rows_sha256(calibration, calibration_label_map)
            if calibration
            else None
        )
        calibration_score_payloads = (
            tuple(
                _score_row_payload(
                    row,
                    calibration_probabilities[index],
                    product_independent=self.product_independent,
                    warm_reference_products=warm_reference_products,
                )
                for index, row in enumerate(calibration)
            )
            if calibration_probabilities is not None
            else ()
        )
        provenance = _freeze(
            {
                "family": self.family,
                "trainingRowSha256": training_sha256,
                "trainingDigestVersion": _TRAINING_DIGEST_VERSION,
                "modelSha256": _hash_model_state(
                    {"encoder": encoder.encoder_manifest, "heads": heads}
                ),
                "calibratorSha256": _hash_model_state(calibrator)
                if calibrator is not None
                else None,
                "cutoff": calibration_cutoff,
                "trainingCutoff": training_cutoff,
                "training": {
                    "rowIds": tuple(sorted(training_ids)),
                    "rowSha256": training_sha256,
                    "cutoff": training_cutoff,
                    "digestVersion": _TRAINING_DIGEST_VERSION,
                },
                "classOrder": ACTION_ORDER,
                "seeds": {
                    "base": self.random_state,
                    "perClass": [
                        self.random_state + index for index in range(len(ACTION_ORDER))
                    ],
                },
                "hyperparameters": self._hyperparameters(),
                "dependencyVersions": _dependency_versions(),
                "featureSchemaSha256": encoder.schema_sha256,
                "productIndependence": self.product_independent,
                "excludedProductIdentityChannels": sorted(_PRODUCT_IDENTITY_FIELDS),
                "warmColdReference": warm_reference,
                "calibration": {
                    "state": calibration_state,
                    "method": _CALIBRATION_METHOD
                    if calibrator is not None
                    else _UNCALIBRATED_METHOD,
                    "rowIds": tuple(sorted(calibration_ids)),
                    "rowSha256": calibration_sha256,
                    "cutoff": resolved_calibration_cutoff,
                    "digestVersion": _TRAINING_DIGEST_VERSION if calibration else None,
                    "scoreRowsSha256": _sha256_json(calibration_score_payloads)
                    if calibration_score_payloads
                    else None,
                },
                "headStates": {
                    action: "class_degenerate"
                    if isinstance(head, _ConstantHead)
                    else "fitted"
                    for action, head in heads.items()
                },
            }
        )
        self._encoder = encoder
        self._heads = heads
        self._calibrator = calibrator
        self._training_ids = training_ids
        self._training_cutoff = training_cutoff
        self._calibration_ids = calibration_ids
        self._calibration_cutoff = resolved_calibration_cutoff
        self._warm_reference_products = warm_reference_products
        self._provenance = provenance
        self._calibration_score_rows = _score_rows_from_payloads(
            calibration_score_payloads,
            family=self.family,
            provenance=provenance,
        )

    def _preflight_fit_resources(self, rows: Sequence[FeatureRow]) -> None:
        estimated_nnz = _estimate_encoder_nnz(rows, self.spec)
        estimated_matrix_bytes = _estimate_sparse_matrix_bytes(len(rows), estimated_nnz)
        estimated_peak_working_bytes = _estimate_sparse_working_set_bytes(
            len(rows), estimated_nnz
        )
        preflight = CandidateFitPreflight(
            row_count=len(rows),
            estimated_nnz=estimated_nnz,
            estimated_matrix_bytes=estimated_matrix_bytes,
            estimated_peak_working_bytes=estimated_peak_working_bytes,
            max_output_nnz=self.spec.max_output_nnz,
            max_matrix_bytes=self.spec.max_matrix_bytes,
            max_peak_working_bytes=self.spec.max_peak_working_bytes,
        )
        self._last_fit_preflight = preflight
        if (
            estimated_nnz > self.spec.max_output_nnz
            or estimated_matrix_bytes > self.spec.max_matrix_bytes
        ):
            raise CandidateFitResourceLimitError(
                "OVR fit pre-allocation budget exceeded: "
                f"estimated_nnz={estimated_nnz} limit={self.spec.max_output_nnz} "
                f"estimated_matrix_bytes={estimated_matrix_bytes} limit={self.spec.max_matrix_bytes}"
            )
        if estimated_peak_working_bytes > self.spec.max_peak_working_bytes:
            raise CandidateFitResourceLimitError(
                "OVR fit pre-allocation max_peak_working_bytes exceeded: "
                f"estimated_peak_working_bytes={estimated_peak_working_bytes} "
                f"limit={self.spec.max_peak_working_bytes}"
            )

    def calibration_score_rows(self) -> tuple[ScoreRow, ...]:
        """Return the frozen, provenance-bound scores for the exact calibration universe."""
        if self._encoder is None or self._provenance is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calibration score access"
            )
        if self._calibrator is None or not self._calibration_score_rows:
            raise RuntimeError(
                "calibration score rows require a calibrated OVR candidate"
            )
        return self._calibration_score_rows

    def _predict_scores(
        self, rows: Iterable[FeatureRow], *, batch_size: int | None = None
    ) -> tuple[ScoreRow, ...]:
        if self._encoder is None or not self._heads or self._provenance is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before prediction"
            )
        materialized = tuple(rows)
        if len(materialized) > self.max_score_rows:
            raise ValueError(
                f"max_score_rows exceeded: {len(materialized)} > {self.max_score_rows}"
            )
        resolved_batch_size = batch_size or self.score_batch_size
        if resolved_batch_size < 1:
            raise ValueError("batch_size must be positive")
        overlap = sorted({row.row_id for row in materialized} & self._training_ids)
        if overlap:
            raise ValueError(f"score rows overlap training identities: {overlap[:3]}")
        if self._training_cutoff is None:
            raise RuntimeError("fitted OVR candidate is missing its training cutoff")
        if any(_row_date(row) <= self._training_cutoff for row in materialized):
            raise ValueError("OVR score rows must be after training cutoff")
        if self._calibrator is not None:
            calibration_overlap = sorted(
                {row.row_id for row in materialized} & self._calibration_ids
            )
            if calibration_overlap:
                raise ValueError(
                    f"score rows overlap calibration identities: {calibration_overlap[:3]}"
                )
            assert self._calibration_cutoff is not None
            if any(_row_date(row) <= self._calibration_cutoff for row in materialized):
                raise ValueError(
                    "calibrated score rows must be after calibration cutoff"
                )
        started = time.perf_counter()
        output: list[ScoreRow] = []
        batch_count = 0
        for start in range(0, len(materialized), resolved_batch_size):
            batch = materialized[start : start + resolved_batch_size]
            matrix = self._encoder.transform(batch)
            probabilities = _ovr_simplex(self._heads, matrix)
            if self._calibrator is not None:
                probabilities = _calibrated_simplex(self._calibrator, probabilities)
            for index, row in enumerate(batch):
                cold_start = row.product_key not in self._warm_reference_products
                output.append(
                    ScoreRow(
                        row_id=row.row_id,
                        scores=MappingProxyType(
                            {
                                action: float(probabilities[index, action_index])
                                for action_index, action in enumerate(ACTION_ORDER)
                            }
                        ),
                        family=self.family,
                        evidence=MappingProxyType(
                            {
                                "coldStart": cold_start,
                                "coldStartMode": "product_independent"
                                if self.product_independent
                                else "product_dependent",
                                # Provenance is frozen once at fit time. Re-freezing it
                                # per row recursively copied the 150k-feature schema and
                                # dominated scoring latency without adding immutability.
                                "provenance": self._provenance,
                            }
                        ),
                    )
                )
            batch_count += 1
        self._last_score_benchmark = CandidateScoreBenchmark(
            row_count=len(materialized),
            batch_count=batch_count,
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
        return tuple(output)

    def persist_artifacts(
        self, output_dir: str | Path, *, stem: str
    ) -> Mapping[str, str]:
        """Persist a registry-verifiable fitted-model receipt and its feature schema."""
        if self._encoder is None or self._provenance is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before artifact persistence"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
            raise ValueError(
                "artifact stem must contain only letters, digits, dot, underscore, or hyphen"
            )
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
            }
        )
        artifact_bytes = pickle.dumps(
            {
                "schemaVersion": _FITTED_ARTIFACT_SCHEMA_VERSION,
                "kind": "candidate_fitted_model",
                "family": self.family,
                "spec": self.spec,
                "validationConfig": _plain(self.validation_config),
                "constructor": {
                    "randomState": self.random_state,
                    "scoreBatchSize": self.score_batch_size,
                    "maxScoreRows": self.max_score_rows,
                    "productIndependent": self.product_independent,
                    "hyperparameters": _plain(self._hyperparameters()),
                },
                "encoder": self._encoder,
                "heads": self._heads,
                "calibrator": self._calibrator,
                "trainingIds": tuple(sorted(self._training_ids)),
                "trainingCutoff": self._training_cutoff,
                "calibrationIds": tuple(sorted(self._calibration_ids)),
                "calibrationCutoff": self._calibration_cutoff,
                "calibrationScoreRows": tuple(
                    _compact_score_row_payload(item)
                    for item in self._calibration_score_rows
                ),
                "warmReferenceProducts": tuple(sorted(self._warm_reference_products)),
                "provenance": _plain(self._provenance),
                "fitPreflight": self._last_fit_preflight,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        feature_schema_sha256 = hashlib.sha256(feature_schema_bytes).hexdigest()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        fitted_manifest_bytes = _canonical_json_bytes(
            {
                "schemaVersion": 1,
                "kind": "fitted_artifact",
                "featureSchemaSha256": feature_schema_sha256,
                "artifactFile": artifact_path.name,
                "artifactSha256": artifact_sha256,
            }
        )
        feature_schema_path.write_bytes(feature_schema_bytes)
        artifact_path.write_bytes(artifact_bytes)
        fitted_manifest_path.write_bytes(fitted_manifest_bytes)
        return MappingProxyType(
            {
                "feature_schema": str(feature_schema_path),
                "fitted_artifact": str(fitted_manifest_path),
            }
        )

    def _head_factory(self, index: int) -> Any:
        raise NotImplementedError

    def _hyperparameters(self) -> Mapping[str, Any]:
        raise NotImplementedError


class OVRLinearCandidate(_OVRCandidate):
    family = "ovr_linear"

    def __init__(
        self,
        *,
        spec: FeatureSpec | None = None,
        validation_config: Mapping[str, Any] | None = None,
        random_state: int = 17,
        max_iter: int = 1500,
        score_batch_size: int = 10_000,
        max_score_rows: int = 500_000,
        product_independent: bool | None = None,
    ) -> None:
        super().__init__(
            spec=spec,
            validation_config=validation_config,
            random_state=random_state,
            score_batch_size=score_batch_size,
            max_score_rows=max_score_rows,
            product_independent=product_independent,
        )
        self.max_iter = max_iter

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        calibration_rows: Iterable[FeatureRow] | None = None,
        calibration_labels: Mapping[str, str] | None = None,
    ) -> "OVRLinearCandidate":
        self._fit(
            rows,
            labels,
            calibration_rows=calibration_rows,
            calibration_labels=calibration_labels,
        )
        return self

    def predict_scores(
        self, rows: Iterable[FeatureRow], *, batch_size: int | None = None
    ) -> tuple[ScoreRow, ...]:
        return self._predict_scores(rows, batch_size=batch_size)

    def _head_factory(self, index: int) -> SGDClassifier:
        return SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-5,
            max_iter=self.max_iter,
            tol=1e-4,
            random_state=self.random_state + index,
        )

    def _hyperparameters(self) -> Mapping[str, Any]:
        return {"maxIter": self.max_iter, "alpha": 1e-5, "classWeight": "balanced"}


class OVRLightGBMCandidate(_OVRCandidate):
    family = "ovr_lightgbm"

    def __init__(
        self,
        *,
        spec: FeatureSpec | None = None,
        validation_config: Mapping[str, Any] | None = None,
        random_state: int = 23,
        n_estimators: int = 120,
        n_jobs: int = 1,
        score_batch_size: int = 10_000,
        max_score_rows: int = 500_000,
        product_independent: bool | None = None,
    ) -> None:
        super().__init__(
            spec=spec,
            validation_config=validation_config,
            random_state=random_state,
            score_batch_size=score_batch_size,
            max_score_rows=max_score_rows,
            product_independent=product_independent,
        )
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        calibration_rows: Iterable[FeatureRow] | None = None,
        calibration_labels: Mapping[str, str] | None = None,
    ) -> "OVRLightGBMCandidate":
        if LGBMClassifier is None:
            raise LightGBMUnavailableError(
                "LightGBM is unavailable; install lightgbm to use OVRLightGBMCandidate"
            )
        self._fit(
            rows,
            labels,
            calibration_rows=calibration_rows,
            calibration_labels=calibration_labels,
        )
        return self

    def predict_scores(
        self, rows: Iterable[FeatureRow], *, batch_size: int | None = None
    ) -> tuple[ScoreRow, ...]:
        return self._predict_scores(rows, batch_size=batch_size)

    def _head_factory(self, index: int) -> Any:
        if LGBMClassifier is None:
            raise LightGBMUnavailableError(
                "LightGBM is unavailable; install lightgbm to use OVRLightGBMCandidate"
            )
        return LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            class_weight="balanced",
            random_state=self.random_state + index,
            n_jobs=self.n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

    def _hyperparameters(self) -> Mapping[str, Any]:
        return {
            "nEstimators": self.n_estimators,
            "nJobs": self.n_jobs,
            "deterministic": True,
            "forceColWise": True,
        }


def load_candidate_artifact(
    fitted_manifest: str | Path,
) -> _OVRCandidate | TemporalSignatureCandidate:
    """Load a trusted, content-verified versioned candidate artifact."""
    manifest_path = Path(fitted_manifest).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("fitted candidate manifest must be readable JSON") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("fitted candidate manifest must be an object")
    artifact_file = manifest.get("artifactFile")
    artifact_sha256 = manifest.get("artifactSha256")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("kind") != "fitted_artifact"
        or not isinstance(artifact_file, str)
        or Path(artifact_file).name != artifact_file
        or not isinstance(artifact_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256)
    ):
        raise ValueError("unsupported or malformed fitted candidate manifest")
    artifact_path = manifest_path.parent / artifact_file
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("fitted candidate artifact bytes are unavailable") from error
    if hashlib.sha256(artifact_bytes).hexdigest() != artifact_sha256:
        raise ValueError("fitted candidate artifact bytes do not match the manifest")
    try:
        payload = pickle.loads(artifact_bytes)
    except Exception as error:  # noqa: BLE001 - normalize pickle/dependency failures at the API boundary.
        raise ValueError("fitted candidate artifact cannot be decoded") from error
    if not isinstance(payload, Mapping):
        raise ValueError("unsupported fitted candidate artifact schema")
    if payload.get("kind") == "temporal_signature_fitted_model":
        return _load_temporal_signature_payload(payload)
    if (
        payload.get("schemaVersion") != _FITTED_ARTIFACT_SCHEMA_VERSION
        or payload.get("kind") != "candidate_fitted_model"
    ):
        raise ValueError("unsupported fitted candidate artifact schema")

    family = payload.get("family")
    spec = payload.get("spec")
    constructor = payload.get("constructor")
    hyperparameters = (
        constructor.get("hyperparameters") if isinstance(constructor, Mapping) else None
    )
    validation_config = payload.get("validationConfig")
    if family not in {"ovr_linear", "ovr_lightgbm"} or not isinstance(
        spec, FeatureSpec
    ):
        raise ValueError("fitted candidate family or feature spec is invalid")
    if not isinstance(constructor, Mapping) or not isinstance(hyperparameters, Mapping):
        raise ValueError("fitted candidate constructor state is invalid")
    if validation_config is not None and not isinstance(validation_config, Mapping):
        raise ValueError("fitted candidate validation config is invalid")
    product_independent = constructor.get("productIndependent")
    if not isinstance(product_independent, bool):
        raise ValueError("fitted candidate product-independence state is invalid")
    try:
        common_arguments = {
            "spec": spec,
            "validation_config": validation_config,
            "random_state": int(constructor["randomState"]),
            "score_batch_size": int(constructor["scoreBatchSize"]),
            "max_score_rows": int(constructor["maxScoreRows"]),
            "product_independent": product_independent,
        }
        if family == "ovr_linear":
            candidate: _OVRCandidate = OVRLinearCandidate(
                **common_arguments,
                max_iter=int(hyperparameters["maxIter"]),
            )
        else:
            candidate = OVRLightGBMCandidate(
                **common_arguments,
                n_estimators=int(hyperparameters["nEstimators"]),
                n_jobs=int(hyperparameters["nJobs"]),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fitted candidate constructor state is incomplete") from error

    encoder = payload.get("encoder")
    heads = payload.get("heads")
    calibrator = payload.get("calibrator")
    provenance_payload = payload.get("provenance")
    if (
        not isinstance(encoder, RawFeatureEncoder)
        or not isinstance(heads, Mapping)
        or set(heads) != set(ACTION_ORDER)
    ):
        raise ValueError("fitted candidate encoder or OVR heads are invalid")
    if calibrator is not None and not isinstance(calibrator, LogisticRegression):
        raise ValueError("fitted candidate calibrator is invalid")
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("fitted candidate provenance is invalid")

    try:
        training_id_values = tuple(payload["trainingIds"])
        calibration_id_values = tuple(payload["calibrationIds"])
        warm_product_values = tuple(payload["warmReferenceProducts"])
        calibration_score_payloads = tuple(payload["calibrationScoreRows"])
        training_cutoff = _normalise_date(payload["trainingCutoff"])
        raw_calibration_cutoff = payload.get("calibrationCutoff")
        calibration_cutoff = (
            _normalise_date(raw_calibration_cutoff)
            if raw_calibration_cutoff is not None
            else None
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fitted candidate chronology state is incomplete") from error
    if any(
        not isinstance(item, str) or not item
        for item in (*training_id_values, *calibration_id_values)
    ):
        raise ValueError("fitted candidate row identities are invalid")
    training_ids = frozenset(training_id_values)
    calibration_ids = frozenset(calibration_id_values)
    warm_reference_products = frozenset(str(item) for item in warm_product_values)
    if not training_ids or len(training_ids) != len(training_id_values):
        raise ValueError("fitted candidate training identities are empty or duplicated")
    if (
        len(calibration_ids) != len(calibration_id_values)
        or training_ids & calibration_ids
    ):
        raise ValueError(
            "fitted candidate calibration identities are duplicated or overlap training"
        )

    provenance = _freeze(_plain(provenance_payload))
    training_provenance = provenance.get("training")
    calibration_provenance = provenance.get("calibration")
    if (
        provenance.get("family") != family
        or provenance.get("featureSchemaSha256") != encoder.schema_sha256
        or not isinstance(training_provenance, Mapping)
        or frozenset(training_provenance.get("rowIds") or ()) != training_ids
        or training_provenance.get("cutoff") != training_cutoff
        or not isinstance(calibration_provenance, Mapping)
        or frozenset(calibration_provenance.get("rowIds") or ()) != calibration_ids
        or calibration_provenance.get("cutoff") != calibration_cutoff
    ):
        raise ValueError("fitted candidate provenance does not match its replay state")
    if calibrator is None:
        if (
            calibration_ids
            or calibration_cutoff is not None
            or calibration_score_payloads
        ):
            raise ValueError(
                "uncalibrated candidate artifact contains calibration replay state"
            )
    elif (
        not calibration_ids
        or calibration_cutoff is None
        or calibration_cutoff <= training_cutoff
        or calibration_provenance.get("state") != "calibrated"
    ):
        raise ValueError(
            "calibrated candidate artifact is missing strict chronology state"
        )

    calibration_score_rows = _score_rows_from_payloads(
        calibration_score_payloads,
        family=family,
        provenance=provenance,
    )
    if calibrator is not None:
        compact_scores = tuple(
            _compact_score_row_payload(item) for item in calibration_score_rows
        )
        if frozenset(
            item.row_id for item in calibration_score_rows
        ) != calibration_ids or calibration_provenance.get(
            "scoreRowsSha256"
        ) != _sha256_json(compact_scores):
            raise ValueError(
                "calibration score snapshot does not match fitted provenance"
            )

    candidate._encoder = encoder
    candidate._heads = dict(heads)
    candidate._calibrator = calibrator
    candidate._training_ids = training_ids
    candidate._training_cutoff = training_cutoff
    candidate._calibration_ids = calibration_ids
    candidate._calibration_cutoff = calibration_cutoff
    candidate._calibration_score_rows = calibration_score_rows
    candidate._warm_reference_products = warm_reference_products
    candidate._provenance = provenance
    fit_preflight = payload.get("fitPreflight")
    candidate._last_fit_preflight = (
        fit_preflight if isinstance(fit_preflight, CandidateFitPreflight) else None
    )
    candidate._last_score_benchmark = None
    return candidate


def _load_temporal_signature_payload(
    payload: Mapping[str, Any],
) -> TemporalSignatureCandidate:
    if (
        payload.get("schemaVersion") != _TEMPORAL_ARTIFACT_SCHEMA_VERSION
        or payload.get("family") != "temporal_signature"
    ):
        raise ValueError("unsupported temporal signature artifact schema")
    constructor = payload.get("constructor")
    raw_counts = payload.get("counts")
    raw_global = payload.get("global")
    provenance_payload = payload.get("provenance")
    if (
        not isinstance(constructor, Mapping)
        or not isinstance(raw_counts, Mapping)
        or not isinstance(raw_global, Mapping)
        or not isinstance(provenance_payload, Mapping)
    ):
        raise ValueError("temporal signature artifact state is invalid")
    try:
        candidate = TemporalSignatureCandidate(
            min_support=int(constructor["minSupport"]),
            smoothing=float(constructor["smoothing"]),
            customer_identity_basis=str(constructor["customerIdentityBasis"]),
        )
        training_id_values = tuple(payload["trainingIds"])
        cutoff = _normalise_date(payload["cutoff"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "temporal signature constructor or chronology state is incomplete"
        ) from error
    if any(not isinstance(item, str) or not item for item in training_id_values):
        raise ValueError("temporal signature training identities are invalid")
    training_ids = frozenset(training_id_values)
    if not training_ids or len(training_ids) != len(training_id_values):
        raise ValueError(
            "temporal signature training identities are empty or duplicated"
        )

    levels = ("product", "customer_transfer", "global_transfer")
    if set(raw_counts) != set(levels):
        raise ValueError("temporal signature count levels are invalid")
    counts: dict[str, dict[tuple[Any, ...], Counter[str]]] = {}
    for level in levels:
        raw_level = raw_counts[level]
        if not isinstance(raw_level, Mapping):
            raise ValueError("temporal signature level counts are invalid")
        restored_level: dict[tuple[Any, ...], Counter[str]] = {}
        for signature, raw_counter in raw_level.items():
            if not isinstance(signature, tuple) or not isinstance(raw_counter, Mapping):
                raise ValueError("temporal signature key or class counts are invalid")
            restored_level[signature] = _restore_action_counter(raw_counter)
        counts[level] = restored_level
    global_counts = _restore_action_counter(raw_global)
    if sum(global_counts.values()) < 1:
        raise ValueError("temporal signature global prior is empty")

    provenance = _freeze(_plain(provenance_payload))
    training_provenance = provenance.get("training")
    if (
        provenance.get("family") != "temporal_signature"
        or provenance.get("cutoff") != cutoff
        or provenance.get("customerIdentityBasis") != candidate.customer_identity_basis
        or tuple(provenance.get("classOrder") or ()) != ACTION_ORDER
        or not isinstance(training_provenance, Mapping)
        or frozenset(training_provenance.get("rowIds") or ()) != training_ids
        or training_provenance.get("cutoff") != cutoff
    ):
        raise ValueError("temporal signature provenance does not match replay state")
    model_sha256 = _temporal_model_sha256(
        counts,
        global_counts,
        cutoff=cutoff,
        min_support=candidate.min_support,
        smoothing=candidate.smoothing,
        customer_identity_basis=candidate.customer_identity_basis,
    )
    if provenance.get("modelSha256") != model_sha256:
        raise ValueError("temporal signature counts do not match fitted provenance")

    candidate._counts = counts
    candidate._global = global_counts
    candidate._training_ids = training_ids
    candidate._cutoff = cutoff
    candidate._provenance = provenance
    return candidate


def _restore_action_counter(raw_counter: Mapping[Any, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for label, count in raw_counter.items():
        if (
            label not in ACTION_ORDER
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("temporal signature class counts are invalid")
        counter[str(label)] = count
    return counter


def abstain_predictions(
    score_rows: Iterable[ScoreRow],
    *,
    thresholds: Mapping[str, float],
) -> tuple[CandidatePrediction, ...]:
    if set(thresholds) != set(ACTION_ORDER):
        raise ValueError("thresholds must contain exactly the three action labels")
    if any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in thresholds.values()
    ):
        raise ValueError("thresholds must be probabilities between zero and one")
    output: list[CandidatePrediction] = []
    for item in score_rows:
        _validate_probability_vector(item.scores, item.row_id)
        _require_calibrated_ovr_for_abstention(item)
        label = max(ACTION_ORDER, key=lambda action: item.scores[action])
        score = float(item.scores[label])
        threshold = float(thresholds[label])
        evidence = {
            **dict(item.evidence),
            "abstained": score < threshold,
            "winningClass": label,
            "threshold": threshold,
            "scoreVector": {
                action: float(item.scores[action]) for action in ACTION_ORDER
            },
        }
        output.append(
            CandidatePrediction(
                row_id=item.row_id,
                label="REVIEW" if score < threshold else label,
                score=score,
                family=item.family,
                evidence=_freeze(evidence),
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class _ConstantHead:
    probability: float


def _fit_binary_head(matrix: Any, target: np.ndarray, estimator: Any) -> Any:
    unique = np.unique(target)
    if len(unique) == 1:
        return _ConstantHead(float(unique[0]))
    estimator.fit(matrix, target)
    return estimator


def _head_probability(head: Any, matrix: Any) -> np.ndarray:
    if isinstance(head, _ConstantHead):
        return np.full(matrix.shape[0], head.probability, dtype=np.float64)
    if LGBMClassifier is not None and isinstance(head, LGBMClassifier):
        return np.asarray(head.booster_.predict(matrix), dtype=np.float64)
    classes = list(head.classes_)
    return np.asarray(head.predict_proba(matrix)[:, classes.index(1)], dtype=np.float64)


def _ovr_simplex(heads: Mapping[str, Any], matrix: Any) -> np.ndarray:
    return _normalize_probabilities(
        np.column_stack(
            [_head_probability(heads[action], matrix) for action in ACTION_ORDER]
        )
    )


def _calibrated_simplex(
    calibrator: LogisticRegression, raw_simplex: np.ndarray
) -> np.ndarray:
    calibrated = calibrator.predict_proba(raw_simplex)
    output = np.zeros((raw_simplex.shape[0], len(ACTION_ORDER)), dtype=np.float64)
    for source_index, target_index in enumerate(calibrator.classes_):
        output[:, int(target_index)] = calibrated[:, source_index]
    return output


def _score_row_payload(
    row: FeatureRow,
    probabilities: Sequence[float],
    *,
    product_independent: bool,
    warm_reference_products: frozenset[str],
) -> Mapping[str, Any]:
    return {
        "rowId": row.row_id,
        "scores": {
            action: float(probabilities[action_index])
            for action_index, action in enumerate(ACTION_ORDER)
        },
        "coldStart": row.product_key not in warm_reference_products,
        "coldStartMode": "product_independent"
        if product_independent
        else "product_dependent",
    }


def _score_rows_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    family: str,
    provenance: Mapping[str, Any],
) -> tuple[ScoreRow, ...]:
    output: list[ScoreRow] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("persisted calibration score row must be an object")
        row_id = payload.get("rowId")
        raw_scores = payload.get("scores")
        cold_start = payload.get("coldStart")
        cold_start_mode = payload.get("coldStartMode")
        if (
            not isinstance(row_id, str)
            or not row_id
            or not isinstance(raw_scores, Mapping)
            or not isinstance(cold_start, bool)
            or cold_start_mode not in {"product_independent", "product_dependent"}
        ):
            raise ValueError("persisted calibration score row is malformed")
        scores = {
            action: float(raw_scores[action])
            for action in ACTION_ORDER
            if action in raw_scores
        }
        _validate_probability_vector(scores, row_id)
        output.append(
            ScoreRow(
                row_id=row_id,
                scores=MappingProxyType(scores),
                family=family,
                evidence=MappingProxyType(
                    {
                        "coldStart": cold_start,
                        "coldStartMode": cold_start_mode,
                        "provenance": provenance,
                    }
                ),
            )
        )
    return tuple(output)


def _compact_score_row_payload(item: ScoreRow) -> Mapping[str, Any]:
    return {
        "rowId": item.row_id,
        "scores": {action: float(item.scores[action]) for action in ACTION_ORDER},
        "coldStart": bool(item.evidence["coldStart"]),
        "coldStartMode": str(item.evidence["coldStartMode"]),
    }


def _normalize_probabilities(raw: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw, 0.0, 1.0)
    totals = clipped.sum(axis=1, keepdims=True)
    return np.divide(
        clipped,
        totals,
        out=np.full_like(clipped, 1 / len(ACTION_ORDER)),
        where=totals > 0,
    )


def _validate_training_rows(
    rows: Iterable[FeatureRow], labels: Mapping[str, str]
) -> tuple[FeatureRow, ...]:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("training rows must not be empty")
    row_ids = [row.row_id for row in materialized]
    if any(not str(row_id).strip() for row_id in row_ids):
        raise ValueError("training row identities must not be empty")
    duplicates = sorted(
        row_id for row_id, count in Counter(row_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate training row identities: {duplicates[:3]}")
    _validate_labels(materialized, labels)
    for row in materialized:
        if _row_date(row) >= _DEVELOPMENT_END.isoformat():
            raise ValueError(
                "2025-12-01 and later rows, including 2026, cannot be used for candidate training or calibration"
            )
    return materialized


def _validate_labels(rows: Sequence[FeatureRow], labels: Mapping[str, str]) -> None:
    row_ids = {row.row_id for row in rows}
    if row_ids != set(labels):
        raise ValueError(
            f"label identity mismatch: rows={len(row_ids)} labels={len(labels)} "
            f"missing={len(row_ids - set(labels))} extra={len(set(labels) - row_ids)}"
        )
    unknown = sorted({label for label in labels.values() if label not in ACTION_LABELS})
    if unknown:
        raise ValueError(f"unknown action label(s): {unknown}")


def _resolve_cutoff(
    rows: Sequence[FeatureRow], cutoff: str | None, *, strict: bool
) -> str:
    resolved = (
        _normalise_date(cutoff)
        if cutoff is not None
        else max(_row_date(row) for row in rows)
    )
    if strict and any(_row_date(row) >= resolved for row in rows):
        raise ValueError(
            "temporal signature training rows must be strictly before cutoff"
        )
    return resolved


def _row_date(row: FeatureRow) -> str:
    return _normalise_date(row.date)


def _normalise_date(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid ISO date: {value!r}") from error


def _validate_probability_vector(scores: Mapping[str, float], row_id: str) -> None:
    if set(scores) != set(ACTION_ORDER):
        raise ValueError(f"score row has an invalid action set: {row_id}")
    values = [float(scores[action]) for action in ACTION_ORDER]
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError(f"score row has invalid probability values: {row_id}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"score row probability vector must sum to one: {row_id}")


def _require_calibrated_ovr_for_abstention(item: ScoreRow) -> None:
    if item.family not in {"ovr_linear", "ovr_lightgbm"}:
        return
    provenance = item.evidence.get("provenance")
    calibration = (
        provenance.get("calibration") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(calibration, Mapping) or calibration.get("state") != "calibrated":
        raise ValueError(
            "OVR scores must be calibrated before probability-threshold abstention"
        )


def _temporal_feature_names(customer_identity_basis: str) -> tuple[str, ...]:
    return (
        f"identity::{customer_identity_basis}",
        "identity::product_key",
        "raw::type",
        "raw::symptom",
        "raw::detail",
        "raw::intake_count",
        "raw::service_count",
        "raw::treatment_count",
    )


def _temporal_model_sha256(
    counts: Mapping[str, Mapping[tuple[Any, ...], Counter[str]]],
    global_counts: Counter[str],
    *,
    cutoff: str,
    min_support: int,
    smoothing: float,
    customer_identity_basis: str,
) -> str:
    return _sha256_json(
        {
            "counts": {
                level: {
                    repr(key): dict(sorted(value.items()))
                    for key, value in values.items()
                }
                for level, values in counts.items()
            },
            "global": dict(sorted(global_counts.items())),
            "cutoff": cutoff,
            "minSupport": min_support,
            "smoothing": smoothing,
            "customerIdentityBasis": customer_identity_basis,
        }
    )


def _signature(
    row: FeatureRow, level: str, customer_identity_basis: str
) -> tuple[Any, ...]:
    base = (
        _raw_text(row.features.get("type")),
        _raw_text(row.features.get("symptom")),
        _raw_text(row.features.get("detail")),
        _count_value(row.features.get("intake_count")),
        _count_value(row.features.get("service_count")),
        _count_value(row.features.get("treatment_count")),
    )
    customer_identity = str(getattr(row, customer_identity_basis))
    if level == "product":
        return customer_identity, row.product_key, *base
    if level == "customer_transfer":
        return customer_identity, *base
    if level == "global_transfer":
        return base
    raise ValueError(f"unknown signature level: {level}")


def _raw_text(value: Any) -> str:
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip()
    ).lower()


def _count_value(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _training_rows_sha256(rows: Sequence[FeatureRow], labels: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{_TRAINING_DIGEST_VERSION}\0".encode("ascii"))
    for row in sorted(rows, key=lambda item: item.row_id):
        row_bytes = _canonical_json_bytes(
            {
                "sourceIdentity": {
                    "rowId": row.row_id,
                    "sourceRow": row.source_row,
                    "sourceDatasetId": _source_dataset_id(row),
                },
                "date": _row_date(row),
                "month": row.month,
                "rawCustomer": row.raw_customer,
                "customerKey": row.customer_key,
                "legacyCustomerKey": row.legacy_customer_key,
                "productKey": row.product_key,
                "episodeId": row.episode_id,
                "episodeBasis": row.episode_basis,
                "features": _plain(row.features),
                "label": labels[row.row_id],
            }
        )
        digest.update(len(row_bytes).to_bytes(8, byteorder="big"))
        digest.update(row_bytes)
    return digest.hexdigest()


def _source_dataset_id(row: FeatureRow) -> str | None:
    source_dataset_id, separator, source_row = row.row_id.rpartition(":")
    if separator and source_dataset_id and source_row == str(row.source_row):
        return source_dataset_id
    return None


def _row_universe_sha256(rows: Sequence[FeatureRow]) -> str:
    return _sha256_json(
        [
            {
                "rowId": row.row_id,
                "sourceRow": row.source_row,
                "date": _row_date(row),
                "productKey": row.product_key,
            }
            for row in sorted(rows, key=lambda item: item.row_id)
        ]
    )


def _direct_product_identity_channels(spec: FeatureSpec) -> frozenset[str]:
    fields = (
        set(spec.text_fields) | set(spec.categorical_fields) | set(spec.numeric_fields)
    )
    if spec.include_product_key:
        fields.add("product_key")
    return frozenset(fields & _PRODUCT_IDENTITY_FIELDS)


def _estimate_encoder_nnz(rows: Sequence[FeatureRow], spec: FeatureSpec) -> int:
    """Upper-bound encoder nnz before vectorizer or sparse-block allocation begins."""
    categorical_channels = (
        len(spec.categorical_fields)
        + int(spec.include_customer_key)
        + int(spec.include_product_key)
    )
    numeric_channels = len(spec.numeric_fields)
    total = 0
    for row in rows:
        total += categorical_channels + numeric_channels
        for field in spec.text_fields:
            text = _raw_text(row.features.get(field))[: spec.max_text_chars_per_field]
            word_count = len(re.findall(r"\w+", text))
            total += word_count + max(word_count - 1, 0)
            total += sum(max(len(text) - ngram + 1, 0) for ngram in (3, 4, 5))
    return total


def _estimate_sparse_matrix_bytes(row_count: int, nnz: int) -> int:
    return (
        nnz * (np.dtype(np.float32).itemsize + np.dtype(np.int32).itemsize)
        + (row_count + 1) * np.dtype(np.int32).itemsize
    )


def _estimate_sparse_working_set_bytes(row_count: int, nnz: int) -> int:
    # Count/TF-IDF/index construction can keep roughly four CSR-sized numeric
    # blocks live together. Raw Python vocabulary memory is checked separately
    # by RawFeatureEncoder before sklearn allocation.
    return 4 * _estimate_sparse_matrix_bytes(row_count, nnz)


def _hash_model_state(value: Any) -> str:
    if isinstance(value, Mapping):
        chunks = [
            {"key": str(key), "valueSha256": _hash_model_state(value[key])}
            for key in sorted(value, key=str)
        ]
        return _sha256_json(chunks)
    if isinstance(value, _ConstantHead):
        return _sha256_json({"constant": value.probability})
    if LGBMClassifier is not None and isinstance(value, LGBMClassifier):
        return hashlib.sha256(
            value.booster_.model_to_string().encode("utf-8")
        ).hexdigest()
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _dependency_versions() -> Mapping[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikitLearn": sklearn.__version__,
        "lightgbm": getattr(sys.modules.get("lightgbm"), "__version__", "unavailable"),
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)

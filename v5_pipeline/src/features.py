from __future__ import annotations

import hashlib
import json
import math
import re
import resource
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .data_contract import FeatureRow, validate_requested_features


DEFAULT_TEXT_FIELDS = ("type", "symptom", "detail")
DEFAULT_CATEGORICAL_FIELDS = (
    "occurrence_category",
    "Area",
    "Maker",
    "material",
    "service_type",
    "customer_process",
    "line",
    "product_category",
    "occurrence_process",
)
DEFAULT_NUMERIC_FIELDS = (
    "intake_count",
    "service_count",
    "treatment_count",
    "usage_hours",
    "cumulative_usage_hours",
    "usage_cycles",
    "quantity",
)

_MISSING_CATEGORY = "__missing__"
_UNKNOWN_CATEGORY = "__unknown__"
_RESERVED_CATEGORIES = frozenset({_MISSING_CATEGORY, _UNKNOWN_CATEGORY})
_NUMERIC_PATTERN = re.compile(r"\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_MATRIX_DTYPE = np.dtype(np.float32)
_NORMALIZATION_VERSION = "nfkc-lower-collapse-whitespace-v1"
_WORD_ANALYZER_VERSION = "field-word-1-2-v1"
_CHAR_ANALYZER_VERSION = "field-local-char-3-5-v1"
_CATEGORY_POLICY_VERSION = "nfkc-explicit-missing-unknown-v1"
_NUMERIC_POLICY_VERSION = "nonnegative-finite-log1p-missing-v1"


class FeatureResourceLimitError(RuntimeError):
    """Raised before a configured feature or matrix resource cap is crossed."""


class FeatureConfigurationError(ValueError):
    """Raised when pruning settings would silently remove a feature channel."""


@dataclass(frozen=True, slots=True)
class EncoderBenchmark:
    operation: str
    row_count: int
    feature_count: int
    nnz: int
    elapsed_seconds: float
    matrix_bytes: int
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable selection, pruning policy, and resource envelope."""

    text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS
    categorical_fields: tuple[str, ...] = DEFAULT_CATEGORICAL_FIELDS
    numeric_fields: tuple[str, ...] = DEFAULT_NUMERIC_FIELDS
    include_customer_key: bool = False
    include_product_key: bool = False
    word_min_df: int | float = 1
    char_min_df: int | float = 1
    word_max_df: int | float = 1.0
    char_max_df: int | float = 1.0
    word_max_features: int = 50_000
    char_max_features: int = 100_000
    max_categories_per_field: int = 50_000
    max_total_features: int = 200_000
    max_fit_rows: int = 500_000
    max_transform_rows: int = 500_000
    max_output_nnz: int = 100_000_000
    max_matrix_bytes: int = 1_200_000_000
    max_peak_working_bytes: int = 8_000_000_000
    max_raw_vocabulary_terms: int = 2_000_000
    max_raw_vocabulary_bytes: int = 2_000_000_000
    max_text_chars_per_field: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "text_fields", tuple(self.text_fields))
        object.__setattr__(self, "categorical_fields", tuple(self.categorical_fields))
        object.__setattr__(self, "numeric_fields", tuple(self.numeric_fields))
        _validate_spec_limits(self)


class RawFeatureEncoder:
    """Deterministic, leakage-closed sparse encoder for `FeatureRow`."""

    def __init__(
        self,
        spec: FeatureSpec | None = None,
        *,
        validation_config: Mapping[str, Any] | None = None,
    ) -> None:
        requested_spec = spec or FeatureSpec()
        config = validation_config or {}
        self.spec = FeatureSpec(
            text_fields=validate_requested_features(requested_spec.text_fields, config),
            categorical_fields=validate_requested_features(requested_spec.categorical_fields, config),
            numeric_fields=validate_requested_features(requested_spec.numeric_fields, config),
            include_customer_key=requested_spec.include_customer_key,
            include_product_key=requested_spec.include_product_key,
            word_min_df=requested_spec.word_min_df,
            char_min_df=requested_spec.char_min_df,
            word_max_df=requested_spec.word_max_df,
            char_max_df=requested_spec.char_max_df,
            word_max_features=requested_spec.word_max_features,
            char_max_features=requested_spec.char_max_features,
            max_categories_per_field=requested_spec.max_categories_per_field,
            max_total_features=requested_spec.max_total_features,
            max_fit_rows=requested_spec.max_fit_rows,
            max_transform_rows=requested_spec.max_transform_rows,
            max_output_nnz=requested_spec.max_output_nnz,
            max_matrix_bytes=requested_spec.max_matrix_bytes,
            max_peak_working_bytes=requested_spec.max_peak_working_bytes,
            max_raw_vocabulary_terms=requested_spec.max_raw_vocabulary_terms,
            max_raw_vocabulary_bytes=requested_spec.max_raw_vocabulary_bytes,
            max_text_chars_per_field=requested_spec.max_text_chars_per_field,
        )
        self._categorical_names = self._channel_names()
        self._word_vectorizer: TfidfVectorizer | None = None
        self._character_vectorizer: TfidfVectorizer | None = None
        self._category_levels: tuple[tuple[str, ...], ...] = ()
        self._feature_names: tuple[str, ...] = ()
        self._schema_sha256: str | None = None
        self._encoder_manifest_json: str | None = None
        self._fitted = False
        self._last_fit_benchmark: EncoderBenchmark | None = None
        self._last_transform_benchmark: EncoderBenchmark | None = None

    @property
    def feature_count(self) -> int:
        self._require_fitted()
        return len(self._feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return self._feature_names

    @property
    def schema_sha256(self) -> str:
        self._require_fitted()
        assert self._schema_sha256 is not None
        return self._schema_sha256

    @property
    def encoder_manifest(self) -> Mapping[str, Any]:
        self._require_fitted()
        assert self._encoder_manifest_json is not None
        return json.loads(self._encoder_manifest_json)

    @property
    def last_fit_benchmark(self) -> EncoderBenchmark | None:
        return self._last_fit_benchmark

    @property
    def last_transform_benchmark(self) -> EncoderBenchmark | None:
        return self._last_transform_benchmark

    def fit(self, rows: Iterable[FeatureRow]) -> "RawFeatureEncoder":
        started = time.perf_counter()
        materialized = _bounded_rows(rows, self.spec.max_fit_rows, "max_fit_rows")
        documents = _text_documents(materialized, self.spec)
        _preflight_matrix_limits(
            documents,
            len(materialized),
            self.spec,
            self._categorical_names,
        )
        word_vectorizer, _ = _fit_vectorizer(
            documents,
            channel="word",
            analyzer=_field_word_ngrams,
            min_df=self.spec.word_min_df,
            max_df=self.spec.word_max_df,
            max_features=self.spec.word_max_features,
            transform=False,
        )
        character_vectorizer, _ = _fit_vectorizer(
            documents,
            channel="character",
            analyzer=_field_char_ngrams,
            min_df=self.spec.char_min_df,
            max_df=self.spec.char_max_df,
            max_features=self.spec.char_max_features,
            transform=False,
        )
        category_levels = _fit_category_levels(
            materialized,
            self._categorical_names,
            self.spec.max_categories_per_field,
        )
        feature_names = _feature_names(
            word_vectorizer,
            character_vectorizer,
            self._categorical_names,
            category_levels,
            self.spec.numeric_fields,
        )
        _ensure_feature_limit(len(feature_names), self.spec)
        manifest_json, schema_sha256 = _fitted_identity(
            self.spec,
            feature_names,
            word_vectorizer,
            character_vectorizer,
            category_levels,
        )
        self._set_fitted_state(
            word_vectorizer,
            character_vectorizer,
            category_levels,
            feature_names,
            manifest_json,
            schema_sha256,
        )
        self._last_fit_benchmark = _benchmark(
            "fit",
            len(materialized),
            len(feature_names),
            None,
            started,
        )
        return self

    def transform(self, rows: Iterable[FeatureRow]) -> sparse.csr_matrix:
        self._require_fitted()
        started = time.perf_counter()
        materialized = _bounded_rows(rows, self.spec.max_transform_rows, "max_transform_rows")
        documents = _text_documents(materialized, self.spec)
        _preflight_matrix_limits(
            documents,
            len(materialized),
            self.spec,
            self._categorical_names,
            word_vocabulary=(
                frozenset(self._word_vectorizer.vocabulary_)
                if self._word_vectorizer is not None
                else frozenset()
            ),
            character_vocabulary=(
                frozenset(self._character_vectorizer.vocabulary_)
                if self._character_vectorizer is not None
                else frozenset()
            ),
        )
        matrix = _empty_matrix(len(materialized))
        if self._word_vectorizer is not None:
            matrix = _append_block(matrix, self._word_vectorizer.transform(documents), self.spec)
        if self._character_vectorizer is not None:
            matrix = _append_block(matrix, self._character_vectorizer.transform(documents), self.spec)
        matrix = _append_block(
            matrix,
            _categorical_matrix(materialized, self._categorical_names, self._category_levels),
            self.spec,
        )
        matrix = _append_block(matrix, _numeric_matrix(materialized, self.spec.numeric_fields), self.spec)
        if matrix.shape[1] != self.feature_count:
            raise RuntimeError(
                f"fitted feature width changed: expected {self.feature_count}, observed {matrix.shape[1]}"
            )
        self._last_transform_benchmark = _benchmark(
            "transform",
            len(materialized),
            self.feature_count,
            matrix,
            started,
        )
        return matrix

    def fit_transform(self, rows: Iterable[FeatureRow]) -> sparse.csr_matrix:
        started = time.perf_counter()
        materialized = _bounded_rows(rows, self.spec.max_fit_rows, "max_fit_rows")
        if len(materialized) > self.spec.max_transform_rows:
            raise FeatureResourceLimitError(
                f"max_transform_rows exceeded: {len(materialized)} > {self.spec.max_transform_rows}"
            )
        documents = _text_documents(materialized, self.spec)
        _preflight_matrix_limits(
            documents,
            len(materialized),
            self.spec,
            self._categorical_names,
        )
        matrix = _empty_matrix(len(materialized))

        word_vectorizer, word_matrix = _fit_vectorizer(
            documents,
            channel="word",
            analyzer=_field_word_ngrams,
            min_df=self.spec.word_min_df,
            max_df=self.spec.word_max_df,
            max_features=self.spec.word_max_features,
            transform=True,
        )
        if word_matrix is not None:
            matrix = _append_block(matrix, word_matrix, self.spec)
        character_vectorizer, character_matrix = _fit_vectorizer(
            documents,
            channel="character",
            analyzer=_field_char_ngrams,
            min_df=self.spec.char_min_df,
            max_df=self.spec.char_max_df,
            max_features=self.spec.char_max_features,
            transform=True,
        )
        if character_matrix is not None:
            matrix = _append_block(matrix, character_matrix, self.spec)

        category_levels = _fit_category_levels(
            materialized,
            self._categorical_names,
            self.spec.max_categories_per_field,
        )
        matrix = _append_block(
            matrix,
            _categorical_matrix(materialized, self._categorical_names, category_levels),
            self.spec,
        )
        matrix = _append_block(matrix, _numeric_matrix(materialized, self.spec.numeric_fields), self.spec)

        feature_names = _feature_names(
            word_vectorizer,
            character_vectorizer,
            self._categorical_names,
            category_levels,
            self.spec.numeric_fields,
        )
        _ensure_feature_limit(len(feature_names), self.spec)
        if matrix.shape[1] != len(feature_names):
            raise RuntimeError(
                f"fitted feature width mismatch: expected {len(feature_names)}, observed {matrix.shape[1]}"
            )
        manifest_json, schema_sha256 = _fitted_identity(
            self.spec,
            feature_names,
            word_vectorizer,
            character_vectorizer,
            category_levels,
        )
        self._set_fitted_state(
            word_vectorizer,
            character_vectorizer,
            category_levels,
            feature_names,
            manifest_json,
            schema_sha256,
        )
        benchmark = _benchmark(
            "fit_transform",
            len(materialized),
            len(feature_names),
            matrix,
            started,
        )
        self._last_fit_benchmark = benchmark
        self._last_transform_benchmark = benchmark
        return matrix

    def _channel_names(self) -> tuple[str, ...]:
        names = list(self.spec.categorical_fields)
        if self.spec.include_customer_key:
            names.append("customer_key")
        if self.spec.include_product_key:
            names.append("product_key")
        return tuple(names)

    def _set_fitted_state(
        self,
        word_vectorizer: TfidfVectorizer | None,
        character_vectorizer: TfidfVectorizer | None,
        category_levels: tuple[tuple[str, ...], ...],
        feature_names: tuple[str, ...],
        manifest_json: str,
        schema_sha256: str,
    ) -> None:
        self._word_vectorizer = word_vectorizer
        self._character_vectorizer = character_vectorizer
        self._category_levels = category_levels
        self._feature_names = feature_names
        self._encoder_manifest_json = manifest_json
        self._schema_sha256 = schema_sha256
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("RawFeatureEncoder must be fitted before accessing the learned schema")


def _fit_vectorizer(
    documents: Sequence[str],
    *,
    channel: str,
    analyzer: Callable[[str], list[str]],
    min_df: int | float,
    max_df: int | float,
    max_features: int,
    transform: bool,
) -> tuple[TfidfVectorizer | None, sparse.csr_matrix | None]:
    _validate_effective_df(min_df, max_df, len(documents), channel)
    if not any(documents) or not any(analyzer(document) for document in documents):
        return None, None
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        dtype=_MATRIX_DTYPE,
    )
    try:
        matrix = vectorizer.fit_transform(documents) if transform else None
        if not transform:
            vectorizer.fit(documents)
    except ValueError as error:
        if "After pruning, no terms remain" in str(error):
            raise FeatureConfigurationError(
                f"{channel} channel pruning removed all terms; revise min_df/max_df"
            ) from error
        raise
    return vectorizer, matrix.tocsr().astype(_MATRIX_DTYPE, copy=False) if matrix is not None else None


def _fit_category_levels(
    rows: Sequence[FeatureRow],
    names: Sequence[str],
    max_categories_per_field: int,
) -> tuple[tuple[str, ...], ...]:
    observed = [set() for _ in names]
    for row in rows:
        for index, name in enumerate(names):
            token = _category_token(_raw_channel_value(row, name))
            if token != _MISSING_CATEGORY:
                observed[index].add(token)
                if len(observed[index]) > max_categories_per_field:
                    raise FeatureResourceLimitError(
                        f"max_categories_per_field exceeded for {name}: "
                        f"{len(observed[index])} > {max_categories_per_field}"
                    )
    return tuple(
        (_MISSING_CATEGORY, _UNKNOWN_CATEGORY, *sorted(field_values))
        for field_values in observed
    )


def _categorical_matrix(
    rows: Sequence[FeatureRow],
    names: Sequence[str],
    levels: Sequence[Sequence[str]],
) -> sparse.csr_matrix:
    if not names:
        return _empty_matrix(len(rows))
    entries = len(rows) * len(names)
    data = np.ones(entries, dtype=_MATRIX_DTYPE)
    indices = np.empty(entries, dtype=np.int32)
    indptr = np.arange(0, entries + 1, len(names), dtype=np.int32)
    offsets: list[int] = []
    level_maps: list[dict[str, int]] = []
    offset = 0
    for field_levels in levels:
        offsets.append(offset)
        level_maps.append({value: index for index, value in enumerate(field_levels)})
        offset += len(field_levels)
    position = 0
    for row in rows:
        for field_index, name in enumerate(names):
            token = _category_token(_raw_channel_value(row, name))
            if token not in level_maps[field_index]:
                token = _UNKNOWN_CATEGORY
            indices[position] = offsets[field_index] + level_maps[field_index][token]
            position += 1
    return sparse.csr_matrix((data, indices, indptr), shape=(len(rows), offset), dtype=_MATRIX_DTYPE)


def _feature_names(
    word_vectorizer: TfidfVectorizer | None,
    character_vectorizer: TfidfVectorizer | None,
    categorical_names: Sequence[str],
    category_levels: Sequence[Sequence[str]],
    numeric_fields: Sequence[str],
) -> tuple[str, ...]:
    names: list[str] = []
    if word_vectorizer is not None:
        names.extend(f"word::{name}" for name in word_vectorizer.get_feature_names_out())
    if character_vectorizer is not None:
        names.extend(f"char::{name}" for name in character_vectorizer.get_feature_names_out())
    for field, levels in zip(categorical_names, category_levels, strict=True):
        names.extend(f"category::{field}={level}" for level in levels)
    for field in numeric_fields:
        names.extend((f"numeric::{field}::log1p", f"numeric::{field}::missing"))
    return tuple(names)


def _text_documents(rows: Sequence[FeatureRow], spec: FeatureSpec) -> list[str]:
    return [_text_document(row, spec.text_fields, spec.max_text_chars_per_field) for row in rows]


def _text_document(row: FeatureRow, fields: Sequence[str], max_chars: int) -> str:
    parts: list[str] = []
    for field in fields:
        value = _normalized_text(row.features.get(field))
        if len(value) > max_chars:
            raise FeatureResourceLimitError(
                f"max_text_chars_per_field exceeded for {field}: {len(value)} > {max_chars}"
            )
        if value:
            parts.append(f"{field}\t{value}")
    return "\n".join(parts)


def _field_word_ngrams(document: str) -> list[str]:
    terms: list[str] = []
    for field, value in _document_fields(document):
        tokens = [f"{field}={token}" for token in _WORD_PATTERN.findall(value)]
        terms.extend(tokens)
        terms.extend(f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1))
    return terms


def _field_char_ngrams(document: str) -> list[str]:
    terms: list[str] = []
    for field, value in _document_fields(document):
        for width in range(3, 6):
            terms.extend(
                f"{field}={value[index:index + width]}"
                for index in range(len(value) - width + 1)
            )
    return terms


def _document_fields(document: str) -> Iterable[tuple[str, str]]:
    for line in document.splitlines():
        field, separator, value = line.partition("\t")
        if separator:
            yield field, value


def _numeric_matrix(rows: Sequence[FeatureRow], fields: Sequence[str]) -> sparse.csr_matrix:
    if not fields:
        return _empty_matrix(len(rows))
    capacity = len(rows) * len(fields)
    data = np.empty(capacity, dtype=_MATRIX_DTYPE)
    indices = np.empty(capacity, dtype=np.int32)
    indptr = np.zeros(len(rows) + 1, dtype=np.int32)
    position = 0
    for row_index, row in enumerate(rows):
        for field_index, field in enumerate(fields):
            parsed = _parse_numeric(row.features.get(field))
            if parsed is None:
                data[position] = 1.0
                indices[position] = field_index * 2 + 1
                position += 1
            else:
                scaled = math.log1p(parsed)
                if scaled:
                    data[position] = scaled
                    indices[position] = field_index * 2
                    position += 1
        indptr[row_index + 1] = position
    return sparse.csr_matrix(
        (data[:position], indices[:position], indptr),
        shape=(len(rows), len(fields) * 2),
        dtype=_MATRIX_DTYPE,
    )


def _append_block(
    matrix: sparse.csr_matrix,
    block: sparse.spmatrix,
    spec: FeatureSpec,
) -> sparse.csr_matrix:
    block = block.tocsr().astype(_MATRIX_DTYPE, copy=False)
    feature_count = matrix.shape[1] + block.shape[1]
    nnz = matrix.nnz + block.nnz
    _ensure_feature_limit(feature_count, spec)
    _ensure_matrix_limits(matrix.shape[0], nnz, spec)
    if matrix.shape[1] == 0:
        result = block
    elif block.shape[1] == 0:
        result = matrix
    else:
        result = sparse.hstack((matrix, block), format="csr", dtype=_MATRIX_DTYPE)
    result.sort_indices()
    _ensure_actual_matrix_limits(result, spec)
    return result


def _empty_matrix(row_count: int) -> sparse.csr_matrix:
    return sparse.csr_matrix((row_count, 0), dtype=_MATRIX_DTYPE)


def _bounded_rows(rows: Iterable[FeatureRow], limit: int, limit_name: str) -> tuple[FeatureRow, ...]:
    if isinstance(rows, Sequence):
        if len(rows) > limit:
            raise FeatureResourceLimitError(f"{limit_name} exceeded: {len(rows)} > {limit}")
        return tuple(rows)
    materialized: list[FeatureRow] = []
    for row in rows:
        materialized.append(row)
        if len(materialized) > limit:
            raise FeatureResourceLimitError(f"{limit_name} exceeded: {len(materialized)} > {limit}")
    return tuple(materialized)


def _raw_channel_value(row: FeatureRow, name: str) -> Any:
    if name == "customer_key":
        return row.customer_key
    if name == "product_key":
        return row.product_key
    return row.features.get(name)


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")).strip()).lower()


def _category_token(value: Any) -> str:
    normalized = _normalized_text(value)
    if not normalized:
        return _MISSING_CATEGORY
    if normalized in _RESERVED_CATEGORIES or normalized.startswith("__value__:"):
        return f"__value__:{normalized}"
    return normalized


def _parse_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        text = unicodedata.normalize("NFKC", str(value)).strip().replace(",", "")
        if not _NUMERIC_PATTERN.fullmatch(text):
            return None
        parsed = float(text)
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _fitted_identity(
    spec: FeatureSpec,
    feature_names: Sequence[str],
    word_vectorizer: TfidfVectorizer | None,
    character_vectorizer: TfidfVectorizer | None,
    category_levels: Sequence[Sequence[str]],
) -> tuple[str, str]:
    word_idf = _canonical_idf_bytes(word_vectorizer)
    char_idf = _canonical_idf_bytes(character_vectorizer)
    manifest = {
        "schemaVersion": 2,
        "implementation": {
            "normalization": _NORMALIZATION_VERSION,
            "wordAnalyzer": _WORD_ANALYZER_VERSION,
            "characterAnalyzer": _CHAR_ANALYZER_VERSION,
            "categoryPolicy": _CATEGORY_POLICY_VERSION,
            "numericPolicy": _NUMERIC_POLICY_VERSION,
            "matrixDtype": _MATRIX_DTYPE.name,
            "tfidfNorm": "l2",
            "tfidfSmoothIdf": True,
            "tfidfSublinearTf": False,
        },
        "dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__,
        },
        "spec": {
            "textFields": list(spec.text_fields),
            "categoricalFields": list(spec.categorical_fields),
            "numericFields": list(spec.numeric_fields),
            "includeCustomerKey": spec.include_customer_key,
            "includeProductKey": spec.include_product_key,
            "wordMinDf": spec.word_min_df,
            "charMinDf": spec.char_min_df,
            "wordMaxDf": spec.word_max_df,
            "charMaxDf": spec.char_max_df,
            "wordMaxFeatures": spec.word_max_features,
            "charMaxFeatures": spec.char_max_features,
            "maxCategoriesPerField": spec.max_categories_per_field,
            "maxTotalFeatures": spec.max_total_features,
            "maxFitRows": spec.max_fit_rows,
            "maxTransformRows": spec.max_transform_rows,
            "maxOutputNnz": spec.max_output_nnz,
            "maxMatrixBytes": spec.max_matrix_bytes,
            "maxPeakWorkingBytes": spec.max_peak_working_bytes,
            "maxRawVocabularyTerms": spec.max_raw_vocabulary_terms,
            "maxRawVocabularyBytes": spec.max_raw_vocabulary_bytes,
            "maxTextCharsPerField": spec.max_text_chars_per_field,
        },
        "learned": {
            "featureNames": list(feature_names),
            "categoryLevels": [list(levels) for levels in category_levels],
            "wordIdfSha256": hashlib.sha256(word_idf).hexdigest(),
            "characterIdfSha256": hashlib.sha256(char_idf).hexdigest(),
        },
    }
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    for payload in (manifest_json.encode("utf-8"), word_idf, char_idf):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return manifest_json, digest.hexdigest()


def _canonical_idf_bytes(vectorizer: TfidfVectorizer | None) -> bytes:
    if vectorizer is None:
        return b""
    return np.asarray(vectorizer.idf_, dtype=np.dtype("<f4")).tobytes(order="C")


def _validate_spec_limits(spec: FeatureSpec) -> None:
    for name in ("word_min_df", "char_min_df", "word_max_df", "char_max_df"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be an int or float")
        if isinstance(value, int) and value < 1:
            raise ValueError(f"{name} integer value must be at least 1")
        if isinstance(value, float) and not 0.0 < value <= 1.0:
            raise ValueError(f"{name} float value must be in (0, 1]")
    for name in (
        "word_max_features",
        "char_max_features",
        "max_categories_per_field",
        "max_total_features",
        "max_fit_rows",
        "max_transform_rows",
        "max_output_nnz",
        "max_matrix_bytes",
        "max_peak_working_bytes",
        "max_raw_vocabulary_terms",
        "max_raw_vocabulary_bytes",
        "max_text_chars_per_field",
    ):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _ensure_feature_limit(feature_count: int, spec: FeatureSpec) -> None:
    if feature_count > spec.max_total_features:
        raise FeatureResourceLimitError(
            f"max_total_features exceeded: {feature_count} > {spec.max_total_features}"
        )


def _validate_effective_df(
    min_df: int | float,
    max_df: int | float,
    document_count: int,
    channel: str,
) -> None:
    minimum = min_df if isinstance(min_df, int) else math.ceil(min_df * document_count)
    maximum = max_df if isinstance(max_df, int) else math.floor(max_df * document_count)
    if maximum < minimum:
        raise FeatureConfigurationError(
            f"{channel} channel effective max_df {maximum} is below min_df {minimum}"
        )


def _preflight_matrix_limits(
    documents: Sequence[str],
    row_count: int,
    spec: FeatureSpec,
    categorical_names: Sequence[str],
    *,
    word_vocabulary: frozenset[str] | None = None,
    character_vocabulary: frozenset[str] | None = None,
) -> None:
    """Conservatively bound CSR size before sklearn allocates a sparse block."""

    projected_nnz = row_count * (len(categorical_names) + len(spec.numeric_fields))
    raw_vocabulary: set[str] | None = (
        set() if word_vocabulary is None and character_vocabulary is None else None
    )
    raw_vocabulary_bytes = 0
    for document in documents:
        word_terms = set(_field_word_ngrams(document))
        character_terms = set(_field_char_ngrams(document))
        if raw_vocabulary is not None:
            for namespace, terms in (("word", word_terms), ("character", character_terms)):
                for token in terms:
                    vocabulary_key = f"{namespace}\0{token}"
                    if vocabulary_key in raw_vocabulary:
                        continue
                    raw_vocabulary.add(vocabulary_key)
                    raw_vocabulary_bytes += 96 + len(vocabulary_key.encode("utf-8"))
                    if len(raw_vocabulary) > spec.max_raw_vocabulary_terms:
                        raise FeatureResourceLimitError(
                            "preflight max_raw_vocabulary_terms exceeded: "
                            f"{len(raw_vocabulary)} > {spec.max_raw_vocabulary_terms}"
                        )
                    if raw_vocabulary_bytes > spec.max_raw_vocabulary_bytes:
                        raise FeatureResourceLimitError(
                            "preflight max_raw_vocabulary_bytes exceeded: "
                            f"{raw_vocabulary_bytes} > {spec.max_raw_vocabulary_bytes}"
                        )
        projected_nnz += (
            len(word_terms & word_vocabulary)
            if word_vocabulary is not None
            else len(word_terms)
        )
        projected_nnz += (
            len(character_terms & character_vocabulary)
            if character_vocabulary is not None
            else len(character_terms)
        )
        if projected_nnz > spec.max_output_nnz:
            raise FeatureResourceLimitError(
                "preflight max_output_nnz exceeded: "
                f"projected at least {projected_nnz} > {spec.max_output_nnz}"
            )
        projected_bytes = projected_nnz * 8 + (row_count + 1) * 4
        if projected_bytes > spec.max_matrix_bytes:
            raise FeatureResourceLimitError(
                "preflight max_matrix_bytes exceeded: "
                f"projected at least {projected_bytes} > {spec.max_matrix_bytes}"
            )
        # CountVectorizer/TfidfVectorizer may retain an int count matrix, a
        # normalized float matrix, index arrays, and the Python vocabulary at
        # the same time before max_features pruning. This conservative envelope
        # is intentionally separate from the final CSR byte budget.
        projected_peak_bytes = (
            projected_nnz * 32
            + (row_count + 1) * 16
            + raw_vocabulary_bytes
        )
        if projected_peak_bytes > spec.max_peak_working_bytes:
            raise FeatureResourceLimitError(
                "preflight max_peak_working_bytes exceeded: "
                f"projected at least {projected_peak_bytes} > {spec.max_peak_working_bytes}"
            )


def _ensure_matrix_limits(row_count: int, nnz: int, spec: FeatureSpec) -> None:
    if nnz > spec.max_output_nnz:
        raise FeatureResourceLimitError(f"max_output_nnz exceeded: {nnz} > {spec.max_output_nnz}")
    projected_bytes = nnz * 8 + (row_count + 1) * 4
    if projected_bytes > spec.max_matrix_bytes:
        raise FeatureResourceLimitError(
            f"max_matrix_bytes exceeded: projected {projected_bytes} > {spec.max_matrix_bytes}"
        )


def _ensure_actual_matrix_limits(matrix: sparse.csr_matrix, spec: FeatureSpec) -> None:
    matrix_bytes = _matrix_bytes(matrix)
    if matrix_bytes > spec.max_matrix_bytes:
        raise FeatureResourceLimitError(
            f"max_matrix_bytes exceeded: observed {matrix_bytes} > {spec.max_matrix_bytes}"
        )


def _matrix_bytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _benchmark(
    operation: str,
    row_count: int,
    feature_count: int,
    matrix: sparse.csr_matrix | None,
    started: float,
) -> EncoderBenchmark:
    return EncoderBenchmark(
        operation=operation,
        row_count=row_count,
        feature_count=feature_count,
        nnz=matrix.nnz if matrix is not None else 0,
        elapsed_seconds=time.perf_counter() - started,
        matrix_bytes=_matrix_bytes(matrix) if matrix is not None else 0,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024

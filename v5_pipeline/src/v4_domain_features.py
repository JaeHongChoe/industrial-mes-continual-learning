from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .data_contract import ACTION_LABELS, FeatureRow


_LABEL_NAMES = {
    "PROCESS": "process",
    "RETURN": "shipment",
    "REPAIR": "repair",
}
_HISTORY_LEVELS = ("product", "product_type", "product_detail_counts", "detail_counts")
# 자유 텍스트(type / symptom / detail)에서 뽑는 의미 신호.
# 아래 어휘는 공개용 예시이며, 실제 적용 시 현장 용어로 교체한다. 키 이름이 곧 feature 이름이 된다.
_SEMANTIC_PATTERNS = {
    "repair": re.compile(r"수리|repair|rework|재작업", re.I),
    "return": re.compile(r"반송|폐기|scrap|return|출고\s*요청", re.I),
    "leak": re.compile(r"leak|누설", re.I),
    "count_limit": re.compile(
        r"(?:사용|누적|처리).{0,20}(?:기준|횟수|초과|over|exceed)",
        re.I,
    ),
    "product_mismatch": re.compile(
        r"제품.{0,8}(?:상이|다름|불일치|오류)|재질\s*상이",
        re.I,
    ),
    "line_change": re.compile(r"라인.{0,8}(?:변경|상이|다름)|>{2,}", re.I),
    "surface_damage": re.compile(
        r"표면.{0,10}(?:박리|손상|얼룩)|surface.{0,10}(?:peel|damage|stain)",
        re.I,
    ),
    "missing_accessory": re.compile(
        r"(?:케이스|부속품|볼트).{0,12}(?:누락|없음|미입고)",
        re.I,
    ),
    "customer_request": re.compile(r"고객\s*요청|customer\s*request", re.I),
}


@dataclass(frozen=True, slots=True)
class V4DomainFeatureSpec:
    smoothing: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.smoothing, bool) or not isinstance(self.smoothing, (int, float)):
            raise ValueError("smoothing must be a non-negative finite number")
        if not math.isfinite(float(self.smoothing)) or float(self.smoothing) < 0:
            raise ValueError("smoothing must be a non-negative finite number")


class V4DomainFeatureTransformer:
    """Prior-only V4-inspired numeric, semantic, and history features."""

    def __init__(self, spec: V4DomainFeatureSpec | None = None) -> None:
        self.spec = spec or V4DomainFeatureSpec()
        self._feature_names = _feature_names()
        self._counts = _empty_counts()
        self._global: Counter[str] = Counter()
        self._cutoff: str | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def schema_sha256(self) -> str:
        payload = {
            "schemaVersion": 1,
            "kind": "v4_domain_features",
            "spec": asdict(self.spec),
            "featureNames": self.feature_names,
            "historyPolicy": "monthly_expanding_oof_then_prior_only",
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def fit_transform_oof(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
    ) -> sparse.csr_matrix:
        materialized = tuple(rows)
        _validate_rows_and_labels(materialized, labels)
        vectors: list[np.ndarray | None] = [None] * len(materialized)
        counts = _empty_counts()
        global_counts: Counter[str] = Counter()
        month_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(materialized):
            month_indices[row.month].append(index)
        for month in sorted(month_indices):
            indices = sorted(
                month_indices[month],
                key=lambda index: (materialized[index].date, materialized[index].source_row),
            )
            for index in indices:
                vectors[index] = _row_vector(
                    materialized[index],
                    counts,
                    global_counts,
                    smoothing=float(self.spec.smoothing),
                )
            for index in indices:
                _update_counts(counts, global_counts, materialized[index], labels[materialized[index].row_id])
        self._counts = counts
        self._global = global_counts
        self._cutoff = max((row.date for row in materialized), default="") or None
        return _matrix(vectors, len(self.feature_names))

    def fit_history(
        self,
        rows: Iterable[FeatureRow],
        labels: Mapping[str, str],
        *,
        cutoff: str,
    ) -> "V4DomainFeatureTransformer":
        materialized = tuple(row for row in rows if row.date <= cutoff)
        _validate_rows_and_labels(materialized, labels)
        counts = _empty_counts()
        global_counts: Counter[str] = Counter()
        for row in sorted(materialized, key=lambda item: (item.date, item.source_row)):
            _update_counts(counts, global_counts, row, labels[row.row_id])
        self._counts = counts
        self._global = global_counts
        self._cutoff = cutoff
        return self

    def transform_with_history(
        self,
        rows: Iterable[FeatureRow],
    ) -> sparse.csr_matrix:
        if self._cutoff is None:
            raise RuntimeError("V4DomainFeatureTransformer must be fitted before transform")
        materialized = tuple(rows)
        if any(row.date <= self._cutoff for row in materialized):
            raise ValueError("scored rows must be strictly after the fitted history cutoff")
        vectors = [
            _row_vector(
                row,
                self._counts,
                self._global,
                smoothing=float(self.spec.smoothing),
            )
            for row in materialized
        ]
        return _matrix(vectors, len(self.feature_names))


def _feature_names() -> tuple[str, ...]:
    names = [
        "numeric::intake_count",
        "numeric::service_count",
        "numeric::treatment_count",
        "numeric::usage_hours",
        "numeric::cum_usage_hours",
        "numeric::intake_minus_service",
        "numeric::service_minus_treatment",
        "numeric::cum_usage_minus_usage",
        "numeric::treatment_to_service_ratio",
        "numeric::usage_to_cum_usage_ratio",
        "numeric::intake_missing",
        "numeric::service_missing",
        "numeric::treatment_missing",
        "numeric::usage_missing",
        "numeric::cum_usage_missing",
        "text::detail_length_log1p",
        "text::defect_count_log1p",
    ]
    names.extend(f"semantic::{name}" for name in _SEMANTIC_PATTERNS)
    names.append("history::cold_start")
    for level in _HISTORY_LEVELS:
        names.extend(
            (
                f"history::{level}::support_log1p",
                f"history::{level}::process_rate",
                f"history::{level}::return_rate",
                f"history::{level}::repair_rate",
            )
        )
    return tuple(names)


def _row_vector(
    row: FeatureRow,
    counts: Mapping[str, Mapping[tuple[Any, ...], Counter[str]]],
    global_counts: Counter[str],
    *,
    smoothing: float,
) -> np.ndarray:
    values: list[float] = []
    intake_count = _number(row.features.get("intake_count"))
    service_count = _number(row.features.get("service_count"))
    treatment_count = _number(row.features.get("treatment_count"))
    usage_hours = _number(row.features.get("usage_hours"))
    cum_usage_hours = _number(row.features.get("cumulative_usage_hours"))
    numeric = (intake_count, service_count, treatment_count, usage_hours, cum_usage_hours)
    values.extend(float(value or 0.0) for value in numeric)
    values.extend(
        (
            float((intake_count or 0.0) - (service_count or 0.0)),
            float((service_count or 0.0) - (treatment_count or 0.0)),
            float((cum_usage_hours or 0.0) - (usage_hours or 0.0)),
            float(((treatment_count or 0.0) + 1.0) / ((service_count or 0.0) + 1.0)),
            float(((usage_hours or 0.0) + 1.0) / ((cum_usage_hours or 0.0) + 1.0)),
        )
    )
    values.extend(1.0 if value is None else 0.0 for value in numeric)
    detail = _normalized(row.features.get("detail"))
    values.append(math.log1p(len(detail)))
    segments = [segment for segment in re.split(r"[,;＋+\n]+", detail) if segment.strip()]
    values.append(math.log1p(len(segments)))
    semantic_text = " ".join(
        _normalized(row.features.get(name)) for name in ("type", "symptom", "detail")
    )
    values.extend(1.0 if pattern.search(semantic_text) else 0.0 for pattern in _SEMANTIC_PATTERNS.values())
    product_support = sum(counts["product"].get(_history_key(row, "product"), Counter()).values())
    values.append(1.0 if product_support == 0 else 0.0)
    global_rates = _global_rates(global_counts)
    for level in _HISTORY_LEVELS:
        observed = counts[level].get(_history_key(row, level), Counter())
        support = sum(observed.values())
        values.append(math.log1p(support))
        rates = _smoothed_rates(observed, support, global_rates, smoothing)
        values.extend(rates[label] for label in ACTION_LABELS_ORDERED)
    return np.asarray(values, dtype=np.float32)


ACTION_LABELS_ORDERED = ("PROCESS", "RETURN", "REPAIR")


def _empty_counts() -> dict[str, dict[tuple[Any, ...], Counter[str]]]:
    return {level: {} for level in _HISTORY_LEVELS}


def _update_counts(
    counts: dict[str, dict[tuple[Any, ...], Counter[str]]],
    global_counts: Counter[str],
    row: FeatureRow,
    label: str,
) -> None:
    global_counts[label] += 1
    for level in _HISTORY_LEVELS:
        key = _history_key(row, level)
        bucket = counts[level].setdefault(key, Counter())
        bucket[label] += 1


def _history_key(row: FeatureRow, level: str) -> tuple[Any, ...]:
    type_value = _normalized(row.features.get("type"))
    detail = _normalized(row.features.get("detail"))
    counts = (
        _count_token(row.features.get("intake_count")),
        _count_token(row.features.get("service_count")),
        _count_token(row.features.get("treatment_count")),
    )
    if level == "product":
        return (row.product_key,)
    if level == "product_type":
        return row.product_key, type_value
    if level == "product_detail_counts":
        return row.product_key, detail, *counts
    if level == "detail_counts":
        return detail, *counts
    raise ValueError(f"unknown history level: {level}")


def _global_rates(global_counts: Counter[str]) -> dict[str, float]:
    total = sum(global_counts.values())
    if not total:
        return {label: 1 / len(ACTION_LABELS_ORDERED) for label in ACTION_LABELS_ORDERED}
    return {label: global_counts[label] / total for label in ACTION_LABELS_ORDERED}


def _smoothed_rates(
    counts: Counter[str],
    support: int,
    global_rates: Mapping[str, float],
    smoothing: float,
) -> dict[str, float]:
    if support == 0:
        return dict(global_rates)
    denominator = support + smoothing
    if denominator == 0:
        return {label: counts[label] / support for label in ACTION_LABELS_ORDERED}
    return {
        label: (counts[label] + smoothing * global_rates[label]) / denominator
        for label in ACTION_LABELS_ORDERED
    }


def _validate_rows_and_labels(rows: Sequence[FeatureRow], labels: Mapping[str, str]) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.row_id in seen:
            raise ValueError(f"duplicate V4 domain feature row: {row.row_id}")
        seen.add(row.row_id)
        label = labels.get(row.row_id)
        if label not in ACTION_LABELS:
            raise ValueError(f"missing or invalid label for V4 domain feature row: {row.row_id}")


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _count_token(value: Any) -> str:
    parsed = _number(value)
    return "__missing__" if parsed is None else f"{parsed:g}"


def _normalized(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def _matrix(vectors: Sequence[np.ndarray | None], width: int) -> sparse.csr_matrix:
    if not vectors:
        return sparse.csr_matrix((0, width), dtype=np.float32)
    if any(vector is None for vector in vectors):
        raise RuntimeError("V4 domain feature vector was not materialized")
    return sparse.csr_matrix(np.vstack(vectors), dtype=np.float32)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

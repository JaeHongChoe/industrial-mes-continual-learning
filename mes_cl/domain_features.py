"""Point-in-time domain features (numeric / text / smoothed label history).

- 수치: 입고·처리 횟수, 사용 시간과 그 차이·비율, 결측 플래그
- 텍스트: 상세내용 길이·항목 수, 의미 패턴 매칭 플래그
- 이력: product / product×type / product×detail×counts / detail×counts 수준의 라벨 비율(전역 사전분포로 smoothing)

`fit_transform_oof` 는 날짜 순 재생으로 out-of-fold 피처를 만들고(같은 날은 score 후 label update),
`fit_history` + `transform_frozen` 은 cutoff 이전 이력만으로 동결된 추론용 피처를 만든다.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from scipy import sparse

from .labels import LabelAvailabilityIndex


ACTION_LABELS_ORDERED = ("PROCESS", "RETURN", "REPAIR")
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


class DomainRow(Protocol):
    row_id: str
    source_row: int
    date: str
    month: str
    product_key: str
    features: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PointInTimeDomainFeatureSpec:
    smoothing: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.smoothing, bool) or not isinstance(
            self.smoothing, (int, float)
        ):
            raise ValueError("smoothing must be a non-negative finite number")
        if not math.isfinite(float(self.smoothing)) or float(self.smoothing) < 0:
            raise ValueError("smoothing must be a non-negative finite number")


class PointInTimeDomainFeatureTransformer:
    def __init__(
        self,
        spec: PointInTimeDomainFeatureSpec | None = None,
    ) -> None:
        self.spec = spec or PointInTimeDomainFeatureSpec()
        self._feature_names = _feature_names()
        self._counts = _empty_counts()
        self._global: Counter[str] = Counter()
        self._cutoff_exclusive: date | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def schema_sha256(self) -> str:
        payload = {
            "schemaVersion": 1,
            "kind": "point_in_time_domain_features",
            "smoothing": float(self.spec.smoothing),
            "featureNames": self.feature_names,
            "historyPolicy": "label_available_date_strictly_before_prediction_date",
            "sameDayOrder": "score_then_label_update",
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    def fit_transform_oof(
        self,
        rows: Sequence[DomainRow],
        *,
        all_observed_rows: Sequence[DomainRow],
        label_index: LabelAvailabilityIndex,
    ) -> sparse.csr_matrix:
        targets = _validated_rows(rows, "target", preserve_order=True)
        observed = _validated_rows(
            all_observed_rows,
            "observed",
            preserve_order=False,
        )
        observed_by_id = {row.row_id: row for row in observed}
        _validate_universes(targets, observed_by_id, label_index)
        if not targets:
            return sparse.csr_matrix((0, len(self.feature_names)), dtype=np.float32)

        max_target_date = max(date.fromisoformat(row.date) for row in targets)
        targets_by_date: dict[date, list[DomainRow]] = {}
        for row in targets:
            targets_by_date.setdefault(date.fromisoformat(row.date), []).append(row)
        labels_by_date: dict[date, list[tuple[DomainRow, str]]] = {}
        for record in label_index.valid_records:
            if record.label_available_date > max_target_date:
                continue
            labels_by_date.setdefault(record.label_available_date, []).append(
                (observed_by_id[record.row_id], record.label)
            )

        counts = _empty_counts()
        global_counts: Counter[str] = Counter()
        vectors: dict[str, np.ndarray] = {}
        for event_date in sorted(set(targets_by_date) | set(labels_by_date)):
            for row in sorted(
                targets_by_date.get(event_date, ()),
                key=lambda item: (item.source_row, item.row_id),
            ):
                vectors[row.row_id] = _row_vector(
                    row,
                    counts,
                    global_counts,
                    smoothing=float(self.spec.smoothing),
                )
            for row, label in sorted(
                labels_by_date.get(event_date, ()),
                key=lambda item: (item[0].source_row, item[0].row_id),
            ):
                _update_counts(counts, global_counts, row, label)

        self._counts = counts
        self._global = global_counts
        self._cutoff_exclusive = max_target_date + date.resolution
        return _matrix(
            [vectors[row.row_id] for row in targets],
            len(self.feature_names),
        )

    def fit_history(
        self,
        *,
        all_observed_rows: Sequence[DomainRow],
        label_index: LabelAvailabilityIndex,
        cutoff_exclusive: date | str,
    ) -> "PointInTimeDomainFeatureTransformer":
        observed = _validated_rows(
            all_observed_rows,
            "observed",
            preserve_order=False,
        )
        observed_by_id = {row.row_id: row for row in observed}
        _validate_universes((), observed_by_id, label_index)
        cutoff = _date_value(cutoff_exclusive)
        counts = _empty_counts()
        global_counts: Counter[str] = Counter()
        for record in label_index.valid_records:
            if record.label_available_date >= cutoff:
                continue
            _update_counts(
                counts,
                global_counts,
                observed_by_id[record.row_id],
                record.label,
            )
        self._counts = counts
        self._global = global_counts
        self._cutoff_exclusive = cutoff
        return self

    def transform_frozen(
        self,
        rows: Sequence[DomainRow],
    ) -> sparse.csr_matrix:
        if self._cutoff_exclusive is None:
            raise RuntimeError("point-in-time domain history is not fitted")
        targets = _validated_rows(rows, "target", preserve_order=True)
        if any(
            date.fromisoformat(row.date) < self._cutoff_exclusive
            for row in targets
        ):
            raise ValueError("frozen rows must be on or after the history cutoff")
        return _matrix(
            [
                _row_vector(
                    row,
                    self._counts,
                    self._global,
                    smoothing=float(self.spec.smoothing),
                )
                for row in targets
            ],
            len(self.feature_names),
        )


def _validate_universes(
    targets: Sequence[DomainRow],
    observed_by_id: Mapping[str, DomainRow],
    label_index: LabelAvailabilityIndex,
) -> None:
    missing_targets = {row.row_id for row in targets} - set(observed_by_id)
    if missing_targets:
        raise ValueError(
            f"target rows are absent from observed universe: {len(missing_targets)}"
        )
    missing_labels = {
        record.row_id for record in label_index.valid_records
    } - set(observed_by_id)
    if missing_labels:
        raise ValueError(
            "label availability rows are absent from observed universe: "
            f"{len(missing_labels)}"
        )


def _validated_rows(
    rows: Sequence[DomainRow],
    name: str,
    *,
    preserve_order: bool,
) -> tuple[DomainRow, ...]:
    materialized = tuple(rows)
    row_ids = [row.row_id for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"duplicate {name} row identity")
    for row in materialized:
        date.fromisoformat(row.date)
    if preserve_order:
        return materialized
    return tuple(
        sorted(materialized, key=lambda row: (row.date, row.source_row, row.row_id))
    )


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
    row: DomainRow,
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
    values.extend(
        1.0 if pattern.search(semantic_text) else 0.0
        for pattern in _SEMANTIC_PATTERNS.values()
    )
    product_support = sum(
        counts["product"].get(_history_key(row, "product"), Counter()).values()
    )
    values.append(1.0 if product_support == 0 else 0.0)
    global_rates = _global_rates(global_counts)
    for level in _HISTORY_LEVELS:
        observed = counts[level].get(_history_key(row, level), Counter())
        support = sum(observed.values())
        values.append(math.log1p(support))
        rates = _smoothed_rates(observed, support, global_rates, smoothing)
        values.extend(rates[label] for label in ACTION_LABELS_ORDERED)
    return np.asarray(values, dtype=np.float32)


def _empty_counts() -> dict[str, dict[tuple[Any, ...], Counter[str]]]:
    return {level: {} for level in _HISTORY_LEVELS}


def _update_counts(
    counts: dict[str, dict[tuple[Any, ...], Counter[str]]],
    global_counts: Counter[str],
    row: DomainRow,
    label: str,
) -> None:
    global_counts[label] += 1
    for level in _HISTORY_LEVELS:
        key = _history_key(row, level)
        counts[level].setdefault(key, Counter())[label] += 1


def _history_key(row: DomainRow, level: str) -> tuple[Any, ...]:
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
        return {
            label: 1 / len(ACTION_LABELS_ORDERED)
            for label in ACTION_LABELS_ORDERED
        }
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


def _matrix(vectors: Sequence[np.ndarray], width: int) -> sparse.csr_matrix:
    if not vectors:
        return sparse.csr_matrix((0, width), dtype=np.float32)
    return sparse.csr_matrix(np.vstack(vectors), dtype=np.float32)


def _date_value(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid cutoff date: {value!r}") from error


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

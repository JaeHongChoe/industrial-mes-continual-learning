"""Point-in-time temporal history features.

제품 / 고객사×제품 / 제품×상세내용 / 라인×제품 범위별로 과거 라벨 분포(전체, 30/90/180일 창)와
최근-전체 위험률 차이, 동일 serial 재입고 간격을 계산한다.
날짜 이벤트를 순서대로 재생하면서 **같은 날은 대상 행 snapshot → 발생 반영 → 라벨 반영** 순으로 처리해
당일·미래 라벨이 피처에 섞이지 않게 한다.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .labels import LabelAvailabilityIndex


LABELS = ("PROCESS", "RETURN", "REPAIR")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
WINDOW_DAYS = (30, 90, 180)
SCOPE_NAMES = ("product", "customer_product", "product_detail", "line_product")


class TemporalRow(Protocol):
    row_id: str
    source_row: int
    date: str
    customer_key: str
    product_key: str
    features: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PointInTimeTemporalMatrix:
    row_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    schema_sha256: str
    _index: Mapping[str, int] = field(repr=False, compare=False)

    def features_for(self, row_id: str) -> Mapping[str, float]:
        if row_id not in self._index:
            raise KeyError(row_id)
        values = self.values[self._index[row_id]]
        return MappingProxyType(
            {
                name: float(values[index])
                for index, name in enumerate(self.feature_names)
            }
        )

    def select(self, row_ids: Sequence[str]) -> np.ndarray:
        indices = [self._index[row_id] for row_id in row_ids]
        return self.values[np.asarray(indices, dtype=np.int64)]


@dataclass(slots=True)
class _ScopeState:
    lifetime: list[int] = field(default_factory=lambda: [0, 0, 0])
    recent: deque[tuple[int, int]] = field(default_factory=deque)

    def snapshot(self, current_ordinal: int) -> tuple[float, ...]:
        minimum = current_ordinal - max(WINDOW_DAYS)
        while self.recent and self.recent[0][0] < minimum:
            self.recent.popleft()
        lifetime_rates = _rates(self.lifetime)
        values: list[float] = [float(sum(self.lifetime)), *lifetime_rates]
        window_rates: dict[int, tuple[float, float, float]] = {}
        for days in WINDOW_DAYS:
            counts = [0, 0, 0]
            cutoff = current_ordinal - days
            for observed_ordinal, label_index in self.recent:
                if observed_ordinal >= cutoff:
                    counts[label_index] += 1
            rates = _rates(counts)
            window_rates[days] = rates
            values.extend((float(sum(counts)), *rates))
        lifetime_risk = sum(lifetime_rates[1:])
        values.append(sum(window_rates[30][1:]) - lifetime_risk)
        values.append(sum(window_rates[90][1:]) - lifetime_risk)
        return tuple(values)

    def update(self, available_ordinal: int, label: str) -> None:
        label_index = LABEL_INDEX[label]
        self.lifetime[label_index] += 1
        self.recent.append((available_ordinal, label_index))


@dataclass(slots=True)
class _SerialState:
    count: int = 0
    last_ordinal: int | None = None
    interval_sum: int = 0
    interval_count: int = 0

    def snapshot(self, current_ordinal: int) -> tuple[float, float, float]:
        days_since_last = (
            current_ordinal - self.last_ordinal
            if self.last_ordinal is not None
            else 0
        )
        mean_interval = (
            self.interval_sum / self.interval_count
            if self.interval_count
            else 0.0
        )
        return float(self.count), float(days_since_last), float(mean_interval)

    def update(self, occurred_ordinal: int) -> None:
        if self.last_ordinal is not None:
            self.interval_sum += occurred_ordinal - self.last_ordinal
            self.interval_count += 1
        self.count += 1
        self.last_ordinal = occurred_ordinal


def build_point_in_time_temporal_matrix(
    target_rows: Sequence[TemporalRow],
    *,
    all_observed_rows: Sequence[TemporalRow],
    label_index: LabelAvailabilityIndex,
    label_cutoff_exclusive: date | str | None = None,
) -> PointInTimeTemporalMatrix:
    targets = _validated_rows(target_rows, "target")
    observed = _validated_rows(all_observed_rows, "observed")
    observed_by_id = {row.row_id: row for row in observed}
    missing_targets = {row.row_id for row in targets} - set(observed_by_id)
    if missing_targets:
        raise ValueError(
            f"target rows are absent from observed universe: {len(missing_targets)}"
        )
    missing_label_rows = {
        record.row_id for record in label_index.valid_records
    } - set(observed_by_id)
    if missing_label_rows:
        raise ValueError(
            "label availability rows are absent from observed universe: "
            f"{len(missing_label_rows)}"
        )
    if not targets:
        return _matrix((), np.empty((0, len(_feature_names())), dtype=np.float32))

    max_target_date = date.fromisoformat(targets[-1].date)
    targets_by_date = _group_rows_by_date(targets, maximum=max_target_date)
    occurrences_by_date = _group_rows_by_date(observed, maximum=max_target_date)
    label_cutoff = (
        _date_value(label_cutoff_exclusive)
        if label_cutoff_exclusive is not None
        else None
    )
    labels_by_date: dict[date, list[tuple[TemporalRow, str]]] = {}
    for record in label_index.valid_records:
        if record.label_available_date > max_target_date:
            continue
        if label_cutoff is not None and record.label_available_date >= label_cutoff:
            continue
        labels_by_date.setdefault(record.label_available_date, []).append(
            (observed_by_id[record.row_id], record.label)
        )

    scope_states: dict[str, dict[tuple[str, ...], _ScopeState]] = {
        name: {} for name in SCOPE_NAMES
    }
    serial_states: dict[str, _SerialState] = {}
    values_by_id: dict[str, tuple[float, ...]] = {}
    event_dates = sorted(
        set(targets_by_date) | set(occurrences_by_date) | set(labels_by_date)
    )
    for event_date in event_dates:
        ordinal = event_date.toordinal()
        for row in targets_by_date.get(event_date, ()):
            values_by_id[row.row_id] = _snapshot_row(
                row,
                ordinal,
                scope_states,
                serial_states,
            )
        for row in occurrences_by_date.get(event_date, ()):
            serial = _normalized(row.features.get("serial"))
            if serial:
                serial_states.setdefault(serial, _SerialState()).update(ordinal)
        for row, label in sorted(
            labels_by_date.get(event_date, ()),
            key=lambda item: (item[0].source_row, item[0].row_id),
        ):
            for scope_name, scope_key in _scope_keys(row).items():
                if scope_key is None:
                    continue
                state = scope_states[scope_name].setdefault(scope_key, _ScopeState())
                state.update(ordinal, label)

    row_ids = tuple(row.row_id for row in targets)
    values = np.asarray([values_by_id[row_id] for row_id in row_ids], dtype=np.float32)
    return _matrix(row_ids, values)


def _validated_rows(
    rows: Sequence[TemporalRow],
    name: str,
) -> tuple[TemporalRow, ...]:
    materialized = tuple(
        sorted(rows, key=lambda row: (row.date, row.source_row, row.row_id))
    )
    row_ids = [row.row_id for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"duplicate {name} row identity")
    for row in materialized:
        date.fromisoformat(row.date)
    return materialized


def _group_rows_by_date(
    rows: Sequence[TemporalRow],
    *,
    maximum: date,
) -> dict[date, tuple[TemporalRow, ...]]:
    grouped: dict[date, list[TemporalRow]] = {}
    for row in rows:
        observed_date = date.fromisoformat(row.date)
        if observed_date <= maximum:
            grouped.setdefault(observed_date, []).append(row)
    return {
        key: tuple(sorted(values, key=lambda row: (row.source_row, row.row_id)))
        for key, values in grouped.items()
    }


def _snapshot_row(
    row: TemporalRow,
    current_ordinal: int,
    scope_states: Mapping[str, Mapping[tuple[str, ...], _ScopeState]],
    serial_states: Mapping[str, _SerialState],
) -> tuple[float, ...]:
    values: list[float] = []
    for scope_name, scope_key in _scope_keys(row).items():
        state = scope_states[scope_name].get(scope_key) if scope_key else None
        values.extend(state.snapshot(current_ordinal) if state else _empty_scope_values())
    serial = _normalized(row.features.get("serial"))
    serial_state = serial_states.get(serial) if serial else None
    values.extend(
        serial_state.snapshot(current_ordinal)
        if serial_state
        else (0.0, 0.0, 0.0)
    )
    return tuple(values)


def _scope_keys(row: TemporalRow) -> Mapping[str, tuple[str, ...] | None]:
    product = _normalized(row.product_key)
    customer = _normalized(row.customer_key)
    detail = _normalized(row.features.get("detail"))
    line = _normalized(row.features.get("line"))
    return {
        "product": (product,) if product else None,
        "customer_product": (customer, product) if customer and product else None,
        "product_detail": (product, detail) if product and detail else None,
        "line_product": (customer, line, product) if customer and line and product else None,
    }


def _feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for scope in SCOPE_NAMES:
        names.extend(
            (
                f"{scope}__lifetime_count",
                f"{scope}__lifetime_process_rate",
                f"{scope}__lifetime_return_rate",
                f"{scope}__lifetime_repair_rate",
            )
        )
        for days in WINDOW_DAYS:
            names.extend(
                (
                    f"{scope}__{days}d_count",
                    f"{scope}__{days}d_process_rate",
                    f"{scope}__{days}d_return_rate",
                    f"{scope}__{days}d_repair_rate",
                )
            )
        names.extend(
            (
                f"{scope}__risk_rate_delta_30_lifetime",
                f"{scope}__risk_rate_delta_90_lifetime",
            )
        )
    names.extend(("serial_prior_count", "serial_days_since_last", "serial_mean_interval_days"))
    return tuple(names)


def _empty_scope_values() -> tuple[float, ...]:
    return (0.0,) * 18


def _rates(counts: Sequence[int]) -> tuple[float, float, float]:
    total = sum(counts)
    if not total:
        return 0.0, 0.0, 0.0
    return tuple(float(value) / total for value in counts)  # type: ignore[return-value]


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(character for character in text if character.isalnum())


def _date_value(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid label cutoff: {value!r}") from error


def _matrix(
    row_ids: tuple[str, ...],
    values: np.ndarray,
) -> PointInTimeTemporalMatrix:
    feature_names = _feature_names()
    if values.shape != (len(row_ids), len(feature_names)):
        raise ValueError(
            "point-in-time temporal matrix shape mismatch: "
            f"expected={(len(row_ids), len(feature_names))} observed={values.shape}"
        )
    values.setflags(write=False)
    schema_payload = {
        "schemaVersion": 1,
        "featureNames": feature_names,
        "scopeLabelEvent": "label_available_date_strictly_before_prediction_date",
        "serialEvent": "occurrence_date_strictly_before_prediction_date",
        "sameDayOrder": "score_then_occurrence_and_label_updates",
    }
    schema_sha256 = hashlib.sha256(
        json.dumps(
            schema_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PointInTimeTemporalMatrix(
        row_ids=row_ids,
        feature_names=feature_names,
        values=values,
        schema_sha256=schema_sha256,
        _index=MappingProxyType(
            {row_id: index for index, row_id in enumerate(row_ids)}
        ),
    )

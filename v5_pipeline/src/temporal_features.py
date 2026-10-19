from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .data_contract import ACTION_LABELS, FeatureRow


LABELS = ("PROCESS", "RETURN", "REPAIR")
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
WINDOW_DAYS = (30, 90, 180)
SCOPE_NAMES = ("product", "customer_product", "product_detail", "line_product")


@dataclass(frozen=True, slots=True)
class TemporalFeatureMatrix:
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
            {name: float(values[index]) for index, name in enumerate(self.feature_names)}
        )

    def select(self, row_ids: Sequence[str]) -> np.ndarray:
        indices = [self._index[row_id] for row_id in row_ids]
        return self.values[np.asarray(indices, dtype=np.int64)]


@dataclass(slots=True)
class _ScopeState:
    lifetime: list[int] = field(default_factory=lambda: [0, 0, 0])
    recent: deque[tuple[int, int]] = field(default_factory=deque)

    def snapshot(self, current_ordinal: int, *, prune: bool = True) -> tuple[float, ...]:
        minimum = current_ordinal - max(WINDOW_DAYS)
        if prune:
            while self.recent and self.recent[0][0] < minimum:
                self.recent.popleft()

        lifetime_count = sum(self.lifetime)
        values: list[float] = [float(lifetime_count), *_rates(self.lifetime)]
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
        lifetime_risk = sum(_rates(self.lifetime)[1:])
        values.append(sum(window_rates[30][1:]) - lifetime_risk)
        values.append(sum(window_rates[90][1:]) - lifetime_risk)
        return tuple(values)

    def update(self, observed_ordinal: int, label_index: int) -> None:
        self.lifetime[label_index] += 1
        self.recent.append((observed_ordinal, label_index))


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

    def update(self, observed_ordinal: int) -> None:
        if self.last_ordinal is not None:
            self.interval_sum += observed_ordinal - self.last_ordinal
            self.interval_count += 1
        self.count += 1
        self.last_ordinal = observed_ordinal


class TemporalFeatureBuilder:
    def __init__(self) -> None:
        self._scopes: dict[str, dict[tuple[str, ...], _ScopeState]] = {
            name: {} for name in SCOPE_NAMES
        }
        self._serials: dict[str, _SerialState] = {}
        self._last_updated_date = ""
        self._pending_date = ""
        self._pending_ids: tuple[str, ...] = ()

    def prequential_transform(
        self,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
    ) -> TemporalFeatureMatrix:
        materialized = _validated_rows(rows)
        if set(labels) != {row.row_id for row in materialized}:
            raise ValueError("temporal labels must match the exact row universe")

        row_ids: list[str] = []
        matrices: list[np.ndarray] = []
        for _, grouped in groupby(materialized, key=lambda row: row.date):
            batch = tuple(grouped)
            transformed = self.transform_date(batch)
            row_ids.extend(transformed.row_ids)
            matrices.append(transformed.values)
            self.update(batch, labels)
        values = (
            np.vstack(matrices).astype(np.float32, copy=False)
            if matrices
            else np.empty((0, len(_feature_names())), dtype=np.float32)
        )
        return _matrix(tuple(row_ids), values)

    def fit_prior(
        self,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
    ) -> "TemporalFeatureBuilder":
        materialized = _validated_rows(rows)
        if set(labels) != {row.row_id for row in materialized}:
            raise ValueError("temporal labels must match the exact row universe")
        # Prior rows only seed state. Keep at most one date's snapshot in memory
        # instead of allocating an otherwise discarded full history matrix.
        for _, grouped in groupby(materialized, key=lambda row: row.date):
            batch = tuple(grouped)
            self.transform_date(batch)
            self.update(batch, labels)
        return self

    def transform_date(self, rows: Sequence[FeatureRow]) -> TemporalFeatureMatrix:
        materialized = _validated_rows(rows)
        if not materialized:
            return _matrix((), np.empty((0, len(_feature_names())), dtype=np.float32))
        observed_dates = {row.date for row in materialized}
        if len(observed_dates) != 1:
            raise ValueError("transform_date requires exactly one observed date")
        observed_date = materialized[0].date
        if self._pending_ids:
            raise RuntimeError("temporal batch must be updated before scoring another date")
        if self._last_updated_date and observed_date <= self._last_updated_date:
            raise ValueError("temporal scoring dates must be strictly increasing")

        observed_ordinal = date.fromisoformat(observed_date).toordinal()
        values = np.asarray(
            [self._snapshot_row(row, observed_ordinal) for row in materialized],
            dtype=np.float32,
        )
        self._pending_date = observed_date
        self._pending_ids = tuple(row.row_id for row in materialized)
        return _matrix(self._pending_ids, values)

    def transform_frozen(self, rows: Sequence[FeatureRow]) -> TemporalFeatureMatrix:
        """Transform future rows without learning from any row in the batch.

        This is the inference path. It intentionally does not create a pending
        update and does not prune the builder's historical window state, so a
        caller cannot accidentally make later predictions depend on earlier
        future labels or mutate the frozen 2025-12-31 snapshot.
        """

        materialized = _validated_rows(rows)
        if not materialized:
            return _matrix((), np.empty((0, len(_feature_names())), dtype=np.float32))
        if self._pending_ids:
            raise RuntimeError("temporal pending batch must be updated before frozen inference")
        if self._last_updated_date and materialized[0].date <= self._last_updated_date:
            raise ValueError("frozen temporal rows must be strictly after fitted history")
        values = np.asarray(
            [
                self._snapshot_row(
                    row,
                    date.fromisoformat(row.date).toordinal(),
                    prune_recent=False,
                )
                for row in materialized
            ],
            dtype=np.float32,
        )
        return _matrix(tuple(row.row_id for row in materialized), values)

    def update(
        self,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
    ) -> None:
        materialized = _validated_rows(rows)
        row_ids = tuple(row.row_id for row in materialized)
        if row_ids != self._pending_ids:
            raise ValueError("temporal update rows do not match the pending scored batch")
        if set(row_ids) - set(labels):
            raise ValueError("temporal update labels do not cover the pending batch")
        observed_ordinal = date.fromisoformat(self._pending_date).toordinal()
        for row in materialized:
            label = str(labels[row.row_id])
            if label not in ACTION_LABELS:
                continue
            label_index = LABEL_INDEX[label]
            for scope_name, scope_key in _scope_keys(row).items():
                if scope_key is None:
                    continue
                state = self._scopes[scope_name].setdefault(scope_key, _ScopeState())
                state.update(observed_ordinal, label_index)
            serial = _normalized(row.features.get("serial"))
            if serial:
                self._serials.setdefault(serial, _SerialState()).update(observed_ordinal)
        self._last_updated_date = self._pending_date
        self._pending_date = ""
        self._pending_ids = ()

    def _snapshot_row(
        self,
        row: FeatureRow,
        current_ordinal: int,
        *,
        prune_recent: bool = True,
    ) -> tuple[float, ...]:
        values: list[float] = []
        for scope_name, scope_key in _scope_keys(row).items():
            state = self._scopes[scope_name].get(scope_key) if scope_key else None
            values.extend(
                state.snapshot(current_ordinal, prune=prune_recent)
                if state
                else _empty_scope_values()
            )
        serial = _normalized(row.features.get("serial"))
        serial_state = self._serials.get(serial) if serial else None
        values.extend(serial_state.snapshot(current_ordinal) if serial_state else (0.0, 0.0, 0.0))
        return tuple(values)


def _validated_rows(rows: Sequence[FeatureRow]) -> tuple[FeatureRow, ...]:
    materialized = tuple(sorted(rows, key=lambda row: (row.date, row.source_row, row.row_id)))
    row_ids = [row.row_id for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("temporal rows contain a duplicate row identity")
    for row in materialized:
        date.fromisoformat(row.date)
    return materialized


def _scope_keys(row: FeatureRow) -> Mapping[str, tuple[str, ...] | None]:
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


def _matrix(row_ids: tuple[str, ...], values: np.ndarray) -> TemporalFeatureMatrix:
    feature_names = _feature_names()
    if values.shape != (len(row_ids), len(feature_names)):
        raise ValueError(
            "temporal matrix shape mismatch: "
            f"expected={(len(row_ids), len(feature_names))} observed={values.shape}"
        )
    values.setflags(write=False)
    schema_bytes = json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return TemporalFeatureMatrix(
        row_ids=row_ids,
        feature_names=feature_names,
        values=values,
        schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        _index=MappingProxyType({row_id: index for index, row_id in enumerate(row_ids)}),
    )

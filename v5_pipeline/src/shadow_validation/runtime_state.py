from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

from experiment_lib.point_in_time_domain_features import (
    PointInTimeDomainFeatureTransformer,
)
from experiment_lib.point_in_time_labels import (
    LabelAvailabilityIndex,
    LabelAvailabilityRecord,
)
from src.data_contract import FeatureRow

from .cache_io import CacheFingerprint, load_cache, store_cache
from .feature_bridge import load_frozen_history, load_label_availability_index


CACHE_KIND = "predictor-history"
RUNTIME_PACKAGES = (
    "lightgbm",
    "numpy",
    "openpyxl",
    "scikit-learn",
    "scipy",
)


@dataclass(frozen=True, slots=True)
class FrozenHistoryState:
    history_rows: tuple[FeatureRow, ...]
    history_labels: Mapping[str, str]
    label_index: LabelAvailabilityIndex
    pit_domain: PointInTimeDomainFeatureTransformer


def current_cache_fingerprint(
    *,
    generation_id: str,
    manifest_sha256: str,
) -> CacheFingerprint:
    runtime_versions: list[tuple[str, str]] = []
    for package in RUNTIME_PACKAGES:
        try:
            observed = version(package)
        except PackageNotFoundError:
            observed = "missing"
        runtime_versions.append((package, observed))
    return CacheFingerprint(
        schema_version=1,
        generation_id=str(generation_id),
        manifest_sha256=str(manifest_sha256),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        runtime_versions=tuple(runtime_versions),
    )


def runtime_cache_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def _feature_row_record(row: FeatureRow) -> tuple[Any, ...]:
    return (
        row.row_id,
        int(row.source_row),
        row.date,
        row.month,
        row.raw_customer,
        row.customer_key,
        row.legacy_customer_key,
        row.product_key,
        row.episode_id,
        row.episode_basis,
        dict(row.features),
    )


def _feature_row_from_record(record: Any) -> FeatureRow:
    if not isinstance(record, tuple) or len(record) != 11:
        raise ValueError("invalid cached FeatureRow record")
    features = record[10]
    if not isinstance(features, dict):
        raise ValueError("invalid cached FeatureRow features")
    return FeatureRow(
        row_id=str(record[0]),
        source_row=int(record[1]),
        date=str(record[2]),
        month=str(record[3]),
        raw_customer=str(record[4]),
        customer_key=str(record[5]),
        legacy_customer_key=str(record[6]),
        product_key=str(record[7]),
        episode_id=str(record[8]),
        episode_basis=str(record[9]),
        features=MappingProxyType(dict(features)),
    )


def _label_index_record(index: LabelAvailabilityIndex) -> dict[str, Any]:
    return {
        "sourceDatasetId": index.source_dataset_id,
        "sourceRows": int(index.source_rows),
        "actionRows": int(index.action_rows),
        "validRecords": tuple(
            (
                record.row_id,
                record.occurrence_date.isoformat(),
                record.label_available_date.isoformat(),
                record.label,
            )
            for record in index.valid_records
        ),
        "staticExcludedRows": dict(index.static_excluded_rows),
        "staticExcludedRowIds": {
            str(reason): tuple(str(value) for value in row_ids)
            for reason, row_ids in index.static_excluded_row_ids.items()
        },
    }


def _label_index_from_record(record: Any) -> LabelAvailabilityIndex:
    if not isinstance(record, dict):
        raise ValueError("invalid cached label index")
    valid_records = tuple(
        LabelAvailabilityRecord(
            row_id=str(value[0]),
            occurrence_date=date.fromisoformat(str(value[1])),
            label_available_date=date.fromisoformat(str(value[2])),
            label=str(value[3]),
        )
        for value in record["validRecords"]
    )
    excluded_rows = {
        str(reason): int(count)
        for reason, count in dict(record["staticExcludedRows"]).items()
    }
    excluded_ids = {
        str(reason): tuple(str(value) for value in row_ids)
        for reason, row_ids in dict(record["staticExcludedRowIds"]).items()
    }
    return LabelAvailabilityIndex(
        source_dataset_id=str(record["sourceDatasetId"]),
        source_rows=int(record["sourceRows"]),
        action_rows=int(record["actionRows"]),
        valid_records=valid_records,
        static_excluded_rows=MappingProxyType(excluded_rows),
        static_excluded_row_ids=MappingProxyType(excluded_ids),
    )


def _state_record(
    state: FrozenHistoryState,
    *,
    source_dataset_id: str,
    cutoff_exclusive: str,
) -> dict[str, Any]:
    return {
        "sourceDatasetId": source_dataset_id,
        "cutoffExclusive": cutoff_exclusive,
        "historyRows": tuple(
            _feature_row_record(row) for row in state.history_rows
        ),
        "historyLabels": dict(state.history_labels),
        "labelIndex": _label_index_record(state.label_index),
        "pitDomain": state.pit_domain,
        "pitDomainSchemaSha256": state.pit_domain.schema_sha256,
    }


def _state_from_record(
    record: Any,
    *,
    source_dataset_id: str,
    cutoff_exclusive: str,
) -> FrozenHistoryState:
    if not isinstance(record, dict):
        raise ValueError("invalid cached predictor history")
    if str(record.get("sourceDatasetId") or "") != source_dataset_id:
        raise ValueError("cached source dataset does not match")
    if str(record.get("cutoffExclusive") or "") != cutoff_exclusive:
        raise ValueError("cached history cutoff does not match")

    rows = tuple(
        _feature_row_from_record(value)
        for value in record["historyRows"]
    )
    row_ids = {row.row_id for row in rows}
    if len(row_ids) != len(rows):
        raise ValueError("cached predictor history contains duplicate row IDs")

    labels = {
        str(row_id): str(label)
        for row_id, label in dict(record["historyLabels"]).items()
    }
    if set(labels) != row_ids:
        raise ValueError("cached predictor history label universe mismatch")

    label_index = _label_index_from_record(record["labelIndex"])
    if label_index.source_dataset_id != source_dataset_id:
        raise ValueError("cached label index dataset does not match")
    if {item.row_id for item in label_index.valid_records} - row_ids:
        raise ValueError("cached label index row universe mismatch")

    pit_domain = record["pitDomain"]
    if not isinstance(pit_domain, PointInTimeDomainFeatureTransformer):
        raise ValueError("cached PIT domain has an invalid type")
    if pit_domain.schema_sha256 != str(record["pitDomainSchemaSha256"]):
        raise ValueError("cached PIT domain schema mismatch")

    return FrozenHistoryState(
        history_rows=rows,
        history_labels=MappingProxyType(labels),
        label_index=label_index,
        pit_domain=pit_domain,
    )


def _build_frozen_history_state(
    *,
    history_path: Path,
    source_dataset_id: str,
    cutoff_exclusive: str,
) -> FrozenHistoryState:
    history_rows, history_labels = load_frozen_history(
        history_path,
        source_dataset_id=source_dataset_id,
    )
    label_index = load_label_availability_index(
        history_path,
        source_dataset_id=source_dataset_id,
    )
    pit_domain = PointInTimeDomainFeatureTransformer().fit_history(
        all_observed_rows=history_rows,
        label_index=label_index,
        cutoff_exclusive=cutoff_exclusive,
    )
    return FrozenHistoryState(
        history_rows=history_rows,
        history_labels=history_labels,
        label_index=label_index,
        pit_domain=pit_domain,
    )


def load_or_build_frozen_history_state(
    *,
    history_path: Path,
    source_dataset_id: str,
    cutoff_exclusive: str,
    cache_path: Path,
    fingerprint: CacheFingerprint,
) -> tuple[FrozenHistoryState, str]:
    existed = cache_path.is_file()
    cached = load_cache(
        cache_path,
        fingerprint,
        expected_kind=CACHE_KIND,
    )
    if cached is not None:
        try:
            return (
                _state_from_record(
                    cached,
                    source_dataset_id=source_dataset_id,
                    cutoff_exclusive=cutoff_exclusive,
                ),
                "hit",
            )
        except (KeyError, TypeError, ValueError):
            pass

    state = _build_frozen_history_state(
        history_path=history_path,
        source_dataset_id=source_dataset_id,
        cutoff_exclusive=cutoff_exclusive,
    )
    store_cache(
        cache_path,
        fingerprint,
        kind=CACHE_KIND,
        payload=_state_record(
            state,
            source_dataset_id=source_dataset_id,
            cutoff_exclusive=cutoff_exclusive,
        ),
    )
    return state, "rebuilt" if existed else "miss"

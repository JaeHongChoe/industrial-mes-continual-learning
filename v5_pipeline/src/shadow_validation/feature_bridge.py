from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from experiment_lib.point_in_time_labels import LabelAvailabilityIndex

from src.data_contract import (
    ACTION_LABELS,
    ALLOWED_RAW_FEATURES,
    FeatureRow,
    _episode_identity,
    _to_feature_row,
    canonical_customer_key,
    legacy_policy_customer_key,
    _parse_date_value,
)

from .input_contract import InputRow


def feature_rows_from_input(rows: Sequence[InputRow]) -> tuple[FeatureRow, ...]:
    """Convert sealed intake rows without exposing their raw post-decision columns."""
    output: list[FeatureRow] = []
    for row in rows:
        payload = dict(row.feature_payload)
        customer_key = canonical_customer_key(payload)
        legacy_key = legacy_policy_customer_key(payload)
        episode_id, episode_basis = _episode_identity(
            payload,
            customer_key=customer_key,
            product_key=row.product_key,
            observed_date=row.date,
            source_row=row.source_row,
        )
        output.append(
            FeatureRow(
                row_id=row.row_id,
                source_row=row.source_row,
                date=row.date,
                month=row.month,
                raw_customer=row.raw_customer,
                customer_key=customer_key,
                legacy_customer_key=legacy_key,
                product_key=row.product_key,
                episode_id=episode_id,
                episode_basis=episode_basis,
                features=MappingProxyType(
                    {name: payload.get(name, "") for name in ALLOWED_RAW_FEATURES}
                ),
            )
        )
    return tuple(output)


def load_frozen_history(
    path: str | Path,
    *,
    source_dataset_id: str,
    end_date: str = "2025-12-31",
) -> tuple[tuple[FeatureRow, ...], Mapping[str, str]]:
    """Load only labeled historical actions through the frozen cutoff."""
    source = Path(path).resolve()
    opener = gzip.open if source.suffix == ".gz" else open
    rows: list[FeatureRow] = []
    labels: dict[str, str] = {}
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            actual = str(raw.get("action_result") or "").strip()
            feature_row = _to_feature_row(raw, source_dataset_id)
            if feature_row.date > end_date or actual not in ACTION_LABELS:
                continue
            if feature_row.row_id in labels:
                raise ValueError(f"duplicate frozen history row at line {line_number}")
            rows.append(feature_row)
            labels[feature_row.row_id] = actual
    return tuple(rows), MappingProxyType(dict(sorted(labels.items())))


def load_label_availability_index(
    path: str | Path,
    *,
    source_dataset_id: str,
) -> LabelAvailabilityIndex:
    """Rebuild the frozen outcome-availability clock from raw history rows."""
    source = Path(path).resolve()
    opener = gzip.open if source.suffix == ".gz" else open

    def rows() -> Iterable[Mapping[str, object]]:
        with opener(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    return LabelAvailabilityIndex.from_rows(
        rows(),
        source_dataset_id=source_dataset_id,
        parse_date_value=_parse_date_value,
    )


def sanitized_raw_rows(rows: Iterable[InputRow]) -> tuple[dict[str, str], ...]:
    """Return the exact allow-listed payload passed to non-Python adapters."""
    return tuple(
        {
            **dict(row.feature_payload),
            "__shadowRowId": row.row_id,
            "__shadowSourceFile": row.source_file,
            "__shadowSourceRow": str(row.source_row),
        }
        for row in rows
    )

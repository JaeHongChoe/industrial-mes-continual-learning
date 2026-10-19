from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .data_contract import FeatureRow, parse_source_date, stable_row_id


@dataclass(frozen=True, slots=True)
class PrivilegedRowMetadata:
    """Training-only metadata that is intentionally absent from FeatureRow."""

    row_id: str
    wait_nonblank: bool


def load_privileged_training_metadata(
    path: str | Path,
    source_dataset_id: str,
    *,
    end_date: str,
) -> dict[str, PrivilegedRowMetadata]:
    """Load only the wait-presence bit for rows available by ``end_date``.

    Outcome and customer-opinion fields are never copied into the returned
    object. Keeping this data in a row-ID side table prevents it from entering
    the runtime feature contract by accident.
    """

    if not str(source_dataset_id).strip():
        raise ValueError("source_dataset_id must not be empty")
    if not _is_iso_date(end_date):
        raise ValueError("end_date must use YYYY-MM-DD")
    output: dict[str, PrivilegedRowMetadata] = {}
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid privileged metadata JSONL at {source}:{line_number}") from error
            if not isinstance(raw, Mapping):
                raise ValueError(f"privileged metadata row must be an object at {source}:{line_number}")
            observed_date = parse_source_date(raw)
            if not observed_date or observed_date > end_date:
                continue
            row_id = stable_row_id(source_dataset_id, raw.get("__sourceRowNumber"))
            if row_id in output:
                raise ValueError(f"duplicate privileged metadata row: {row_id}")
            wait_value = raw.get("on hold", raw.get("on_hold"))
            output[row_id] = PrivilegedRowMetadata(
                row_id=row_id,
                wait_nonblank=bool(str(wait_value or "").strip()),
            )
    return output


def gray_zone_sample_weights(
    rows: Iterable[FeatureRow],
    metadata: Mapping[str, PrivilegedRowMetadata],
    *,
    gray_weight: float,
    default_weight: float = 1.0,
) -> np.ndarray:
    """Return training weights without mutating or decorating feature rows."""

    _validate_weight(gray_weight, "gray_weight")
    _validate_weight(default_weight, "default_weight")
    output = []
    for row in rows:
        item = metadata.get(row.row_id)
        output.append(
            float(gray_weight)
            if item is not None and item.wait_nonblank
            else float(default_weight)
        )
    return np.asarray(output, dtype=np.float64)


def _validate_weight(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _is_iso_date(value: str) -> bool:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        return False
    try:
        year, month, day = (int(part) for part in value.split("-"))
    except ValueError:
        return False
    return year >= 1 and 1 <= month <= 12 and 1 <= day <= 31

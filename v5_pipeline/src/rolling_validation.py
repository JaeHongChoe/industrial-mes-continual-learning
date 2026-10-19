from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data_contract import (
    AuditLabelsLockedError,
    FeatureRow,
    parse_source_date,
    stable_row_id,
)
from .freeze import FreezeReceipt


@dataclass(frozen=True, slots=True)
class RollingFold:
    evaluation_month: str
    train_end: str
    calibration_month: str
    calibration_start: str
    calibration_end: str
    evaluation_start: str
    evaluation_end: str


@dataclass(frozen=True, slots=True)
class FoldRows:
    train: tuple[FeatureRow, ...]
    calibration: tuple[FeatureRow, ...]
    evaluation: tuple[FeatureRow, ...]


def build_monthly_folds(start_month: str, end_month: str) -> tuple[RollingFold, ...]:
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    if start > end:
        raise ValueError("rolling fold start month must not be after end month")

    folds: list[RollingFold] = []
    current = start
    while current <= end:
        calibration = _shift_month(current, -1)
        train_last_month = _shift_month(current, -2)
        folds.append(
            RollingFold(
                evaluation_month=_month_token(current),
                train_end=_month_end(train_last_month),
                calibration_month=_month_token(calibration),
                calibration_start=_month_start(calibration),
                calibration_end=_month_end(calibration),
                evaluation_start=_month_start(current),
                evaluation_end=_month_end(current),
            )
        )
        current = _shift_month(current, 1)
    return tuple(folds)


def partition_fold(rows: Sequence[FeatureRow], fold: RollingFold) -> FoldRows:
    materialized = tuple(rows)
    row_ids = [row.row_id for row in materialized]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("rolling rows contain a duplicate row identity")
    ordered = tuple(sorted(materialized, key=lambda row: (row.date, row.source_row, row.row_id)))
    train = tuple(row for row in ordered if row.date <= fold.train_end)
    calibration = tuple(
        row
        for row in ordered
        if fold.calibration_start <= row.date <= fold.calibration_end
    )
    evaluation = tuple(
        row
        for row in ordered
        if fold.evaluation_start <= row.date <= fold.evaluation_end
    )
    _assert_disjoint(train, calibration, evaluation)
    if train and calibration and train[-1].date >= calibration[0].date:
        raise ValueError("rolling train rows are not strictly before calibration rows")
    if calibration and evaluation and calibration[-1].date >= evaluation[0].date:
        raise ValueError("rolling calibration rows are not strictly before evaluation rows")
    return FoldRows(train=train, calibration=calibration, evaluation=evaluation)


def seal_selection_receipt(payload: Mapping[str, Any], output: Path) -> FreezeReceipt:
    if _contains_audit_material(payload):
        raise AuditLabelsLockedError("selection receipt contains audit-label material")
    normalized = _plain_json_value(payload)
    payload_bytes = _canonical_json_bytes(normalized)
    sha256 = hashlib.sha256(payload_bytes).hexdigest()
    sealed = {**normalized, "sha256": sha256}
    final_bytes = json.dumps(
        sealed,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(final_bytes)
    temporary.replace(output)
    return FreezeReceipt(policy_bytes=final_bytes, sha256=sha256)


def assert_audit_labels_locked(bundle: Any) -> None:
    try:
        bundle.audit_labels()
    except AuditLabelsLockedError:
        return
    raise AuditLabelsLockedError("audit labels are unexpectedly available during selection")


def load_development_labels(
    config_path: str | Path,
    rows: Sequence[FeatureRow],
) -> Mapping[str, str]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    audit_start = str(config["auditStart"])
    materialized = tuple(rows)
    if any(row.date >= audit_start for row in materialized):
        raise AuditLabelsLockedError("2026 audit rows are forbidden in development labels")
    expected_ids = {row.row_id for row in materialized}
    if len(expected_ids) != len(materialized):
        raise ValueError("development rows contain a duplicate row identity")

    history_path = Path(str(config["historyRows"]))
    observed_sha = _sha256_file(history_path)
    expected_sha = str(config.get("historyRowsSha256") or "")
    if observed_sha != expected_sha:
        raise ValueError("historyRowsSha256 does not match before development label loading")

    source_dataset_id = str(config["sourceDatasetId"])
    labels: dict[str, str] = {}
    with history_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid history JSONL at line {line_number}") from error
            observed_date = parse_source_date(raw)
            if not observed_date or observed_date >= audit_start:
                continue
            row_id = stable_row_id(source_dataset_id, raw.get("__sourceRowNumber"))
            if row_id not in expected_ids:
                continue
            if row_id in labels:
                raise ValueError(f"duplicate development label row: {row_id}")
            labels[row_id] = str(raw.get("action_result") or "").strip()
    if set(labels) != expected_ids:
        missing = sorted(expected_ids - set(labels))
        raise ValueError(f"development label universe mismatch: missing={len(missing)}")
    return MappingProxyType(labels)


def _parse_month(value: str) -> date:
    try:
        year_token, month_token = value.split("-", 1)
        parsed = date(int(year_token), int(month_token), 1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid YYYY-MM month token: {value!r}") from error
    if value != f"{parsed.year:04d}-{parsed.month:02d}":
        raise ValueError(f"invalid YYYY-MM month token: {value!r}")
    return parsed


def _shift_month(value: date, offset: int) -> date:
    absolute = value.year * 12 + value.month - 1 + offset
    return date(absolute // 12, absolute % 12 + 1, 1)


def _month_token(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _month_start(value: date) -> str:
    return f"{_month_token(value)}-01"


def _month_end(value: date) -> str:
    last_day = calendar.monthrange(value.year, value.month)[1]
    return f"{_month_token(value)}-{last_day:02d}"


def _assert_disjoint(*parts: Sequence[FeatureRow]) -> None:
    seen: set[str] = set()
    for part in parts:
        current = {row.row_id for row in part}
        if seen & current:
            raise ValueError("rolling partitions contain overlapping row identities")
        seen.update(current)


def _contains_audit_material(value: Any) -> bool:
    forbidden = ("auditlabel", "auditmetric", "auditoutcome", "actual2026", "2026label")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(char for char in str(key).lower() if char.isalnum())
            if any(token in normalized for token in forbidden):
                return True
            if _contains_audit_material(nested):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_audit_material(item) for item in value)
    return False


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(nested) for key, nested in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"selection receipt contains a non-JSON value: {type(value).__name__}")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

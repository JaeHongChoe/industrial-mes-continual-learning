from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping


ACTION_LABELS = frozenset({"PROCESS", "RETURN", "REPAIR"})
_STATIC_EXCLUSION_REASONS = (
    "actionBeforeOccurrence",
    "invalidOrMissingLabelAvailableDate",
    "invalidOrMissingOccurrenceDate",
    "nonActionLabel",
)


@dataclass(frozen=True, slots=True)
class LabelAvailabilityRecord:
    row_id: str
    occurrence_date: date
    label_available_date: date
    label: str


@dataclass(frozen=True, slots=True)
class LabelAvailabilityIndex:
    source_dataset_id: str
    source_rows: int
    action_rows: int
    valid_records: tuple[LabelAvailabilityRecord, ...]
    static_excluded_rows: Mapping[str, int]
    static_excluded_row_ids: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_dataset_id: str,
        parse_date_value: Callable[[Any], date | str | None],
    ) -> "LabelAvailabilityIndex":
        dataset_id = str(source_dataset_id).strip()
        if not dataset_id:
            raise ValueError("source_dataset_id must be non-empty")

        seen_ids: set[str] = set()
        valid_records: list[LabelAvailabilityRecord] = []
        excluded_ids: dict[str, list[str]] = {
            reason: [] for reason in _STATIC_EXCLUSION_REASONS
        }
        source_rows = 0
        action_rows = 0

        for raw in rows:
            source_rows += 1
            row_id = _stable_row_id(dataset_id, raw.get("__sourceRowNumber"))
            if row_id in seen_ids:
                raise ValueError(f"duplicate row identity: {row_id}")
            seen_ids.add(row_id)

            label = str(raw.get("action_result") or "").strip()
            if label not in ACTION_LABELS:
                excluded_ids["nonActionLabel"].append(row_id)
                continue
            action_rows += 1

            occurrence = _first_date(
                parse_date_value,
                raw.get("occurrence_date"),
                raw.get("registered_date"),
            )
            if occurrence is None:
                excluded_ids["invalidOrMissingOccurrenceDate"].append(row_id)
                continue

            available = _parsed_date(parse_date_value, raw.get("action_result_date"))
            if available is None:
                excluded_ids["invalidOrMissingLabelAvailableDate"].append(row_id)
                continue
            if available < occurrence:
                excluded_ids["actionBeforeOccurrence"].append(row_id)
                continue

            valid_records.append(
                LabelAvailabilityRecord(
                    row_id=row_id,
                    occurrence_date=occurrence,
                    label_available_date=available,
                    label=label,
                )
            )

        valid_records.sort(key=lambda item: (item.label_available_date, item.row_id))
        frozen_excluded_ids = {
            reason: tuple(sorted(row_ids))
            for reason, row_ids in excluded_ids.items()
        }
        return cls(
            source_dataset_id=dataset_id,
            source_rows=source_rows,
            action_rows=action_rows,
            valid_records=tuple(valid_records),
            static_excluded_rows=MappingProxyType(
                {
                    reason: len(row_ids)
                    for reason, row_ids in frozen_excluded_ids.items()
                }
            ),
            static_excluded_row_ids=MappingProxyType(frozen_excluded_ids),
        )

    def eligible_row_ids(self, cutoff_exclusive: date | str) -> tuple[str, ...]:
        cutoff = _cutoff_date(cutoff_exclusive)
        return tuple(
            sorted(
                record.row_id
                for record in self.valid_records
                if record.label_available_date < cutoff
            )
        )

    def eligible_labels(self, cutoff_exclusive: date | str) -> Mapping[str, str]:
        cutoff = _cutoff_date(cutoff_exclusive)
        return MappingProxyType(
            {
                record.row_id: record.label
                for record in sorted(self.valid_records, key=lambda item: item.row_id)
                if record.label_available_date < cutoff
            }
        )

    def receipt_payload(self, cutoff_exclusive: date | str) -> Mapping[str, Any]:
        cutoff = _cutoff_date(cutoff_exclusive)
        eligible = tuple(
            record
            for record in self.valid_records
            if record.occurrence_date < cutoff
            and record.label_available_date < cutoff
        )
        future_occurrence_count = sum(
            record.occurrence_date >= cutoff for record in self.valid_records
        )
        unavailable_prior_count = sum(
            record.occurrence_date < cutoff
            and record.label_available_date >= cutoff
            for record in self.valid_records
        )
        row_ids = tuple(sorted(record.row_id for record in eligible))
        labels = tuple(sorted((record.row_id, record.label) for record in eligible))
        eligible_evidence = tuple(
            sorted(
                (
                    record.row_id,
                    record.occurrence_date.isoformat(),
                    record.label_available_date.isoformat(),
                    record.label,
                )
                for record in eligible
            )
        )
        record_map = tuple(
            sorted(
                (
                    record.row_id,
                    record.occurrence_date.isoformat(),
                    record.label_available_date.isoformat(),
                    record.label,
                )
                for record in self.valid_records
            )
        )
        excluded = dict(self.static_excluded_rows)
        excluded["futureOccurrenceDate"] = future_occurrence_count
        excluded["notAvailableBeforeCutoff"] = unavailable_prior_count
        future_occurrence_ids = tuple(
            sorted(
                record.row_id
                for record in self.valid_records
                if record.occurrence_date >= cutoff
            )
        )
        unavailable_prior_ids = tuple(
            sorted(
                record.row_id
                for record in self.valid_records
                if record.occurrence_date < cutoff
                and record.label_available_date >= cutoff
            )
        )
        excluded_row_ids = {
            **self.static_excluded_row_ids,
            "futureOccurrenceDate": future_occurrence_ids,
            "notAvailableBeforeCutoff": unavailable_prior_ids,
        }
        contract = {
            "sourceDatasetId": self.source_dataset_id,
            "cutoffExclusive": cutoff.isoformat(),
            "eligibility": (
                "occurrence_date<=label_available_date<cutoff_exclusive "
                "and action_label"
            ),
        }
        return MappingProxyType(
            {
                "schemaVersion": 1,
                "kind": "point_in_time_label_availability_receipt",
                "sourceDatasetId": self.source_dataset_id,
                "cutoffExclusive": cutoff.isoformat(),
                "sourceRows": self.source_rows,
                "actionRows": self.action_rows,
                "eligibleRows": len(eligible),
                "excludedRows": MappingProxyType(excluded),
                "eligibleRowIdsSha256": _sha256_json(row_ids),
                "eligibleLabelsSha256": _sha256_json(labels),
                "eligibleEvidenceSha256": _sha256_json(eligible_evidence),
                "recordMapSha256": _sha256_json(record_map),
                "excludedRowIdsSha256": MappingProxyType(
                    {
                        reason: _sha256_json(excluded_row_ids[reason])
                        for reason in excluded
                    }
                ),
                "contractSha256": _sha256_json(contract),
            }
        )


def _first_date(
    parser: Callable[[Any], date | str | None],
    *values: Any,
) -> date | None:
    for value in values:
        parsed = _parsed_date(parser, value)
        if parsed is not None:
            return parsed
    return None


def _parsed_date(
    parser: Callable[[Any], date | str | None],
    value: Any,
) -> date | None:
    parsed = parser(value)
    if parsed is None:
        return None
    if isinstance(parsed, date):
        return parsed
    try:
        return date.fromisoformat(str(parsed))
    except ValueError:
        return None


def _cutoff_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"invalid cutoff date: {value!r}") from error


def _stable_row_id(source_dataset_id: str, source_row: Any) -> str:
    try:
        number = int(float(source_row))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid source row number: {source_row!r}") from error
    if number < 1:
        raise ValueError(f"invalid source row number: {source_row!r}")
    return f"{source_dataset_id}:{number}"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

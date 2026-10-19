"""Frozen intake-text signals for a prior-supported repair guard."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data_contract import ACTION_LABELS, FeatureRow


@dataclass(frozen=True, slots=True)
class RepairSignal:
    signal_id: str
    pattern: str


@dataclass(frozen=True, slots=True)
class PriorSupportEvidence:
    customer_key: str
    signal_id: str
    support: int
    process_rows: int
    ship_rows: int
    repair_rows: int
    risk_rows: int
    risk_rate: float
    enabled: bool


@dataclass(frozen=True, slots=True)
class PriorSupportedRepairGuardResult:
    predictions: Mapping[str, str]
    demoted_ids: tuple[str, ...]
    reasons_by_row: Mapping[str, tuple[Mapping[str, Any], ...]]


_SIGNALS = (
    RepairSignal(
        signal_id="electrical_short",
        pattern=r"단락|SHORT\s*CIRCUIT",
    ),
    RepairSignal(
        signal_id="fastener_functional_damage",
        pattern=r"(?:볼트|나사).{0,12}(?:마모|손상|풀림)",
    ),
)

FROZEN_REPAIR_SIGNALS: Mapping[str, RepairSignal] = MappingProxyType(
    {signal.signal_id: signal for signal in _SIGNALS}
)
_COMPILED_SIGNALS = {
    signal_id: re.compile(signal.pattern)
    for signal_id, signal in FROZEN_REPAIR_SIGNALS.items()
}


@dataclass(frozen=True, slots=True)
class PriorSupportedRepairGuardParams:
    signal_ids: tuple[str, ...] = tuple(FROZEN_REPAIR_SIGNALS)
    min_rows: int = 20
    min_risk_rows: int = 5
    min_risk_rate: float = 0.30

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_ids", _validated_signal_ids(self.signal_ids))
        if (
            isinstance(self.min_rows, bool)
            or not isinstance(self.min_rows, int)
            or self.min_rows < 1
        ):
            raise ValueError("min_rows must be a positive integer")
        if (
            isinstance(self.min_risk_rows, bool)
            or not isinstance(self.min_risk_rows, int)
            or self.min_risk_rows < 1
        ):
            raise ValueError("min_risk_rows must be a positive integer")
        if isinstance(self.min_risk_rate, bool) or not isinstance(
            self.min_risk_rate, Real
        ):
            raise ValueError("min_risk_rate must be between zero and one")
        risk_rate = float(self.min_risk_rate)
        if not math.isfinite(risk_rate) or not 0.0 <= risk_rate <= 1.0:
            raise ValueError("min_risk_rate must be between zero and one")
        object.__setattr__(self, "min_risk_rate", risk_rate)


@dataclass(frozen=True, slots=True)
class PriorSupportedRepairGuard:
    training_end: str
    params: PriorSupportedRepairGuardParams
    fit_row_ids_sha256: str
    enabled_by_company: Mapping[str, tuple[str, ...]]
    evidence_by_company: Mapping[str, Mapping[str, PriorSupportEvidence]]

    def apply(
        self,
        rows: Sequence[FeatureRow],
        predictions: Mapping[str, str],
    ) -> PriorSupportedRepairGuardResult:
        _validate_enabled_policy(self)
        expected: set[str] = set()
        for row in rows:
            if row.row_id in expected:
                raise ValueError(f"duplicate apply row_id: {row.row_id}")
            expected.add(row.row_id)
        if set(predictions) != expected:
            raise ValueError("apply prediction universe mismatch")

        output: dict[str, str] = {}
        demoted: list[str] = []
        reasons: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for row in rows:
            predicted = predictions[row.row_id]
            matched = matched_repair_signal_ids(
                row,
                self.enabled_by_company.get(row.customer_key, ()),
            )
            if predicted == "PROCESS" and matched:
                output[row.row_id] = ""
                demoted.append(row.row_id)
                reasons[row.row_id] = tuple(
                    _demotion_reason(
                        self.training_end,
                        self.evidence_by_company[row.customer_key][signal_id],
                        self.fit_row_ids_sha256,
                    )
                    for signal_id in matched
                )
            else:
                output[row.row_id] = predicted
        return PriorSupportedRepairGuardResult(
            predictions=MappingProxyType(output),
            demoted_ids=tuple(demoted),
            reasons_by_row=MappingProxyType(reasons),
        )

    @classmethod
    def fit(
        cls,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
        *,
        training_end: str,
        signal_ids: Sequence[str] = tuple(FROZEN_REPAIR_SIGNALS),
        min_rows: int = 20,
        min_risk_rows: int = 5,
        min_risk_rate: float = 0.30,
    ) -> "PriorSupportedRepairGuard":
        params = PriorSupportedRepairGuardParams(
            signal_ids=signal_ids,
            min_rows=min_rows,
            min_risk_rows=min_risk_rows,
            min_risk_rate=min_risk_rate,
        )
        cutoff = _parse_iso_date(training_end, "training_end")
        seen_ids: set[str] = set()
        for row in rows:
            if row.row_id in seen_ids:
                raise ValueError(f"duplicate fit row_id: {row.row_id}")
            seen_ids.add(row.row_id)
            observed = _parse_iso_date(row.date, f"date for row {row.row_id}")
            if observed > cutoff:
                raise ValueError(
                    f"fit row {row.row_id} is after training_end {training_end}"
                )
        if set(labels) != seen_ids:
            raise ValueError("fit label universe mismatch")
        unsupported = sorted(set(labels.values()) - ACTION_LABELS)
        if unsupported:
            raise ValueError(f"unsupported fit action label: {unsupported}")

        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for row in rows:
            for signal_id in matched_repair_signal_ids(row, params.signal_ids):
                counts[(row.customer_key, signal_id)][labels[row.row_id]] += 1

        evidence: dict[str, dict[str, PriorSupportEvidence]] = defaultdict(dict)
        enabled: dict[str, list[str]] = defaultdict(list)
        for (company, signal_id), label_counts in sorted(counts.items()):
            support = sum(label_counts.values())
            ship_rows = label_counts["RETURN"]
            repair_rows = label_counts["REPAIR"]
            risk_rows = ship_rows + repair_rows
            risk_rate = risk_rows / support
            is_enabled = (
                support >= params.min_rows
                and risk_rows >= params.min_risk_rows
                and risk_rate >= params.min_risk_rate
            )
            evidence[company][signal_id] = PriorSupportEvidence(
                customer_key=company,
                signal_id=signal_id,
                support=support,
                process_rows=label_counts["PROCESS"],
                ship_rows=ship_rows,
                repair_rows=repair_rows,
                risk_rows=risk_rows,
                risk_rate=risk_rate,
                enabled=is_enabled,
            )
            if is_enabled:
                enabled[company].append(signal_id)

        return cls(
            training_end=training_end,
            params=params,
            fit_row_ids_sha256=_row_ids_sha256(seen_ids),
            enabled_by_company=MappingProxyType(
                {
                    company: tuple(signal_ids)
                    for company, signal_ids in sorted(enabled.items())
                }
            ),
            evidence_by_company=MappingProxyType(
                {
                    company: MappingProxyType(dict(sorted(by_signal.items())))
                    for company, by_signal in sorted(evidence.items())
                }
            ),
        )


def normalize_repair_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.sub(r"\s+", " ", text).strip()


def matched_repair_signal_ids(
    row: FeatureRow,
    signal_ids: Sequence[str] = tuple(FROZEN_REPAIR_SIGNALS),
) -> tuple[str, ...]:
    selected = frozenset(_validated_signal_ids(signal_ids))
    detail = normalize_repair_text(row.features.get("detail", ""))
    return tuple(
        signal_id
        for signal_id in FROZEN_REPAIR_SIGNALS
        if signal_id in selected and _COMPILED_SIGNALS[signal_id].search(detail)
    )


def _validated_signal_ids(signal_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(signal_ids, str):
        raise ValueError("signal_ids must be a sequence of signal IDs")
    try:
        values = tuple(signal_ids)
    except TypeError as error:
        raise ValueError("signal_ids must be a sequence of signal IDs") from error
    if any(not isinstance(signal_id, str) for signal_id in values):
        raise ValueError("signal IDs must be strings")
    if len(set(values)) != len(values):
        raise ValueError("duplicate signal ID")
    unknown = sorted(set(values) - set(FROZEN_REPAIR_SIGNALS))
    if unknown:
        raise ValueError(f"unknown repair signal ID: {unknown}")
    return values


def _validate_enabled_policy(guard: PriorSupportedRepairGuard) -> None:
    fitted_signals = set(guard.params.signal_ids)
    for company, raw_signal_ids in guard.enabled_by_company.items():
        if not str(company).strip():
            raise ValueError("enabled repair guard company cannot be blank")
        signal_ids = _validated_signal_ids(raw_signal_ids)
        not_fitted = sorted(set(signal_ids) - fitted_signals)
        if not_fitted:
            raise ValueError(f"enabled repair signal ID was not fitted: {not_fitted}")
        for signal_id in signal_ids:
            evidence = guard.evidence_by_company.get(company, {}).get(signal_id)
            if evidence is None or not evidence.enabled:
                raise ValueError(
                    "enabled repair signal is missing enabled fit evidence: "
                    f"{company}/{signal_id}"
                )


def _demotion_reason(
    training_end: str,
    evidence: PriorSupportEvidence,
    fit_row_ids_sha256: str,
) -> Mapping[str, Any]:
    reason = {
        "reason": "enabled_prior_supported_repair_signal",
        "customerKey": evidence.customer_key,
        "signalId": evidence.signal_id,
        "support": evidence.support,
        "processRows": evidence.process_rows,
        "shipRows": evidence.ship_rows,
        "repairRows": evidence.repair_rows,
        "riskRows": evidence.risk_rows,
        "riskRate": evidence.risk_rate,
        "trainingEnd": training_end,
        "fitRowIdsSha256": fit_row_ids_sha256,
    }
    return MappingProxyType(reason)


def _parse_iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date") from error


def _row_ids_sha256(row_ids: Sequence[str] | set[str]) -> str:
    payload = json.dumps(
        sorted(row_ids),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()

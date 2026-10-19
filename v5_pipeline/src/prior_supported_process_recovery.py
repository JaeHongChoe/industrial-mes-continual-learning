"""Recover review rows with exact, historically pure process signatures.

The policy is deliberately narrow.  It learns only exact signatures whose
pre-cutoff action history contains no shipment or repair counterexample.  At
inference time it may change a review row to ``공정투입``; it never rewrites an
existing automatic decision and never reopens a row protected by an upstream
danger guard.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast
import unicodedata

from .data_contract import ACTION_LABELS, FeatureRow


PROCESS_LABEL = "PROCESS"
REVIEW_LABELS = frozenset({"", "REVIEW_NEEDED", "REVIEW_REQUIRED", "REVIEW_SHORT"})
SUPPORTED_SCOPES = (
    "product_detail_counts",
    "company_detail_counts",
)


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _json_ready(self))


@dataclass(frozen=True, slots=True)
class PriorSupportedProcessRecoveryParams(_Serializable):
    scopes: tuple[str, ...] = SUPPORTED_SCOPES
    min_support: int = 10
    min_months: int = 2
    min_recent_support: int = 3
    recent_window_months: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "scopes", _validated_scopes(self.scopes))
        for name in (
            "min_support",
            "min_months",
            "min_recent_support",
            "recent_window_months",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ProcessSignature(_Serializable):
    signature_id: str
    scope: str
    key: tuple[str, ...]
    support: int
    process_rows: int
    months: tuple[str, ...]
    recent_support: int
    training_end: str


@dataclass(frozen=True, slots=True)
class ProcessRecoveryResult(_Serializable):
    predictions: Mapping[str, str]
    recovered_ids: tuple[str, ...]
    reasons_by_row: Mapping[str, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictions",
            MappingProxyType(dict(sorted(self.predictions.items()))),
        )
        object.__setattr__(
            self,
            "reasons_by_row",
            MappingProxyType(dict(sorted(self.reasons_by_row.items()))),
        )


@dataclass(frozen=True, slots=True)
class PriorSupportedProcessRecovery(_Serializable):
    training_end: str
    params: PriorSupportedProcessRecoveryParams
    fit_row_ids_sha256: str
    signatures_by_scope: Mapping[str, tuple[ProcessSignature, ...]]

    def __post_init__(self) -> None:
        expected = set(self.params.scopes)
        if set(self.signatures_by_scope) != expected:
            raise ValueError("signature scope universe differs from fitted parameters")
        frozen: dict[str, tuple[ProcessSignature, ...]] = {}
        seen_ids: set[str] = set()
        for scope in sorted(expected):
            values = tuple(self.signatures_by_scope[scope])
            for item in values:
                if item.scope != scope:
                    raise ValueError("signature is stored under the wrong scope")
                if item.signature_id in seen_ids:
                    raise ValueError("duplicate process signature ID")
                seen_ids.add(item.signature_id)
            frozen[scope] = tuple(sorted(values, key=lambda item: item.key))
        object.__setattr__(self, "signatures_by_scope", MappingProxyType(frozen))

    @classmethod
    def fit(
        cls,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
        *,
        training_end: str,
        scopes: Sequence[str] = SUPPORTED_SCOPES,
        min_support: int = 10,
        min_months: int = 2,
        min_recent_support: int = 3,
        recent_window_months: int = 12,
    ) -> "PriorSupportedProcessRecovery":
        params = PriorSupportedProcessRecoveryParams(
            scopes=tuple(scopes),
            min_support=min_support,
            min_months=min_months,
            min_recent_support=min_recent_support,
            recent_window_months=recent_window_months,
        )
        cutoff = _parse_iso_date(training_end, "training_end")
        cutoff_month = _month_index(training_end[:7])
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

        counts: dict[str, dict[tuple[str, ...], Counter[str]]] = {
            scope: defaultdict(Counter) for scope in params.scopes
        }
        months: dict[str, dict[tuple[str, ...], set[str]]] = {
            scope: defaultdict(set) for scope in params.scopes
        }
        recent: dict[str, Counter[tuple[str, ...]]] = {
            scope: Counter() for scope in params.scopes
        }
        for row in rows:
            for scope in params.scopes:
                key = process_signature_key(row, scope)
                if not key[1]:
                    continue
                counts[scope][key][labels[row.row_id]] += 1
                months[scope][key].add(row.month)
                age = cutoff_month - _month_index(row.month)
                if 0 <= age < params.recent_window_months:
                    recent[scope][key] += 1

        by_scope: dict[str, tuple[ProcessSignature, ...]] = {}
        for scope in params.scopes:
            signatures: list[ProcessSignature] = []
            for key, observed in sorted(counts[scope].items()):
                support = sum(observed.values())
                if (
                    support < params.min_support
                    or observed["RETURN"]
                    or observed["REPAIR"]
                    or len(months[scope][key]) < params.min_months
                    or recent[scope][key] < params.min_recent_support
                ):
                    continue
                basis = {
                    "scope": scope,
                    "key": key,
                    "support": support,
                    "months": sorted(months[scope][key]),
                    "recentSupport": recent[scope][key],
                    "trainingEnd": training_end,
                }
                signatures.append(
                    ProcessSignature(
                        signature_id=hashlib.sha256(
                            _canonical_json_bytes(basis)
                        ).hexdigest(),
                        scope=scope,
                        key=key,
                        support=support,
                        process_rows=observed[PROCESS_LABEL],
                        months=tuple(sorted(months[scope][key])),
                        recent_support=recent[scope][key],
                        training_end=training_end,
                    )
                )
            by_scope[scope] = tuple(signatures)
        return cls(
            training_end=training_end,
            params=params,
            fit_row_ids_sha256=_row_ids_sha256(seen_ids),
            signatures_by_scope=MappingProxyType(by_scope),
        )

    def apply(
        self,
        rows: Sequence[FeatureRow],
        predictions: Mapping[str, str],
        *,
        protected_row_ids: Sequence[str] = (),
    ) -> ProcessRecoveryResult:
        expected: set[str] = set()
        for row in rows:
            if row.row_id in expected:
                raise ValueError(f"duplicate apply row_id: {row.row_id}")
            expected.add(row.row_id)
        if set(predictions) != expected:
            raise ValueError("apply prediction universe mismatch")
        protected = tuple(str(value) for value in protected_row_ids)
        if len(protected) != len(set(protected)) or not set(protected) <= expected:
            raise ValueError("protected row universe is invalid")

        protected_set = set(protected)
        signature_index = {
            scope: {signature.key: signature for signature in signatures}
            for scope, signatures in self.signatures_by_scope.items()
        }
        output: dict[str, str] = {}
        recovered: list[str] = []
        reasons: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for row in rows:
            raw_prediction = str(predictions[row.row_id] or "").strip()
            matched = tuple(
                signature_index[scope][key]
                for scope in sorted(signature_index)
                if (key := process_signature_key(row, scope))
                in signature_index[scope]
            )
            if (
                raw_prediction in REVIEW_LABELS
                and row.row_id not in protected_set
                and matched
            ):
                output[row.row_id] = PROCESS_LABEL
                recovered.append(row.row_id)
                reasons[row.row_id] = tuple(
                    _recovery_reason(
                        signature,
                        fit_row_ids_sha256=self.fit_row_ids_sha256,
                    )
                    for signature in matched
                )
            else:
                output[row.row_id] = raw_prediction
        return ProcessRecoveryResult(
            predictions=output,
            recovered_ids=tuple(sorted(recovered)),
            reasons_by_row=reasons,
        )


def process_signature_key(row: FeatureRow, scope: str) -> tuple[str, ...]:
    if scope not in SUPPORTED_SCOPES:
        raise ValueError(f"unsupported process signature scope: {scope}")
    detail = _normalized(row.features.get("detail"))
    counts = (
        _count_token(row.features.get("intake_count")),
        _count_token(row.features.get("service_count")),
        _count_token(row.features.get("treatment_count")),
    )
    if scope == "product_detail_counts":
        return row.product_key, detail, *counts
    return row.customer_key, detail, *counts


def matched_process_signature_ids(
    row: FeatureRow,
    signatures_by_scope: Mapping[str, Sequence[ProcessSignature]],
) -> tuple[ProcessSignature, ...]:
    output: list[ProcessSignature] = []
    for scope in sorted(signatures_by_scope):
        if scope not in SUPPORTED_SCOPES:
            raise ValueError(f"unsupported process signature scope: {scope}")
        key = process_signature_key(row, scope)
        output.extend(
            signature
            for signature in signatures_by_scope[scope]
            if signature.key == key
        )
    return tuple(sorted(output, key=lambda item: item.signature_id))


def _recovery_reason(
    signature: ProcessSignature,
    *,
    fit_row_ids_sha256: str,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "reason": "prior_pure_process_signature",
            "signatureId": signature.signature_id,
            "scope": signature.scope,
            "key": signature.key,
            "support": signature.support,
            "processRows": signature.process_rows,
            "months": signature.months,
            "recentSupport": signature.recent_support,
            "trainingEnd": signature.training_end,
            "fitRowIdsSha256": fit_row_ids_sha256,
        }
    )


def _validated_scopes(scopes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(scopes, str):
        raise ValueError("scopes must be a sequence")
    values = tuple(scopes)
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError("scopes must contain supported strings")
    if len(values) != len(set(values)):
        raise ValueError("duplicate process signature scope")
    unknown = sorted(set(values) - set(SUPPORTED_SCOPES))
    if unknown:
        raise ValueError(f"unsupported process signature scope: {unknown}")
    return tuple(sorted(values))


def _normalized(value: Any) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).lower().split()
    )


def _count_token(value: Any) -> str:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return "__missing__"
    return f"{number:g}" if math.isfinite(number) else "__missing__"


def _month_index(value: str) -> int:
    try:
        year, month = (int(part) for part in value.split("-"))
    except (TypeError, ValueError) as error:
        raise ValueError("month must use YYYY-MM") from error
    if not 1 <= month <= 12:
        raise ValueError("month must use YYYY-MM")
    return year * 12 + month


def _parse_iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date") from error


def _row_ids_sha256(row_ids: Sequence[str] | set[str]) -> str:
    payload = json.dumps(
        sorted(row_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

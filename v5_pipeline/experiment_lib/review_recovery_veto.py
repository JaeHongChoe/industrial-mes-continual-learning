"""Semantic vetoes that must run before any review-to-process recovery."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import unicodedata


RepairMatcher = Callable[[Any, Sequence[str]], Sequence[str]]
SemanticGroupMatcher = Callable[[Any], Sequence[str]]


# These terms come from the pre-existing V4 recovery vetoes.  This pattern is
# deliberately broad because it is used only to stop an extra review->process
# promotion; it never changes a baseline automatic action.
_LEGACY_V4_REPAIR_CONTEXT = re.compile(
    r"(?:수리|REPAIR|REWORK|재작업|볼트|BOLT|나사|SCREW|분해|분리\s*불가|파손|BROKEN|마모)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReviewRecoveryVetoResult:
    vetoed_row_ids: tuple[str, ...]
    reasons_by_row: Mapping[str, tuple[str, ...]]


def semantic_review_veto(
    rows: Sequence[Any],
    *,
    repair_signal_ids: Sequence[str],
    repair_matcher: RepairMatcher,
    semantic_group_ids: Sequence[str] = (),
    semantic_group_matcher: SemanticGroupMatcher | None = None,
    include_legacy_v4_recovery_risk: bool = False,
) -> ReviewRecoveryVetoResult:
    signals = tuple(str(value) for value in repair_signal_ids)
    if not signals or len(signals) != len(set(signals)):
        raise ValueError("repair signal IDs must be non-empty and unique")
    semantic_groups = tuple(str(value) for value in semantic_group_ids)
    if len(semantic_groups) != len(set(semantic_groups)):
        raise ValueError("semantic group IDs must be unique")
    if semantic_groups and semantic_group_matcher is None:
        raise ValueError("semantic group matcher is required")
    materialized = tuple(rows)
    row_ids = tuple(str(row.row_id) for row in materialized)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate review-veto row ID")

    reasons: dict[str, tuple[str, ...]] = {}
    for row in materialized:
        row_reasons: list[str] = []
        features = getattr(row, "features", {})
        if (
            _token(features.get("type")) == "customer_request"
            or _token(features.get("symptom")) == "customer_request"
        ):
            row_reasons.append("customer_request_gray_zone")
        matched = sorted(set(str(value) for value in repair_matcher(row, signals)))
        row_reasons.extend(f"repair_signal:{value}" for value in matched)
        if semantic_group_matcher is not None:
            matched_groups = sorted(
                set(str(value) for value in semantic_group_matcher(row))
                & set(semantic_groups)
            )
            row_reasons.extend(
                f"semantic_group:{value}" for value in matched_groups
            )
        if include_legacy_v4_recovery_risk and _legacy_v4_recovery_risk(row):
            row_reasons.append("legacy_v4_recovery_risk:repair_context")
        if row_reasons:
            reasons[str(row.row_id)] = tuple(row_reasons)
    return ReviewRecoveryVetoResult(
        vetoed_row_ids=tuple(sorted(reasons)),
        reasons_by_row=MappingProxyType(dict(sorted(reasons.items()))),
    )


def _token(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.sub(r"[^0-9A-Z가-힣]+", "", text)


def _legacy_v4_recovery_risk(row: Any) -> bool:
    features = getattr(row, "features", {})
    text = " | ".join(
        unicodedata.normalize("NFKC", str(features.get(field, "")))
        for field in ("type", "symptom", "detail")
    )
    return bool(_LEGACY_V4_REPAIR_CONTEXT.search(text))

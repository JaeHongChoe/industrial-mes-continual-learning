from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar


T = TypeVar("T")


def select_company_rows(rows: Sequence[T], scope_company: str | None) -> tuple[T, ...]:
    """Return the exact scoring universe for a global or company-local model."""

    materialized = tuple(rows)
    if scope_company is None:
        return materialized
    selected = tuple(
        row for row in materialized if getattr(row, "customer_key") == scope_company
    )
    if not selected:
        raise ValueError(f"candidate company scope has no rows: {scope_company}")
    return selected


def prediction_or_review(predictions: Mapping[str, str], row_id: str) -> str:
    """Company-local candidates must abstain outside their scored row universe."""

    return predictions.get(row_id, "REVIEW")


def validate_route_company(
    candidate_registry: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    route_company: str,
) -> None:
    candidate = candidate_registry.get(candidate_id)
    if candidate is None:
        raise ValueError(f"candidate route is not registered: {candidate_id}")
    owner = candidate.get("scopeCompany")
    if owner is not None and str(owner) != route_company:
        raise ValueError(
            f"candidate company scope mismatch: {candidate_id} owns {owner}, "
            f"route requested {route_company}"
        )

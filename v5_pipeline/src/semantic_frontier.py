"""Exhaustive 2025 trade-off frontier for frozen semantic veto groups."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from .data_contract import FeatureRow
from .hierarchical_veto import FROZEN_SEMANTIC_GROUPS, matched_semantic_groups
from .metrics import OutcomeRow, compute_metrics


def build_company_semantic_frontier(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    predictions: Mapping[str, str],
    *,
    group_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    expected = {row.row_id for row in rows}
    if set(labels) != expected:
        raise ValueError("semantic frontier label universe mismatch")
    if set(predictions) != expected:
        raise ValueError("semantic frontier prediction universe mismatch")
    selected_groups = tuple(group_ids or FROZEN_SEMANTIC_GROUPS)
    unknown = sorted(set(selected_groups) - set(FROZEN_SEMANTIC_GROUPS))
    if unknown:
        raise ValueError(f"unknown semantic veto group: {unknown}")
    if len(set(selected_groups)) != len(selected_groups):
        raise ValueError("semantic veto groups must be unique")

    matched = {
        row.row_id: frozenset(matched_semantic_groups(row))
        for row in rows
    }
    candidates: list[dict[str, Any]] = []
    ordered_groups = tuple(sorted(selected_groups))
    for size in range(len(ordered_groups) + 1):
        for active_tuple in combinations(ordered_groups, size):
            active = frozenset(active_tuple)
            adjusted = {
                row.row_id: (
                    "REVIEW"
                    if predictions[row.row_id] == "PROCESS"
                    and bool(matched[row.row_id] & active)
                    else predictions[row.row_id]
                )
                for row in rows
            }
            metrics = compute_metrics(
                OutcomeRow(
                    row_id=row.row_id,
                    customer_key=row.customer_key,
                    raw_customer=row.raw_customer,
                    legacy_customer_key=row.legacy_customer_key,
                    month=row.month,
                    product_key=row.product_key,
                    episode_id=row.episode_id,
                    actual=labels[row.row_id],
                    predicted=adjusted[row.row_id],
                )
                for row in rows
            )
            candidates.append(
                {
                    "groups": list(active_tuple),
                    "autoRows": metrics.auto_rows,
                    "automationRate": metrics.automation_rate,
                    "correctProcessRows": metrics.correct_process_rows,
                    "actualProcessAutomationRate": metrics.actual_process_automation_rate,
                    "autoAccuracy": metrics.auto_accuracy,
                    "shipToProcess": metrics.ship_to_process,
                    "repairToProcess": metrics.repair_to_process,
                    "riskToProcess": metrics.risk_to_process,
                    "processToNg": metrics.process_to_ng,
                }
            )

    baseline = next(item for item in candidates if not item["groups"])
    for item in candidates:
        item["automationLossRows"] = baseline["autoRows"] - item["autoRows"]
        item["correctProcessLossRows"] = (
            baseline["correctProcessRows"] - item["correctProcessRows"]
        )
        item["riskRemovedRows"] = baseline["riskToProcess"] - item["riskToProcess"]
    frontier = [
        item
        for item in candidates
        if not any(
            other is not item and _dominates(other, item)
            for other in candidates
        )
    ]
    frontier.sort(
        key=lambda item: (
            item["riskToProcess"],
            item["correctProcessLossRows"],
            item["automationLossRows"],
            tuple(item["groups"]),
        )
    )
    return {
        "baseline": baseline,
        "frontier": frontier,
        "candidates": sorted(candidates, key=lambda item: tuple(item["groups"])),
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_accuracy = float(left["autoAccuracy"] or 0.0)
    right_accuracy = float(right["autoAccuracy"] or 0.0)
    no_worse = (
        int(left["autoRows"]) >= int(right["autoRows"])
        and int(left["correctProcessRows"]) >= int(right["correctProcessRows"])
        and int(left["shipToProcess"]) <= int(right["shipToProcess"])
        and int(left["repairToProcess"]) <= int(right["repairToProcess"])
        and left_accuracy >= right_accuracy
    )
    strictly_better = (
        int(left["autoRows"]) > int(right["autoRows"])
        or int(left["correctProcessRows"]) > int(right["correctProcessRows"])
        or int(left["shipToProcess"]) < int(right["shipToProcess"])
        or int(left["repairToProcess"]) < int(right["repairToProcess"])
        or left_accuracy > right_accuracy
    )
    return no_worse and strictly_better

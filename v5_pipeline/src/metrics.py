from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .data_contract import ACTION_LABELS, DataContractError


REVIEW_LABELS = frozenset({"", "REVIEW", "REVIEW_NEEDED", "REVIEW_SHORT", "REVIEW_REQUIRED"})


class GroupingBasis(str, Enum):
    ACTUAL_CUSTOMER = "customer_key"
    RAW_CUSTOMER = "raw_customer"
    LEGACY_POLICY_CUSTOMER = "legacy_customer_key"
    MONTH = "month"
    PRODUCT = "product_key"
    WARM_COLD = "warm_cold"


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    row_id: str
    customer_key: str
    month: str
    episode_id: str
    actual: str
    predicted: str
    raw_customer: str = ""
    legacy_customer_key: str = ""
    product_key: str = ""
    is_cold: bool = False


@dataclass(frozen=True, slots=True)
class Metrics:
    evaluation_rows: int
    auto_rows: int
    review_rows: int
    automation_rate: float
    correct_auto_rows: int
    auto_accuracy: float | None
    actual_process_rows: int
    correct_process_rows: int
    correct_process_recovery: float | None
    actual_process_automatic_rows: int
    actual_process_automation_rate: float | None
    process_action_rows: int
    process_prediction_rate: float
    ship_to_process: int
    repair_to_process: int
    risk_to_process: int
    process_to_ship: int
    process_to_repair: int
    process_to_ng: int
    ship_to_repair: int
    repair_to_ship: int
    action_confusion: Mapping[str, Mapping[str, int]]
    automation_by_actual: Mapping[str, float | None]
    risk_to_process_episodes: int
    process_prediction_episodes: int


def compute_metrics(rows: Iterable[OutcomeRow]) -> Metrics:
    materialized = tuple(rows)
    row_ids = [row.row_id for row in materialized]
    if len(row_ids) != len(set(row_ids)):
        raise DataContractError("metrics input contains duplicate row IDs")
    for row in materialized:
        if row.actual not in ACTION_LABELS:
            raise DataContractError(f"unknown actual action label: {row.actual}")
        if row.predicted not in ACTION_LABELS and row.predicted not in REVIEW_LABELS:
            raise DataContractError(f"unknown predicted label: {row.predicted}")
        if not row.episode_id:
            raise DataContractError(f"missing episode ID for row: {row.row_id}")

    evaluation_rows = len(materialized)
    automatic = [row for row in materialized if row.predicted in ACTION_LABELS]
    auto_rows = len(automatic)
    correct_auto_rows = sum(row.actual == row.predicted for row in automatic)
    actual_process_rows = sum(row.actual == "PROCESS" for row in materialized)
    correct_process_rows = sum(
        row.actual == "PROCESS" and row.predicted == "PROCESS"
        for row in materialized
    )
    process_action_rows = sum(row.predicted == "PROCESS" for row in materialized)
    actual_process_automatic_rows = sum(
        row.actual == "PROCESS" and row.predicted in ACTION_LABELS
        for row in materialized
    )
    ship_to_process = sum(
        row.actual == "RETURN" and row.predicted == "PROCESS"
        for row in materialized
    )
    repair_to_process = sum(
        row.actual == "REPAIR" and row.predicted == "PROCESS"
        for row in materialized
    )
    process_to_ship = sum(
        row.actual == "PROCESS" and row.predicted == "RETURN"
        for row in materialized
    )
    process_to_repair = sum(
        row.actual == "PROCESS" and row.predicted == "REPAIR"
        for row in materialized
    )
    ship_to_repair = sum(
        row.actual == "RETURN" and row.predicted == "REPAIR"
        for row in materialized
    )
    repair_to_ship = sum(
        row.actual == "REPAIR" and row.predicted == "RETURN"
        for row in materialized
    )
    action_order = ("PROCESS", "RETURN", "REPAIR")
    confusion = {
        actual: MappingProxyType(
            {
                predicted: sum(
                    row.actual == actual and row.predicted == predicted
                    for row in materialized
                )
                for predicted in action_order
            }
        )
        for actual in action_order
    }

    automation_by_actual: dict[str, float | None] = {}
    for label in ("PROCESS", "RETURN", "REPAIR"):
        denominator = sum(row.actual == label for row in materialized)
        numerator = sum(
            row.actual == label and row.predicted in ACTION_LABELS
            for row in materialized
        )
        automation_by_actual[label] = _ratio(numerator, denominator)

    process_prediction_episodes = {
        row.episode_id for row in materialized if row.predicted == "PROCESS"
    }
    risk_to_process_episodes = {
        row.episode_id
        for row in materialized
        if row.actual in {"RETURN", "REPAIR"} and row.predicted == "PROCESS"
    }
    return Metrics(
        evaluation_rows=evaluation_rows,
        auto_rows=auto_rows,
        review_rows=evaluation_rows - auto_rows,
        automation_rate=_ratio(auto_rows, evaluation_rows) or 0.0,
        correct_auto_rows=correct_auto_rows,
        auto_accuracy=_ratio(correct_auto_rows, auto_rows),
        actual_process_rows=actual_process_rows,
        correct_process_rows=correct_process_rows,
        correct_process_recovery=_ratio(correct_process_rows, actual_process_rows),
        actual_process_automatic_rows=actual_process_automatic_rows,
        actual_process_automation_rate=_ratio(actual_process_automatic_rows, actual_process_rows),
        process_action_rows=process_action_rows,
        process_prediction_rate=_ratio(process_action_rows, evaluation_rows) or 0.0,
        ship_to_process=ship_to_process,
        repair_to_process=repair_to_process,
        risk_to_process=ship_to_process + repair_to_process,
        process_to_ship=process_to_ship,
        process_to_repair=process_to_repair,
        process_to_ng=process_to_ship + process_to_repair,
        ship_to_repair=ship_to_repair,
        repair_to_ship=repair_to_ship,
        action_confusion=MappingProxyType(confusion),
        automation_by_actual=MappingProxyType(automation_by_actual),
        risk_to_process_episodes=len(risk_to_process_episodes),
        process_prediction_episodes=len(process_prediction_episodes),
    )


def grouped_metrics(
    rows: Iterable[OutcomeRow],
    basis: GroupingBasis | str,
) -> dict[str, Metrics]:
    try:
        grouping = basis if isinstance(basis, GroupingBasis) else GroupingBasis(basis)
    except ValueError as error:
        raise DataContractError(f"unsupported grouping basis: {basis}") from error
    grouped: dict[str, list[OutcomeRow]] = {}
    for row in rows:
        if grouping is GroupingBasis.WARM_COLD:
            key = "cold" if row.is_cold else "warm"
        else:
            key = str(getattr(row, grouping.value))
        grouped.setdefault(key, []).append(row)
    return {key: compute_metrics(grouped[key]) for key in sorted(grouped)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

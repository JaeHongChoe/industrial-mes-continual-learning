from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .hierarchical_evidence import ScopeEvidence
from .risk import U_CP


ScopeLevel = Literal["global", "company", "company_line"]


@dataclass(frozen=True, slots=True)
class RiskBudget:
    company_min_validation_rows: int = 50
    line_min_validation_rows: int = 30
    max_observed_danger_rate: float = 0.0025
    max_exact_danger_ucb: float = 0.0025
    min_automatic_accuracy: float = 0.994
    exact_alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class ScopeRiskAssessment:
    candidate_id: str
    scope_id: str
    scope_level: ScopeLevel
    months: tuple[str, ...]
    validation_rows: int
    auto_rows: int
    correct_auto_rows: int
    actual_process_rows: int
    correct_process_rows: int
    unsafe_label_rows: int
    process_prediction_episodes: int
    risk_episodes: int
    ship_to_process: int
    repair_to_process: int
    process_to_ng: int
    automation_rate: float
    automatic_accuracy: float | None
    correct_process_recovery: float | None
    observed_danger_rate: float | None
    exact_danger_ucb: float
    passed: bool
    rejection_reasons: tuple[str, ...]


def assess_scope(
    evidence: Iterable[ScopeEvidence],
    *,
    scope_level: ScopeLevel,
    budget: RiskBudget,
) -> ScopeRiskAssessment:
    records = tuple(evidence)
    if not records:
        raise ValueError("scope evidence cannot be empty")
    if scope_level not in {"global", "company", "company_line"}:
        raise ValueError("unsupported scope level")
    candidate_ids = {item.candidate_id for item in records}
    scope_ids = {item.scope_id for item in records}
    months = tuple(sorted({item.month for item in records}))
    if len(candidate_ids) != 1 or len(scope_ids) != 1:
        raise ValueError("scope assessment requires one candidate and one scope")
    if len(months) != len(records):
        raise ValueError("scope assessment contains a duplicate month")

    validation_rows = sum(item.rows for item in records)
    auto_rows = sum(item.auto_rows for item in records)
    correct_auto_rows = sum(item.correct_auto_rows for item in records)
    actual_process_rows = sum(item.actual_process_rows for item in records)
    correct_process_rows = sum(item.correct_process_rows for item in records)
    process_prediction_episodes = sum(
        item.process_prediction_episodes for item in records
    )
    risk_episodes = sum(item.risk_episodes for item in records)
    unsafe_label_rows = validation_rows - actual_process_rows
    ship_to_process = sum(item.ship_to_process for item in records)
    repair_to_process = sum(item.repair_to_process for item in records)
    process_to_ng = sum(item.process_to_ng for item in records)

    automation_rate = auto_rows / validation_rows if validation_rows else 0.0
    automatic_accuracy = correct_auto_rows / auto_rows if auto_rows else None
    correct_process_recovery = (
        correct_process_rows / actual_process_rows if actual_process_rows else None
    )
    observed_danger_rate = (
        risk_episodes / process_prediction_episodes
        if process_prediction_episodes
        else None
    )
    exact_danger_ucb = U_CP(
        risk_episodes,
        process_prediction_episodes,
        budget.exact_alpha,
    )

    reasons: list[str] = []
    required_rows = (
        budget.line_min_validation_rows
        if scope_level == "company_line"
        else budget.company_min_validation_rows
        if scope_level == "company"
        else 1
    )
    if validation_rows < required_rows:
        reasons.append("insufficient_validation_rows")
    if scope_level == "company_line" and unsafe_label_rows == 0:
        reasons.append("missing_unsafe_label_evidence")
    if process_prediction_episodes == 0:
        reasons.append("missing_process_prediction_support")
    if (
        observed_danger_rate is not None
        and observed_danger_rate > budget.max_observed_danger_rate
    ):
        reasons.append("observed_danger_rate_exceeds_budget")
    if exact_danger_ucb > budget.max_exact_danger_ucb:
        reasons.append("exact_danger_ucb_exceeds_budget")
    if (
        automatic_accuracy is None
        or automatic_accuracy < budget.min_automatic_accuracy
    ):
        reasons.append("automatic_accuracy_below_floor")

    return ScopeRiskAssessment(
        candidate_id=next(iter(candidate_ids)),
        scope_id=next(iter(scope_ids)),
        scope_level=scope_level,
        months=months,
        validation_rows=validation_rows,
        auto_rows=auto_rows,
        correct_auto_rows=correct_auto_rows,
        actual_process_rows=actual_process_rows,
        correct_process_rows=correct_process_rows,
        unsafe_label_rows=unsafe_label_rows,
        process_prediction_episodes=process_prediction_episodes,
        risk_episodes=risk_episodes,
        ship_to_process=ship_to_process,
        repair_to_process=repair_to_process,
        process_to_ng=process_to_ng,
        automation_rate=automation_rate,
        automatic_accuracy=automatic_accuracy,
        correct_process_recovery=correct_process_recovery,
        observed_danger_rate=observed_danger_rate,
        exact_danger_ucb=exact_danger_ucb,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )

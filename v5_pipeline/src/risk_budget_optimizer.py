from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .hierarchical_evidence import ScopeEvidence
from .hierarchical_risk import RiskBudget, ScopeRiskAssessment, assess_scope
from .hierarchical_select import (
    PolicyChoice,
    REVIEW_ALL_CANDIDATE_ID,
)


@dataclass(frozen=True, slots=True)
class TransitionCaps:
    ship_to_process: int
    repair_to_process: int
    total: int

    def __post_init__(self) -> None:
        if min(self.ship_to_process, self.repair_to_process, self.total) < 0:
            raise ValueError("transition caps cannot be negative")
        if self.total > self.ship_to_process + self.repair_to_process:
            raise ValueError("combined cap cannot exceed component caps sum")


@dataclass(frozen=True, slots=True)
class BudgetedLineChoice:
    company: str
    line: str
    candidate_id: str
    parent_candidate_id: str
    assessment: ScopeRiskAssessment | None


@dataclass(frozen=True, slots=True)
class BudgetedLinePlan:
    line_choices: Mapping[tuple[str, str], BudgetedLineChoice]
    ship_to_process: int
    repair_to_process: int
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int


@dataclass(frozen=True, slots=True)
class _LineOption:
    candidate_id: str
    assessment: ScopeRiskAssessment | None
    ship_to_process: int
    repair_to_process: int
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int


@dataclass(frozen=True, slots=True)
class _PlanState:
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int
    selections: tuple[str, ...]


def optimize_line_budget(
    evidence: Iterable[ScopeEvidence],
    company_choices: Mapping[str, PolicyChoice],
    *,
    budget: RiskBudget,
    caps: TransitionCaps,
) -> BudgetedLinePlan:
    records = tuple(evidence)
    grouped: dict[tuple[str, str], list[ScopeEvidence]] = defaultdict(list)
    for item in records:
        if item.scope_id.startswith("company_line:"):
            grouped[(item.candidate_id, item.scope_id)].append(item)
    assessments = {
        key: assess_scope(items, scope_level="company_line", budget=budget)
        for key, items in grouped.items()
    }

    fixed_ship = 0
    fixed_repair = 0
    fixed_process = 0
    fixed_auto = 0
    fixed_correct = 0
    cells: list[tuple[str, str, str, tuple[_LineOption, ...]]] = []
    for company, company_choice in sorted(company_choices.items()):
        parent_id = company_choice.candidate_id
        parent_lines = sorted(
            (
                scope_id,
                assessment,
            )
            for (candidate_id, scope_id), assessment in assessments.items()
            if candidate_id == parent_id
            and scope_id.startswith(f"company_line:{company}:")
        )
        parent_line_ship = sum(item.ship_to_process for _, item in parent_lines)
        parent_line_repair = sum(item.repair_to_process for _, item in parent_lines)
        parent_line_process = sum(item.correct_process_rows for _, item in parent_lines)
        parent_line_auto = sum(item.auto_rows for _, item in parent_lines)
        parent_line_correct = sum(item.correct_auto_rows for _, item in parent_lines)
        parent_company = company_choice.assessment
        residuals = (
            parent_company.ship_to_process - parent_line_ship,
            parent_company.repair_to_process - parent_line_repair,
            parent_company.correct_process_rows - parent_line_process,
            parent_company.auto_rows - parent_line_auto,
            parent_company.correct_auto_rows - parent_line_correct,
        )
        if min(residuals) < 0:
            raise ValueError(f"company-line evidence exceeds company evidence: {company}")
        fixed_ship += residuals[0]
        fixed_repair += residuals[1]
        fixed_process += residuals[2]
        fixed_auto += residuals[3]
        fixed_correct += residuals[4]

        for scope_id, parent_assessment in parent_lines:
            _, _, line = scope_id.split(":", 2)
            options: dict[str, _LineOption] = {
                parent_id: _option(parent_id, parent_assessment),
                REVIEW_ALL_CANDIDATE_ID: _LineOption(
                    candidate_id=REVIEW_ALL_CANDIDATE_ID,
                    assessment=None,
                    ship_to_process=0,
                    repair_to_process=0,
                    correct_process_rows=0,
                    auto_rows=0,
                    correct_auto_rows=0,
                ),
            }
            for (candidate_id, candidate_scope), assessment in assessments.items():
                if candidate_scope != scope_id or not _observed_eligible(assessment):
                    continue
                options[candidate_id] = _option(candidate_id, assessment)
            cells.append(
                (
                    company,
                    line,
                    parent_id,
                    tuple(sorted(options.values(), key=lambda item: item.candidate_id)),
                )
            )

    if not _within_caps(fixed_ship, fixed_repair, caps):
        raise ValueError("non-line residual risk already exceeds the global caps")
    states: dict[tuple[int, int], _PlanState] = {
        (fixed_ship, fixed_repair): _PlanState(
            correct_process_rows=fixed_process,
            auto_rows=fixed_auto,
            correct_auto_rows=fixed_correct,
            selections=(),
        )
    }
    for _, _, _, options in cells:
        next_states: dict[tuple[int, int], _PlanState] = {}
        for (ship, repair), state in states.items():
            for option in options:
                next_ship = ship + option.ship_to_process
                next_repair = repair + option.repair_to_process
                if not _within_caps(next_ship, next_repair, caps):
                    continue
                candidate = _PlanState(
                    correct_process_rows=(
                        state.correct_process_rows + option.correct_process_rows
                    ),
                    auto_rows=state.auto_rows + option.auto_rows,
                    correct_auto_rows=(
                        state.correct_auto_rows + option.correct_auto_rows
                    ),
                    selections=(*state.selections, option.candidate_id),
                )
                key = (next_ship, next_repair)
                current = next_states.get(key)
                if current is None or _state_score(candidate) > _state_score(current):
                    next_states[key] = candidate
        if not next_states:
            raise ValueError("no line-policy plan can satisfy the global risk caps")
        states = next_states

    best_key, best = max(
        states.items(),
        key=lambda item: (*_state_score(item[1]), -sum(item[0]), -item[0][0]),
    )
    choices: dict[tuple[str, str], BudgetedLineChoice] = {}
    for (company, line, parent_id, options), selected_id in zip(
        cells, best.selections, strict=True
    ):
        if selected_id == parent_id:
            continue
        selected = next(item for item in options if item.candidate_id == selected_id)
        choices[(company, line)] = BudgetedLineChoice(
            company=company,
            line=line,
            candidate_id=selected_id,
            parent_candidate_id=parent_id,
            assessment=selected.assessment,
        )
    return BudgetedLinePlan(
        line_choices=choices,
        ship_to_process=best_key[0],
        repair_to_process=best_key[1],
        correct_process_rows=best.correct_process_rows,
        auto_rows=best.auto_rows,
        correct_auto_rows=best.correct_auto_rows,
    )


def _option(candidate_id: str, assessment: ScopeRiskAssessment) -> _LineOption:
    return _LineOption(
        candidate_id=candidate_id,
        assessment=assessment,
        ship_to_process=assessment.ship_to_process,
        repair_to_process=assessment.repair_to_process,
        correct_process_rows=assessment.correct_process_rows,
        auto_rows=assessment.auto_rows,
        correct_auto_rows=assessment.correct_auto_rows,
    )


def _observed_eligible(assessment: ScopeRiskAssessment) -> bool:
    return not any(
        reason != "exact_danger_ucb_exceeds_budget"
        for reason in assessment.rejection_reasons
    )


def _within_caps(ship: int, repair: int, caps: TransitionCaps) -> bool:
    return (
        ship <= caps.ship_to_process
        and repair <= caps.repair_to_process
        and ship + repair <= caps.total
    )


def _state_score(state: _PlanState) -> tuple[int, int, int]:
    return state.correct_process_rows, state.auto_rows, state.correct_auto_rows

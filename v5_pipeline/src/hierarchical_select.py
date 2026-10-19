from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .hierarchical_evidence import ScopeEvidence
from .hierarchical_risk import RiskBudget, ScopeRiskAssessment, assess_scope


REVIEW_ALL_CANDIDATE_ID = "__review_all__"


@dataclass(frozen=True, slots=True)
class PolicyChoice:
    scope_id: str
    scope_level: str
    candidate_id: str
    parent_candidate_id: str | None
    assessment: ScopeRiskAssessment
    gain_correct_process_rows: int
    gain_auto_rows: int


def resolve_candidate(
    company: str,
    line: str,
    company_choices: dict[str, PolicyChoice],
    line_overrides: dict[tuple[str, str], PolicyChoice],
) -> str | None:
    override = line_overrides.get((company, line))
    if override is not None:
        return override.candidate_id
    company_choice = company_choices.get(company)
    return company_choice.candidate_id if company_choice is not None else None


def select_company_policies(
    evidence: Iterable[ScopeEvidence],
    *,
    budget: RiskBudget,
    require_exact: bool = True,
) -> dict[str, PolicyChoice]:
    grouped = _group_scopes(evidence, prefix="company:")
    by_company: dict[str, list[PolicyChoice]] = defaultdict(list)
    for (candidate_id, scope_id), records in grouped.items():
        company = scope_id.split(":", 1)[1]
        assessment = assess_scope(records, scope_level="company", budget=budget)
        if not _eligible(assessment, require_exact=require_exact):
            continue
        by_company[company].append(
            PolicyChoice(
                scope_id=scope_id,
                scope_level="company",
                candidate_id=candidate_id,
                parent_candidate_id=None,
                assessment=assessment,
                gain_correct_process_rows=assessment.correct_process_rows,
                gain_auto_rows=assessment.auto_rows,
            )
        )
    return {
        company: _best_choice(choices)
        for company, choices in sorted(by_company.items())
        if choices
    }


def select_line_overrides(
    evidence: Iterable[ScopeEvidence],
    company_choices: dict[str, PolicyChoice],
    *,
    budget: RiskBudget,
    require_exact: bool = True,
) -> dict[tuple[str, str], PolicyChoice]:
    grouped = _group_scopes(evidence, prefix="company_line:")
    assessments: dict[tuple[str, str], ScopeRiskAssessment] = {}
    for (candidate_id, scope_id), records in grouped.items():
        assessments[(candidate_id, scope_id)] = assess_scope(
            records,
            scope_level="company_line",
            budget=budget,
        )

    by_line: dict[tuple[str, str], list[PolicyChoice]] = defaultdict(list)
    for (candidate_id, scope_id), assessment in assessments.items():
        _, company, line = scope_id.split(":", 2)
        parent = company_choices.get(company)
        if parent is None or not _eligible(
            assessment, require_exact=require_exact
        ):
            continue
        parent_assessment = assessments.get((parent.candidate_id, scope_id))
        if parent_assessment is None:
            continue
        gain_process = (
            assessment.correct_process_rows
            - parent_assessment.correct_process_rows
        )
        gain_auto = assessment.auto_rows - parent_assessment.auto_rows
        if gain_process < 0 or (gain_process == 0 and gain_auto <= 0):
            continue
        by_line[(company, line)].append(
            PolicyChoice(
                scope_id=scope_id,
                scope_level="company_line",
                candidate_id=candidate_id,
                parent_candidate_id=parent.candidate_id,
                assessment=assessment,
                gain_correct_process_rows=gain_process,
                gain_auto_rows=gain_auto,
            )
        )
    return {
        key: _best_override(choices)
        for key, choices in sorted(by_line.items())
        if choices
    }


def _group_scopes(
    evidence: Iterable[ScopeEvidence], *, prefix: str
) -> dict[tuple[str, str], list[ScopeEvidence]]:
    grouped: dict[tuple[str, str], list[ScopeEvidence]] = defaultdict(list)
    for item in evidence:
        if item.scope_id.startswith(prefix):
            grouped[(item.candidate_id, item.scope_id)].append(item)
    return grouped


def _eligible(
    assessment: ScopeRiskAssessment, *, require_exact: bool
) -> bool:
    if require_exact:
        return assessment.passed
    return not any(
        reason != "exact_danger_ucb_exceeds_budget"
        for reason in assessment.rejection_reasons
    )


def _best_choice(choices: list[PolicyChoice]) -> PolicyChoice:
    return sorted(
        choices,
        key=lambda item: (
            -item.assessment.correct_process_rows,
            -item.assessment.auto_rows,
            item.assessment.risk_episodes,
            item.candidate_id,
        ),
    )[0]


def _best_override(choices: list[PolicyChoice]) -> PolicyChoice:
    return sorted(
        choices,
        key=lambda item: (
            -item.gain_correct_process_rows,
            -item.gain_auto_rows,
            item.assessment.risk_episodes,
            item.candidate_id,
        ),
    )[0]

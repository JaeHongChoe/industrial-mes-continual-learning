"""Robust company-threshold selection using development-year evidence only.

Candidates carry exact per-fold metrics for the ensemble and every development
seed.  A policy is feasible only when the full window and every leave-one-fold-
out subset satisfy the reserved annual caps on the fixed 140,477-row reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


REFERENCE_ROWS = 140_477
ENSEMBLE_SCENARIO = "ensemble"
HARD_SHIP_TO_PROCESS = 65
HARD_REPAIR_TO_PROCESS = 35
HARD_TOTAL = 100


@dataclass(frozen=True, slots=True)
class TransitionCaps:
    ship_to_process: int
    repair_to_process: int
    total: int

    def __post_init__(self) -> None:
        if min(self.ship_to_process, self.repair_to_process, self.total) < 0:
            raise ValueError("transition caps must be non-negative")

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.ship_to_process, self.repair_to_process, self.total)


@dataclass(frozen=True, slots=True)
class ReserveCapPolicy:
    """Annual hard caps with an exact rational reserve fraction."""

    annual_ship_to_process: int = HARD_SHIP_TO_PROCESS
    annual_repair_to_process: int = HARD_REPAIR_TO_PROCESS
    annual_total: int = HARD_TOTAL
    reserve_numerator: int = 95
    reserve_denominator: int = 100

    def __post_init__(self) -> None:
        if (
            min(
                self.annual_ship_to_process,
                self.annual_repair_to_process,
                self.annual_total,
            )
            < 0
        ):
            raise ValueError("annual caps must be non-negative")
        if self.reserve_denominator <= 0:
            raise ValueError("reserve denominator must be positive")
        if not 0 <= self.reserve_numerator <= self.reserve_denominator:
            raise ValueError("reserve fraction must be between zero and one")
        if (
            self.annual_ship_to_process > HARD_SHIP_TO_PROCESS
            or self.annual_repair_to_process > HARD_REPAIR_TO_PROCESS
            or self.annual_total > HARD_TOTAL
        ):
            raise ValueError("cap policy cannot exceed the 65/35/100 hard caps")
        if self.reserve_numerator * 100 > 95 * self.reserve_denominator:
            raise ValueError("cap policy cannot exceed the 95% reserve fraction")


DEFAULT_CAP_POLICY = ReserveCapPolicy()


def scaled_reserve_caps(
    evaluation_rows: int,
    *,
    policy: ReserveCapPolicy = DEFAULT_CAP_POLICY,
) -> TransitionCaps:
    """Scale reserved caps against the immutable 140,477-row reference.

    Each component is floored independently, including total, as required for
    full-window and leave-one-month-out feasibility checks.
    """

    if evaluation_rows < 0:
        raise ValueError("evaluation_rows must be non-negative")
    denominator = policy.reserve_denominator * REFERENCE_ROWS

    def scale(annual_cap: int) -> int:
        return annual_cap * policy.reserve_numerator * evaluation_rows // denominator

    return TransitionCaps(
        ship_to_process=scale(policy.annual_ship_to_process),
        repair_to_process=scale(policy.annual_repair_to_process),
        total=scale(policy.annual_total),
    )


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    """Exact evidence for one model scenario and one development fold."""

    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    correct_auto_rows: int
    ship_to_process: int
    repair_to_process: int

    def __post_init__(self) -> None:
        values = (
            self.evaluation_rows,
            self.actual_process_rows,
            self.correct_process_rows,
            self.correct_auto_rows,
            self.ship_to_process,
            self.repair_to_process,
        )
        if min(values) < 0:
            raise ValueError("fold metrics must be non-negative")
        if self.actual_process_rows > self.evaluation_rows:
            raise ValueError("actual process rows cannot exceed evaluation rows")
        if self.correct_process_rows > self.actual_process_rows:
            raise ValueError("correct process rows cannot exceed actual process rows")
        if self.correct_auto_rows > self.evaluation_rows:
            raise ValueError("correct auto rows cannot exceed evaluation rows")
        if self.total_danger > self.evaluation_rows:
            raise ValueError("danger transitions cannot exceed evaluation rows")

    @property
    def total_danger(self) -> int:
        return self.ship_to_process + self.repair_to_process


def _freeze_scenario_folds(
    scenario_folds: Mapping[str, Mapping[str, FoldMetrics]],
) -> Mapping[str, Mapping[str, FoldMetrics]]:
    return MappingProxyType(
        {
            scenario: MappingProxyType(dict(folds))
            for scenario, folds in scenario_folds.items()
        }
    )


@dataclass(frozen=True, slots=True)
class CompanyThresholdCandidate:
    """One company's threshold pair with ensemble and seed fold evidence."""

    company: str
    option_id: str
    shipment_threshold: float
    repair_threshold: float
    scenario_folds: Mapping[str, Mapping[str, FoldMetrics]]

    def __post_init__(self) -> None:
        if not self.company:
            raise ValueError("company must not be empty")
        if not self.option_id:
            raise ValueError("option_id must not be empty")
        for threshold in (self.shipment_threshold, self.repair_threshold):
            if not isfinite(threshold) or threshold < 0:
                raise ValueError("thresholds must be finite and non-negative")
        if not self.scenario_folds:
            raise ValueError("scenario_folds must not be empty")

        frozen = _freeze_scenario_folds(self.scenario_folds)
        fold_sets = {frozenset(folds) for folds in frozen.values()}
        if len(fold_sets) != 1 or not next(iter(fold_sets)):
            raise ValueError("every scenario must contain the same non-empty folds")

        first_scenario = next(iter(frozen.values()))
        for scenario, folds in frozen.items():
            for fold, metrics in folds.items():
                reference = first_scenario[fold]
                if (
                    metrics.evaluation_rows != reference.evaluation_rows
                    or metrics.actual_process_rows != reference.actual_process_rows
                ):
                    raise ValueError(
                        "evaluation and actual-process populations must be "
                        f"identical across scenarios ({scenario}, {fold})"
                    )
        object.__setattr__(self, "scenario_folds", frozen)


@dataclass(frozen=True, slots=True)
class SubsetAssessment:
    scenario: str
    subset_id: str
    included_folds: tuple[str, ...]
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    correct_auto_rows: int
    ship_to_process: int
    repair_to_process: int
    caps: TransitionCaps

    @property
    def total_danger(self) -> int:
        return self.ship_to_process + self.repair_to_process

    @property
    def correct_process_recall(self) -> Fraction:
        if self.actual_process_rows == 0:
            return Fraction(1, 1)
        return Fraction(self.correct_process_rows, self.actual_process_rows)

    @property
    def feasible(self) -> bool:
        return (
            self.ship_to_process <= self.caps.ship_to_process
            and self.repair_to_process <= self.caps.repair_to_process
            and self.total_danger <= self.caps.total
        )

    @property
    def max_cap_utilization(self) -> Fraction:
        return max(
            _utilization(self.ship_to_process, self.caps.ship_to_process),
            _utilization(self.repair_to_process, self.caps.repair_to_process),
            _utilization(self.total_danger, self.caps.total),
        )


@dataclass(frozen=True, slots=True)
class RobustCompanyPolicy:
    choices: Mapping[str, CompanyThresholdCandidate]
    assessments: Mapping[str, Mapping[str, SubsetAssessment]]
    worst_correct_process_recall: Fraction
    full_correct_process_rows: int
    full_correct_auto_rows: int
    max_cap_utilization: Fraction
    full_danger: int


def _utilization(observed: int, cap: int) -> Fraction:
    if cap == 0:
        return Fraction(0, 1) if observed == 0 else Fraction(observed, 1)
    return Fraction(observed, cap)


def _validate_candidate_universe(
    options_by_company: Mapping[str, Sequence[CompanyThresholdCandidate]],
    eligible_companies: frozenset[str],
    fallback_option_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not options_by_company:
        raise ValueError("options_by_company must not be empty")
    if not fallback_option_id:
        raise ValueError("fallback_option_id must not be empty")

    unknown_eligible = eligible_companies.difference(options_by_company)
    if unknown_eligible:
        raise ValueError(
            f"eligible companies have no candidates: {sorted(unknown_eligible)}"
        )

    expected_scenarios: tuple[str, ...] | None = None
    expected_folds: tuple[str, ...] | None = None
    population_by_company: dict[str, dict[str, tuple[int, int]]] = {}

    for company, candidates in options_by_company.items():
        if not candidates:
            raise ValueError(f"company {company!r} has no candidates")
        option_ids = [candidate.option_id for candidate in candidates]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError(f"company {company!r} has duplicate option ids")
        if fallback_option_id not in option_ids:
            raise ValueError(
                f"company {company!r} is missing fallback option {fallback_option_id!r}"
            )

        for candidate in candidates:
            if candidate.company != company:
                raise ValueError(
                    f"candidate company {candidate.company!r} does not match "
                    f"company key {company!r}"
                )
            scenarios = tuple(sorted(candidate.scenario_folds))
            folds = tuple(sorted(next(iter(candidate.scenario_folds.values()))))
            if expected_scenarios is None:
                expected_scenarios = scenarios
                expected_folds = folds
            elif scenarios != expected_scenarios or folds != expected_folds:
                raise ValueError(
                    "all candidates must share the same scenarios and folds"
                )

            populations = (
                {
                    fold: (
                        candidate.scenario_folds[ENSEMBLE_SCENARIO][
                            fold
                        ].evaluation_rows,
                        candidate.scenario_folds[ENSEMBLE_SCENARIO][
                            fold
                        ].actual_process_rows,
                    )
                    for fold in folds
                }
                if ENSEMBLE_SCENARIO in candidate.scenario_folds
                else {}
            )
            previous = population_by_company.setdefault(company, populations)
            if previous != populations:
                raise ValueError(
                    f"all candidates for {company!r} must use identical populations"
                )

    assert expected_scenarios is not None and expected_folds is not None
    if ENSEMBLE_SCENARIO not in expected_scenarios:
        raise ValueError("candidates must include an ensemble scenario")
    if len(expected_scenarios) < 2:
        raise ValueError("candidates must include the ensemble and at least one seed")
    if len(expected_folds) < 2:
        raise ValueError(
            "at least two folds are required for leave-one-fold-out checks"
        )
    return expected_scenarios, expected_folds


def _subset_specs(folds: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (("full", folds),) + tuple(
        (f"leave_out:{omitted}", tuple(fold for fold in folds if fold != omitted))
        for omitted in folds
    )


def _assess_choices(
    choices: Mapping[str, CompanyThresholdCandidate],
    scenarios: tuple[str, ...],
    folds: tuple[str, ...],
    cap_policy: ReserveCapPolicy,
) -> Mapping[str, Mapping[str, SubsetAssessment]]:
    assessments: dict[str, Mapping[str, SubsetAssessment]] = {}
    for scenario in scenarios:
        scenario_assessments: dict[str, SubsetAssessment] = {}
        for subset_id, included_folds in _subset_specs(folds):
            selected_metrics = [
                choice.scenario_folds[scenario][fold]
                for choice in choices.values()
                for fold in included_folds
            ]
            evaluation_rows = sum(item.evaluation_rows for item in selected_metrics)
            assessment = SubsetAssessment(
                scenario=scenario,
                subset_id=subset_id,
                included_folds=included_folds,
                evaluation_rows=evaluation_rows,
                actual_process_rows=sum(
                    item.actual_process_rows for item in selected_metrics
                ),
                correct_process_rows=sum(
                    item.correct_process_rows for item in selected_metrics
                ),
                correct_auto_rows=sum(
                    item.correct_auto_rows for item in selected_metrics
                ),
                ship_to_process=sum(item.ship_to_process for item in selected_metrics),
                repair_to_process=sum(
                    item.repair_to_process for item in selected_metrics
                ),
                caps=scaled_reserve_caps(evaluation_rows, policy=cap_policy),
            )
            scenario_assessments[subset_id] = assessment
        assessments[scenario] = MappingProxyType(scenario_assessments)
    return MappingProxyType(assessments)


def _fraction_from_float(value: float) -> Fraction:
    return Fraction(str(value))


def _selection_score(
    choices: Mapping[str, CompanyThresholdCandidate],
    assessments: Mapping[str, Mapping[str, SubsetAssessment]],
    eligible_companies: frozenset[str],
    global_fallback_thresholds: tuple[float, float],
) -> tuple[Fraction | int, ...]:
    all_assessments = [
        assessment
        for scenario_assessments in assessments.values()
        for assessment in scenario_assessments.values()
    ]
    full_assessments = [
        scenario_assessments["full"] for scenario_assessments in assessments.values()
    ]
    worst_recall = min(
        assessment.correct_process_recall for assessment in all_assessments
    )
    full_correct_process = min(
        assessment.correct_process_rows for assessment in full_assessments
    )
    full_correct_auto = min(
        assessment.correct_auto_rows for assessment in full_assessments
    )
    max_utilization = max(
        assessment.max_cap_utilization for assessment in all_assessments
    )
    ensemble_full = assessments[ENSEMBLE_SCENARIO]["full"]

    eligible_choices = [choices[company] for company in sorted(eligible_companies)]
    conservative_threshold = sum(
        (
            _fraction_from_float(choice.shipment_threshold)
            + _fraction_from_float(choice.repair_threshold)
            for choice in eligible_choices
        ),
        Fraction(0, 1),
    )
    fallback_ship = _fraction_from_float(global_fallback_thresholds[0])
    fallback_repair = _fraction_from_float(global_fallback_thresholds[1])
    fallback_distance = sum(
        (
            abs(_fraction_from_float(choice.shipment_threshold) - fallback_ship)
            + abs(_fraction_from_float(choice.repair_threshold) - fallback_repair)
            for choice in eligible_choices
        ),
        Fraction(0, 1),
    )

    return (
        worst_recall,
        full_correct_process,
        full_correct_auto,
        -max_utilization,
        -ensemble_full.total_danger,
        -conservative_threshold,
        -fallback_distance,
    )


def _build_policy(
    choices: Mapping[str, CompanyThresholdCandidate],
    assessments: Mapping[str, Mapping[str, SubsetAssessment]],
) -> RobustCompanyPolicy:
    all_assessments = [
        assessment
        for scenario_assessments in assessments.values()
        for assessment in scenario_assessments.values()
    ]
    full_assessments = [
        scenario_assessments["full"] for scenario_assessments in assessments.values()
    ]
    ensemble_full = assessments[ENSEMBLE_SCENARIO]["full"]
    return RobustCompanyPolicy(
        choices=MappingProxyType(dict(choices)),
        assessments=assessments,
        worst_correct_process_recall=min(
            assessment.correct_process_recall for assessment in all_assessments
        ),
        full_correct_process_rows=min(
            assessment.correct_process_rows for assessment in full_assessments
        ),
        full_correct_auto_rows=min(
            assessment.correct_auto_rows for assessment in full_assessments
        ),
        max_cap_utilization=max(
            assessment.max_cap_utilization for assessment in all_assessments
        ),
        full_danger=ensemble_full.total_danger,
    )


def optimize_robust_company_thresholds(
    options_by_company: Mapping[str, Sequence[CompanyThresholdCandidate]],
    *,
    eligible_companies: Iterable[str],
    fallback_option_id: str,
    global_fallback_thresholds: tuple[float, float] = (0.01, 0.015),
    cap_policy: ReserveCapPolicy = DEFAULT_CAP_POLICY,
) -> RobustCompanyPolicy:
    """Select the lexicographically best robust development-only policy.

    Non-eligible companies are fixed to ``fallback_option_id``.  Eligible
    companies are searched exhaustively.  The objective is, in order:

    1. worst exact process recall over every scenario and full/LOFO subset;
    2. worst full-window exact process count across scenarios;
    3. worst full-window exact automation count across scenarios;
    4. minimum maximum component/total cap utilization;
    5. minimum ensemble full-window danger;
    6. lower (more conservative) thresholds;
    7. minimum distance to the global fallback thresholds.
    """

    eligible = frozenset(eligible_companies)
    scenarios, folds = _validate_candidate_universe(
        options_by_company, eligible, fallback_option_id
    )
    fallback_ship, fallback_repair = global_fallback_thresholds
    if not all(
        isfinite(value) and value >= 0 for value in (fallback_ship, fallback_repair)
    ):
        raise ValueError("global fallback thresholds must be finite and non-negative")

    candidates_by_company = {
        company: tuple(candidates) for company, candidates in options_by_company.items()
    }
    fallback_by_company = {
        company: next(
            candidate
            for candidate in candidates
            if candidate.option_id == fallback_option_id
        )
        for company, candidates in candidates_by_company.items()
    }
    eligible_names = tuple(sorted(eligible))
    eligible_pools = tuple(candidates_by_company[name] for name in eligible_names)
    combinations = product(*eligible_pools) if eligible_pools else ((),)

    best_choices: dict[str, CompanyThresholdCandidate] | None = None
    best_assessments: Mapping[str, Mapping[str, SubsetAssessment]] | None = None
    best_score: tuple[Fraction | int, ...] | None = None
    best_signature: tuple[tuple[str, str], ...] | None = None

    for eligible_selection in combinations:
        choices = dict(fallback_by_company)
        choices.update(zip(eligible_names, eligible_selection, strict=True))
        assessments = _assess_choices(choices, scenarios, folds, cap_policy)
        if not all(
            assessment.feasible
            for scenario_assessments in assessments.values()
            for assessment in scenario_assessments.values()
        ):
            continue

        score = _selection_score(
            choices, assessments, eligible, global_fallback_thresholds
        )
        signature = tuple(
            (company, choices[company].option_id) for company in sorted(choices)
        )
        if (
            best_score is None
            or score > best_score
            or (score == best_score and signature < best_signature)
        ):
            best_choices = choices
            best_assessments = assessments
            best_score = score
            best_signature = signature

    if best_choices is None or best_assessments is None:
        raise ValueError(
            "no candidate policy passes reserved full and leave-one-fold-out "
            "caps for the ensemble and every seed"
        )
    return _build_policy(best_choices, best_assessments)


__all__ = [
    "CompanyThresholdCandidate",
    "DEFAULT_CAP_POLICY",
    "ENSEMBLE_SCENARIO",
    "FoldMetrics",
    "HARD_REPAIR_TO_PROCESS",
    "HARD_SHIP_TO_PROCESS",
    "HARD_TOTAL",
    "REFERENCE_ROWS",
    "ReserveCapPolicy",
    "RobustCompanyPolicy",
    "SubsetAssessment",
    "TransitionCaps",
    "optimize_robust_company_thresholds",
    "scaled_reserve_caps",
]

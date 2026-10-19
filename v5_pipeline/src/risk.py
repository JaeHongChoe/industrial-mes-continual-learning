from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping

from scipy.stats import beta

from .data_contract import ACTION_LABELS
from .metrics import REVIEW_LABELS


PROCESS_LABEL = "PROCESS"
SHIP_LABEL = "RETURN"
REPAIR_LABEL = "REPAIR"


@dataclass(frozen=True, slots=True)
class EpisodeDecision:
    """A row-level decision associated with a pre-frozen episode identifier."""

    row_id: str
    episode_id: str
    company: str
    month: str
    actual: str
    predicted: str
    is_added: bool


@dataclass(frozen=True, slots=True)
class EpisodeRiskSummary:
    """Episode-level safety numerator, denominator, and exact upper bound."""

    added_process_predicted_episodes: int
    ship_to_process_episodes: int
    repair_to_process_episodes: int
    unsafe_episodes: int
    alpha: float
    ucb: float


@dataclass(frozen=True, slots=True)
class FrozenHypothesisRegistry:
    """Validated identity and exact hypothesis family from a frozen registry."""

    registry_sha256: str
    hypothesis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.registry_sha256):
            raise ValueError("registry_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.hypothesis_ids, tuple):
            raise TypeError("hypothesis_ids must be a tuple")
        if not self.hypothesis_ids:
            raise ValueError("hypothesis_ids cannot be empty")
        if any(not isinstance(item, str) or not item for item in self.hypothesis_ids):
            raise ValueError("hypothesis_ids must contain non-empty strings")
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("hypothesis_ids must be unique")

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypothesis_ids)


@dataclass(frozen=True, slots=True)
class RiskGateResult:
    """A single company-month safety decision retaining its raw evidence."""

    summary: EpisodeRiskSummary
    budget: float
    adjusted_alpha: float
    registry: FrozenHypothesisRegistry
    passed: bool
    rejection_reasons: tuple[str, ...]

    @property
    def registry_sha256(self) -> str:
        return self.registry.registry_sha256

    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        return self.registry.hypothesis_ids

    @property
    def hypothesis_count(self) -> int:
        return self.registry.hypothesis_count


@dataclass(frozen=True, slots=True)
class CompanyRiskGateResult:
    """Separate selector-month gates for one company; pooled acceptance is unavailable."""

    company: str
    month_gates: Mapping[str, RiskGateResult]
    passed: bool
    rejection_reasons: tuple[str, ...]


def U_CP(events: int, trials: int, alpha: float) -> float:
    """Return the one-sided Clopper-Pearson upper confidence bound.

    A zero-trial cell has no safety evidence, so its upper bound is one.
    """

    _validate_counts(events, trials)
    _validate_alpha(alpha)
    if trials == 0 or events == trials:
        return 1.0
    return float(beta.ppf(1 - alpha, events + 1, trials - events))


def bonferroni_discovery_alpha(
    alpha: float,
    registry: FrozenHypothesisRegistry,
) -> float:
    """Allocate discovery alpha across every tested safety hypothesis."""

    _validate_alpha(alpha)
    if not isinstance(registry, FrozenHypothesisRegistry):
        raise TypeError("registry must be FrozenHypothesisRegistry metadata")
    return alpha / registry.hypothesis_count


def summarize_episode_risk(
    decisions: Iterable[EpisodeDecision],
    *,
    alpha: float,
) -> EpisodeRiskSummary:
    """Collapse added process predictions to worst-case episode-level risk."""

    _validate_alpha(alpha)
    materialized = _validate_decisions(decisions)
    by_episode: dict[str, list[EpisodeDecision]] = {}
    for decision in materialized:
        if decision.is_added and decision.predicted == PROCESS_LABEL:
            by_episode.setdefault(decision.episode_id, []).append(decision)

    ship_to_process_episodes = sum(
        any(decision.actual == SHIP_LABEL for decision in episode)
        for episode in by_episode.values()
    )
    repair_to_process_episodes = sum(
        any(decision.actual == REPAIR_LABEL for decision in episode)
        for episode in by_episode.values()
    )
    unsafe_episodes = sum(
        any(decision.actual in {SHIP_LABEL, REPAIR_LABEL} for decision in episode)
        for episode in by_episode.values()
    )
    added_process_predicted_episodes = len(by_episode)
    return EpisodeRiskSummary(
        added_process_predicted_episodes=added_process_predicted_episodes,
        ship_to_process_episodes=ship_to_process_episodes,
        repair_to_process_episodes=repair_to_process_episodes,
        unsafe_episodes=unsafe_episodes,
        alpha=alpha,
        ucb=U_CP(unsafe_episodes, added_process_predicted_episodes, alpha),
    )


def evaluate_selector_risk_gates(
    decisions: Iterable[EpisodeDecision],
    *,
    budget: float,
    registry: FrozenHypothesisRegistry,
    expected_companies: frozenset[str],
    alpha: float = 0.05,
    required_months: tuple[str, ...] = ("2025-10", "2025-11"),
) -> Mapping[str, CompanyRiskGateResult]:
    """Evaluate each required selector month independently for every company."""

    _validate_budget(budget)
    adjusted_alpha = bonferroni_discovery_alpha(alpha, registry)
    if not required_months or len(required_months) != len(set(required_months)):
        raise ValueError("required_months must contain unique month values")
    if any(not month for month in required_months):
        raise ValueError("required_months cannot contain an empty month")
    _validate_expected_companies(expected_companies)

    materialized = _validate_decisions(decisions)
    observed_companies = {decision.company for decision in materialized}
    unexpected_companies = observed_companies - expected_companies
    if unexpected_companies:
        raise ValueError(
            f"decisions contain unexpected companies: {', '.join(sorted(unexpected_companies))}"
        )
    _validate_episode_ownership(materialized)
    results: dict[str, CompanyRiskGateResult] = {}
    for company in sorted(expected_companies):
        company_decisions = [decision for decision in materialized if decision.company == company]
        month_gates: dict[str, RiskGateResult] = {}
        company_reasons = [] if company_decisions else ["missing_company_data"]
        for month in required_months:
            month_decisions = [
                decision
                for decision in company_decisions
                if decision.month == month
            ]
            summary = summarize_episode_risk(month_decisions, alpha=adjusted_alpha)
            reasons = _rejection_reasons(summary, budget, has_month_data=bool(month_decisions))
            gate = RiskGateResult(
                summary=summary,
                budget=budget,
                adjusted_alpha=adjusted_alpha,
                registry=registry,
                passed=not reasons,
                rejection_reasons=tuple(reasons),
            )
            month_gates[month] = gate
            company_reasons.extend(f"{month}:{reason}" for reason in reasons)
        results[company] = CompanyRiskGateResult(
            company=company,
            month_gates=MappingProxyType(month_gates),
            passed=not company_reasons,
            rejection_reasons=tuple(company_reasons),
        )
    return MappingProxyType(results)


clopper_pearson_upper_bound = U_CP
bonferroni_alpha = bonferroni_discovery_alpha
compute_episode_risk = summarize_episode_risk


def _rejection_reasons(
    summary: EpisodeRiskSummary,
    budget: float,
    *,
    has_month_data: bool,
) -> list[str]:
    reasons: list[str] = []
    if not has_month_data:
        reasons.append("missing_month_data")
    if summary.ship_to_process_episodes:
        reasons.append("ship_to_process_episodes_nonzero")
    if summary.repair_to_process_episodes:
        reasons.append("repair_to_process_episodes_nonzero")
    if summary.ucb > budget:
        reasons.append("ucb_exceeds_budget")
    return reasons


def _validate_decisions(decisions: Iterable[EpisodeDecision]) -> tuple[EpisodeDecision, ...]:
    materialized = tuple(decisions)
    row_ids: set[str] = set()
    for decision in materialized:
        if not isinstance(decision, EpisodeDecision):
            raise TypeError("decisions must contain EpisodeDecision values")
        if not decision.row_id or not decision.episode_id or not decision.company or not decision.month:
            raise ValueError("EpisodeDecision requires non-empty row_id, episode_id, company, and month")
        if decision.row_id in row_ids:
            raise ValueError(f"duplicate row ID: {decision.row_id}")
        row_ids.add(decision.row_id)
        if decision.actual not in ACTION_LABELS:
            raise ValueError(f"unknown actual action label: {decision.actual}")
        if decision.predicted not in ACTION_LABELS and decision.predicted not in REVIEW_LABELS:
            raise ValueError(f"unknown predicted action label: {decision.predicted}")
        if not isinstance(decision.is_added, bool):
            raise ValueError("is_added must be a bool")
    return materialized


def _validate_counts(events: int, trials: int) -> None:
    if (
        isinstance(events, bool)
        or isinstance(trials, bool)
        or not isinstance(events, Integral)
        or not isinstance(trials, Integral)
        or not 0 <= events <= trials
    ):
        raise ValueError("events and trials must satisfy 0 <= events <= trials")


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")


def _validate_budget(budget: float) -> None:
    if not 0 < budget < 1:
        raise ValueError("budget must be between zero and one")


def _validate_expected_companies(expected_companies: frozenset[str]) -> None:
    if not isinstance(expected_companies, frozenset):
        raise TypeError("expected_companies must be a frozenset")
    if not expected_companies:
        raise ValueError("expected_companies cannot be empty")
    if any(not isinstance(company, str) or not company for company in expected_companies):
        raise ValueError("expected_companies must contain non-empty strings")


def _validate_episode_ownership(decisions: tuple[EpisodeDecision, ...]) -> None:
    ownership: dict[str, tuple[str, str]] = {}
    for decision in decisions:
        owner = (decision.company, decision.month)
        previous_owner = ownership.setdefault(decision.episode_id, owner)
        if previous_owner != owner:
            raise ValueError(
                "episode ID shared across company/month cells: "
                f"{decision.episode_id} belongs to {previous_owner} and {owner}"
            )

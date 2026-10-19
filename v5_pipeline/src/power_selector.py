from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import binom

from .data_contract import ACTION_LABELS
from .risk import PROCESS_LABEL, REPAIR_LABEL, SHIP_LABEL
from .select import FALLBACK_POLICY, CandidateDecision


RISK_BUDGET = 0.005
FAMILYWISE_ALPHA = 0.05
MIN_PROCESS_EPISODES_PER_MONTH = 598
SELECTOR_MONTHS = ("2025-10", "2025-11")
ENDPOINT = "unsafe_to_process_union"


@dataclass(frozen=True, slots=True)
class ProcessDecision:
    """A selector decision that deliberately contains no outcome label."""

    candidate_id: str
    row_id: str
    episode_id: str
    company: str
    month: str
    predicted: str
    baseline_predicted: str = "REVIEW"


@dataclass(frozen=True, slots=True)
class PowerHypothesis:
    hypothesis_id: str
    decision_hash: str
    candidate_ids: tuple[str, ...]
    company: str
    month: str
    endpoint: str
    risk_budget: float
    process_episodes: int


@dataclass(frozen=True, slots=True)
class FrozenPowerRegistry:
    registry_sha256: str
    source_selector_sha256: str
    selector_months: tuple[str, ...]
    expected_companies: tuple[str, ...]
    present_companies: tuple[str, ...]
    absent_companies: tuple[str, ...]
    raw_candidate_count: int
    unique_decision_function_count: int
    candidate_to_decision_hash: Mapping[str, str]
    decision_classes: Mapping[str, tuple[str, ...]]
    support_by_pair: Mapping[tuple[str, str], Mapping[str, int]]
    row_count_by_pair: Mapping[tuple[str, str], Mapping[str, int]]
    eligible_pairs: tuple[tuple[str, str], ...]
    hypotheses: tuple[PowerHypothesis, ...]
    selection_mode: str
    risk_budget: float = RISK_BUDGET
    familywise_alpha: float = FAMILYWISE_ALPHA
    minimum_process_episodes_per_month: int = MIN_PROCESS_EPISODES_PER_MONTH
    endpoint: str = ENDPOINT
    _process_decisions: tuple[ProcessDecision, ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)


@dataclass(frozen=True, slots=True)
class PValueRecord:
    hypothesis_id: str
    decision_hash: str
    candidate_ids: tuple[str, ...]
    company: str
    month: str
    endpoint: str
    risk_budget: float
    process_episodes: int
    unsafe_episodes: int
    ship_to_process_episodes: int
    repair_to_process_episodes: int
    raw_p_value: float


@dataclass(frozen=True, slots=True)
class HolmDecision:
    hypothesis_id: str
    raw_p_value: float
    rank: int
    threshold: float
    adjusted_p_value: float
    rejected: bool


@dataclass(frozen=True, slots=True)
class AdjustedDecision:
    hypothesis_id: str
    decision_hash: str
    candidate_ids: tuple[str, ...]
    company: str
    month: str
    endpoint: str
    process_episodes: int
    unsafe_episodes: int
    raw_p_value: float
    rank: int
    holm_threshold: float
    adjusted_p_value: float
    rejected: bool


@dataclass(frozen=True, slots=True)
class PowerSelectionResult:
    registry_sha256: str
    p_values: tuple[PValueRecord, ...]
    adjusted_decisions: tuple[AdjustedDecision, ...]
    strict_assignments: Mapping[str, str]
    shadow_assignments: Mapping[str, str]
    status_by_company: Mapping[str, str]
    rejections: Mapping[str, Mapping[str, tuple[str, ...]]]
    shadow_rejections: Mapping[str, Mapping[str, tuple[str, ...]]]
    scores: Mapping[str, Mapping[str, tuple[float, int, int, int]]]


@dataclass(frozen=True, slots=True)
class _RiskCell:
    process_episodes: int
    unsafe_episodes: int
    ship_to_process_episodes: int
    repair_to_process_episodes: int


def freeze_power_hypotheses(
    decisions: Iterable[ProcessDecision | Mapping[str, Any]],
    *,
    expected_companies: Iterable[str],
    selector_months: Sequence[str] = SELECTOR_MONTHS,
    source_selector_sha256: str,
    selection_mode: str = "standalone_clean",
) -> FrozenPowerRegistry:
    """Freeze a label-free, support-filtered candidate-company-month family."""

    months = tuple(selector_months)
    if months != SELECTOR_MONTHS:
        raise ValueError(
            "power selector is restricted to the viewed 2025-10/2025-11 selector"
        )
    if selection_mode not in {"standalone_clean", "monotone_overlay"}:
        raise ValueError("selection_mode must be standalone_clean or monotone_overlay")
    _validate_sha256(source_selector_sha256, "source_selector_sha256")
    materialized = tuple(_coerce_process_decision(item) for item in decisions)
    if not materialized:
        raise ValueError("process decisions cannot be empty")

    expected = tuple(sorted(set(expected_companies)))
    if not expected or any(not item for item in expected):
        raise ValueError("expected_companies must contain non-empty company names")

    by_candidate: dict[str, dict[str, ProcessDecision]] = {}
    row_identity: dict[str, tuple[str, str, str]] = {}
    for item in materialized:
        if item.month not in months:
            raise ValueError(f"process decision month is outside Oct-Nov: {item.month}")
        candidate_rows = by_candidate.setdefault(item.candidate_id, {})
        if item.row_id in candidate_rows:
            raise ValueError(
                f"duplicate candidate-row process decision: {item.candidate_id}/{item.row_id}"
            )
        candidate_rows[item.row_id] = item
        identity = (item.episode_id, item.company, item.month)
        previous = row_identity.setdefault(item.row_id, identity)
        if previous != identity:
            raise ValueError(f"row identity differs across candidates: {item.row_id}")

    candidate_ids = tuple(sorted(by_candidate))
    reference_rows = frozenset(by_candidate[candidate_ids[0]])
    for candidate_id in candidate_ids[1:]:
        if frozenset(by_candidate[candidate_id]) != reference_rows:
            raise ValueError(
                "every candidate must cover the same frozen selector row universe"
            )
    episode_owners: dict[str, tuple[str, str]] = {}
    for item in by_candidate[candidate_ids[0]].values():
        owner = (item.company, item.month)
        previous_owner = episode_owners.setdefault(item.episode_id, owner)
        if previous_owner != owner:
            raise ValueError(
                "episode ID shared across company-month cells: "
                f"{item.episode_id} belongs to {previous_owner} and {owner}"
            )

    present = tuple(sorted({identity[1] for identity in row_identity.values()}))
    unexpected = sorted(set(present) - set(expected))
    if unexpected:
        raise ValueError(
            f"selector contains companies outside expected_companies: {unexpected}"
        )
    absent = tuple(sorted(set(expected) - set(present)))

    candidate_to_hash: dict[str, str] = {}
    classes: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        rows = by_candidate[candidate_id]
        decision_hash = _process_function_hash(
            rows.values(), selection_mode=selection_mode
        )
        candidate_to_hash[candidate_id] = decision_hash
        classes.setdefault(decision_hash, []).append(candidate_id)

    frozen_classes = {
        decision_hash: tuple(sorted(aliases))
        for decision_hash, aliases in sorted(classes.items())
    }
    support_by_pair: dict[tuple[str, str], Mapping[str, int]] = {}
    row_count_by_pair: dict[tuple[str, str], Mapping[str, int]] = {}
    eligible_pairs: list[tuple[str, str]] = []
    hypotheses: list[PowerHypothesis] = []
    for decision_hash, aliases in frozen_classes.items():
        representative_rows = tuple(by_candidate[aliases[0]].values())
        for company in present:
            support = {
                month: len(
                    {
                        row.episode_id
                        for row in representative_rows
                        if row.company == company
                        and row.month == month
                        and _is_added_process(row, selection_mode)
                    }
                )
                for month in months
            }
            row_counts = {
                month: sum(
                    row.company == company and row.month == month
                    for row in representative_rows
                )
                for month in months
            }
            pair = (decision_hash, company)
            support_by_pair[pair] = MappingProxyType(support)
            row_count_by_pair[pair] = MappingProxyType(row_counts)
            eligible = all(
                row_counts[month] > 0
                and support[month] >= MIN_PROCESS_EPISODES_PER_MONTH
                for month in months
            )
            if not eligible:
                continue
            eligible_pairs.append(pair)
            for month in months:
                hypothesis_payload = {
                    "decisionHash": decision_hash,
                    "company": company,
                    "month": month,
                    "endpoint": ENDPOINT,
                    "riskBudget": RISK_BUDGET,
                }
                hypotheses.append(
                    PowerHypothesis(
                        hypothesis_id=hashlib.sha256(
                            _canonical_json_bytes(hypothesis_payload)
                        ).hexdigest(),
                        decision_hash=decision_hash,
                        candidate_ids=aliases,
                        company=company,
                        month=month,
                        endpoint=ENDPOINT,
                        risk_budget=RISK_BUDGET,
                        process_episodes=support[month],
                    )
                )

    hypotheses.sort(key=lambda item: item.hypothesis_id)
    eligible_pairs.sort()
    registry_payload = _registry_payload(
        registry_sha256=None,
        source_selector_sha256=source_selector_sha256,
        selector_months=months,
        expected_companies=expected,
        present_companies=present,
        absent_companies=absent,
        candidate_to_decision_hash=candidate_to_hash,
        decision_classes=frozen_classes,
        support_by_pair=support_by_pair,
        row_count_by_pair=row_count_by_pair,
        eligible_pairs=tuple(eligible_pairs),
        hypotheses=tuple(hypotheses),
        selection_mode=selection_mode,
    )
    registry_sha256 = hashlib.sha256(
        _canonical_json_bytes(registry_payload)
    ).hexdigest()
    return FrozenPowerRegistry(
        registry_sha256=registry_sha256,
        source_selector_sha256=source_selector_sha256,
        selector_months=months,
        expected_companies=expected,
        present_companies=present,
        absent_companies=absent,
        raw_candidate_count=len(candidate_ids),
        unique_decision_function_count=len(frozen_classes),
        candidate_to_decision_hash=MappingProxyType(dict(candidate_to_hash)),
        decision_classes=MappingProxyType(frozen_classes),
        support_by_pair=MappingProxyType(support_by_pair),
        row_count_by_pair=MappingProxyType(row_count_by_pair),
        eligible_pairs=tuple(eligible_pairs),
        hypotheses=tuple(hypotheses),
        selection_mode=selection_mode,
        _process_decisions=materialized,
    )


def exact_binomial_p_value(events: int, trials: int, q: float = RISK_BUDGET) -> float:
    """Return P[X <= events] for X ~ Binomial(trials, q)."""

    if isinstance(events, bool) or isinstance(trials, bool):
        raise TypeError("events and trials must be integers")
    if not isinstance(events, int) or not isinstance(trials, int):
        raise TypeError("events and trials must be integers")
    if trials < 1 or events < 0 or events > trials:
        raise ValueError(
            "exact binomial counts require 0 <= events <= trials and trials >= 1"
        )
    if not math.isclose(q, RISK_BUDGET, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the exploratory protocol fixes q at 0.005")
    return float(binom.cdf(events, trials, q))


def holm_step_down(
    p_values: Mapping[str, float],
    *,
    alpha: float = FAMILYWISE_ALPHA,
) -> tuple[HolmDecision, ...]:
    """Apply deterministic Holm step-down with monotone adjusted p-values."""

    if not math.isclose(alpha, FAMILYWISE_ALPHA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the exploratory protocol fixes family-wise alpha at 0.05")
    if len(p_values) != len(set(p_values)):
        raise ValueError("hypothesis IDs must be unique")
    ordered = sorted((float(value), str(key)) for key, value in p_values.items())
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0 for value, _ in ordered
    ):
        raise ValueError("p-values must be finite values in [0, 1]")
    total = len(ordered)
    decisions: list[HolmDecision] = []
    running_adjusted = 0.0
    still_rejecting = True
    for index, (raw_p, hypothesis_id) in enumerate(ordered):
        rank = index + 1
        remaining = total - index
        threshold = alpha / remaining
        running_adjusted = max(running_adjusted, min(1.0, remaining * raw_p))
        rejected = still_rejecting and raw_p <= threshold
        if not rejected:
            still_rejecting = False
        decisions.append(
            HolmDecision(
                hypothesis_id=hypothesis_id,
                raw_p_value=raw_p,
                rank=rank,
                threshold=threshold,
                adjusted_p_value=running_adjusted,
                rejected=rejected,
            )
        )
    return tuple(decisions)


def evaluate_power_selector(
    registry: FrozenPowerRegistry,
    outcomes: Iterable[CandidateDecision | Mapping[str, Any]],
    *,
    candidate_complexity: Mapping[str, int] | None = None,
    accuracy_floor: float = 0.98,
) -> PowerSelectionResult:
    """Attach viewed selector outcomes only after the discovery family is frozen."""

    if not isinstance(registry, FrozenPowerRegistry):
        raise TypeError("registry must be a FrozenPowerRegistry")
    if not 0.0 <= accuracy_floor <= 1.0:
        raise ValueError("accuracy_floor must be in [0, 1]")
    complexity = dict(candidate_complexity or {})
    materialized = tuple(_coerce_outcome(item) for item in outcomes)
    by_candidate = _validate_outcomes_against_registry(registry, materialized)

    frozen_by_candidate: dict[str, dict[str, ProcessDecision]] = {}
    for row in registry._process_decisions:
        frozen_by_candidate.setdefault(row.candidate_id, {})[row.row_id] = row

    risk_cells: dict[tuple[str, str, str], _RiskCell] = {}
    for decision_hash, aliases in registry.decision_classes.items():
        representative = aliases[0]
        for company in registry.present_companies:
            for month in registry.selector_months:
                rows = [
                    item
                    for item in by_candidate[representative].values()
                    if item.company == company and item.month == month
                ]
                frozen_rows = frozen_by_candidate[representative]
                risk_cells[(decision_hash, company, month)] = _summarize_risk_cell(
                    rows,
                    frozen_rows=frozen_rows,
                    selection_mode=registry.selection_mode,
                )

    p_values: list[PValueRecord] = []
    for hypothesis in registry.hypotheses:
        cell = risk_cells[
            (hypothesis.decision_hash, hypothesis.company, hypothesis.month)
        ]
        if cell.process_episodes != hypothesis.process_episodes:
            raise ValueError(
                "outcome episode support differs from the frozen preflight"
            )
        p_values.append(
            PValueRecord(
                hypothesis_id=hypothesis.hypothesis_id,
                decision_hash=hypothesis.decision_hash,
                candidate_ids=hypothesis.candidate_ids,
                company=hypothesis.company,
                month=hypothesis.month,
                endpoint=hypothesis.endpoint,
                risk_budget=hypothesis.risk_budget,
                process_episodes=cell.process_episodes,
                unsafe_episodes=cell.unsafe_episodes,
                ship_to_process_episodes=cell.ship_to_process_episodes,
                repair_to_process_episodes=cell.repair_to_process_episodes,
                raw_p_value=exact_binomial_p_value(
                    cell.unsafe_episodes,
                    cell.process_episodes,
                    hypothesis.risk_budget,
                ),
            )
        )

    holm = holm_step_down(
        {item.hypothesis_id: item.raw_p_value for item in p_values},
        alpha=registry.familywise_alpha,
    )
    p_by_id = {item.hypothesis_id: item for item in p_values}
    adjusted = tuple(
        AdjustedDecision(
            hypothesis_id=item.hypothesis_id,
            decision_hash=p_by_id[item.hypothesis_id].decision_hash,
            candidate_ids=p_by_id[item.hypothesis_id].candidate_ids,
            company=p_by_id[item.hypothesis_id].company,
            month=p_by_id[item.hypothesis_id].month,
            endpoint=p_by_id[item.hypothesis_id].endpoint,
            process_episodes=p_by_id[item.hypothesis_id].process_episodes,
            unsafe_episodes=p_by_id[item.hypothesis_id].unsafe_episodes,
            raw_p_value=item.raw_p_value,
            rank=item.rank,
            holm_threshold=item.threshold,
            adjusted_p_value=item.adjusted_p_value,
            rejected=item.rejected,
        )
        for item in holm
    )
    adjusted_by_cell = {
        (item.decision_hash, item.company, item.month): item for item in adjusted
    }

    strict_assignments: dict[str, str] = {}
    shadow_assignments: dict[str, str] = {}
    status_by_company: dict[str, str] = {}
    rejections: dict[str, Mapping[str, tuple[str, ...]]] = {}
    shadow_rejections: dict[str, Mapping[str, tuple[str, ...]]] = {}
    scores: dict[str, Mapping[str, tuple[float, int, int, int]]] = {}
    for company in registry.expected_companies:
        if company in registry.absent_companies:
            strict_assignments[company] = FALLBACK_POLICY
            shadow_assignments[company] = FALLBACK_POLICY
            status_by_company[company] = "forced_review_absent_from_selector"
            rejections[company] = MappingProxyType({})
            shadow_rejections[company] = MappingProxyType({})
            scores[company] = MappingProxyType({})
            continue

        strict_candidates: list[tuple[tuple[float, int, int, int, str], str]] = []
        shadow_candidates: list[tuple[tuple[float, int, int, int, str], str]] = []
        company_rejections: dict[str, tuple[str, ...]] = {}
        company_shadow_rejections: dict[str, tuple[str, ...]] = {}
        company_scores: dict[str, tuple[float, int, int, int]] = {}
        for candidate_id in sorted(by_candidate):
            decision_hash = registry.candidate_to_decision_hash[candidate_id]
            empirical_reasons: list[str] = []
            strict_reasons: list[str] = []
            recoveries: list[float] = []
            total_correct_process = 0
            total_correct_auto = 0
            pair = (decision_hash, company)
            pair_eligible = pair in registry.eligible_pairs
            if not pair_eligible:
                strict_reasons.append("insufficient_support")
            for month in registry.selector_months:
                month_rows = [
                    item
                    for item in by_candidate[candidate_id].values()
                    if item.company == company and item.month == month
                ]
                automatic = [
                    item for item in month_rows if item.predicted in ACTION_LABELS
                ]
                correct_auto = sum(item.actual == item.predicted for item in automatic)
                accuracy = correct_auto / len(automatic) if automatic else 0.0
                if accuracy < accuracy_floor:
                    empirical_reasons.append(f"{month}:accuracy_below_floor")
                actual_process = sum(
                    item.actual == PROCESS_LABEL for item in month_rows
                )
                correct_process = sum(
                    item.actual == PROCESS_LABEL
                    and _is_added_process(
                        frozen_by_candidate[candidate_id][item.row_id],
                        registry.selection_mode,
                    )
                    for item in month_rows
                )
                recovery = correct_process / actual_process if actual_process else 0.0
                if correct_process == 0:
                    empirical_reasons.append(f"{month}:no_correct_process_recovery")
                cell = risk_cells[(decision_hash, company, month)]
                if cell.unsafe_episodes:
                    empirical_reasons.append(f"{month}:{ENDPOINT}_nonzero")
                if pair_eligible:
                    decision = adjusted_by_cell[(decision_hash, company, month)]
                    if not decision.rejected:
                        strict_reasons.append(f"{month}:holm_not_rejected")
                recoveries.append(recovery)
                total_correct_process += correct_process
                total_correct_auto += correct_auto

            empirical_reasons = sorted(set(empirical_reasons))
            strict_reasons = sorted(set((*strict_reasons, *empirical_reasons)))
            score = (
                min(recoveries) if recoveries else 0.0,
                total_correct_process,
                total_correct_auto,
                -int(complexity.get(candidate_id, 0)),
                candidate_id,
            )
            company_scores[candidate_id] = score[:4]
            company_rejections[candidate_id] = tuple(strict_reasons)
            company_shadow_rejections[candidate_id] = tuple(empirical_reasons)
            if not empirical_reasons:
                shadow_candidates.append((score, candidate_id))
            if not strict_reasons:
                strict_candidates.append((score, candidate_id))

        strict_id = max(strict_candidates)[1] if strict_candidates else FALLBACK_POLICY
        shadow_id = max(shadow_candidates)[1] if shadow_candidates else FALLBACK_POLICY
        strict_assignments[company] = strict_id
        shadow_assignments[company] = shadow_id
        if strict_id != FALLBACK_POLICY:
            status = "exploratory_strict_requires_new_prospective_shadow"
        elif shadow_id != FALLBACK_POLICY:
            shadow_strict_reasons = company_rejections[shadow_id]
            status = (
                "exploratory_shadow_insufficient_support"
                if "insufficient_support" in shadow_strict_reasons
                else "exploratory_shadow_holm_not_rejected"
            )
        else:
            status = "forced_review_rejected"
        status_by_company[company] = status
        rejections[company] = MappingProxyType(company_rejections)
        shadow_rejections[company] = MappingProxyType(company_shadow_rejections)
        scores[company] = MappingProxyType(company_scores)

    return PowerSelectionResult(
        registry_sha256=registry.registry_sha256,
        p_values=tuple(p_values),
        adjusted_decisions=adjusted,
        strict_assignments=MappingProxyType(strict_assignments),
        shadow_assignments=MappingProxyType(shadow_assignments),
        status_by_company=MappingProxyType(status_by_company),
        rejections=MappingProxyType(rejections),
        shadow_rejections=MappingProxyType(shadow_rejections),
        scores=MappingProxyType(scores),
    )


def build_structural_power_preflight(
    registry: FrozenPowerRegistry,
) -> Mapping[str, Any]:
    if not isinstance(registry, FrozenPowerRegistry):
        raise TypeError("registry must be a FrozenPowerRegistry")
    unadjusted_minimum = _minimum_zero_event_episodes(FAMILYWISE_ALPHA, RISK_BUDGET)
    first_holm_minimum = (
        _minimum_zero_event_episodes(
            FAMILYWISE_ALPHA / registry.hypothesis_count,
            RISK_BUDGET,
        )
        if registry.hypothesis_count
        else None
    )
    excluded_pairs = []
    for decision_hash, aliases in registry.decision_classes.items():
        for company in registry.present_companies:
            pair = (decision_hash, company)
            if pair in registry.eligible_pairs:
                continue
            excluded_pairs.append(
                {
                    "decisionHash": decision_hash,
                    "candidateIds": list(aliases),
                    "company": company,
                    "rowCountByMonth": dict(registry.row_count_by_pair[pair]),
                    "processEpisodeSupportByMonth": dict(
                        registry.support_by_pair[pair]
                    ),
                    "reason": "insufficient_support",
                }
            )
    first_holm_capable = (
        sum(
            hypothesis.process_episodes >= first_holm_minimum
            for hypothesis in registry.hypotheses
        )
        if first_holm_minimum is not None
        else 0
    )
    return MappingProxyType(
        {
            "schemaVersion": 1,
            "kind": "power_selector_structural_preflight",
            "protocolStatus": "exploratory_historical_replay_only",
            "outcomesAccessed": False,
            "selectorMonths": list(registry.selector_months),
            "riskBudget": registry.risk_budget,
            "familywiseAlpha": registry.familywise_alpha,
            "endpointFamily": [registry.endpoint],
            "minimumProcessEpisodesPerMonth": registry.minimum_process_episodes_per_month,
            "zeroEventMinimumEpisodesUnadjusted": unadjusted_minimum,
            "zeroEventMinimumEpisodesAtFirstHolmThreshold": first_holm_minimum,
            "rawCandidateCount": registry.raw_candidate_count,
            "uniqueDecisionFunctionCount": registry.unique_decision_function_count,
            "expectedCompanyCount": len(registry.expected_companies),
            "selectorPresentCompanyCount": len(registry.present_companies),
            "absentCompanies": list(registry.absent_companies),
            "eligibleDecisionCompanyPairCount": len(registry.eligible_pairs),
            "excludedDecisionCompanyPairs": excluded_pairs,
            "mAll": registry.hypothesis_count,
            "zeroEventFirstHolmCapableHypothesisCount": first_holm_capable,
            "registrySha256": registry.registry_sha256,
        }
    )


def persist_frozen_power_preflight(
    output_dir: str | Path,
    *,
    registry: FrozenPowerRegistry,
    preflight: Mapping[str, Any],
) -> Mapping[str, Mapping[str, str]]:
    """Persist the label-free registry and preflight before outcomes are attached."""

    if preflight.get("outcomesAccessed") is not False:
        raise ValueError(
            "structural preflight must be persisted before outcomes are accessed"
        )
    if preflight.get("registrySha256") != registry.registry_sha256:
        raise ValueError("structural preflight does not match the frozen registry")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = _persist_payloads(
        destination,
        {
            "structuralPowerPreflight": (
                "structural_power_preflight.json",
                dict(preflight),
            ),
            "hypothesisRegistry": (
                "hypothesis_registry.json",
                _registry_payload_from_registry(registry),
            ),
        },
    )
    return MappingProxyType(artifacts)


def persist_power_selector_artifacts(
    output_dir: str | Path,
    *,
    registry: FrozenPowerRegistry,
    preflight: Mapping[str, Any],
    result: PowerSelectionResult,
    source_output_sha256: str,
) -> Mapping[str, Any]:
    """Persist the complete exploratory audit trail and return its summary."""

    _validate_sha256(source_output_sha256, "source_output_sha256")
    if result.registry_sha256 != registry.registry_sha256:
        raise ValueError("selection result does not match the frozen power registry")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = dict(
        persist_frozen_power_preflight(
            destination,
            registry=registry,
            preflight=preflight,
        )
    )
    payloads = {
        "pValues": (
            "p_values.json",
            {
                "schemaVersion": 1,
                "kind": "power_selector_exact_p_values",
                "registrySha256": registry.registry_sha256,
                "riskBudget": registry.risk_budget,
                "endpointFamily": [registry.endpoint],
                "mAll": registry.hypothesis_count,
                "pValues": [_p_value_payload(item) for item in result.p_values],
            },
        ),
        "adjustedDecisions": (
            "adjusted_decisions.json",
            {
                "schemaVersion": 1,
                "kind": "power_selector_holm_decisions",
                "registrySha256": registry.registry_sha256,
                "familywiseAlpha": registry.familywise_alpha,
                "mAll": registry.hypothesis_count,
                "decisions": [
                    _adjusted_decision_payload(item)
                    for item in result.adjusted_decisions
                ],
            },
        ),
        "assignments": (
            "assignments.json",
            {
                "schemaVersion": 1,
                "kind": "power_selector_assignments",
                "protocolStatus": "exploratory_historical_replay_only",
                "promotionEligible": False,
                "fallbackPolicy": FALLBACK_POLICY,
                "strictAssignments": dict(result.strict_assignments),
                "shadowAssignments": dict(result.shadow_assignments),
                "statusByCompany": dict(result.status_by_company),
                "rejectionReasons": _jsonable(result.rejections),
                "shadowRejectionReasons": _jsonable(result.shadow_rejections),
                "scores": _jsonable(result.scores),
            },
        ),
    }
    artifacts.update(_persist_payloads(destination, payloads))

    summary: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "power_adjusted_exploratory_selector",
        "protocolStatus": "exploratory_historical_replay_only",
        "promotionEligible": False,
        "selectorMonths": list(registry.selector_months),
        "riskBudget": registry.risk_budget,
        "familywiseAlpha": registry.familywise_alpha,
        "minimumProcessEpisodesPerMonth": registry.minimum_process_episodes_per_month,
        "endpointFamily": [registry.endpoint],
        "mAll": registry.hypothesis_count,
        "registrySha256": registry.registry_sha256,
        "sourceOutputSha256": source_output_sha256,
        "artifacts": artifacts,
    }
    summary["resultSha256"] = hashlib.sha256(_canonical_json_bytes(summary)).hexdigest()
    _atomic_write(
        destination / "power_adjusted_selector.json", _canonical_json_bytes(summary)
    )
    return MappingProxyType(summary)


def _coerce_process_decision(
    item: ProcessDecision | Mapping[str, Any],
) -> ProcessDecision:
    if isinstance(item, ProcessDecision):
        result = item
    elif isinstance(item, Mapping):
        try:
            result = ProcessDecision(
                candidate_id=str(item["candidate_id"]),
                row_id=str(item["row_id"]),
                episode_id=str(item["episode_id"]),
                company=str(item["company"]),
                month=str(item["month"]),
                predicted=str(item["predicted"]),
                baseline_predicted=str(item.get("baseline_predicted") or "REVIEW"),
            )
        except KeyError as error:
            raise ValueError("process decision mapping is incomplete") from error
    else:
        raise TypeError("process decisions must be ProcessDecision values or mappings")
    if not all(
        (
            result.candidate_id,
            result.row_id,
            result.episode_id,
            result.company,
            result.month,
            result.predicted,
        )
    ):
        raise ValueError(
            "process decisions require non-empty identity and prediction fields"
        )
    return result


def _coerce_outcome(item: CandidateDecision | Mapping[str, Any]) -> CandidateDecision:
    if isinstance(item, CandidateDecision):
        result = item
    elif isinstance(item, Mapping):
        try:
            result = CandidateDecision(
                candidate_id=str(item["candidate_id"]),
                row_id=str(item["row_id"]),
                episode_id=str(item["episode_id"]),
                company=str(item["company"]),
                month=str(item["month"]),
                actual=str(item["actual"]),
                predicted=str(item["predicted"]),
            )
        except KeyError as error:
            raise ValueError("outcome mapping is incomplete") from error
    else:
        raise TypeError("outcomes must be CandidateDecision values or mappings")
    if result.actual not in ACTION_LABELS:
        raise ValueError(
            f"outcome contains an unsupported actual label: {result.actual}"
        )
    return result


def _validate_outcomes_against_registry(
    registry: FrozenPowerRegistry,
    outcomes: tuple[CandidateDecision, ...],
) -> dict[str, dict[str, CandidateDecision]]:
    frozen: dict[str, dict[str, ProcessDecision]] = {}
    for item in registry._process_decisions:
        frozen.setdefault(item.candidate_id, {})[item.row_id] = item
    materialized: dict[str, dict[str, CandidateDecision]] = {}
    for item in outcomes:
        candidate_rows = materialized.setdefault(item.candidate_id, {})
        if item.row_id in candidate_rows:
            raise ValueError(
                f"duplicate outcome row: {item.candidate_id}/{item.row_id}"
            )
        candidate_rows[item.row_id] = item
    if set(materialized) != set(frozen):
        raise ValueError("outcome candidate family differs from the frozen registry")
    for candidate_id, frozen_rows in frozen.items():
        outcome_rows = materialized[candidate_id]
        if set(outcome_rows) != set(frozen_rows):
            raise ValueError("outcome row universe differs from the frozen registry")
        for row_id, item in outcome_rows.items():
            source = frozen_rows[row_id]
            if (
                item.episode_id,
                item.company,
                item.month,
                item.predicted,
            ) != (
                source.episode_id,
                source.company,
                source.month,
                source.predicted,
            ):
                raise ValueError(
                    f"outcome process decision changed after freeze: {row_id}"
                )
    return materialized


def _process_function_hash(
    rows: Iterable[ProcessDecision],
    *,
    selection_mode: str,
) -> str:
    payload = [
        [row.row_id, int(_is_added_process(row, selection_mode))]
        for row in sorted(rows, key=lambda item: item.row_id)
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _is_added_process(row: ProcessDecision, selection_mode: str) -> bool:
    return row.predicted == PROCESS_LABEL and (
        selection_mode == "standalone_clean"
        or row.baseline_predicted not in ACTION_LABELS
    )


def _summarize_risk_cell(
    rows: Iterable[CandidateDecision],
    *,
    frozen_rows: Mapping[str, ProcessDecision],
    selection_mode: str,
) -> _RiskCell:
    by_episode: dict[str, set[str]] = {}
    for row in rows:
        if _is_added_process(frozen_rows[row.row_id], selection_mode):
            by_episode.setdefault(row.episode_id, set()).add(row.actual)
    return _RiskCell(
        process_episodes=len(by_episode),
        unsafe_episodes=sum(
            bool(actuals & {SHIP_LABEL, REPAIR_LABEL})
            for actuals in by_episode.values()
        ),
        ship_to_process_episodes=sum(
            SHIP_LABEL in actuals for actuals in by_episode.values()
        ),
        repair_to_process_episodes=sum(
            REPAIR_LABEL in actuals for actuals in by_episode.values()
        ),
    )


def _minimum_zero_event_episodes(alpha: float, q: float) -> int:
    return math.ceil(math.log(alpha) / math.log1p(-q))


def _registry_payload_from_registry(registry: FrozenPowerRegistry) -> dict[str, Any]:
    return _registry_payload(
        registry_sha256=registry.registry_sha256,
        source_selector_sha256=registry.source_selector_sha256,
        selector_months=registry.selector_months,
        expected_companies=registry.expected_companies,
        present_companies=registry.present_companies,
        absent_companies=registry.absent_companies,
        candidate_to_decision_hash=registry.candidate_to_decision_hash,
        decision_classes=registry.decision_classes,
        support_by_pair=registry.support_by_pair,
        row_count_by_pair=registry.row_count_by_pair,
        eligible_pairs=registry.eligible_pairs,
        hypotheses=registry.hypotheses,
        selection_mode=registry.selection_mode,
    )


def _registry_payload(
    *,
    registry_sha256: str | None,
    source_selector_sha256: str,
    selector_months: tuple[str, ...],
    expected_companies: tuple[str, ...],
    present_companies: tuple[str, ...],
    absent_companies: tuple[str, ...],
    candidate_to_decision_hash: Mapping[str, str],
    decision_classes: Mapping[str, tuple[str, ...]],
    support_by_pair: Mapping[tuple[str, str], Mapping[str, int]],
    row_count_by_pair: Mapping[tuple[str, str], Mapping[str, int]],
    eligible_pairs: tuple[tuple[str, str], ...],
    hypotheses: tuple[PowerHypothesis, ...],
    selection_mode: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "power_selector_hypothesis_registry",
        "protocolStatus": "exploratory_historical_replay_only",
        "labelsUsed": False,
        "selectionMode": selection_mode,
        "sourceSelectorSha256": source_selector_sha256,
        "selectorMonths": list(selector_months),
        "riskBudget": RISK_BUDGET,
        "familywiseAlpha": FAMILYWISE_ALPHA,
        "minimumProcessEpisodesPerMonth": MIN_PROCESS_EPISODES_PER_MONTH,
        "endpointFamily": [ENDPOINT],
        "expectedCompanies": list(expected_companies),
        "presentCompanies": list(present_companies),
        "absentCompanies": list(absent_companies),
        "rawCandidateCount": len(candidate_to_decision_hash),
        "uniqueDecisionFunctionCount": len(decision_classes),
        "candidateToDecisionHash": dict(sorted(candidate_to_decision_hash.items())),
        "decisionClasses": [
            {
                "decisionHash": decision_hash,
                "candidateIds": list(aliases),
            }
            for decision_hash, aliases in sorted(decision_classes.items())
        ],
        "decisionCompanySupport": [
            {
                "decisionHash": decision_hash,
                "candidateIds": list(decision_classes[decision_hash]),
                "company": company,
                "rowCountByMonth": dict(row_count_by_pair[(decision_hash, company)]),
                "processEpisodeSupportByMonth": dict(
                    support_by_pair[(decision_hash, company)]
                ),
                "eligible": (decision_hash, company) in eligible_pairs,
            }
            for decision_hash, company in sorted(support_by_pair)
        ],
        "hypotheses": [_hypothesis_payload(item) for item in hypotheses],
        "mAll": len(hypotheses),
    }
    if registry_sha256 is not None:
        payload["registrySha256"] = registry_sha256
    return payload


def _hypothesis_payload(item: PowerHypothesis) -> dict[str, Any]:
    return {
        "hypothesisId": item.hypothesis_id,
        "decisionHash": item.decision_hash,
        "candidateIds": list(item.candidate_ids),
        "company": item.company,
        "month": item.month,
        "endpoint": item.endpoint,
        "riskBudget": item.risk_budget,
        "processEpisodes": item.process_episodes,
    }


def _p_value_payload(item: PValueRecord) -> dict[str, Any]:
    return {
        "hypothesisId": item.hypothesis_id,
        "decisionHash": item.decision_hash,
        "candidateIds": list(item.candidate_ids),
        "company": item.company,
        "month": item.month,
        "endpoint": item.endpoint,
        "riskBudget": item.risk_budget,
        "processEpisodes": item.process_episodes,
        "unsafeEpisodes": item.unsafe_episodes,
        "shipToProcessEpisodes": item.ship_to_process_episodes,
        "repairToProcessEpisodes": item.repair_to_process_episodes,
        "rawPValue": item.raw_p_value,
    }


def _adjusted_decision_payload(item: AdjustedDecision) -> dict[str, Any]:
    return {
        "hypothesisId": item.hypothesis_id,
        "decisionHash": item.decision_hash,
        "candidateIds": list(item.candidate_ids),
        "company": item.company,
        "month": item.month,
        "endpoint": item.endpoint,
        "processEpisodes": item.process_episodes,
        "unsafeEpisodes": item.unsafe_episodes,
        "rawPValue": item.raw_p_value,
        "rank": item.rank,
        "holmThreshold": item.holm_threshold,
        "adjustedPValue": item.adjusted_p_value,
        "rejected": item.rejected,
    }


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _persist_payloads(
    destination: Path,
    payloads: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for name, (filename, payload) in payloads.items():
        data = _canonical_json_bytes(payload)
        _atomic_write(destination / filename, data)
        artifacts[name] = {
            "path": filename,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return artifacts

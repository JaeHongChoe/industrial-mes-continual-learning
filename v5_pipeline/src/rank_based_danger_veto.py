"""Leakage-safe rank-based danger demotion and development selection."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Sequence, cast


PROCESS_LABEL = "PROCESS"
SHIPMENT_LABEL = "RETURN"
REPAIR_LABEL = "REPAIR"
REVIEW_LABEL = ""
ENSEMBLE_SCENARIO = "ensemble"
_MONTH_PATTERN = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _json_ready(self))


@dataclass(frozen=True, slots=True)
class RankDangerScoreRow(_Serializable):
    row_id: str
    month: str
    shipment_score: float
    repair_score: float
    company: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.row_id, str) or not self.row_id.strip():
            raise ValueError("row_id is required")
        if not isinstance(self.month, str) or _MONTH_PATTERN.fullmatch(self.month) is None:
            raise ValueError("month must use canonical YYYY-MM format")
        if self.company is not None and (
            not isinstance(self.company, str) or not self.company.strip()
        ):
            raise ValueError("company must be non-empty when provided")
        _validate_finite_number(self.shipment_score, "shipment_score")
        _validate_finite_number(self.repair_score, "repair_score")


@dataclass(frozen=True, slots=True)
class RankVetoCandidate(_Serializable):
    candidate_id: str
    shipment_fraction: float
    repair_fraction: float
    scope_by_company: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id is required")
        _validate_fraction(self.shipment_fraction, "shipment_fraction")
        _validate_fraction(self.repair_fraction, "repair_fraction")
        if not isinstance(self.scope_by_company, bool):
            raise ValueError("scope_by_company must be boolean")


@dataclass(frozen=True, slots=True)
class RankScopeEvidence(_Serializable):
    month: str
    company: str | None
    process_rows: int
    shipment_k: int
    repair_k: int
    shipment_demoted_row_ids: tuple[str, ...]
    repair_demoted_row_ids: tuple[str, ...]
    demoted_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankVetoEvidence(_Serializable):
    candidate: RankVetoCandidate
    rounding: str
    ordering: str
    score_rows: int
    process_rows: int
    scopes: tuple[RankScopeEvidence, ...]
    shipment_demoted_row_ids: tuple[str, ...]
    repair_demoted_row_ids: tuple[str, ...]
    demoted_row_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankVetoApplication(_Serializable):
    predictions: Mapping[str, str]
    evidence: RankVetoEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictions",
            MappingProxyType(dict(sorted(self.predictions.items()))),
        )


@dataclass(frozen=True, slots=True)
class DangerCaps(_Serializable):
    max_shipment_to_process: int
    max_repair_to_process: int
    max_total_danger: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_shipment_to_process", self.max_shipment_to_process),
            ("max_repair_to_process", self.max_repair_to_process),
            ("max_total_danger", self.max_total_danger),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def allows(self, shipment: int, repair: int) -> bool:
        return (
            shipment <= self.max_shipment_to_process
            and repair <= self.max_repair_to_process
            and shipment + repair <= self.max_total_danger
        )


@dataclass(frozen=True, slots=True)
class FoldVetoEvaluation(_Serializable):
    scenario: str
    fold_id: str
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    shipment_to_process: int
    repair_to_process: int
    total_danger: int
    veto: RankVetoEvidence


@dataclass(frozen=True, slots=True)
class ScenarioVetoEvaluation(_Serializable):
    scenario: str
    folds: tuple[FoldVetoEvaluation, ...]
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    shipment_to_process: int
    repair_to_process: int
    total_danger: int
    caps_satisfied: bool


@dataclass(frozen=True, slots=True)
class RankVetoCandidateEvaluation(_Serializable):
    candidate: RankVetoCandidate
    caps: DangerCaps
    scenarios: tuple[ScenarioVetoEvaluation, ...]
    cap_failures: tuple[str, ...]
    eligible: bool
    worst_scenario_correct_process: int
    worst_scenario_danger: int
    total_scenario_danger: int
    evidence_source: str = "development_folds_only"


@dataclass(frozen=True, slots=True)
class RankVetoSelection(_Serializable):
    selected: RankVetoCandidateEvaluation
    evaluations: tuple[RankVetoCandidateEvaluation, ...]
    evidence_source: str = "development_folds_only"


def _validate_finite_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")


def _validate_fraction(value: object, name: str) -> None:
    _validate_finite_number(value, name)
    if not 0.0 <= float(cast(float, value)) <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _top_k(
    rows: Sequence[RankDangerScoreRow],
    *,
    score_name: str,
    fraction: float,
) -> tuple[str, ...]:
    """Return exactly floor(group size * fraction) rows by (-score, row_id)."""

    product = len(rows) * fraction
    nearest_integer = round(product)
    if abs(product - nearest_integer) <= 2 * math.ulp(product):
        product = float(nearest_integer)
    k = math.floor(product)
    ranked = sorted(rows, key=lambda item: (-getattr(item, score_name), item.row_id))
    return tuple(item.row_id for item in ranked[:k])


def apply_rank_based_danger_veto(
    baseline_predictions: Mapping[str, str],
    scores: Sequence[RankDangerScoreRow],
    candidate: RankVetoCandidate,
    *,
    process_label: str = PROCESS_LABEL,
    review_label: str = REVIEW_LABEL,
) -> RankVetoApplication:
    """Demote the exact top fraction of baseline process rows in each scope."""

    score_by_id: dict[str, RankDangerScoreRow] = {}
    for score in scores:
        if score.row_id in score_by_id:
            raise ValueError(f"duplicate danger score row: {score.row_id}")
        score_by_id[score.row_id] = score
    process_ids = {
        row_id
        for row_id, prediction in baseline_predictions.items()
        if prediction == process_label
    }
    score_ids = set(score_by_id)
    missing = sorted(process_ids - score_ids)
    extra = sorted(score_ids - process_ids)
    if missing or extra:
        raise ValueError(
            "process score row universe mismatch: "
            f"missing={missing}; extra={extra}"
        )
    if candidate.scope_by_company:
        missing_company = sorted(
            score.row_id for score in scores if score.company is None
        )
        if missing_company:
            raise ValueError(
                "company is required for company-scoped danger scores: "
                f"row_ids={missing_company}"
            )
    grouped: dict[tuple[str, str | None], list[RankDangerScoreRow]] = defaultdict(list)
    for row_id in process_ids:
        score = score_by_id[row_id]
        scope = (
            score.month,
            score.company if candidate.scope_by_company else None,
        )
        grouped[scope].append(score)

    shipment_demoted: set[str] = set()
    repair_demoted: set[str] = set()
    shipment_ranked_ids: list[str] = []
    repair_ranked_ids: list[str] = []
    scopes: list[RankScopeEvidence] = []
    for (month, company), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        shipment_ids = _top_k(
            rows,
            score_name="shipment_score",
            fraction=candidate.shipment_fraction,
        )
        repair_ids = _top_k(
            rows,
            score_name="repair_score",
            fraction=candidate.repair_fraction,
        )
        union_ids = tuple(sorted(set(shipment_ids) | set(repair_ids)))
        shipment_demoted.update(shipment_ids)
        repair_demoted.update(repair_ids)
        shipment_ranked_ids.extend(shipment_ids)
        repair_ranked_ids.extend(repair_ids)
        scopes.append(
            RankScopeEvidence(
                month=month,
                company=company,
                process_rows=len(rows),
                shipment_k=len(shipment_ids),
                repair_k=len(repair_ids),
                shipment_demoted_row_ids=shipment_ids,
                repair_demoted_row_ids=repair_ids,
                demoted_row_ids=union_ids,
            )
        )

    demoted = shipment_demoted | repair_demoted
    predictions = {
        row_id: review_label
        if row_id in demoted and prediction == process_label
        else prediction
        for row_id, prediction in baseline_predictions.items()
    }
    evidence = RankVetoEvidence(
        candidate=candidate,
        rounding="floor(n * fraction)",
        ordering="per-head (-score, row_id); union row_id",
        score_rows=len(scores),
        process_rows=len(process_ids),
        scopes=tuple(scopes),
        shipment_demoted_row_ids=tuple(shipment_ranked_ids),
        repair_demoted_row_ids=tuple(repair_ranked_ids),
        demoted_row_ids=tuple(sorted(demoted)),
    )
    return RankVetoApplication(predictions=predictions, evidence=evidence)


def _validate_development_universe(
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    scores_by_scenario: Mapping[
        str,
        Mapping[str, Sequence[RankDangerScoreRow]],
    ],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    fold_ids = tuple(sorted(baseline_by_fold))
    if not fold_ids:
        raise ValueError("development folds must not be empty")
    if set(fold_ids) != set(labels_by_fold):
        raise ValueError("development fold universe mismatch between predictions and labels")

    if any(not isinstance(name, str) or not name.strip() for name in scores_by_scenario):
        raise ValueError("development scenario names must be non-empty strings")
    scenarios = tuple(sorted(scores_by_scenario))
    if ENSEMBLE_SCENARIO not in scenarios or len(scenarios) < 2:
        raise ValueError("development scores require the ensemble and at least one seed")
    expected_folds = set(fold_ids)
    for scenario in scenarios:
        if set(scores_by_scenario[scenario]) != expected_folds:
            raise ValueError(
                "development fold universe mismatch for scenario "
                f"{scenario}"
            )

    seen_row_ids: set[str] = set()
    for fold_id in fold_ids:
        baseline = baseline_by_fold[fold_id]
        labels = labels_by_fold[fold_id]
        baseline_ids = set(baseline)
        label_ids = set(labels)
        if baseline_ids != label_ids:
            missing = sorted(baseline_ids - label_ids)
            extra = sorted(label_ids - baseline_ids)
            raise ValueError(
                f"prediction universe mismatch for fold {fold_id}: "
                f"missing_labels={missing}; extra_labels={extra}"
            )
        overlap = seen_row_ids & baseline_ids
        if overlap:
            raise ValueError(
                "development folds contain duplicate prediction row IDs: "
                f"{sorted(overlap)}"
            )
        seen_row_ids.update(baseline_ids)
        unknown_labels = set(labels.values()) - {
            PROCESS_LABEL,
            SHIPMENT_LABEL,
            REPAIR_LABEL,
        }
        if unknown_labels:
            raise ValueError(
                f"unknown actual labels for fold {fold_id}: {sorted(unknown_labels)}"
            )
    return fold_ids, scenarios


def _score_fold(
    *,
    scenario: str,
    fold_id: str,
    baseline: Mapping[str, str],
    labels: Mapping[str, str],
    scores: Sequence[RankDangerScoreRow],
    candidate: RankVetoCandidate,
    process_label: str,
    review_label: str,
) -> FoldVetoEvaluation:
    application = apply_rank_based_danger_veto(
        baseline,
        scores,
        candidate,
        process_label=process_label,
        review_label=review_label,
    )
    predictions = application.predictions
    actual_process = sum(actual == process_label for actual in labels.values())
    correct_process = sum(
        labels[row_id] == process_label and predictions[row_id] == process_label
        for row_id in labels
    )
    shipment_to_process = sum(
        labels[row_id] == SHIPMENT_LABEL and predictions[row_id] == process_label
        for row_id in labels
    )
    repair_to_process = sum(
        labels[row_id] == REPAIR_LABEL and predictions[row_id] == process_label
        for row_id in labels
    )
    return FoldVetoEvaluation(
        scenario=scenario,
        fold_id=fold_id,
        evaluation_rows=len(labels),
        actual_process_rows=actual_process,
        correct_process_rows=correct_process,
        shipment_to_process=shipment_to_process,
        repair_to_process=repair_to_process,
        total_danger=shipment_to_process + repair_to_process,
        veto=application.evidence,
    )


def evaluate_rank_veto_candidate(
    candidate: RankVetoCandidate,
    *,
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    scores_by_scenario: Mapping[
        str,
        Mapping[str, Sequence[RankDangerScoreRow]],
    ],
    caps: DangerCaps,
    process_label: str = PROCESS_LABEL,
    review_label: str = REVIEW_LABEL,
) -> RankVetoCandidateEvaluation:
    """Evaluate one candidate using development folds, never future labels."""

    fold_ids, scenarios = _validate_development_universe(
        baseline_by_fold,
        labels_by_fold,
        scores_by_scenario,
    )
    scenario_evaluations: list[ScenarioVetoEvaluation] = []
    for scenario in scenarios:
        folds = tuple(
            _score_fold(
                scenario=scenario,
                fold_id=fold_id,
                baseline=baseline_by_fold[fold_id],
                labels=labels_by_fold[fold_id],
                scores=scores_by_scenario[scenario][fold_id],
                candidate=candidate,
                process_label=process_label,
                review_label=review_label,
            )
            for fold_id in fold_ids
        )
        shipment_to_process = sum(item.shipment_to_process for item in folds)
        repair_to_process = sum(item.repair_to_process for item in folds)
        scenario_evaluations.append(
            ScenarioVetoEvaluation(
                scenario=scenario,
                folds=folds,
                evaluation_rows=sum(item.evaluation_rows for item in folds),
                actual_process_rows=sum(item.actual_process_rows for item in folds),
                correct_process_rows=sum(item.correct_process_rows for item in folds),
                shipment_to_process=shipment_to_process,
                repair_to_process=repair_to_process,
                total_danger=shipment_to_process + repair_to_process,
                caps_satisfied=caps.allows(
                    shipment_to_process,
                    repair_to_process,
                ),
            )
        )

    materialized = tuple(scenario_evaluations)
    failures = tuple(item.scenario for item in materialized if not item.caps_satisfied)
    return RankVetoCandidateEvaluation(
        candidate=candidate,
        caps=caps,
        scenarios=materialized,
        cap_failures=failures,
        eligible=not failures,
        worst_scenario_correct_process=min(
            item.correct_process_rows for item in materialized
        ),
        worst_scenario_danger=max(item.total_danger for item in materialized),
        total_scenario_danger=sum(item.total_danger for item in materialized),
    )


def _candidate_selection_key(
    evaluation: RankVetoCandidateEvaluation,
) -> tuple[int | Fraction | str, ...]:
    shipment_fraction = Fraction(str(evaluation.candidate.shipment_fraction))
    repair_fraction = Fraction(str(evaluation.candidate.repair_fraction))
    return (
        -evaluation.worst_scenario_correct_process,
        evaluation.worst_scenario_danger,
        evaluation.total_scenario_danger,
        shipment_fraction + repair_fraction,
        shipment_fraction,
        repair_fraction,
        evaluation.candidate.candidate_id,
    )


def select_rank_veto_candidate(
    candidates: Sequence[RankVetoCandidate],
    *,
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    scores_by_scenario: Mapping[
        str,
        Mapping[str, Sequence[RankDangerScoreRow]],
    ],
    caps: DangerCaps,
    process_label: str = PROCESS_LABEL,
    review_label: str = REVIEW_LABEL,
) -> RankVetoSelection:
    """Select only from candidates passing caps for ensemble and every seed."""

    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("rank veto candidates must not be empty")
    candidate_ids = [candidate.candidate_id for candidate in materialized]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("rank veto candidates contain duplicate candidate IDs")

    evaluations = tuple(
        evaluate_rank_veto_candidate(
            candidate,
            baseline_by_fold=baseline_by_fold,
            labels_by_fold=labels_by_fold,
            scores_by_scenario=scores_by_scenario,
            caps=caps,
            process_label=process_label,
            review_label=review_label,
        )
        for candidate in sorted(materialized, key=lambda item: item.candidate_id)
    )
    eligible = tuple(item for item in evaluations if item.eligible)
    if not eligible:
        raise ValueError(
            "no rank veto candidate satisfies caps for the ensemble and every seed"
        )
    return RankVetoSelection(
        selected=min(eligible, key=_candidate_selection_key),
        evaluations=evaluations,
    )

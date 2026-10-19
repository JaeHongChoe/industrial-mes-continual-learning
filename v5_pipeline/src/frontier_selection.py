from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from .baseline_risk_veto import DangerScoreRow, apply_danger_veto
from .data_contract import ACTION_LABELS


@dataclass(frozen=True, slots=True)
class ThresholdCandidate:
    candidate_id: str
    shipment_threshold: float
    repair_threshold: float
    complexity: int = 0

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id is required")
        for name, value in (
            ("shipment_threshold", self.shipment_threshold),
            ("repair_threshold", self.repair_threshold),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if isinstance(self.complexity, bool) or self.complexity < 0:
            raise ValueError("complexity must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_id: str
    evaluation_rows: int
    auto_rows: int
    correct_auto_rows: int
    actual_process_rows: int
    correct_process_rows: int
    ship_to_process: int
    repair_to_process: int

    def __post_init__(self) -> None:
        if not str(self.fold_id).strip():
            raise ValueError("fold_id is required")
        counts = (
            self.evaluation_rows,
            self.auto_rows,
            self.correct_auto_rows,
            self.actual_process_rows,
            self.correct_process_rows,
            self.ship_to_process,
            self.repair_to_process,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("fold metric counts must be non-negative integers")
        if self.auto_rows > self.evaluation_rows:
            raise ValueError("auto_rows exceeds evaluation_rows")
        if self.correct_auto_rows > self.auto_rows:
            raise ValueError("correct_auto_rows exceeds auto_rows")
        if self.actual_process_rows > self.evaluation_rows:
            raise ValueError("actual_process_rows exceeds evaluation_rows")
        if self.correct_process_rows > self.actual_process_rows:
            raise ValueError("correct_process_rows exceeds actual_process_rows")

    @property
    def process_automation_rate(self) -> float:
        return _ratio(self.correct_process_rows, self.actual_process_rows)

    @property
    def automation_rate(self) -> float:
        return _ratio(self.auto_rows, self.evaluation_rows)

    @property
    def accuracy(self) -> float:
        return _ratio(self.correct_auto_rows, self.auto_rows)

    @property
    def danger_errors(self) -> int:
        return self.ship_to_process + self.repair_to_process


@dataclass(frozen=True, slots=True)
class FrontierRecord:
    candidate: ThresholdCandidate
    fold_metrics: tuple[FoldMetrics, ...]
    evaluation_rows: int
    auto_rows: int
    correct_auto_rows: int
    actual_process_rows: int
    correct_process_rows: int
    ship_to_process: int
    repair_to_process: int
    automation_rate: float
    accuracy: float
    aggregate_process_automation: float
    worst_fold_process_automation: float
    budget_passed: bool = True
    stress_failed: bool = False
    stress_reasons: tuple[str, ...] = ()

    @classmethod
    def from_fold_metrics(
        cls,
        candidate: ThresholdCandidate,
        metrics: Sequence[FoldMetrics],
    ) -> "FrontierRecord":
        materialized = tuple(sorted(metrics, key=lambda item: item.fold_id))
        if not materialized:
            raise ValueError("frontier record requires at least one fold")
        fold_ids = [item.fold_id for item in materialized]
        if len(fold_ids) != len(set(fold_ids)):
            raise ValueError("frontier record contains duplicate fold IDs")
        evaluation_rows = sum(item.evaluation_rows for item in materialized)
        auto_rows = sum(item.auto_rows for item in materialized)
        correct_auto_rows = sum(item.correct_auto_rows for item in materialized)
        actual_process_rows = sum(item.actual_process_rows for item in materialized)
        correct_process_rows = sum(item.correct_process_rows for item in materialized)
        ship_to_process = sum(item.ship_to_process for item in materialized)
        repair_to_process = sum(item.repair_to_process for item in materialized)
        return cls(
            candidate=candidate,
            fold_metrics=materialized,
            evaluation_rows=evaluation_rows,
            auto_rows=auto_rows,
            correct_auto_rows=correct_auto_rows,
            actual_process_rows=actual_process_rows,
            correct_process_rows=correct_process_rows,
            ship_to_process=ship_to_process,
            repair_to_process=repair_to_process,
            automation_rate=_ratio(auto_rows, evaluation_rows),
            accuracy=_ratio(correct_auto_rows, auto_rows),
            aggregate_process_automation=_ratio(correct_process_rows, actual_process_rows),
            worst_fold_process_automation=min(
                item.process_automation_rate for item in materialized
            ),
        )

    @property
    def total_danger_errors(self) -> int:
        return self.ship_to_process + self.repair_to_process


@dataclass(frozen=True, slots=True)
class RiskBudget:
    reference_rows: int = 140_477
    max_ship_to_process: int = 65
    max_repair_to_process: int = 35
    max_total_danger: int = 100
    reserve_fraction: float = 0.9
    monthly_stress_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if isinstance(self.reference_rows, bool) or self.reference_rows <= 0:
            raise ValueError("reference_rows must be a positive integer")
        for name, value in (
            ("max_ship_to_process", self.max_ship_to_process),
            ("max_repair_to_process", self.max_repair_to_process),
            ("max_total_danger", self.max_total_danger),
        ):
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not 0.0 < float(self.reserve_fraction) <= 1.0:
            raise ValueError("reserve_fraction must be greater than zero and at most one")
        if not math.isfinite(float(self.monthly_stress_multiplier)) or float(
            self.monthly_stress_multiplier
        ) <= 0.0:
            raise ValueError("monthly_stress_multiplier must be positive")

    def assess(self, record: FrontierRecord) -> FrontierRecord:
        scale = record.evaluation_rows / self.reference_rows
        ship_limit = math.floor(self.max_ship_to_process * self.reserve_fraction * scale + 1e-12)
        repair_limit = math.floor(
            self.max_repair_to_process * self.reserve_fraction * scale + 1e-12
        )
        total_limit = math.floor(self.max_total_danger * self.reserve_fraction * scale + 1e-12)
        budget_passed = (
            record.ship_to_process <= ship_limit
            and record.repair_to_process <= repair_limit
            and record.total_danger_errors <= total_limit
        )

        stress_reasons: list[str] = []
        for fold in record.fold_metrics:
            fold_scale = fold.evaluation_rows / self.reference_rows
            limits = {
                "ship_to_process": math.ceil(
                    self.max_ship_to_process * self.monthly_stress_multiplier * fold_scale
                ),
                "repair_to_process": math.ceil(
                    self.max_repair_to_process * self.monthly_stress_multiplier * fold_scale
                ),
                "total_danger": math.ceil(
                    self.max_total_danger * self.monthly_stress_multiplier * fold_scale
                ),
            }
            observed = {
                "ship_to_process": fold.ship_to_process,
                "repair_to_process": fold.repair_to_process,
                "total_danger": fold.danger_errors,
            }
            for name in ("ship_to_process", "repair_to_process", "total_danger"):
                if observed[name] > limits[name]:
                    stress_reasons.append(f"{fold.fold_id}:{name}")
        return replace(
            record,
            budget_passed=budget_passed,
            stress_failed=bool(stress_reasons),
            stress_reasons=tuple(stress_reasons),
        )


def pareto_frontier(records: Iterable[FrontierRecord]) -> tuple[FrontierRecord, ...]:
    materialized = tuple(records)
    candidate_ids = [item.candidate.candidate_id for item in materialized]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("frontier contains duplicate candidate IDs")
    frontier = [
        candidate
        for candidate in materialized
        if not any(
            challenger is not candidate and _dominates(challenger, candidate)
            for challenger in materialized
        )
    ]
    return tuple(sorted(frontier, key=_stable_frontier_key))


def select_robust_candidate(records: Iterable[FrontierRecord]) -> FrontierRecord:
    eligible = tuple(
        record for record in records if record.budget_passed and not record.stress_failed
    )
    if not eligible:
        raise ValueError("no frontier candidate satisfies the risk and monthly stress budget")
    return max(eligible, key=_robust_key)


def enumerate_threshold_frontier(
    scores_by_fold: Mapping[str, Sequence[DangerScoreRow]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    grid: Iterable[ThresholdCandidate],
    risk_budget: RiskBudget,
) -> tuple[FrontierRecord, ...]:
    records = score_threshold_grid(
        scores_by_fold,
        labels_by_fold,
        baseline_by_fold,
        grid,
        risk_budget,
    )
    return pareto_frontier(
        record for record in records if record.budget_passed and not record.stress_failed
    )


def score_threshold_grid(
    scores_by_fold: Mapping[str, Sequence[DangerScoreRow]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    grid: Iterable[ThresholdCandidate],
    risk_budget: RiskBudget,
) -> tuple[FrontierRecord, ...]:
    fold_ids = tuple(sorted(scores_by_fold))
    if set(fold_ids) != set(labels_by_fold) or set(fold_ids) != set(baseline_by_fold):
        raise ValueError("threshold frontier fold universes do not match")
    candidates = tuple(sorted(grid, key=lambda item: item.candidate_id))
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("threshold grid contains duplicate candidate IDs")

    records: list[FrontierRecord] = []
    for candidate in candidates:
        fold_metrics: list[FoldMetrics] = []
        for fold_id in fold_ids:
            scores = tuple(scores_by_fold[fold_id])
            labels = labels_by_fold[fold_id]
            baseline = baseline_by_fold[fold_id]
            predictions = apply_danger_veto(
                baseline,
                scores,
                shipment_threshold=candidate.shipment_threshold,
                repair_threshold=candidate.repair_threshold,
            )
            fold_metrics.append(_score_predictions(fold_id, labels, predictions))
        assessed = risk_budget.assess(
            FrontierRecord.from_fold_metrics(candidate, fold_metrics)
        )
        records.append(assessed)
    return tuple(sorted(records, key=lambda item: item.candidate.candidate_id))


def _score_predictions(
    fold_id: str,
    labels: Mapping[str, str],
    predictions: Mapping[str, str],
) -> FoldMetrics:
    if set(labels) != set(predictions):
        raise ValueError(f"prediction universe mismatch for fold {fold_id}")
    unknown_labels = set(labels.values()) - set(ACTION_LABELS)
    if unknown_labels:
        raise ValueError(f"unknown action labels for fold {fold_id}: {sorted(unknown_labels)}")
    unknown_predictions = {
        value for value in predictions.values() if value and value not in ACTION_LABELS
    }
    if unknown_predictions:
        raise ValueError(
            f"unknown predictions for fold {fold_id}: {sorted(unknown_predictions)}"
        )
    auto_ids = {row_id for row_id, value in predictions.items() if value in ACTION_LABELS}
    return FoldMetrics(
        fold_id=fold_id,
        evaluation_rows=len(labels),
        auto_rows=len(auto_ids),
        correct_auto_rows=sum(labels[row_id] == predictions[row_id] for row_id in auto_ids),
        actual_process_rows=sum(value == "PROCESS" for value in labels.values()),
        correct_process_rows=sum(
            actual == "PROCESS" and predictions[row_id] == "PROCESS"
            for row_id, actual in labels.items()
        ),
        ship_to_process=sum(
            actual == "RETURN" and predictions[row_id] == "PROCESS"
            for row_id, actual in labels.items()
        ),
        repair_to_process=sum(
            actual == "REPAIR" and predictions[row_id] == "PROCESS"
            for row_id, actual in labels.items()
        ),
    )


def _dominates(left: FrontierRecord, right: FrontierRecord) -> bool:
    left_max = (
        left.worst_fold_process_automation,
        left.aggregate_process_automation,
        left.automation_rate,
        left.accuracy,
    )
    right_max = (
        right.worst_fold_process_automation,
        right.aggregate_process_automation,
        right.automation_rate,
        right.accuracy,
    )
    left_min = (
        left.ship_to_process,
        left.repair_to_process,
        left.total_danger_errors,
        left.candidate.complexity,
    )
    right_min = (
        right.ship_to_process,
        right.repair_to_process,
        right.total_danger_errors,
        right.candidate.complexity,
    )
    no_worse = all(a >= b for a, b in zip(left_max, right_max)) and all(
        a <= b for a, b in zip(left_min, right_min)
    )
    strictly_better = any(a > b for a, b in zip(left_max, right_max)) or any(
        a < b for a, b in zip(left_min, right_min)
    )
    return no_worse and strictly_better


def _robust_key(record: FrontierRecord) -> tuple[float | int | str, ...]:
    return (
        record.worst_fold_process_automation,
        record.aggregate_process_automation,
        record.automation_rate,
        record.accuracy,
        -record.total_danger_errors,
        -record.candidate.complexity,
        _reverse_lexicographic(record.candidate.candidate_id),
    )


def _stable_frontier_key(record: FrontierRecord) -> tuple[object, ...]:
    return (
        -record.worst_fold_process_automation,
        -record.aggregate_process_automation,
        -record.automation_rate,
        -record.accuracy,
        record.total_danger_errors,
        record.candidate.complexity,
        record.candidate.candidate_id,
    )


def _reverse_lexicographic(value: str) -> str:
    # max() is used for the robust key. Invert code points so the smaller stable
    # candidate ID wins an otherwise exact tie.
    return "".join(chr(0x10FFFF - ord(char)) for char in value)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

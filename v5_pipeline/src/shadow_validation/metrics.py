from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping


ACTION_LABELS = ("PROCESS", "RETURN", "REPAIR")
PROCESS_LABEL = "PROCESS"
REVIEW_LABEL = "REVIEW"
_ACTION_ALIASES = {
    "PROCESS": PROCESS_LABEL,
    "고객사출고": "RETURN",
    "출고": "RETURN",
    "REPAIR": "REPAIR",
    "수리": "REPAIR",
    "수리처리": "REPAIR",
}
_REVIEW_ALIASES = {"", "REVIEW", "검토필요", "REVIEW_SHORT", "리뷰필요"}

MIN_COMPLETE_MONTHS = 2
MIN_EVALUATION_ROWS = 30_000
MIN_ACTUAL_SHIP_ROWS = 100
MIN_ACTUAL_REPAIR_ROWS = 100
MIN_AUTOMATION_RATE = 0.885
MIN_CORRECT_PROCESS_RECALL = 0.915
MIN_AUTO_ACCURACY = 0.995
MAX_SHIP_TO_PROCESS_PER_10K = 4.63
MAX_REPAIR_TO_PROCESS_PER_10K = 2.50
MAX_COMBINED_CRITICAL_PER_10K = 7.12
MIN_COMPANY_SAMPLE_ROWS = 500


class MetricsError(ValueError):
    """Raised when an outcome row cannot be measured under the frozen contract."""


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    row_id: str
    company: str
    month: str
    actual: str
    predicted: str


@dataclass(frozen=True, slots=True)
class Metrics:
    evaluation_rows: int
    auto_rows: int
    review_rows: int
    automation_rate: float | None
    correct_auto_rows: int
    auto_accuracy: float | None
    actual_process_rows: int
    correct_process_rows: int
    correct_process_recall: float | None
    actual_process_automatic_rows: int
    actual_process_automation_rate: float | None
    actual_ship_rows: int
    actual_repair_rows: int
    ship_to_process: int
    repair_to_process: int
    risk_to_process: int
    process_to_ship: int
    process_to_repair: int
    process_to_ng: int
    action_confusion: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    observed: int | float
    threshold: int | float
    comparator: str
    passed: bool


@dataclass(frozen=True, slots=True)
class CompanySample:
    evaluation_rows: int
    status: str


@dataclass(frozen=True, slots=True)
class PromotionGates:
    checks: Mapping[str, GateCheck]
    company_samples: Mapping[str, CompanySample]

    @property
    def eligible(self) -> bool:
        return all(check.passed for check in self.checks.values())


def normalize_action_label(value: object, *, allow_review: bool = False) -> str:
    """Normalize a displayed action while keeping unknown outcomes fail-closed."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    compact = re.sub(r"\s+", "", text).strip()
    if compact in _ACTION_ALIASES:
        return _ACTION_ALIASES[compact]
    if allow_review and compact in _REVIEW_ALIASES:
        return REVIEW_LABEL
    raise MetricsError(f"unknown action label: {text.strip() or '<blank>'}")


def compute_metrics(rows: Iterable[OutcomeRow]) -> Metrics:
    materialized = tuple(rows)
    row_ids = [row.row_id for row in materialized]
    if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise MetricsError("metrics input must contain unique nonblank row IDs")

    normalized = tuple(
        OutcomeRow(
            row_id=row.row_id,
            company=str(row.company or "UNCLASSIFIED"),
            month=str(row.month or ""),
            actual=normalize_action_label(row.actual),
            predicted=normalize_action_label(row.predicted, allow_review=True),
        )
        for row in materialized
    )
    automatic = tuple(row for row in normalized if row.predicted in ACTION_LABELS)
    evaluation_rows = len(normalized)
    auto_rows = len(automatic)
    actual_process_rows = sum(row.actual == PROCESS_LABEL for row in normalized)
    correct_process_rows = sum(
        row.actual == PROCESS_LABEL and row.predicted == PROCESS_LABEL for row in normalized
    )
    ship_to_process = sum(
        row.actual == "RETURN" and row.predicted == PROCESS_LABEL for row in normalized
    )
    repair_to_process = sum(
        row.actual == "REPAIR" and row.predicted == PROCESS_LABEL for row in normalized
    )
    process_to_ship = sum(
        row.actual == PROCESS_LABEL and row.predicted == "RETURN" for row in normalized
    )
    process_to_repair = sum(
        row.actual == PROCESS_LABEL and row.predicted == "REPAIR" for row in normalized
    )
    labels = (*ACTION_LABELS, REVIEW_LABEL)
    confusion = {
        actual: MappingProxyType(
            {
                predicted: sum(
                    row.actual == actual and row.predicted == predicted for row in normalized
                )
                for predicted in labels
            }
        )
        for actual in ACTION_LABELS
    }
    correct_auto_rows = sum(row.actual == row.predicted for row in automatic)
    actual_process_automatic_rows = sum(
        row.actual == PROCESS_LABEL and row.predicted in ACTION_LABELS for row in normalized
    )
    actual_ship_rows = sum(row.actual == "RETURN" for row in normalized)
    actual_repair_rows = sum(row.actual == "REPAIR" for row in normalized)
    return Metrics(
        evaluation_rows=evaluation_rows,
        auto_rows=auto_rows,
        review_rows=evaluation_rows - auto_rows,
        automation_rate=_ratio(auto_rows, evaluation_rows),
        correct_auto_rows=correct_auto_rows,
        auto_accuracy=_ratio(correct_auto_rows, auto_rows),
        actual_process_rows=actual_process_rows,
        correct_process_rows=correct_process_rows,
        correct_process_recall=_ratio(correct_process_rows, actual_process_rows),
        actual_process_automatic_rows=actual_process_automatic_rows,
        actual_process_automation_rate=_ratio(actual_process_automatic_rows, actual_process_rows),
        actual_ship_rows=actual_ship_rows,
        actual_repair_rows=actual_repair_rows,
        ship_to_process=ship_to_process,
        repair_to_process=repair_to_process,
        risk_to_process=ship_to_process + repair_to_process,
        process_to_ship=process_to_ship,
        process_to_repair=process_to_repair,
        process_to_ng=process_to_ship + process_to_repair,
        action_confusion=MappingProxyType(confusion),
    )


def grouped_metrics(rows: Iterable[OutcomeRow], basis: str) -> Mapping[str, Metrics]:
    attribute = {"company": "company", "month": "month"}.get(basis)
    if attribute is None:
        raise MetricsError(f"unsupported metric grouping: {basis}")
    grouped: dict[str, list[OutcomeRow]] = {}
    for row in rows:
        key = str(getattr(row, attribute) or "UNCLASSIFIED")
        grouped.setdefault(key, []).append(row)
    return MappingProxyType(
        {key: compute_metrics(grouped[key]) for key in sorted(grouped)}
    )


def reconcile_group_metrics(overall: Metrics, groups: Mapping[str, Metrics]) -> bool:
    count_fields = (
        "evaluation_rows", "auto_rows", "review_rows", "correct_auto_rows",
        "actual_process_rows", "correct_process_rows", "actual_process_automatic_rows",
        "actual_ship_rows", "actual_repair_rows", "ship_to_process", "repair_to_process",
        "risk_to_process", "process_to_ship", "process_to_repair", "process_to_ng",
    )
    return all(
        getattr(overall, field) == sum(getattr(metric, field) for metric in groups.values())
        for field in count_fields
    )


def evaluate_promotion_gates(
    overall: Metrics,
    monthly_metrics: Mapping[str, Metrics],
    company_metrics: Mapping[str, Metrics],
    *,
    complete_months: Iterable[str],
    unresolved_rows: int = 0,
) -> PromotionGates:
    complete_count = len(set(complete_months))
    return_rate = _per_10k(overall.ship_to_process, overall.evaluation_rows)
    repair_rate = _per_10k(overall.repair_to_process, overall.evaluation_rows)
    combined_rate = _per_10k(overall.risk_to_process, overall.evaluation_rows)
    checks = {
        # Interim reports may contain unresolved labels, but they must never be
        # eligible for promotion on a selectively resolved denominator.
        "unresolved_rows": _at_most("unresolved_rows", unresolved_rows, 0),
        "complete_months": _at_least("complete_months", complete_count, MIN_COMPLETE_MONTHS),
        "evaluation_rows": _at_least("evaluation_rows", overall.evaluation_rows, MIN_EVALUATION_ROWS),
        "actual_ship_rows": _at_least("actual_ship_rows", overall.actual_ship_rows, MIN_ACTUAL_SHIP_ROWS),
        "actual_repair_rows": _at_least("actual_repair_rows", overall.actual_repair_rows, MIN_ACTUAL_REPAIR_ROWS),
        "automation_rate": _at_least("automation_rate", overall.automation_rate, MIN_AUTOMATION_RATE),
        "correct_process_recall": _at_least("correct_process_recall", overall.correct_process_recall, MIN_CORRECT_PROCESS_RECALL),
        "auto_accuracy": _at_least("auto_accuracy", overall.auto_accuracy, MIN_AUTO_ACCURACY),
        "ship_to_process_per_10k": _at_most("ship_to_process_per_10k", return_rate, MAX_SHIP_TO_PROCESS_PER_10K),
        "repair_to_process_per_10k": _at_most("repair_to_process_per_10k", repair_rate, MAX_REPAIR_TO_PROCESS_PER_10K),
        "combined_critical_per_10k": _at_most("combined_critical_per_10k", combined_rate, MAX_COMBINED_CRITICAL_PER_10K),
    }
    samples = {
        company: CompanySample(
            evaluation_rows=metric.evaluation_rows,
            status=("sufficient" if metric.evaluation_rows >= MIN_COMPANY_SAMPLE_ROWS else "insufficient_sample"),
        )
        for company, metric in company_metrics.items()
    }
    return PromotionGates(
        checks=MappingProxyType(checks),
        company_samples=MappingProxyType(samples),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _per_10k(numerator: int, denominator: int) -> float:
    return numerator * 10_000 / denominator if denominator else float("inf")


def _at_least(name: str, observed: int | float | None, threshold: int | float) -> GateCheck:
    value = float("-inf") if observed is None else observed
    return GateCheck(name, value, threshold, ">=", value >= threshold)


def _at_most(name: str, observed: float, threshold: float) -> GateCheck:
    return GateCheck(name, observed, threshold, "<=", observed <= threshold)

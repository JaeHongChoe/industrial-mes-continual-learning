from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .baseline_risk_veto import DangerScoreRow
from .data_contract import ACTION_LABELS


@dataclass(frozen=True, slots=True)
class TransitionCaps:
    ship_to_process: int
    repair_to_process: int
    total: int

    def __post_init__(self) -> None:
        values = (self.ship_to_process, self.repair_to_process, self.total)
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("transition caps must be non-negative integers")


@dataclass(frozen=True, slots=True)
class CompanyThresholdOption:
    company: str
    option_id: str
    shipment_threshold: float
    repair_threshold: float
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int
    ship_to_process: int
    repair_to_process: int
    fold_danger: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        if not str(self.company).strip():
            raise ValueError("company is required")
        if not str(self.option_id).strip():
            raise ValueError("option_id is required")
        for name, value in (
            ("shipment_threshold", self.shipment_threshold),
            ("repair_threshold", self.repair_threshold),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        counts = (
            self.evaluation_rows,
            self.actual_process_rows,
            self.correct_process_rows,
            self.auto_rows,
            self.correct_auto_rows,
            self.ship_to_process,
            self.repair_to_process,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("company option counts must be non-negative integers")
        if self.actual_process_rows > self.evaluation_rows:
            raise ValueError("actual process rows exceed evaluation rows")
        if self.correct_process_rows > self.actual_process_rows:
            raise ValueError("correct process rows exceed actual process rows")
        if self.auto_rows > self.evaluation_rows:
            raise ValueError("automatic rows exceed evaluation rows")
        if self.correct_auto_rows > self.auto_rows:
            raise ValueError("correct automatic rows exceed automatic rows")
        for fold, danger in self.fold_danger.items():
            if not str(fold).strip() or len(danger) != 2:
                raise ValueError("fold danger entries require a fold and two counts")
            if any(isinstance(value, bool) or value < 0 for value in danger):
                raise ValueError("fold danger counts must be non-negative integers")

    @property
    def total_danger(self) -> int:
        return self.ship_to_process + self.repair_to_process


@dataclass(frozen=True, slots=True)
class CompanyThresholdPolicy:
    choices: Mapping[str, CompanyThresholdOption]
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int
    ship_to_process: int
    repair_to_process: int

    @property
    def total_danger(self) -> int:
        return self.ship_to_process + self.repair_to_process


@dataclass(frozen=True, slots=True)
class _PlanState:
    evaluation_rows: int
    actual_process_rows: int
    correct_process_rows: int
    auto_rows: int
    correct_auto_rows: int
    option_ids: tuple[str, ...]


def build_company_threshold_options(
    *,
    scores_by_fold: Mapping[str, Sequence[DangerScoreRow]],
    labels_by_fold: Mapping[str, Mapping[str, str]],
    baseline_by_fold: Mapping[str, Mapping[str, str]],
    company_by_row: Mapping[str, str],
    threshold_grid: Iterable[tuple[str, float, float]],
    fallback_option_id: str,
    eligible_companies: set[str] | frozenset[str] | None = None,
) -> Mapping[str, tuple[CompanyThresholdOption, ...]]:
    """Aggregate per-company threshold evidence over chronological folds."""

    fold_ids = tuple(sorted(scores_by_fold))
    if not fold_ids or set(fold_ids) != set(labels_by_fold) or set(fold_ids) != set(
        baseline_by_fold
    ):
        raise ValueError("company threshold fold universes do not match")
    row_ids: set[str] = set()
    for fold_id in fold_ids:
        labels = labels_by_fold[fold_id]
        baseline = baseline_by_fold[fold_id]
        if set(labels) != set(baseline):
            raise ValueError(f"label and baseline row universes differ: {fold_id}")
        if row_ids & set(labels):
            raise ValueError("company threshold folds contain overlapping row IDs")
        row_ids.update(labels)
        unknown_labels = set(labels.values()) - set(ACTION_LABELS)
        if unknown_labels:
            raise ValueError(f"unknown action labels in {fold_id}: {sorted(unknown_labels)}")
        score_ids: set[str] = set()
        for score in scores_by_fold[fold_id]:
            if score.row_id in score_ids:
                raise ValueError(f"duplicate danger score in {fold_id}: {score.row_id}")
            score_ids.add(score.row_id)
            if score.row_id not in baseline:
                raise ValueError(f"danger score is outside fold {fold_id}: {score.row_id}")
            if baseline[score.row_id] != "PROCESS":
                raise ValueError("danger scores may only target baseline process approvals")
    if set(company_by_row) != row_ids:
        raise ValueError("company mapping must match the exact development row universe")

    grid = tuple(
        (str(option_id), float(ship), float(repair))
        for option_id, ship, repair in threshold_grid
    )
    option_ids = [item[0] for item in grid]
    if not grid or len(option_ids) != len(set(option_ids)):
        raise ValueError("threshold grid must have unique option IDs")
    if fallback_option_id not in set(option_ids):
        raise ValueError("fallback option is absent from the threshold grid")
    eligible = set(company_by_row.values()) if eligible_companies is None else set(
        eligible_companies
    )

    companies = tuple(sorted(set(company_by_row.values())))
    company_ids_by_fold: dict[str, dict[str, tuple[str, ...]]] = {}
    company_scores_by_fold: dict[str, dict[str, tuple[DangerScoreRow, ...]]] = {}
    for fold_id in fold_ids:
        ids_index: dict[str, list[str]] = {company: [] for company in companies}
        score_index: dict[str, list[DangerScoreRow]] = {
            company: [] for company in companies
        }
        for row_id in labels_by_fold[fold_id]:
            ids_index[company_by_row[row_id]].append(row_id)
        for score in scores_by_fold[fold_id]:
            score_index[company_by_row[score.row_id]].append(score)
        company_ids_by_fold[fold_id] = {
            company: tuple(values) for company, values in ids_index.items()
        }
        company_scores_by_fold[fold_id] = {
            company: tuple(values) for company, values in score_index.items()
        }

    output: dict[str, tuple[CompanyThresholdOption, ...]] = {}
    for company in companies:
        company_grid = (
            grid
            if company in eligible
            else tuple(item for item in grid if item[0] == fallback_option_id)
        )
        options: list[CompanyThresholdOption] = []
        for option_id, ship_threshold, repair_threshold in company_grid:
            evaluation_rows = 0
            actual_process_rows = 0
            correct_process_rows = 0
            auto_rows = 0
            correct_auto_rows = 0
            ship_to_process = 0
            repair_to_process = 0
            fold_danger: dict[str, tuple[int, int]] = {}
            for fold_id in fold_ids:
                labels = labels_by_fold[fold_id]
                baseline = baseline_by_fold[fold_id]
                company_ids = company_ids_by_fold[fold_id][company]
                score_rows = company_scores_by_fold[fold_id][company]
                predictions = apply_company_threshold_policy(
                    {row_id: baseline[row_id] for row_id in company_ids},
                    score_rows,
                    {row_id: company for row_id in company_ids},
                    {
                        company: _threshold_only_option(
                            company,
                            option_id,
                            ship_threshold,
                            repair_threshold,
                        )
                    },
                )
                automatic = {
                    row_id for row_id, value in predictions.items() if value in ACTION_LABELS
                }
                fold_ship = sum(
                    labels[row_id] == "RETURN"
                    and predictions[row_id] == "PROCESS"
                    for row_id in company_ids
                )
                fold_repair = sum(
                    labels[row_id] == "REPAIR"
                    and predictions[row_id] == "PROCESS"
                    for row_id in company_ids
                )
                fold_danger[fold_id] = (fold_ship, fold_repair)
                evaluation_rows += len(company_ids)
                actual_process_rows += sum(
                    labels[row_id] == "PROCESS" for row_id in company_ids
                )
                correct_process_rows += sum(
                    labels[row_id] == "PROCESS"
                    and predictions[row_id] == "PROCESS"
                    for row_id in company_ids
                )
                auto_rows += len(automatic)
                correct_auto_rows += sum(
                    labels[row_id] == predictions[row_id] for row_id in automatic
                )
                ship_to_process += fold_ship
                repair_to_process += fold_repair
            options.append(
                CompanyThresholdOption(
                    company=company,
                    option_id=option_id,
                    shipment_threshold=ship_threshold,
                    repair_threshold=repair_threshold,
                    evaluation_rows=evaluation_rows,
                    actual_process_rows=actual_process_rows,
                    correct_process_rows=correct_process_rows,
                    auto_rows=auto_rows,
                    correct_auto_rows=correct_auto_rows,
                    ship_to_process=ship_to_process,
                    repair_to_process=repair_to_process,
                    fold_danger=MappingProxyType(fold_danger),
                )
            )
        output[company] = tuple(options)
    return MappingProxyType(output)


def apply_company_threshold_policy(
    baseline_predictions: Mapping[str, str],
    scores: Sequence[DangerScoreRow],
    company_by_row: Mapping[str, str],
    choices: Mapping[str, CompanyThresholdOption],
) -> dict[str, str]:
    """Apply company thresholds as a process-to-review veto only."""

    if set(company_by_row) != set(baseline_predictions):
        raise ValueError("company mapping must match the prediction universe")
    score_index: dict[str, DangerScoreRow] = {}
    for score in scores:
        if score.row_id in score_index:
            raise ValueError(f"duplicate danger score: {score.row_id}")
        if score.row_id not in baseline_predictions:
            raise ValueError(f"danger score is outside the prediction universe: {score.row_id}")
        if baseline_predictions[score.row_id] != "PROCESS":
            raise ValueError("danger scores may only target baseline process approvals")
        score_index[score.row_id] = score

    output: dict[str, str] = {}
    for row_id, original in baseline_predictions.items():
        company = company_by_row[row_id]
        if company not in choices:
            raise ValueError(f"company threshold choice is missing: {company}")
        choice = choices[company]
        if choice.company != company:
            raise ValueError(f"company threshold choice key mismatch: {company}")
        score = score_index.get(row_id)
        if original != "PROCESS":
            output[row_id] = original
            continue
        if score is None:
            # Missing danger evidence must never silently preserve an automatic
            # process approval. Route the row to human review instead.
            output[row_id] = ""
            continue
        if (
            score.shipment_probability >= choice.shipment_threshold
            or score.repair_probability >= choice.repair_threshold
        ):
            output[row_id] = ""
        else:
            output[row_id] = original
    return dict(sorted(output.items()))


def optimize_company_thresholds(
    options_by_company: Mapping[str, Sequence[CompanyThresholdOption]],
    *,
    caps: TransitionCaps,
) -> CompanyThresholdPolicy:
    """Select one independent threshold option per company under global caps.

    The allocator consumes already-scored development evidence. It never sees
    feature rows, current labels, or future data, which keeps policy selection
    separate from model fitting and the later 2026 retrospective audit.
    """

    companies = tuple(sorted(str(company) for company in options_by_company))
    if not companies:
        raise ValueError("at least one company option set is required")

    normalized: dict[str, tuple[CompanyThresholdOption, ...]] = {}
    for company in companies:
        raw_options = tuple(options_by_company[company])
        if not raw_options:
            raise ValueError(f"company has no threshold options: {company}")
        if any(option.company != company for option in raw_options):
            raise ValueError(f"company key differs from an option: {company}")
        identifiers = [option.option_id for option in raw_options]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"company has duplicate option IDs: {company}")
        normalized[company] = _deduplicate_risk_cells(raw_options)

    states: dict[tuple[int, int], _PlanState] = {
        (0, 0): _PlanState(0, 0, 0, 0, 0, ())
    }
    for company in companies:
        next_states: dict[tuple[int, int], _PlanState] = {}
        for (ship, repair), state in states.items():
            for option in normalized[company]:
                next_ship = ship + option.ship_to_process
                next_repair = repair + option.repair_to_process
                if not _within_caps(next_ship, next_repair, caps):
                    continue
                candidate = _PlanState(
                    evaluation_rows=state.evaluation_rows + option.evaluation_rows,
                    actual_process_rows=(
                        state.actual_process_rows + option.actual_process_rows
                    ),
                    correct_process_rows=(
                        state.correct_process_rows + option.correct_process_rows
                    ),
                    auto_rows=state.auto_rows + option.auto_rows,
                    correct_auto_rows=state.correct_auto_rows + option.correct_auto_rows,
                    option_ids=(*state.option_ids, option.option_id),
                )
                key = (next_ship, next_repair)
                current = next_states.get(key)
                if current is None or _state_score(candidate) > _state_score(current):
                    next_states[key] = candidate
        if not next_states:
            raise ValueError(
                f"no company threshold policy satisfies the caps after {company}"
            )
        states = next_states

    (ship, repair), best = max(
        states.items(),
        key=lambda item: _final_state_score(
            item[1],
            ship=item[0][0],
            repair=item[0][1],
            caps=caps,
        ),
    )
    choices: dict[str, CompanyThresholdOption] = {}
    for company, option_id in zip(companies, best.option_ids, strict=True):
        choices[company] = next(
            option for option in normalized[company] if option.option_id == option_id
        )
    return CompanyThresholdPolicy(
        choices=MappingProxyType(choices),
        evaluation_rows=best.evaluation_rows,
        actual_process_rows=best.actual_process_rows,
        correct_process_rows=best.correct_process_rows,
        auto_rows=best.auto_rows,
        correct_auto_rows=best.correct_auto_rows,
        ship_to_process=ship,
        repair_to_process=repair,
    )


def _threshold_only_option(
    company: str,
    option_id: str,
    shipment_threshold: float,
    repair_threshold: float,
) -> CompanyThresholdOption:
    return CompanyThresholdOption(
        company=company,
        option_id=option_id,
        shipment_threshold=shipment_threshold,
        repair_threshold=repair_threshold,
        evaluation_rows=0,
        actual_process_rows=0,
        correct_process_rows=0,
        auto_rows=0,
        correct_auto_rows=0,
        ship_to_process=0,
        repair_to_process=0,
        fold_danger=MappingProxyType({}),
    )


def _deduplicate_risk_cells(
    options: Sequence[CompanyThresholdOption],
) -> tuple[CompanyThresholdOption, ...]:
    best: dict[tuple[int, int], CompanyThresholdOption] = {}
    for option in sorted(
        options,
        key=lambda item: (
            item.shipment_threshold,
            item.repair_threshold,
            item.option_id,
        ),
    ):
        key = (option.ship_to_process, option.repair_to_process)
        current = best.get(key)
        if current is None or _option_score(option) > _option_score(current):
            best[key] = option
    return tuple(
        sorted(
            best.values(),
            key=lambda item: (
                item.ship_to_process,
                item.repair_to_process,
                item.option_id,
            ),
        )
    )


def _option_score(
    option: CompanyThresholdOption,
) -> tuple[int, int, int, float, float]:
    return (
        option.correct_process_rows,
        option.correct_auto_rows,
        -option.auto_rows,
        -option.shipment_threshold,
        -option.repair_threshold,
    )


def _state_score(state: _PlanState) -> tuple[int, int, int]:
    return state.correct_process_rows, state.correct_auto_rows, -state.auto_rows


def _final_state_score(
    state: _PlanState,
    *,
    ship: int,
    repair: int,
    caps: TransitionCaps,
) -> tuple[int, int, float, int, int, int, int]:
    utilization = max(
        _safe_utilization(ship, caps.ship_to_process),
        _safe_utilization(repair, caps.repair_to_process),
        _safe_utilization(ship + repair, caps.total),
    )
    return (
        state.correct_process_rows,
        state.correct_auto_rows,
        -utilization,
        -(ship + repair),
        -repair,
        -ship,
        -state.auto_rows,
    )


def _safe_utilization(value: int, cap: int) -> float:
    if cap == 0:
        return 0.0 if value == 0 else math.inf
    return value / cap


def _within_caps(ship: int, repair: int, caps: TransitionCaps) -> bool:
    return (
        ship <= caps.ship_to_process
        and repair <= caps.repair_to_process
        and ship + repair <= caps.total
    )

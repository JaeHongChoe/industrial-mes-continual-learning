from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

from .data_contract import ACTION_LABELS
from .hierarchy import ScopeKey, scope_chain


REVIEW_LABEL = "REVIEW"


@dataclass(frozen=True, slots=True)
class PredictionDecision:
    candidate_id: str
    row_id: str
    episode_id: str
    company: str
    raw_customer: str
    line: str
    month: str
    actual: str
    predicted: str
    train_end: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScopeEvidence:
    candidate_id: str
    scope_id: str
    month: str
    rows: int
    auto_rows: int
    correct_auto_rows: int
    actual_process_rows: int
    correct_process_rows: int
    ship_to_process: int
    repair_to_process: int
    process_to_ng: int
    process_prediction_episodes: int
    risk_episodes: int

    @property
    def automation_rate(self) -> float:
        return self.auto_rows / self.rows if self.rows else 0.0

    @property
    def auto_accuracy(self) -> float | None:
        return self.correct_auto_rows / self.auto_rows if self.auto_rows else None

    @property
    def correct_process_recovery(self) -> float | None:
        return (
            self.correct_process_rows / self.actual_process_rows
            if self.actual_process_rows
            else None
        )


def build_scope_evidence(
    decisions: Iterable[PredictionDecision],
) -> tuple[ScopeEvidence, ...]:
    materialized = tuple(decisions)
    if not materialized:
        return ()
    seen: set[tuple[str, str]] = set()
    states: dict[tuple[str, str, str], dict[str, object]] = {}
    for item in materialized:
        _validate_decision(item)
        identity = item.candidate_id, item.row_id
        if identity in seen:
            raise ValueError("duplicate candidate decision row")
        seen.add(identity)
        scopes = scope_chain(ScopeKey(item.company, item.line))
        for scope_id in scopes:
            key = item.candidate_id, scope_id, item.month
            state = states.setdefault(
                key,
                {
                    "rows": 0,
                    "auto_rows": 0,
                    "correct_auto_rows": 0,
                    "actual_process_rows": 0,
                    "correct_process_rows": 0,
                    "ship_to_process": 0,
                    "repair_to_process": 0,
                    "process_to_ng": 0,
                    "process_episodes": set(),
                    "risk_episodes": set(),
                },
            )
            state["rows"] = int(state["rows"]) + 1
            automatic = item.predicted in ACTION_LABELS
            if automatic:
                state["auto_rows"] = int(state["auto_rows"]) + 1
                if item.predicted == item.actual:
                    state["correct_auto_rows"] = int(state["correct_auto_rows"]) + 1
            if item.actual == "PROCESS":
                state["actual_process_rows"] = int(state["actual_process_rows"]) + 1
                if item.predicted == "PROCESS":
                    state["correct_process_rows"] = int(state["correct_process_rows"]) + 1
            if item.predicted == "PROCESS":
                process_episodes = state["process_episodes"]
                assert isinstance(process_episodes, set)
                process_episodes.add(item.episode_id)
            if item.actual == "RETURN" and item.predicted == "PROCESS":
                state["ship_to_process"] = int(state["ship_to_process"]) + 1
            if item.actual == "REPAIR" and item.predicted == "PROCESS":
                state["repair_to_process"] = int(state["repair_to_process"]) + 1
            if item.actual == "PROCESS" and item.predicted in {
                "RETURN",
                "REPAIR",
            }:
                state["process_to_ng"] = int(state["process_to_ng"]) + 1
            if item.actual in {"RETURN", "REPAIR"} and item.predicted == "PROCESS":
                risk_episodes = state["risk_episodes"]
                assert isinstance(risk_episodes, set)
                risk_episodes.add(item.episode_id)

    output: list[ScopeEvidence] = []
    for (candidate_id, scope_id, month), state in sorted(states.items()):
        process_episodes = state.pop("process_episodes")
        risk_episodes = state.pop("risk_episodes")
        assert isinstance(process_episodes, set)
        assert isinstance(risk_episodes, set)
        output.append(
            ScopeEvidence(
                candidate_id=candidate_id,
                scope_id=scope_id,
                month=month,
                process_prediction_episodes=len(process_episodes),
                risk_episodes=len(risk_episodes),
                **{name: int(value) for name, value in state.items()},
            )
        )
    return tuple(output)


def _validate_decision(item: PredictionDecision) -> None:
    if item.actual not in ACTION_LABELS:
        raise ValueError("decision actual label is not an action")
    if item.predicted not in ACTION_LABELS | {REVIEW_LABEL}:
        raise ValueError("decision prediction is not supported")
    try:
        month_start = date.fromisoformat(f"{item.month}-01")
    except ValueError as error:
        raise ValueError("decision chronology must use ISO dates") from error
    if month_start.year >= 2026:
        raise ValueError("selector evidence ends in 2025")
    if not item.train_end and item.predicted == REVIEW_LABEL:
        return
    try:
        train_end = date.fromisoformat(item.train_end)
    except ValueError as error:
        raise ValueError("decision chronology must use ISO dates") from error
    if train_end >= month_start:
        raise ValueError("training end must precede scored month")
    if not item.candidate_id or not item.row_id or not item.episode_id:
        raise ValueError("decision identity fields cannot be empty")

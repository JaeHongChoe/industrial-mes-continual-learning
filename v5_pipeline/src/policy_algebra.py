from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .data_contract import ACTION_LABELS
from .hierarchical_evidence import REVIEW_LABEL


@dataclass(frozen=True, slots=True)
class AlgebraPolicy:
    policy_id: str
    guard_candidates: tuple[str, ...]
    guard_votes_required: int
    recovery_candidates: tuple[str, ...]
    recovery_votes_required: int

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("algebra policy id cannot be empty")
        if len(set(self.guard_candidates)) != len(self.guard_candidates):
            raise ValueError("guard candidates must be unique")
        if len(set(self.recovery_candidates)) != len(self.recovery_candidates):
            raise ValueError("recovery candidates must be unique")
        if not 0 <= self.guard_votes_required <= len(self.guard_candidates):
            raise ValueError("guard vote requirement is out of range")
        if not 0 <= self.recovery_votes_required <= len(self.recovery_candidates):
            raise ValueError("recovery vote requirement is out of range")


def combine_prediction(
    baseline_prediction: str,
    candidate_predictions: Mapping[str, str],
    policy: AlgebraPolicy,
) -> str:
    if baseline_prediction not in ACTION_LABELS | {REVIEW_LABEL}:
        raise ValueError("unsupported baseline prediction")
    required = set(policy.guard_candidates) | set(policy.recovery_candidates)
    missing = required - set(candidate_predictions)
    if missing:
        raise ValueError(f"missing algebra candidate predictions: {sorted(missing)}")
    if baseline_prediction in {"RETURN", "REPAIR"}:
        return baseline_prediction
    if baseline_prediction == "PROCESS":
        if not policy.guard_candidates:
            return "PROCESS"
        votes = sum(
            candidate_predictions[candidate_id] == "PROCESS"
            for candidate_id in policy.guard_candidates
        )
        return "PROCESS" if votes >= policy.guard_votes_required else REVIEW_LABEL
    if not policy.recovery_candidates:
        return REVIEW_LABEL
    votes = sum(
        candidate_predictions[candidate_id] == "PROCESS"
        for candidate_id in policy.recovery_candidates
    )
    return "PROCESS" if votes >= policy.recovery_votes_required else REVIEW_LABEL

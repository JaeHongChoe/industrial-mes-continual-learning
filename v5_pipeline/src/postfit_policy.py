from __future__ import annotations

from typing import Mapping

from .data_contract import ACTION_LABELS
from .hierarchical_evidence import REVIEW_LABEL
from .hierarchical_select import REVIEW_ALL_CANDIDATE_ID
from .policy_algebra import AlgebraPolicy, combine_prediction


LOOP93_CANDIDATE_ID = "loop93_current_policy_crossfit"


def resolve_postfit_prediction(
    route_candidate_id: str,
    baseline_prediction: str,
    base_predictions: Mapping[str, str],
    algebra_policies: Mapping[str, AlgebraPolicy],
) -> str:
    if route_candidate_id == REVIEW_ALL_CANDIDATE_ID:
        return REVIEW_LABEL
    if route_candidate_id == LOOP93_CANDIDATE_ID:
        return baseline_prediction
    algebra = algebra_policies.get(route_candidate_id)
    if algebra is not None:
        return combine_prediction(baseline_prediction, base_predictions, algebra)
    predicted = base_predictions.get(route_candidate_id)
    if predicted not in ACTION_LABELS | {REVIEW_LABEL}:
        raise ValueError(f"missing postfit candidate prediction: {route_candidate_id}")
    return predicted

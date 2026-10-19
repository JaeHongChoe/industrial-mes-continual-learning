from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from src.data_contract import ACTION_LABELS
from src.data_contract import FeatureRow
from src.v4_derived_policy import ModelRiskGuard, apply_model_risk_guard
from src.v4_hybrid_candidate import HybridThresholdFit, apply_hybrid_thresholds

from .integrity import sha256_file


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"


@dataclass(frozen=True, slots=True)
class V4GreenResult:
    predictions: Mapping[str, str]
    model_predictions: Mapping[str, str]
    model_scores: tuple[object, ...]
    risk_guard_demoted_ids: tuple[str, ...]


def fill_reviews(
    baseline: Mapping[str, str], candidate: Mapping[str, str]
) -> dict[str, str]:
    if set(baseline) != set(candidate):
        raise ValueError("review-fill inputs must share one row universe")
    return {
        row_id: baseline[row_id]
        if baseline[row_id] in ACTION_LABELS
        else candidate[row_id]
        for row_id in baseline
    }


class V4GreenPredictor:
    def __init__(self, frozen_root: str | Path = FROZEN_ROOT) -> None:
        root = Path(frozen_root).resolve()
        model_dir = root / "models/v4_green/gray_auxiliary"
        model_path = model_dir / "gray_auxiliary.pkl"
        sidecar = json.loads(
            (model_dir / "gray_auxiliary.pkl.contract.json").read_text(
                encoding="utf-8"
            )
        )
        threshold_payload = json.loads(
            (model_dir / "gray_auxiliary.threshold.json").read_text(encoding="utf-8")
        )
        self.threshold = HybridThresholdFit(**threshold_payload)
        self.policy = json.loads(
            (root / "v4_green/green_frozen_policy.json").read_text(encoding="utf-8")
        )
        expected_model_sha = str(self.policy["modelParams"]["modelSha256"])
        observed_model_sha = sha256_file(model_path)
        if (
            observed_model_sha != expected_model_sha
            or observed_model_sha != str(sidecar.get("artifactSha256") or "")
        ):
            raise ValueError("V4 Green model SHA mismatch")
        with model_path.open("rb") as handle:
            self.model = pickle.load(handle)
        expected_receipt = str(self.policy["modelParams"]["modelFitReceiptSha256"])
        if (
            str(self.model.fit_receipt.get("receiptSha256") or "")
            != expected_receipt
            or str(sidecar.get("contract", {}).get("modelFitReceiptSha256") or "")
            != expected_receipt
        ):
            raise ValueError("V4 Green fit receipt mismatch")
        recipe = self.policy["modelParams"]["recipe"]
        self.guard = ModelRiskGuard(**recipe["guardThresholds"])

    def predict(
        self,
        rows: Sequence[FeatureRow],
        hierarchy_predictions: Mapping[str, str],
        *,
        prior_history_rows: Sequence[FeatureRow],
        prior_history_labels: Mapping[str, str],
    ) -> V4GreenResult:
        expected = {row.row_id for row in rows}
        if set(hierarchy_predictions) != expected:
            raise ValueError("V4 Green hierarchy row universe mismatch")
        prior = tuple(row for row in prior_history_rows if row.date <= "2025-11-30")
        labels = {row.row_id: prior_history_labels[row.row_id] for row in prior}
        scores = self.model.score_rows(
            rows,
            prior_history_rows=prior,
            prior_history_labels=labels,
            prior_history_cutoff="2025-11-30",
        )
        model_predictions = apply_hybrid_thresholds(scores, self.threshold)
        filled = fill_reviews(hierarchy_predictions, model_predictions)
        guarded = apply_model_risk_guard(filled, scores, self.guard)
        demoted = tuple(
            sorted(
                row_id
                for row_id in expected
                if filled[row_id] == "PROCESS" and guarded[row_id] != "PROCESS"
            )
        )
        return V4GreenResult(
            predictions=MappingProxyType(dict(sorted(guarded.items()))),
            model_predictions=MappingProxyType(dict(sorted(model_predictions.items()))),
            model_scores=tuple(scores),
            risk_guard_demoted_ids=demoted,
        )

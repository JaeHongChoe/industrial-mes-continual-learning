"""Stage D: OVR candidates, calibration, threshold fitting and application."""

from __future__ import annotations

import hashlib

import pytest

from src.candidates import (
    ACTION_ORDER,
    OVRLightGBMCandidate,
    OVRLinearCandidate,
    ScoreRow,
    abstain_predictions,
)
from src.thresholds import (
    THRESHOLD_FIT_PHASE,
    apply_threshold_fit,
    apply_threshold_policy,
    fit_thresholds,
    threshold_fit_receipt,
)

_TEXT = {
    "PROCESS": ("visual", "clean", "surface ok no defect"),
    "RETURN": ("scrap", "obsolete", "scrap requested by owner"),
    "REPAIR": ("crack", "damage", "crack on flange rework"),
}


def _sha(tag: str) -> str:
    return hashlib.sha256(f"synthetic:{tag}".encode()).hexdigest()


def _rows(make_row, start: int, month: str, per_class: int):
    rows, labels = [], {}
    number = start
    for day in range(1, per_class + 1):
        for label in ACTION_ORDER:
            type_, symptom, detail = _TEXT[label]
            row = make_row(
                number,
                f"{month}-{day:02d}",
                type_=type_,
                symptom=symptom,
                detail=detail,
                line=f"L{number % 2}",
            )
            rows.append(row)
            labels[row.row_id] = label
            number += 1
    return rows, labels


@pytest.fixture
def splits(make_row):
    train, train_labels = [], {}
    number = 1
    for month in ("2025-03", "2025-04", "2025-05", "2025-06"):
        rows, labels = _rows(make_row, number, month, 4)
        train += rows
        train_labels.update(labels)
        number += len(rows)
    calibration, calibration_labels = _rows(make_row, 1000, "2025-07", 5)
    threshold, threshold_labels = _rows(make_row, 2000, "2025-08", 5)
    selector, selector_labels = _rows(make_row, 3000, "2025-10", 2)
    return {
        "train": (train, train_labels),
        "calibration": (calibration, calibration_labels),
        "threshold": (threshold, threshold_labels),
        "selector": (selector, selector_labels),
    }


def _fit_linear(splits):
    train, train_labels = splits["train"]
    calibration, calibration_labels = splits["calibration"]
    return OVRLinearCandidate(random_state=17).fit(
        train,
        train_labels,
        calibration_rows=calibration,
        calibration_labels=calibration_labels,
    )


def test_ovr_linear_candidate_scores_are_calibrated_simplex(splits):
    candidate = _fit_linear(splits)
    threshold_rows, threshold_labels = splits["threshold"]
    scores = candidate.predict_scores(threshold_rows)
    assert [item.row_id for item in scores] == [row.row_id for row in threshold_rows]
    for item in scores:
        assert set(item.scores) == set(ACTION_ORDER)
        assert sum(item.scores.values()) == pytest.approx(1.0, abs=1e-9)
        assert item.family == "ovr_linear"
        assert item.evidence["provenance"]["calibration"]["state"] == "calibrated"
        assert item.evidence["coldStart"] is False
    correct = sum(
        max(ACTION_ORDER, key=item.scores.__getitem__) == threshold_labels[item.row_id]
        for item in scores
    )
    assert correct == len(scores)  # perfectly separable synthetic text
    assert len(candidate.calibration_score_rows()) == len(splits["calibration"][0])


def test_ovr_linear_candidate_is_deterministic(splits):
    first = _fit_linear(splits).predict_scores(splits["threshold"][0])
    second = _fit_linear(splits).predict_scores(splits["threshold"][0])
    assert [dict(item.scores) for item in first] == [dict(item.scores) for item in second]
    assert (
        first[0].evidence["provenance"]["modelSha256"]
        == second[0].evidence["provenance"]["modelSha256"]
    )


def test_ovr_candidate_temporal_guards(splits, make_row):
    candidate = _fit_linear(splits)
    # Rows on/before the calibration cutoff cannot be scored.
    with pytest.raises(ValueError):
        candidate.predict_scores([make_row(9000, "2025-07-03")])
    # December 2025 onward can never be used for training.
    late = make_row(9001, "2025-12-02")
    with pytest.raises(ValueError):
        OVRLinearCandidate().fit([late], {late.row_id: "PROCESS"})


def test_threshold_fit_and_apply_end_to_end(splits):
    candidate = _fit_linear(splits)
    threshold_rows, threshold_labels = splits["threshold"]
    scores = candidate.predict_scores(threshold_rows)
    provenance = scores[0].evidence["provenance"]
    kwargs = dict(
        policy="calibration_zero_unsafe",
        min_support=2,
        source_dataset_id="synthetic_ds",
        history_sha256=_sha("history"),
        phase=THRESHOLD_FIT_PHASE,
        row_dates={row.row_id: row.date for row in threshold_rows},
        fitted_artifact_sha256=_sha("artifact"),
        feature_schema_sha256=provenance["featureSchemaSha256"],
    )
    fitted = fit_thresholds(scores, threshold_labels, **kwargs)
    assert set(fitted.thresholds) == set(ACTION_ORDER)
    assert fitted.enabled["PROCESS"] is True
    assert fitted.calibration_metrics["PROCESS"].unsafe == 0
    receipt = threshold_fit_receipt(fitted)
    assert receipt["sha256"] == fitted.sha256 and len(fitted.sha256) == 64
    assert fit_thresholds(scores, threshold_labels, **kwargs).sha256 == fitted.sha256

    selector_rows, selector_labels = splits["selector"]
    predictions = apply_threshold_fit(candidate.predict_scores(selector_rows), fitted)
    assert len(predictions) == len(selector_rows)
    for item in predictions:
        assert item.label in (*ACTION_ORDER, "REVIEW")
        assert item.evidence["thresholdFitSha256"] == fitted.sha256
        if item.label == "PROCESS":
            assert selector_labels[item.row_id] == "PROCESS"


def test_threshold_fit_rejects_selector_months(splits):
    candidate = _fit_linear(splits)
    selector_rows, selector_labels = splits["selector"]
    scores = candidate.predict_scores(selector_rows)
    with pytest.raises(ValueError):
        fit_thresholds(
            scores,
            selector_labels,
            policy="calibration_zero_unsafe",
            min_support=1,
            source_dataset_id="synthetic_ds",
            history_sha256=_sha("history"),
            phase=THRESHOLD_FIT_PHASE,
            row_dates={row.row_id: row.date for row in selector_rows},
            fitted_artifact_sha256=_sha("artifact"),
            feature_schema_sha256=_sha("schema"),
        )


def test_abstention_and_disabled_class_route_to_review(splits):
    candidate = _fit_linear(splits)
    scores = candidate.predict_scores(splits["threshold"][0])
    strict = abstain_predictions(scores, thresholds={action: 1.0 for action in ACTION_ORDER})
    assert all(item.label == "REVIEW" for item in strict if item.score < 1.0)
    loose = abstain_predictions(scores, thresholds={action: 0.0 for action in ACTION_ORDER})
    assert all(item.label in ACTION_ORDER for item in loose)

    plain = [ScoreRow("r1", {"PROCESS": 0.9, "RETURN": 0.05, "REPAIR": 0.05}, "synthetic", {})]
    disabled = apply_threshold_policy(
        plain,
        thresholds={action: 0.5 for action in ACTION_ORDER},
        enabled={"PROCESS": False, "RETURN": True, "REPAIR": True},
        policy="synthetic",
    )
    assert disabled[0].label == "REVIEW"
    assert disabled[0].evidence["classEnabled"] is False


def test_ovr_lightgbm_candidate_smoke(splits):
    pytest.importorskip("lightgbm")
    train, train_labels = splits["train"]
    candidate = OVRLightGBMCandidate(random_state=23, n_estimators=10).fit(train, train_labels)
    scores = candidate.predict_scores(splits["threshold"][0])
    assert len(scores) == len(splits["threshold"][0])
    for item in scores:
        assert sum(item.scores.values()) == pytest.approx(1.0, abs=1e-9)
        assert item.evidence["provenance"]["calibration"]["state"] == "uncalibrated"
    # Uncalibrated OVR scores must not be used for probability abstention.
    with pytest.raises(ValueError):
        abstain_predictions(scores, thresholds={action: 0.5 for action in ACTION_ORDER})

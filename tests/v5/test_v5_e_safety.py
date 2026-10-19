"""Stage E: demotion-only safety layers (semantic veto, repair guard, rank veto)."""

from __future__ import annotations

import pytest

from src.data_contract import ACTION_LABELS
from src.hierarchical_veto import (
    FROZEN_SEMANTIC_GROUPS,
    apply_semantic_veto,
    matched_semantic_groups,
    semantic_policy_sha256,
)
from src.metrics import REVIEW_LABELS
from src.prior_supported_repair_guard import PriorSupportedRepairGuard
from src.rank_based_danger_veto import (
    RankDangerScoreRow,
    RankVetoCandidate,
    apply_rank_based_danger_veto,
)


def _assert_demotion_only(before, after):
    for row_id, original in before.items():
        updated = after[row_id]
        if updated != original:
            assert original == "PROCESS", f"{row_id}: only PROCESS may change"
            assert updated not in ACTION_LABELS, f"{row_id}: must become a review label"
            assert updated in REVIEW_LABELS


def test_semantic_veto_demotes_process_and_never_promotes(make_row):
    rows = [
        make_row(1, "2025-10-01", detail="crack on flange"),
        make_row(2, "2025-10-01", customer="customer_b", detail="crack on flange"),
        make_row(3, "2025-10-01", detail="crack on flange"),
        make_row(4, "2025-10-01", detail="crack on flange"),
        make_row(5, "2025-10-01", detail="surface ok"),
        make_row(6, "2025-10-01", symptom="Leak", detail="pressure test"),
    ]
    predictions = {
        "synthetic_ds:1": "PROCESS",
        "synthetic_ds:2": "PROCESS",
        "synthetic_ds:3": "REPAIR",
        "synthetic_ds:4": "REVIEW",
        "synthetic_ds:5": "PROCESS",
        "synthetic_ds:6": "PROCESS",
    }
    policy = {"customer_a": ["structural_crack_or_breakage", "leak_or_hole_repair"]}
    result = apply_semantic_veto(rows, predictions, policy)

    assert result.predictions == {
        "synthetic_ds:1": "REVIEW",
        "synthetic_ds:2": "PROCESS",  # group not enabled for customer_b
        "synthetic_ds:3": "REPAIR",  # non-PROCESS actions untouched
        "synthetic_ds:4": "REVIEW",  # never promoted
        "synthetic_ds:5": "PROCESS",
        "synthetic_ds:6": "REVIEW",
    }
    assert result.demoted_rows == ("synthetic_ds:1", "synthetic_ds:6")
    assert result.matched_groups_by_row["synthetic_ds:1"] == ("structural_crack_or_breakage",)
    _assert_demotion_only(predictions, result.predictions)
    assert "structural_crack_or_breakage" in matched_semantic_groups(rows[1])


def test_semantic_policy_digest_and_validation(make_row):
    a = semantic_policy_sha256({"customer_a": ["count_limit", "leak_or_hole_repair"]})
    b = semantic_policy_sha256({"customer_a": ["leak_or_hole_repair", "count_limit"]})
    assert a == b and len(a) == 64
    assert a != semantic_policy_sha256({"customer_a": ["count_limit"]})
    assert len(FROZEN_SEMANTIC_GROUPS) == 7
    row = make_row(1, "2025-10-01")
    with pytest.raises(ValueError):
        apply_semantic_veto([row], {row.row_id: "PROCESS"}, {"customer_a": ["no_such_group"]})
    with pytest.raises(ValueError):
        apply_semantic_veto([row], {}, {})


def _guard_fit_rows(make_row):
    rows, labels = [], {}
    plan = [
        ("customer_a", "SHORT CIRCUIT on terminal", "REPAIR"),
        ("customer_a", "short circuit detected", "REPAIR"),
        ("customer_a", "Short Circuit", "RETURN"),
        ("customer_a", "short circuit minor", "PROCESS"),
        ("customer_b", "short circuit", "PROCESS"),
        ("customer_b", "short circuit", "PROCESS"),
        ("customer_b", "short circuit", "PROCESS"),
        ("customer_a", "surface ok", "PROCESS"),
    ]
    for index, (customer, detail, label) in enumerate(plan, start=1):
        row = make_row(index, f"2025-08-{index:02d}", customer=customer, detail=detail)
        rows.append(row)
        labels[row.row_id] = label
    return rows, labels


def test_prior_supported_repair_guard_fit_and_apply(make_row):
    rows, labels = _guard_fit_rows(make_row)
    guard = PriorSupportedRepairGuard.fit(
        rows,
        labels,
        training_end="2025-09-30",
        min_rows=3,
        min_risk_rows=2,
        min_risk_rate=0.5,
    )
    assert guard.enabled_by_company == {"customer_a": ("electrical_short",)}
    evidence = guard.evidence_by_company["customer_a"]["electrical_short"]
    assert (evidence.support, evidence.risk_rows, evidence.enabled) == (4, 3, True)
    assert guard.evidence_by_company["customer_b"]["electrical_short"].enabled is False
    assert len(guard.fit_row_ids_sha256) == 64

    scored = [
        make_row(101, "2025-10-01", detail="short circuit found"),
        make_row(102, "2025-10-01", customer="customer_b", detail="short circuit found"),
        make_row(103, "2025-10-01", detail="short circuit found"),
        make_row(104, "2025-10-01", detail="surface ok"),
    ]
    predictions = {
        "synthetic_ds:101": "PROCESS",
        "synthetic_ds:102": "PROCESS",
        "synthetic_ds:103": "RETURN",
        "synthetic_ds:104": "PROCESS",
    }
    result = guard.apply(scored, predictions)
    assert result.demoted_ids == ("synthetic_ds:101",)
    assert result.predictions["synthetic_ds:101"] == ""  # package-wide review alias
    assert result.predictions["synthetic_ds:102"] == "PROCESS"
    assert result.predictions["synthetic_ds:103"] == "RETURN"
    reason = result.reasons_by_row["synthetic_ds:101"][0]
    assert reason["signalId"] == "electrical_short" and reason["trainingEnd"] == "2025-09-30"
    _assert_demotion_only(predictions, result.predictions)


def test_prior_supported_repair_guard_rejects_future_fit_rows(make_row):
    rows, labels = _guard_fit_rows(make_row)
    with pytest.raises(ValueError):
        PriorSupportedRepairGuard.fit(rows, labels, training_end="2025-08-05")


def _rank_inputs():
    baseline = {f"r{index}": "PROCESS" for index in range(8)}
    baseline.update({"r8": "RETURN", "r9": ""})
    shipment = [0.10, 0.90, 0.20, 0.80, 0.30, 0.05, 0.40, 0.15]
    repair = [0.70, 0.10, 0.20, 0.30, 0.95, 0.10, 0.10, 0.10]
    scores = [
        RankDangerScoreRow(
            row_id=f"r{index}",
            month="2025-10",
            shipment_score=shipment[index],
            repair_score=repair[index],
            company="customer_a" if index < 4 else "customer_b",
        )
        for index in range(8)
    ]
    return baseline, scores


def test_rank_based_danger_veto_demotes_exact_top_fraction():
    baseline, scores = _rank_inputs()
    candidate = RankVetoCandidate("toy", shipment_fraction=0.25, repair_fraction=0.125)
    application = apply_rank_based_danger_veto(baseline, scores, candidate)

    # floor(8 * 0.25) = 2 shipment rows (r1, r3); floor(8 * 0.125) = 1 repair row (r4).
    assert application.evidence.shipment_demoted_row_ids == ("r1", "r3")
    assert application.evidence.repair_demoted_row_ids == ("r4",)
    assert application.evidence.demoted_row_ids == ("r1", "r3", "r4")
    for row_id in ("r1", "r3", "r4"):
        assert application.predictions[row_id] == ""
    assert application.predictions["r8"] == "RETURN"
    _assert_demotion_only(baseline, application.predictions)
    assert application.to_dict()["evidence"]["rounding"] == "floor(n * fraction)"

    again = apply_rank_based_danger_veto(baseline, list(reversed(scores)), candidate)
    assert dict(again.predictions) == dict(application.predictions)


def test_rank_based_danger_veto_company_scope_and_universe_checks():
    baseline, scores = _rank_inputs()
    scoped = apply_rank_based_danger_veto(
        baseline,
        scores,
        RankVetoCandidate("toy_company", 0.25, 0.0, scope_by_company=True),
    )
    # floor(4 * 0.25) = 1 per company: r1 (customer_a), r6 (customer_b).
    assert scoped.evidence.demoted_row_ids == ("r1", "r6")
    assert len(scoped.evidence.scopes) == 2

    with pytest.raises(ValueError):
        apply_rank_based_danger_veto(baseline, scores[:-1], RankVetoCandidate("toy", 0.25, 0.0))
    unscoped = [
        RankDangerScoreRow(item.row_id, item.month, item.shipment_score, item.repair_score)
        for item in scores
    ]
    with pytest.raises(ValueError):
        apply_rank_based_danger_veto(
            baseline, unscoped, RankVetoCandidate("toy", 0.25, 0.0, scope_by_company=True)
        )
    with pytest.raises(ValueError):
        RankVetoCandidate("bad", shipment_fraction=1.5, repair_fraction=0.0)

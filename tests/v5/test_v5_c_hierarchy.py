"""Stage C: company/line hierarchy, company scoping and hierarchical selection."""

from __future__ import annotations

import pytest

from src.company_scope import (
    prediction_or_review,
    select_company_rows,
    validate_route_company,
)
from src.hierarchical_evidence import PredictionDecision, build_scope_evidence
from src.hierarchical_risk import RiskBudget, assess_scope
from src.hierarchical_select import (
    resolve_candidate,
    select_company_policies,
    select_line_overrides,
)
from src.hierarchy import (
    ScopeKey,
    canonical_line_key,
    line_collision_report,
    normalize_line_name,
    scope_chain,
)


def test_scope_key_normalization_and_chain():
    key = ScopeKey("  customer_a ", " l_1 ")
    assert key.company == "customer_a"
    assert key.line == "L-1"
    assert scope_chain(key) == ("company_line:customer_a:L-1", "company:customer_a", "global")
    assert scope_chain(ScopeKey("customer_b", "N/A")) == ("company:customer_b", "global")
    with pytest.raises(ValueError):
        ScopeKey("   ")
    assert normalize_line_name("line —  2") == "line-2"
    assert canonical_line_key("line - 2") == "LINE-2"


def test_line_collision_report_lists_raw_aliases(make_row):
    rows = [
        make_row(1, "2025-10-01", line="L_1"),
        make_row(2, "2025-10-01", line="l-1"),
        make_row(3, "2025-10-01", line="L2"),
        make_row(4, "2025-10-01", customer="customer_b", line="L_1"),
    ]
    report = line_collision_report(rows)
    assert report == {"customer_a": {"L-1": ("L_1", "l-1")}}


def test_company_scope_selection_and_routing(make_row):
    rows = [
        make_row(1, "2025-10-01", customer="customer_a"),
        make_row(2, "2025-10-01", customer="customer_b"),
        make_row(3, "2025-10-02", customer="customer_b"),
    ]
    assert select_company_rows(rows, None) == tuple(rows)
    assert [row.row_id for row in select_company_rows(rows, "customer_b")] == [
        "synthetic_ds:2",
        "synthetic_ds:3",
    ]
    with pytest.raises(ValueError):
        select_company_rows(rows, "customer_e")

    assert prediction_or_review({"synthetic_ds:1": "PROCESS"}, "synthetic_ds:2") == "REVIEW"
    registry = {"cand_local": {"scopeCompany": "customer_a"}, "cand_global": {"scopeCompany": None}}
    validate_route_company(registry, "cand_local", "customer_a")
    validate_route_company(registry, "cand_global", "customer_c")
    with pytest.raises(ValueError):
        validate_route_company(registry, "cand_local", "customer_b")
    with pytest.raises(ValueError):
        validate_route_company(registry, "cand_missing", "customer_a")


# Per month for customer_a: L1 = 4 PROCESS + 1 REPAIR, L2 = 4 PROCESS + 1 RETURN.
# cand_safe automates all L1 PROCESS rows but only half of the L2 PROCESS rows.
# cand_line reviews L1 but automates every L2 PROCESS row.
# cand_risky sends an unsafe row to PROCESS.
_PLANS = {
    "cand_safe": {"L1": ["PROCESS"] * 4 + ["REPAIR"], "L2": ["PROCESS", "PROCESS", "REVIEW", "REVIEW", "RETURN"]},
    "cand_line": {"L1": ["REVIEW"] * 4 + ["REPAIR"], "L2": ["PROCESS"] * 4 + ["RETURN"]},
    "cand_risky": {"L1": ["PROCESS"] * 5, "L2": ["PROCESS"] * 4 + ["RETURN"]},
}
_ACTUALS = {"L1": ["PROCESS"] * 4 + ["REPAIR"], "L2": ["PROCESS"] * 4 + ["RETURN"]}


def _decisions():
    output = []
    for candidate_id, plan in _PLANS.items():
        for month in ("2025-10", "2025-11"):
            for line, actuals in _ACTUALS.items():
                for index, (actual, predicted) in enumerate(zip(actuals, plan[line])):
                    row_id = f"{month}:{line}:{index}"
                    output.append(
                        PredictionDecision(
                            candidate_id=candidate_id,
                            row_id=row_id,
                            episode_id=f"ep:{row_id}",
                            company="customer_a",
                            raw_customer="customer_a",
                            line=line,
                            month=month,
                            actual=actual,
                            predicted=predicted,
                            train_end="2025-09-30",
                        )
                    )
    return output


_BUDGET = RiskBudget(
    company_min_validation_rows=5,
    line_min_validation_rows=3,
    max_observed_danger_rate=0.0,
    max_exact_danger_ucb=0.6,
    min_automatic_accuracy=0.99,
)


def test_scope_evidence_rolls_up_company_line_global():
    evidence = build_scope_evidence(_decisions())
    scopes = {item.scope_id for item in evidence}
    assert scopes == {
        "global",
        "company:customer_a",
        "company_line:customer_a:L1",
        "company_line:customer_a:L2",
    }
    risky = [
        item for item in evidence
        if item.candidate_id == "cand_risky" and item.scope_id == "company:customer_a"
    ]
    assert len(risky) == 2
    assert all(item.repair_to_process == 1 and item.risk_episodes == 1 for item in risky)

    assessment = assess_scope(risky, scope_level="company", budget=_BUDGET)
    assert not assessment.passed
    assert "observed_danger_rate_exceeds_budget" in assessment.rejection_reasons


def test_hierarchical_selection_picks_company_policy_and_line_override():
    evidence = build_scope_evidence(_decisions())
    company = select_company_policies(evidence, budget=_BUDGET)
    assert set(company) == {"customer_a"}
    assert company["customer_a"].candidate_id == "cand_safe"
    assert company["customer_a"].assessment.correct_process_rows == 12

    overrides = select_line_overrides(evidence, company, budget=_BUDGET)
    assert set(overrides) == {("customer_a", "L2")}
    override = overrides[("customer_a", "L2")]
    assert override.candidate_id == "cand_line"
    assert override.parent_candidate_id == "cand_safe"
    assert override.gain_correct_process_rows == 4

    assert resolve_candidate("customer_a", "L2", company, overrides) == "cand_line"
    assert resolve_candidate("customer_a", "L1", company, overrides) == "cand_safe"
    assert resolve_candidate("customer_d", "L1", company, overrides) is None


def test_evidence_rejects_2026_months_and_training_overlap():
    base = _decisions()[0]
    from dataclasses import replace

    with pytest.raises(ValueError):
        build_scope_evidence([replace(base, month="2026-01", train_end="2025-12-31")])
    with pytest.raises(ValueError):
        build_scope_evidence([replace(base, train_end="2025-10-01")])

"""후보 비교 게이트: 평균 개선이 있어도 그룹 단위 악화·표본 부족 변경은 차단한다."""
import pytest

from mes_cl.gate import compare_predictions


def cohort(customer, n, *, start=0):
    """PROCESS/RETURN/REPAIR 를 고르게 가진 행. baseline 은 전부 정답."""
    labels = ("PROCESS", "RETURN", "REPAIR")
    return [dict(id=f"{customer}-{i}", date="2026-05-20", customer=customer, line="l1",
                 product_key="p1", actual=labels[i % 3], baseline=labels[i % 3]) for i in range(start, start + n)]


def predictions(rows, overrides=None):
    overrides = overrides or {}
    return [dict(id=r["id"], predicted=overrides.get(r["id"], r["baseline"])) for r in rows]


def test_identical_predictions_have_no_meaningful_improvement():
    rows = cohort("a", 60)
    report = compare_predictions(rows, predictions(rows))
    assert not report["gate"]["passed"]
    assert "no_meaningful_improvement" in report["gate"]["reasons"]
    assert report["gate"]["production_eligible"] is False


def test_customer_regression_blocks_even_if_overall_improves():
    a, b = cohort("a", 90), cohort("b", 60)
    # a: baseline 오답 6건(REVIEW)을 후보가 맞춤 → 전체 개선
    for r in a[:6]:
        r["baseline"] = "REVIEW"
    # b: 정답이 REPAIR 인 1건을 후보가 PROCESS 로 보냄 → 위험 방향 신규 발생
    repair_b = next(r for r in b if r["actual"] == "REPAIR")
    rows = a + b
    preds = predictions(rows, {**{r["id"]: r["actual"] for r in a[:6]}, repair_b["id"]: "PROCESS"})
    report = compare_predictions(rows, preds)
    assert report["overall"]["delta"]["correct_count"] > 0
    assert "customer:b:new_dangerous_cases" in report["gate"]["reasons"]
    assert "customer:b:increased_repair_to_process" in report["gate"]["reasons"]
    assert not report["gate"]["passed"]


def test_changed_prediction_in_small_group_is_not_treated_as_safe():
    big, small = cohort("a", 90), cohort("tiny", 6)
    for r in big[:3]:
        r["baseline"] = "REVIEW"
    rows = big + small
    preds = predictions(rows, {**{r["id"]: r["actual"] for r in big[:3]}, small[0]["id"]: "REVIEW"})
    report = compare_predictions(rows, preds)
    assert "tiny" in report["gate"]["insufficient_support"]
    assert "customer:tiny:insufficient_support_changed" in report["gate"]["reasons"]


def test_recovering_reviews_without_danger_passes_same_cohort_gate():
    rows = cohort("a", 90)
    for r in rows[:3]:
        r["baseline"] = "REVIEW"
    report = compare_predictions(rows, predictions(rows, {r["id"]: r["actual"] for r in rows[:3]}))
    assert report["gate"]["passed"], report["gate"]["reasons"]
    assert report["changes"]["recovered_reviews"] == 3
    assert report["gate"]["production_eligible"] is False     # 사람 승인 전에는 항상 False


def test_prediction_cohort_must_match_exactly():
    rows = cohort("a", 30)
    with pytest.raises(ValueError):
        compare_predictions(rows, predictions(rows)[:-1])
    with pytest.raises(ValueError):
        compare_predictions(rows, predictions(rows) + [dict(id="a-0", predicted="PROCESS")])

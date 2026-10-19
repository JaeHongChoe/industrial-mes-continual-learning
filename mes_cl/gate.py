"""Candidate-vs-baseline safety gate on the same evaluation cohort.

재학습 후보 모델을 현재(기준) 모델과 **같은 평가 행**에서 비교하고, 전체 평균이 좋아져도
특정 고객사(customer) 단위에서 위험 방향 오판정이 늘면 차단한다.

위험 방향(danger)은 "실제와 다르게 PROCESS(공정투입)로 보내거나, PROCESS 여야 할 것을 NG 처리"하는 경우다.
표본이 부족한 그룹은 "안전하다"가 아니라 "판단 보류"로 다루며, 그 그룹에서 판정이 바뀌면 차단 사유가 된다.
이 게이트를 통과해도 production_eligible 은 False 로 고정된다: 사람 승인과 전향적(prospective) 근거가 필요하다.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date

ACTION_LABELS = ("PROCESS", "RETURN", "REPAIR")
PROCESS, RETURN, REPAIR = ACTION_LABELS
REVIEW_LABELS = frozenset({"REVIEW"})

MIN_SUPPORT = 30          # 그룹 판정에 필요한 최소 행 수
MIN_CLASS_SUPPORT = 5     # 각 라벨별 최소 행 수
MAX_AUTO_LOSS = 0.02      # 자동 처리율 감소 허용폭
DANGERS = ("return_to_process", "repair_to_process", "process_to_return",
           "process_to_repair", "ng_to_process", "process_to_ng")


def ratio(a, b):
    return a / b if b else None


def summarize(rows):
    """(actual, predicted) 행 목록 → 자동처리율, 정확도, 방향별 오판정 수, 라벨별 지표."""
    evaluated = [r for r in rows if r.get("actual") in ACTION_LABELS]
    auto = [r for r in evaluated if r.get("predicted") in ACTION_LABELS]
    correct = sum(r["actual"] == r["predicted"] for r in auto)
    result = dict(evaluation_count=len(evaluated), auto_count=len(auto), correct_count=correct,
                  review_count=sum(r.get("predicted") in REVIEW_LABELS for r in evaluated),
                  auto_rate=ratio(len(auto), len(evaluated)), accuracy=ratio(correct, len(auto)))
    for key, actual, predicted in (("return_to_process", RETURN, PROCESS),
                                   ("repair_to_process", REPAIR, PROCESS),
                                   ("process_to_return", PROCESS, RETURN),
                                   ("process_to_repair", PROCESS, REPAIR)):
        result[key] = sum(r["actual"] == actual and r.get("predicted") == predicted for r in evaluated)
    classes = {}
    for label in ACTION_LABELS:
        actual = [r for r in evaluated if r["actual"] == label]
        predicted = [r for r in evaluated if r.get("predicted") == label]
        hit = sum(r.get("predicted") == label for r in actual)
        classes[label] = dict(actual=len(actual), predicted=len(predicted), correct=hit,
                              automation=ratio(hit, len(actual)), precision=ratio(hit, len(predicted)))
    result["classes"] = classes
    return result


def _id(value):
    if type(value) not in (str, int) or (isinstance(value, str) and not value.strip()):
        raise ValueError("row id must be a nonempty string or integer")
    return (type(value).__name__, value)


def _validated_rows(rows):
    checked, seen = [], set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("row must be a dictionary")
        identity = _id(raw.get("id"))
        if identity in seen:
            raise ValueError("duplicate row id")
        seen.add(identity)
        day = raw.get("date")
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("row date must be YYYY-MM-DD")
        date.fromisoformat(day)
        for key in ("customer", "line", "product_key"):
            if not isinstance(raw.get(key), str):
                raise ValueError(f"row {key} must be a string")
        if raw.get("baseline") not in (*ACTION_LABELS, *REVIEW_LABELS):
            raise ValueError("unrecognized baseline action")
        if raw.get("actual") not in ACTION_LABELS:
            raise ValueError("unrecognized actual action")
        checked.append(raw)
    return checked


def _support(rows):
    counts = Counter(r["actual"] for r in rows)
    reasons = []
    if len(rows) < MIN_SUPPORT:
        reasons.append("insufficient_rows")
    if any(counts[label] < MIN_CLASS_SUPPORT for label in ACTION_LABELS):
        reasons.append("insufficient_class_support")
    return dict(supported=not reasons, count=len(rows), classes={k: counts[k] for k in ACTION_LABELS},
                minimum_rows=MIN_SUPPORT, minimum_per_class=MIN_CLASS_SUPPORT, reasons=reasons)


def _metrics(rows, labels):
    metrics = summarize([dict(actual=r["actual"], predicted=p) for r, p in zip(rows, labels, strict=True)])
    metrics["incorrect_auto_count"] = metrics["auto_count"] - metrics["correct_count"]
    metrics["process_to_ng"] = metrics["process_to_return"] + metrics["process_to_repair"]
    metrics["ng_to_process"] = metrics["return_to_process"] + metrics["repair_to_process"]
    metrics["correct_rate"] = metrics["correct_count"] / len(rows) if rows else None
    return metrics


def _pair(rows, labels):
    baseline = _metrics(rows, [r["baseline"] for r in rows])
    candidate = _metrics(rows, labels)
    fields = (*DANGERS, "auto_count", "correct_count", "incorrect_auto_count", "accuracy", "auto_rate")
    delta = {key: candidate[key] - baseline[key]
             if candidate[key] is not None and baseline[key] is not None else None for key in fields}
    delta["auto_loss"] = max(0.0, baseline["auto_rate"] - candidate["auto_rate"]) if rows else None
    return dict(count=len(rows), support=_support(rows), baseline=baseline, candidate=candidate, delta=delta)


def _danger(actual, predicted):
    return predicted in ACTION_LABELS and actual != predicted and (actual == PROCESS or predicted == PROCESS)


def _safety_reasons(rows, labels, pair):
    reasons = [f"increased_{key}" for key in DANGERS if pair["delta"][key] > 0]
    if any(_danger(row["actual"], label) and not _danger(row["actual"], row["baseline"])
           for row, label in zip(rows, labels, strict=True)):
        reasons.append("new_dangerous_cases")
    base, candidate = pair["baseline"], pair["candidate"]
    if candidate["incorrect_auto_count"] > base["incorrect_auto_count"]:
        reasons.append("increased_incorrect_auto")
    if base["accuracy"] is not None and (candidate["accuracy"] is None or candidate["accuracy"] < base["accuracy"]):
        reasons.append("accuracy_regression")
    if pair["delta"]["auto_loss"] is not None and pair["delta"]["auto_loss"] > MAX_AUTO_LOSS + 1e-12:
        reasons.append("excessive_auto_loss")
    return reasons


def compare_predictions(rows, predictions):
    """한 행당 정확히 하나의 후보 예측을 짝지어 같은 cohort 기준 안전성을 보고한다.

    rows: id, date, customer, line, product_key, actual(확정 라벨), baseline(기준 모델 판정)
    predictions: id, predicted(후보 모델 판정), reason(선택)
    """
    rows = _validated_rows(rows)
    if not rows:
        raise ValueError("comparison requires nonempty labeled rows")
    by_id = {}
    for prediction in predictions:
        key = _id(prediction.get("id"))
        if key in by_id:
            raise ValueError("duplicate prediction id")
        if prediction.get("predicted") not in (*ACTION_LABELS, *REVIEW_LABELS):
            raise ValueError("unrecognized predicted action")
        by_id[key] = prediction
    if set(by_id) != {_id(row["id"]) for row in rows}:
        raise ValueError("prediction ids must match the exact row cohort")
    labels = [by_id[_id(row["id"])]["predicted"] for row in rows]

    overall = _pair(rows, labels)
    reasons = ["overall:" + reason for reason in _safety_reasons(rows, labels, overall)]
    if not overall["support"]["supported"]:
        reasons.append("overall:insufficient_support")

    customer_groups, line_groups = defaultdict(list), defaultdict(list)
    for index, row in enumerate(rows):
        customer_groups[row["customer"]].append(index)
        line_groups[(row["customer"], row["line"])].append(index)
    customers, lines, insufficient = [], [], []
    for customer, indexes in sorted(customer_groups.items()):
        group, predicted = [rows[i] for i in indexes], [labels[i] for i in indexes]
        pair = _pair(group, predicted)
        customers.append(dict(customer=customer, **pair))
        if pair["support"]["supported"]:
            reasons.extend(f"customer:{customer}:{reason}" for reason in _safety_reasons(group, predicted, pair))
        else:
            # 표본 부족은 안전 입증이 아니다: 판정이 하나라도 바뀌었으면 차단
            insufficient.append(customer)
            if any(row["baseline"] != label for row, label in zip(group, predicted, strict=True)):
                reasons.append(f"customer:{customer}:insufficient_support_changed")
    for (customer, line), indexes in sorted(line_groups.items()):
        lines.append(dict(customer=customer, line=line,
                          **_pair([rows[i] for i in indexes], [labels[i] for i in indexes])))

    changes = [dict(id=r["id"], date=r["date"], customer=r["customer"], line=r["line"],
                    actual=r["actual"], baseline=r["baseline"], predicted=p,
                    reason=by_id[_id(r["id"])].get("reason", "comparison"))
               for r, p in zip(rows, labels, strict=True) if r["baseline"] != p]
    recovered = sum(r["baseline"] in REVIEW_LABELS and r["predicted"] in ACTION_LABELS for r in changes)
    guarded = sum(r["baseline"] in ACTION_LABELS and r["predicted"] in REVIEW_LABELS for r in changes)
    meaningful = (overall["delta"]["correct_count"] > 0 or
                  overall["delta"]["ng_to_process"] < 0 or overall["delta"]["process_to_ng"] < 0)
    if not changes or not meaningful:
        reasons.append("no_meaningful_improvement")

    return dict(overall=overall, customers=customers, lines=lines,
                row_ids=[r["id"] for r in rows],
                changes=dict(count=len(changes), recovered_reviews=recovered,
                             guarded_baseline=guarded, rows=changes),
                gate=dict(passed=not reasons, reasons=reasons,
                          insufficient_support=insufficient, max_auto_loss=MAX_AUTO_LOSS,
                          production_eligible=False,
                          production_blockers=["prospective_evidence_not_established",
                                               "human_approval_required"],
                          scope="same_cohort_validation_only"))

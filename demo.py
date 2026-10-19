"""합성 MES 데이터로 지속 학습 한 사이클을 끝까지 재현하는 데모.

1. 원천 행 → point-in-time 라벨 가용성 인덱스 (+ cutoff 별 receipt)
2. 기준 모델: 오래된 cutoff 까지의 이력·라벨로 학습 (동결 상태)
3. 후보 모델: 최신 cutoff 까지 "그 시점에 확정된" 라벨만으로 재학습
4. 같은 평가 행에서 기준 vs 후보 비교 → 고객사 단위 악화 차단 게이트

실행: python3 demo.py [--rows 6000] [--seed 0]
"""
import argparse
import json

import numpy as np
from scipy import sparse

from mes_cl.domain_features import PointInTimeDomainFeatureTransformer
from mes_cl.gate import ACTION_LABELS, compare_predictions
from mes_cl.labels import LabelAvailabilityIndex
from mes_cl.synthetic import generate, parse_date
from mes_cl.temporal import build_point_in_time_temporal_matrix

REVIEW_THRESHOLD = 0.55   # 최대 확률이 이보다 낮으면 사람 검토(REVIEW)로 보낸다


def features(rows, observed, label_index, cutoff, *, fit_rows=None):
    """cutoff 이전에 확정된 라벨만 쓰는 피처 행렬 (domain + temporal)."""
    domain = PointInTimeDomainFeatureTransformer()
    if fit_rows is not None:
        # 학습 행: 날짜 순 재생으로 out-of-fold 피처 (같은 날 라벨은 score 이후 반영)
        x_domain = domain.fit_transform_oof(fit_rows, all_observed_rows=observed, label_index=label_index)
        targets = fit_rows
    else:
        domain.fit_history(all_observed_rows=observed, label_index=label_index, cutoff_exclusive=cutoff)
        x_domain = domain.transform_frozen(rows)
        targets = rows
    temporal = build_point_in_time_temporal_matrix(
        targets, all_observed_rows=observed, label_index=label_index, label_cutoff_exclusive=cutoff)
    order = [temporal.row_ids.index(r.row_id) for r in targets] if targets else []
    x = sparse.hstack([x_domain, sparse.csr_matrix(temporal.values[order])]).toarray()
    return x


class SoftmaxModel:
    """의존성 최소화를 위한 numpy 다항 로지스틱 회귀 (표준화 + L2)."""

    def fit(self, x, y, *, epochs=400, lr=0.3, l2=1e-3):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mu) / self.sd
        onehot = np.eye(len(ACTION_LABELS))[y]
        self.w = np.zeros((z.shape[1], len(ACTION_LABELS)))
        self.b = np.zeros(len(ACTION_LABELS))
        for _ in range(epochs):
            p = self._softmax(z @ self.w + self.b)
            grad = p - onehot
            self.w -= lr * (z.T @ grad / len(z) + l2 * self.w)
            self.b -= lr * grad.mean(0)
        return self

    def predict(self, x):
        p = self._softmax(((x - self.mu) / self.sd) @ self.w + self.b)
        labels = [ACTION_LABELS[i] for i in p.argmax(1)]
        return [label if conf >= REVIEW_THRESHOLD else "REVIEW" for label, conf in zip(labels, p.max(1))]

    @staticmethod
    def _softmax(logits):
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def train(rows, observed, label_index, cutoff):
    labels = label_index.eligible_labels(cutoff)           # cutoff 시점에 이미 확정된 라벨만
    fit_rows = [r for r in rows if r.date < cutoff and r.row_id in labels]
    x = features(None, observed, label_index, cutoff, fit_rows=fit_rows)
    y = np.array([ACTION_LABELS.index(labels[r.row_id]) for r in fit_rows])
    return SoftmaxModel().fit(x, y), len(fit_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw_rows, rows, truth = generate(args.rows, drift_after="2026-04-01", seed=args.seed)
    index = LabelAvailabilityIndex.from_rows(raw_rows, source_dataset_id="synthetic", parse_date_value=parse_date)

    baseline_cutoff, candidate_cutoff, eval_start = "2026-03-01", "2026-05-15", "2026-05-15"
    receipt = index.receipt_payload(candidate_cutoff)
    print("label receipt @", candidate_cutoff, json.dumps(
        {k: receipt[k] for k in ("sourceRows", "actionRows", "eligibleRows")} |
        {"excludedRows": dict(receipt["excludedRows"])}, ensure_ascii=False))

    baseline, n_base = train(rows, rows, index, baseline_cutoff)
    candidate, n_cand = train(rows, rows, index, candidate_cutoff)
    print(f"train rows: baseline={n_base} (cutoff {baseline_cutoff}), candidate={n_cand} (cutoff {candidate_cutoff})")

    confirmed = {record.row_id for record in index.valid_records}
    eval_rows = sorted((r for r in rows if r.date >= eval_start and r.row_id in confirmed),
                       key=lambda r: (r.date, r.source_row))
    base_pred = baseline.predict(features(eval_rows, rows, index, baseline_cutoff))
    cand_pred = candidate.predict(features(eval_rows, rows, index, candidate_cutoff))

    cohort = [dict(id=r.row_id, date=r.date, customer=r.customer_key, line=r.features["line"],
                   product_key=r.product_key, actual=truth[r.row_id], baseline=b)
              for r, b in zip(eval_rows, base_pred)]
    report = compare_predictions(cohort, [dict(id=r.row_id, predicted=p) for r, p in zip(eval_rows, cand_pred)])

    o = report["overall"]
    print(f"eval rows: {o['count']}")
    for name in ("baseline", "candidate"):
        m = o[name]
        print(f"  {name:9s} auto_rate={m['auto_rate']:.3f} accuracy={m['accuracy']:.3f} "
              f"ng_to_process={m['ng_to_process']} process_to_ng={m['process_to_ng']}")
    for c in report["customers"]:
        d = c["delta"]
        print(f"  {c['customer']}: Δaccuracy={d['accuracy']:+.3f} Δng_to_process={d['ng_to_process']:+d} "
              f"supported={c['support']['supported']}")
    gate = report["gate"]
    print("gate passed:", gate["passed"], "| production_eligible:", gate["production_eligible"])
    for reason in gate["reasons"]:
        print("  -", reason)


if __name__ == "__main__":
    main()

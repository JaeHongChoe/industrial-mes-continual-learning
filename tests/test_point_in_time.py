"""Point-in-time 계약: 미래(또는 당일 확정) 라벨이 피처·학습 라벨로 새지 않는다."""
import pytest

from mes_cl.domain_features import PointInTimeDomainFeatureTransformer
from mes_cl.labels import LabelAvailabilityIndex
from mes_cl.synthetic import Row, parse_date
from mes_cl.temporal import build_point_in_time_temporal_matrix


def raw(n, occurred, available, label="REPAIR"):
    return {"__sourceRowNumber": n, "occurrence_date": occurred, "registered_date": occurred,
            "action_result": label, "action_result_date": available}


def row(n, day, product="p1", customer="c1"):
    return Row(row_id=f"ds:{n}", source_row=n, date=day, month=day[:7], customer_key=customer,
               product_key=product, features={"line": "l1", "type": "t", "detail": "d", "serial": f"s{n}"})


def index(raws):
    return LabelAvailabilityIndex.from_rows(raws, source_dataset_id="ds", parse_date_value=parse_date)


def test_labels_only_eligible_when_confirmed_before_cutoff():
    idx = index([raw(1, "2026-01-01", "2026-01-10"), raw(2, "2026-01-05", "2026-02-01")])
    assert dict(idx.eligible_labels("2026-01-20")) == {"ds:1": "REPAIR"}
    assert set(idx.eligible_labels("2026-02-02")) == {"ds:1", "ds:2"}


def test_receipt_counts_future_and_unavailable_rows():
    idx = index([raw(1, "2026-01-01", "2026-01-10"),   # eligible
                 raw(2, "2026-01-05", "2026-02-01"),   # occurred before cutoff, label after
                 raw(3, "2026-03-01", "2026-03-02")])  # occurred after cutoff
    receipt = idx.receipt_payload("2026-01-20")
    assert receipt["eligibleRows"] == 1
    assert receipt["excludedRows"]["notAvailableBeforeCutoff"] == 1
    assert receipt["excludedRows"]["futureOccurrenceDate"] == 1
    assert len(receipt["eligibleRowIdsSha256"]) == 64


def test_static_exclusions():
    idx = index([raw(1, "2026-01-10", "2026-01-01"),             # 확정일이 발생일보다 빠름
                 raw(2, "2026-01-10", None),                     # 확정일 누락
                 raw(3, "2026-01-10", "2026-01-12", label="")])  # 미확정 라벨
    assert idx.static_excluded_rows["actionBeforeOccurrence"] == 1
    assert idx.static_excluded_rows["invalidOrMissingLabelAvailableDate"] == 1
    assert idx.static_excluded_rows["nonActionLabel"] == 1
    assert idx.valid_records == ()


def test_temporal_snapshot_excludes_same_day_label():
    # 과거 행 1의 라벨은 2026-01-10에 확정. 같은 날 들어온 행 2는 그 라벨을 보면 안 되고, 다음 날 행 3은 봐야 한다.
    raws = [raw(1, "2026-01-01", "2026-01-10")]
    observed = [row(1, "2026-01-01"), row(2, "2026-01-10"), row(3, "2026-01-11")]
    matrix = build_point_in_time_temporal_matrix(observed[1:], all_observed_rows=observed, label_index=index(raws))
    assert matrix.features_for("ds:2")["product__lifetime_count"] == 0.0
    assert matrix.features_for("ds:3")["product__lifetime_count"] == 1.0
    assert matrix.features_for("ds:3")["product__lifetime_repair_rate"] == 1.0


def test_temporal_label_cutoff_freezes_history():
    raws = [raw(1, "2026-01-01", "2026-01-10")]
    observed = [row(1, "2026-01-01"), row(3, "2026-02-01")]
    matrix = build_point_in_time_temporal_matrix([observed[1]], all_observed_rows=observed,
                                                 label_index=index(raws), label_cutoff_exclusive="2026-01-05")
    assert matrix.features_for("ds:3")["product__lifetime_count"] == 0.0


def test_domain_oof_scores_before_same_day_update():
    raws = [raw(1, "2026-01-01", "2026-01-10")]
    observed = [row(1, "2026-01-01"), row(2, "2026-01-10"), row(3, "2026-01-11")]
    transformer = PointInTimeDomainFeatureTransformer()
    x = transformer.fit_transform_oof(observed[1:], all_observed_rows=observed, label_index=index(raws)).toarray()
    support = transformer.feature_names.index("history::product::support_log1p")
    assert x[0, support] == 0.0          # 당일 확정 라벨은 아직 반영 전
    assert x[1, support] > 0.0           # 다음 날에는 반영


def test_domain_frozen_rejects_rows_before_cutoff():
    raws = [raw(1, "2026-01-01", "2026-01-10")]
    observed = [row(1, "2026-01-01"), row(2, "2026-01-20")]
    transformer = PointInTimeDomainFeatureTransformer().fit_history(
        all_observed_rows=observed, label_index=index(raws), cutoff_exclusive="2026-01-15")
    with pytest.raises(ValueError):
        transformer.transform_frozen([row(9, "2026-01-14")])
    assert transformer.transform_frozen([observed[1]]).shape == (1, len(transformer.feature_names))

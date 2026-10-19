"""Stage B: features, temporal features and V4 domain features (point-in-time)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import sparse

from experiment_lib.point_in_time_labels import LabelAvailabilityIndex
from experiment_lib.point_in_time_temporal import build_point_in_time_temporal_matrix
from src.data_contract import parse_source_date
from src.features import FeatureSpec, RawFeatureEncoder
from src.temporal_features import TemporalFeatureBuilder
from src.v4_domain_features import V4DomainFeatureTransformer


def _encoder_rows(make_row):
    texts = [
        ("visual", "scratch", "minor scratch on surface"),
        ("visual", "stain", "stain near edge"),
        ("leak", "leak", "leak test fail"),
        ("crack", "crack", "crack on flange"),
        ("visual", "ok", "no defect found"),
        ("count", "limit", "usage count exceeded"),
    ]
    return [
        make_row(
            index + 1,
            f"2025-03-{index + 1:02d}",
            type_=type_,
            symptom=symptom,
            detail=detail,
            intake_count=str(index),
            usage_hours=str(10 * index),
        )
        for index, (type_, symptom, detail) in enumerate(texts)
    ]


def test_raw_feature_encoder_shapes_and_deterministic_schema(make_row):
    rows = _encoder_rows(make_row)
    encoder = RawFeatureEncoder(FeatureSpec())
    matrix = encoder.fit_transform(rows[:4])
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (4, encoder.feature_count)
    assert len(encoder.feature_names) == encoder.feature_count

    unseen = encoder.transform(rows[4:])
    assert unseen.shape == (2, encoder.feature_count)
    assert np.isfinite(unseen.data).all()

    again = RawFeatureEncoder(FeatureSpec())
    again.fit(rows[:4])
    assert again.schema_sha256 == encoder.schema_sha256
    assert again.feature_names == encoder.feature_names


def test_raw_feature_encoder_requires_fit(make_row):
    with pytest.raises(RuntimeError):
        RawFeatureEncoder().transform(_encoder_rows(make_row)[:1])


def test_temporal_prequential_same_day_labels_do_not_leak(make_row):
    r1 = make_row(1, "2025-05-01", detail="scratch")
    r2 = make_row(2, "2025-05-01", detail="scratch")
    r3 = make_row(3, "2025-05-02", detail="scratch")
    labels = {r1.row_id: "PROCESS", r2.row_id: "RETURN", r3.row_id: "PROCESS"}

    matrix = TemporalFeatureBuilder().prequential_transform([r3, r1, r2], labels)
    assert matrix.values.shape == (3, len(matrix.feature_names))
    assert len(matrix.schema_sha256) == 64

    # Rows on the same day see neither their own nor each other's label.
    for row in (r1, r2):
        features = matrix.features_for(row.row_id)
        assert features["product__lifetime_count"] == 0.0
        assert features["serial_prior_count"] == 0.0
    later = matrix.features_for(r3.row_id)
    assert later["product__lifetime_count"] == 2.0
    assert later["product__lifetime_return_rate"] == pytest.approx(0.5)


def test_temporal_frozen_transform_rejects_rows_not_after_history(make_row):
    history = [make_row(1, "2025-05-01"), make_row(2, "2025-05-02")]
    builder = TemporalFeatureBuilder().fit_prior(
        history, {row.row_id: "PROCESS" for row in history}
    )
    future = builder.transform_frozen([make_row(3, "2025-05-10")])
    assert future.features_for("synthetic_ds:3")["product__lifetime_count"] == 2.0
    with pytest.raises(ValueError):
        builder.transform_frozen([make_row(4, "2025-05-02")])


def _labels_index(raws):
    return LabelAvailabilityIndex.from_rows(
        raws, source_dataset_id="synthetic_ds", parse_date_value=parse_source_date_value
    )


def parse_source_date_value(value):
    return parse_source_date({"occurrence_date": value}) or None


def test_label_index_cutoff_is_exclusive(raw_factory):
    raws = [
        raw_factory(1, "2025-06-01", label="REPAIR", action_result_date="2025-06-05"),
        raw_factory(2, "2025-06-01", label="PROCESS", action_result_date="2025-06-10"),
        raw_factory(3, "2025-06-02", label="REVIEW", action_result_date="2025-06-03"),
        raw_factory(4, "2025-06-05", label="RETURN", action_result_date="2025-06-04"),
    ]
    index = _labels_index(raws)
    # Confirmed on the cutoff day -> not yet available.
    assert dict(index.eligible_labels("2025-06-10")) == {"synthetic_ds:1": "REPAIR"}
    assert set(index.eligible_labels("2025-06-11")) == {"synthetic_ds:1", "synthetic_ds:2"}
    assert index.static_excluded_rows["nonActionLabel"] == 1
    assert index.static_excluded_rows["actionBeforeOccurrence"] == 1


def test_point_in_time_temporal_matrix_hides_same_day_confirmations(raw_factory, make_row):
    # Two earlier intakes of the same product; one label is confirmed the day
    # before the target, the other ON the target day (must not be visible).
    raws = [
        raw_factory(1, "2025-06-01", label="REPAIR", action_result_date="2025-06-09"),
        raw_factory(2, "2025-06-02", label="PROCESS", action_result_date="2025-06-10"),
    ]
    observed = [make_row(1, "2025-06-01"), make_row(2, "2025-06-02"), make_row(3, "2025-06-10")]
    target = [observed[2]]
    matrix = build_point_in_time_temporal_matrix(
        target, all_observed_rows=observed, label_index=_labels_index(raws)
    )
    assert matrix.values.shape == (1, len(matrix.feature_names))
    features = matrix.features_for("synthetic_ds:3")
    assert features["product__lifetime_count"] == 1.0
    assert features["product__lifetime_repair_rate"] == 1.0
    assert features["product__lifetime_process_rate"] == 0.0

    # An explicit label cutoff hides even the earlier confirmation.
    cut = build_point_in_time_temporal_matrix(
        target,
        all_observed_rows=observed,
        label_index=_labels_index(raws),
        label_cutoff_exclusive="2025-06-09",
    )
    assert cut.features_for("synthetic_ds:3")["product__lifetime_count"] == 0.0


def test_v4_domain_features_oof_is_month_prior_only(make_row):
    january = [
        make_row(1, "2025-01-05", detail="leak test fail", intake_count="2", service_count="1"),
        make_row(2, "2025-01-20", detail="minor scratch"),
    ]
    february = [make_row(3, "2025-02-03", detail="minor scratch")]
    rows = (*january, *february)
    labels = {rows[0].row_id: "REPAIR", rows[1].row_id: "PROCESS", rows[2].row_id: "PROCESS"}

    transformer = V4DomainFeatureTransformer()
    matrix = transformer.fit_transform_oof(rows, labels).toarray()
    names = transformer.feature_names
    assert matrix.shape == (3, len(names))
    col = {name: index for index, name in enumerate(names)}

    # Same-month rows do not see each other's labels.
    for index in (0, 1):
        assert matrix[index, col["history::cold_start"]] == 1.0
        assert matrix[index, col["history::product::support_log1p"]] == 0.0
    assert matrix[2, col["history::cold_start"]] == 0.0
    assert matrix[2, col["history::product::support_log1p"]] == pytest.approx(math.log1p(2))
    assert matrix[0, col["semantic::leak"]] == 1.0
    assert matrix[1, col["semantic::leak"]] == 0.0
    assert matrix[0, col["numeric::intake_minus_service"]] == 1.0
    assert len(transformer.schema_sha256) == 64


def test_v4_domain_transform_with_history_rejects_rows_on_cutoff(make_row):
    history = [make_row(1, "2025-01-05"), make_row(2, "2025-01-10")]
    labels = {history[0].row_id: "PROCESS", history[1].row_id: "RETURN"}
    transformer = V4DomainFeatureTransformer().fit_history(history, labels, cutoff="2025-01-10")
    scored = transformer.transform_with_history([make_row(3, "2025-01-11")])
    assert scored.shape == (1, len(transformer.feature_names))
    with pytest.raises(ValueError):
        transformer.transform_with_history([make_row(4, "2025-01-10")])

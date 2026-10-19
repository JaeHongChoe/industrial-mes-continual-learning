"""Stage A: input contract (field canonicalization, row identity, dates, FeatureRow)."""

from __future__ import annotations

import select as stdlib_select
from datetime import date, timedelta

import pytest

from src import data_contract
from src.data_contract import (
    ALLOWED_RAW_FEATURES,
    DataContractError,
    FeatureSchemaError,
    canonical_field_name,
    parse_source_date,
    stable_row_id,
    validate_requested_features,
)
from src.shadow_validation import input_contract
from src.shadow_validation.input_contract import InputContractError, load_input_files


def test_import_path_keeps_stdlib_select():
    # src/select.py must not shadow the stdlib module (see conftest).
    assert hasattr(stdlib_select, "select")
    assert "v5_pipeline" not in str(getattr(stdlib_select, "__file__", ""))


def test_canonical_field_name_is_nfkc_lower_alnum_and_consistent():
    assert canonical_field_name("Action_Result ") == "actionresult"
    assert canonical_field_name("ＵＳＡＧＥ　Hours") == "usagehours"  # full-width
    assert canonical_field_name(None) == ""
    for value in ("Occurrence Date", "intake-count", "P/R"):
        assert canonical_field_name(value) == input_contract.canonical_field_name(value)


def test_stable_row_id_and_invalid_numbers():
    assert stable_row_id("synthetic_ds", "12.0") == "synthetic_ds:12"
    assert stable_row_id("synthetic_ds", 3) == "synthetic_ds:3"
    for bad in (0, -1, "x", None):
        with pytest.raises(DataContractError):
            stable_row_id("synthetic_ds", bad)


def test_parse_source_date_formats_and_fail_closed():
    assert parse_source_date({"occurrence_date": "2025/03/04"}) == "2025-03-04"
    # Falls back to registered_date when occurrence_date is blank.
    assert parse_source_date({"occurrence_date": "", "registered_date": "20250305"}) == "2025-03-05"
    excel_serial = 45000
    expected = (date(1899, 12, 30) + timedelta(days=excel_serial)).isoformat()
    assert parse_source_date({"occurrence_date": excel_serial}) == expected
    # Post-decision dates must never place a row on the timeline.
    assert parse_source_date({"action_result_date": "2025-03-06"}) == ""
    assert parse_source_date({"occurrence_date": "2025-02-30"}) == ""


def test_feature_row_uses_intake_allow_list_only(make_row):
    row = make_row(
        7,
        "2025-04-02",
        detail="minor scratch",
        label="REPAIR",
        action_result_date="2025-04-10",
        customer_comment="should never leak",
        intake_count="3",
    )
    assert row.row_id == "synthetic_ds:7"
    assert row.date == "2025-04-02" and row.month == "2025-04"
    assert row.product_key == "PC01-PT01"
    assert row.customer_key == "customer_a"
    assert row.episode_basis == "serial"
    assert len(row.episode_id) == 64
    assert tuple(row.features) == ALLOWED_RAW_FEATURES
    for leaked in ("action_result", "action_result_date", "customer_comment"):
        assert leaked not in row.features
    assert row.features["intake_count"] == "3"


def test_feature_row_episode_is_deterministic(make_row):
    first = make_row(1, "2025-04-02", serial="SN-X")
    second = make_row(2, "2025-04-02", serial="SN-X")
    # Same customer/serial/product/date -> same episode, different row identity.
    assert first.episode_id == second.episode_id
    assert first.row_id != second.row_id


def test_validate_requested_features_rejects_forbidden_and_unknown():
    config = {"forbiddenFeatureNames": ["action_result"], "forbiddenFeaturePatterns": [r"comment"]}
    assert validate_requested_features(["Detail", "intake count"], config) == ("detail", "intake_count")
    with pytest.raises(FeatureSchemaError):
        validate_requested_features(["Action Result"], config)
    with pytest.raises(FeatureSchemaError):
        validate_requested_features(["customer_comment"], config)
    with pytest.raises(FeatureSchemaError):
        validate_requested_features(["not_a_field"], {})


def _write_csv(path, rows):
    header = ["Occurrence Date", "Customer", "Process Code", "Part Code", "Serial", "Detail"]
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_input_files_dry_run_is_deterministic(tmp_path):
    path = tmp_path / "intake.csv"
    _write_csv(
        path,
        [
            ["2025-09-01", "customer_a", "PC01", "PT01", "SN0001", "minor scratch"],
            ["2025-09-02", "customer_b", "PC02", "", "SN0002", "leak test fail"],
        ],
    )
    first = load_input_files([path], mode="dry_run")
    second = load_input_files([path], mode="dry_run")
    assert [row.row_id for row in first.rows] == [row.row_id for row in second.rows]
    assert len(first.rows) == 2
    by_customer = {row.raw_customer: row for row in first.rows}
    assert by_customer["customer_a"].product_key == "PC01-PT01"
    assert by_customer["customer_b"].product_key == "PC02"
    # Header canonicalization: "Occurrence Date" -> occurrence_date.
    assert by_customer["customer_a"].feature_payload["occurrence_date"] == "2025-09-01"
    assert all(row.run_scope == "dry_run_only" for row in first.rows)
    assert first.input_files[0]["rowCount"] == 2
    assert len(first.input_files[0]["sha256"]) == 64


def test_load_input_files_prospective_rejects_pre_shadow_dates(tmp_path):
    path = tmp_path / "intake.csv"
    _write_csv(path, [["2025-09-01", "customer_a", "PC01", "PT01", "SN0001", "ok"]])
    with pytest.raises(InputContractError):
        load_input_files([path], mode="prospective")
    with pytest.raises(InputContractError):
        load_input_files([path], mode="unknown_mode")


def test_prospective_input_rejects_nonblank_postdecision_field(tmp_path):
    path = tmp_path / "intake.csv"
    path.write_text(
        "occurrence_date,customer,process_code,part_code,detail,action_result\n"
        "2026-08-03,customer_a,PC01,PT01,ok,PROCESS\n",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError):
        load_input_files([path], mode="prospective")


def test_data_contract_module_constants_are_generic():
    assert data_contract.ACTION_LABELS == frozenset({"PROCESS", "RETURN", "REPAIR"})
    assert "action_result" not in ALLOWED_RAW_FEATURES

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from . import GENERATION_ID
from .input_contract import InputRow, canonical_field_name, load_input_files
from .integrity import IntegrityError, verify_manifest
from .metrics import (
    Metrics,
    MetricsError,
    OutcomeRow,
    PromotionGates,
    compute_metrics,
    evaluate_promotion_gates,
    grouped_metrics,
    normalize_action_label,
    reconcile_group_metrics,
)
from .receipt import PREDICTIONS_FILE, PredictionReceiptError, verify_prediction_receipt


class AuditError(RuntimeError):
    """Raised when sealed predictions and later labels cannot be audited safely."""


@dataclass(frozen=True, slots=True)
class AuditOutcomeRow:
    row_id: str
    customer: str
    month: str
    date: str
    source_file: str
    source_row: int
    predicted: str
    actual: str | None


@dataclass(frozen=True, slots=True)
class AuditResult:
    as_of_date: str
    receipt: Mapping[str, Any]
    rows: tuple[AuditOutcomeRow, ...]
    unresolved_row_ids: tuple[str, ...]
    overall: Metrics
    company_metrics: Mapping[str, Metrics]
    monthly_metrics: Mapping[str, Metrics]
    critical_errors: tuple[Mapping[str, Any], ...]
    promotion_gates: PromotionGates
    reconciled: bool


def audit_run(
    run_dir: str | Path,
    label_paths: Iterable[str | Path],
    *,
    final: bool = False,
    as_of_date: str | None = None,
) -> AuditResult:
    """Audit a sealed prediction run without ever reopening labels before receipt checks."""
    try:
        receipt = verify_prediction_receipt(run_dir)
    except PredictionReceiptError:
        raise
    if receipt.get("generationId") != GENERATION_ID:
        raise AuditError("prediction receipt generation does not match the frozen generation")
    _verify_frozen_generation(receipt)

    label_batch = load_input_files(label_paths, mode="audit")
    sealed_ids = _receipt_ids(receipt)
    label_by_id = _validate_label_universe(label_batch.rows, sealed_ids)
    _validate_feature_hashes(label_by_id, receipt.get("featureHashes"))
    predictions = _load_predictions(Path(run_dir) / PREDICTIONS_FILE, sealed_ids)

    rows: list[AuditOutcomeRow] = []
    outcome_rows: list[OutcomeRow] = []
    unresolved: list[str] = []
    for row_id in sealed_ids:
        label_row = label_by_id[row_id]
        actual = _label_action(label_row)
        predicted = _prediction_action(predictions[row_id], row_id)
        audit_row = AuditOutcomeRow(
            row_id=row_id,
            customer=label_row.raw_customer,
            month=label_row.month,
            date=label_row.date,
            source_file=label_row.source_file,
            source_row=label_row.source_row,
            predicted=predicted,
            actual=actual,
        )
        rows.append(audit_row)
        if actual is None:
            unresolved.append(row_id)
            continue
        outcome_rows.append(
            OutcomeRow(
                row_id=row_id,
                company=audit_row.customer,
                month=audit_row.month,
                actual=actual,
                predicted=predicted,
            )
        )
    if final and unresolved:
        raise AuditError(f"final audit has unresolved labels: {len(unresolved)}")

    overall = compute_metrics(outcome_rows)
    by_company = grouped_metrics(outcome_rows, "company")
    by_month = grouped_metrics(outcome_rows, "month")
    reconciled = reconcile_group_metrics(overall, by_company) and reconcile_group_metrics(overall, by_month)
    audit_date = _audit_date(as_of_date)
    current_month = audit_date[:7]
    complete_months = ({
        row.month for row in rows if row.actual is not None
    } - {
        row.month for row in rows if row.actual is None
    })
    complete_months = {month for month in complete_months if month < current_month}
    promotion_gates = evaluate_promotion_gates(
        overall,
        by_month,
        by_company,
        complete_months=complete_months,
        unresolved_rows=len(unresolved),
    )
    return AuditResult(
        as_of_date=audit_date,
        receipt=MappingProxyType(dict(receipt)),
        rows=tuple(rows),
        unresolved_row_ids=tuple(unresolved),
        overall=overall,
        company_metrics=by_company,
        monthly_metrics=by_month,
        critical_errors=_critical_errors(rows),
        promotion_gates=promotion_gates,
        reconciled=reconciled,
    )


def _audit_date(value: str | None) -> str:
    text = value or calendar_date.today().isoformat()
    try:
        return calendar_date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise AuditError(f"invalid audit as-of date: {text}") from error


def _verify_frozen_generation(receipt: Mapping[str, Any]) -> None:
    input_hashes = receipt.get("inputHashes")
    if not isinstance(input_hashes, Mapping):
        raise AuditError("prediction receipt input hashes are malformed")
    manifest_entry = input_hashes.get("frozenManifest")
    if not isinstance(manifest_entry, Mapping):
        raise AuditError("prediction receipt has no frozen manifest")
    manifest_path = Path(str(manifest_entry.get("path") or "")).resolve()
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "frozen":
        raise AuditError("frozen manifest path is outside the expected package layout")
    package_root = manifest_path.parent.parent
    try:
        verify_manifest(
            package_root,
            manifest_path,
            expected_generation=GENERATION_ID,
        )
    except IntegrityError as error:
        raise AuditError(f"frozen artifact verification failed: {error}") from error


def _receipt_ids(receipt: Mapping[str, Any]) -> tuple[str, ...]:
    raw = receipt.get("sortedRowIds")
    if not isinstance(raw, list) or any(not isinstance(row_id, str) for row_id in raw):
        raise AuditError("prediction receipt row IDs are malformed")
    return tuple(raw)


def _validate_label_universe(
    rows: Iterable[InputRow],
    sealed_ids: tuple[str, ...],
) -> Mapping[str, InputRow]:
    output: dict[str, InputRow] = {}
    for row in rows:
        if row.row_id in output:
            raise AuditError(f"duplicate label row ID: {row.row_id}")
        output[row.row_id] = row
    if set(output) != set(sealed_ids):
        missing = len(set(sealed_ids) - set(output))
        extra = len(set(output) - set(sealed_ids))
        raise AuditError(f"label row universe mismatch: missing={missing}, extra={extra}")
    return MappingProxyType(output)


def _validate_feature_hashes(rows: Mapping[str, InputRow], raw_hashes: Any) -> None:
    if not isinstance(raw_hashes, Mapping):
        raise AuditError("prediction receipt feature hashes are malformed")
    expected = {str(row_id): str(value) for row_id, value in raw_hashes.items()}
    observed = {row_id: row.feature_sha256 for row_id, row in rows.items()}
    if observed != expected:
        raise AuditError("label feature hash mismatch")


def _load_predictions(path: Path, sealed_ids: tuple[str, ...]) -> Mapping[str, Mapping[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "rowId" not in reader.fieldnames:
                raise AuditError("predictions CSV has no rowId column")
            output = {str(row.get("rowId") or ""): row for row in reader}
    except (OSError, csv.Error) as error:
        raise AuditError("predictions CSV is unreadable") from error
    if set(output) != set(sealed_ids) or len(output) != len(sealed_ids):
        raise AuditError("prediction row universe differs from receipt")
    return MappingProxyType(output)


def _label_action(row: InputRow) -> str | None:
    value = row.raw.get("action_result", "")
    if canonical_field_name(value) in {"", "미기재", "na", "없음", "unconfirmed"}:
        return None
    try:
        return normalize_action_label(value)
    except MetricsError as error:
        raise AuditError(f"unknown 조치결과 for {row.row_id}: {value}") from error


def _prediction_action(prediction: Mapping[str, str], row_id: str) -> str:
    try:
        return normalize_action_label(prediction.get("finalPrediction", ""), allow_review=True)
    except MetricsError as error:
        raise AuditError(f"unknown finalPrediction for {row_id}") from error


def _critical_errors(rows: Iterable[AuditOutcomeRow]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        MappingProxyType(
            {
                "rowId": row.row_id,
                "customer": row.customer,
                "month": row.month,
                "date": row.date,
                "sourceFile": row.source_file,
                "sourceRow": row.source_row,
                "actual": row.actual,
                "predicted": row.predicted,
            }
        )
        for row in rows
        if row.actual in {"RETURN", "REPAIR"} and row.predicted == "PROCESS"
    )

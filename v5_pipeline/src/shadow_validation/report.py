from __future__ import annotations

import csv
import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .audit import AuditResult
from .metrics import GateCheck, Metrics


ARTIFACT_NAMES = (
    "audit_summary.json",
    "audit_rows.csv",
    "company_metrics.csv",
    "monthly_metrics.csv",
    "critical_errors.csv",
    "report.html",
)

_METRIC_COLUMNS = (
    ("evaluationRows", "평가행", "evaluation_rows"),
    ("autoRows", "자동판정", "auto_rows"),
    ("reviewRows", "검토행", "review_rows"),
    ("automationRate", "자동화율", "automation_rate"),
    ("actualProcessAutomationRate", "실제 공정투입 자동판정률", "actual_process_automation_rate"),
    ("correctProcessRecall", "정확 공정투입 회수율", "correct_process_recall"),
    ("autoAccuracy", "자동판정 정확도", "auto_accuracy"),
    ("shipToProcess", "고객사 출고 -> 공정투입", "ship_to_process"),
    ("repairToProcess", "수리요청 -> 공정투입", "repair_to_process"),
    ("riskToProcess", "치명오검 합", "risk_to_process"),
    ("processToShip", "공정투입 -> 고객사 출고", "process_to_ship"),
    ("processToRepair", "공정투입 -> 수리요청", "process_to_repair"),
)
_CRITICAL_COLUMNS = (
    ("rowId", "rowId"),
    ("customer", "customer"),
    ("month", "month"),
    ("date", "date"),
    ("sourceFile", "sourceFile"),
    ("sourceRow", "sourceRow"),
    ("actual", "actual"),
    ("predicted", "predicted"),
)
_AUDIT_ROW_COLUMNS = (
    ("rowId", "row_id"),
    ("customer", "customer"),
    ("month", "month"),
    ("date", "date"),
    ("sourceFile", "source_file"),
    ("sourceRow", "source_row"),
    ("actual", "actual"),
    ("predicted", "predicted"),
)


def write_audit_artifacts(
    result: AuditResult,
    output_dir: str | Path,
) -> Mapping[str, Path]:
    """Write the static, receipt-bound audit surfaces without changing audit results."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload(result)
    paths = {name: root / name for name in ARTIFACT_NAMES}
    _write_json(paths["audit_summary.json"], summary)
    _write_audit_rows_csv(paths["audit_rows.csv"], result)
    _write_metrics_csv(paths["company_metrics.csv"], "customer", result.company_metrics)
    _write_metrics_csv(paths["monthly_metrics.csv"], "month", result.monthly_metrics)
    _write_critical_csv(paths["critical_errors.csv"], result.critical_errors)
    _write_text(paths["report.html"], _render_html(result, summary))
    return paths


def _summary_payload(result: AuditResult) -> dict[str, Any]:
    mode = str(result.receipt.get("mode") or "")
    is_dry_run = mode == "dry_run"
    return {
        "schemaVersion": 1,
        "evidenceStatus": "dry_run_only" if is_dry_run else "prospective_shadow_observation",
        "prospectiveEvidence": not is_dry_run,
        "prospectiveStartDate": "2026-08-01",
        "asOfDate": result.as_of_date,
        "closedMonthRule": "fully_labelled_months_strictly_before_as_of_month",
        "receipt": _receipt_payload(result.receipt),
        "overall": _metric_payload(result.overall),
        "unresolvedRows": len(result.unresolved_row_ids),
        "criticalErrorRows": len(result.critical_errors),
        "reconciled": result.reconciled,
        "promotionEligible": result.promotion_gates.eligible,
        "promotionGates": {
            name: _gate_payload(check)
            for name, check in sorted(result.promotion_gates.checks.items())
        },
        "companySamples": {
            name: {"evaluationRows": sample.evaluation_rows, "status": sample.status}
            for name, sample in sorted(result.promotion_gates.company_samples.items())
        },
    }


def _receipt_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generationId": receipt.get("generationId"),
        "mode": receipt.get("mode"),
        "sealedAtUtc": receipt.get("sealedAtUtc"),
        "integrityLevel": receipt.get("integrityLevel"),
        "receiptSha256": receipt.get("receiptSha256"),
        "predictionCommitmentSha256": receipt.get("predictionCommitmentSha256"),
        "rowCount": receipt.get("rowCount"),
    }


def _metric_payload(metric: Metrics) -> dict[str, int | float | None]:
    return {key: getattr(metric, attribute) for key, _, attribute in _METRIC_COLUMNS}


def _gate_payload(check: GateCheck) -> dict[str, int | float | str | bool]:
    return {
        "observed": check.observed,
        "threshold": check.threshold,
        "comparator": check.comparator,
        "passed": check.passed,
    }


def _write_metrics_csv(path: Path, key_name: str, metrics: Mapping[str, Metrics]) -> None:
    fieldnames = [key_name, *(key for key, _, _ in _METRIC_COLUMNS)]
    rows = [
        {key_name: name, **_metric_payload(metric)}
        for name, metric in sorted(metrics.items())
    ]
    _write_csv(path, fieldnames, rows)


def _write_critical_csv(path: Path, rows: tuple[Mapping[str, Any], ...]) -> None:
    _write_csv(
        path,
        [key for key, _ in _CRITICAL_COLUMNS],
        [
            {key: row.get(source, "") for key, source in _CRITICAL_COLUMNS}
            for row in rows
        ],
    )


def _write_audit_rows_csv(path: Path, result: AuditResult) -> None:
    """Export the receipt-bound row universe used by the audit metrics.

    The status distinguishes a human-review decision from a wrong automatic
    decision.  This prevents the browser from treating review rows as model
    errors when it filters or paginates the detailed table.
    """
    fieldnames = [key for key, _ in _AUDIT_ROW_COLUMNS] + ["status"]
    rows = []
    for row in result.rows:
        payload = {
            key: getattr(row, attribute) if getattr(row, attribute) is not None else ""
            for key, attribute in _AUDIT_ROW_COLUMNS
        }
        payload["status"] = _audit_row_status(row.actual, row.predicted)
        rows.append(payload)
    _write_csv(path, fieldnames, rows)


def _audit_row_status(actual: str | None, predicted: str) -> str:
    if actual is None:
        return "UNCONFIRMED"
    if predicted == "REVIEW_REQUIRED":
        return "REVIEW"
    return "일치" if actual == predicted else "불일치"


def _write_csv(path: Path, fieldnames: list[str], rows: list[Mapping[str, Any]]) -> None:
    with _temporary_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, value: str) -> None:
    with _temporary_path(path) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())


class _temporary_path:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.path: Path | None = None

    def __enter__(self) -> Path:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{self.target.name}.", dir=self.target.parent)
        os.close(descriptor)
        self.path = Path(raw_path)
        return self.path

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.path is None:
            return
        if exc_type is None:
            os.replace(self.path, self.target)
        else:
            self.path.unlink(missing_ok=True)


def _render_html(result: AuditResult, summary: Mapping[str, Any]) -> str:
    is_dry_run = summary["evidenceStatus"] == "dry_run_only"
    evidence = (
        "DRY RUN - 기능 검증 전용. 이 결과는 실제 Shadow 관찰 근거가 아닙니다."
        if is_dry_run
        else "실제 Shadow 관찰 - 2026-08-01 이후 예측 시점에 미관측이었던 행만 포함합니다."
    )
    status = "미승격" if not summary["promotionEligible"] else "승격 기준 충족"
    overall = _metric_payload(result.overall)
    kpis = (
        ("평가행", overall["evaluationRows"]),
        ("UNCONFIRMED", summary["unresolvedRows"]),
        ("자동화율", _percent(overall["automationRate"])),
        ("정확 공정투입 회수율", _percent(overall["correctProcessRecall"])),
        ("자동판정 정확도", _percent(overall["autoAccuracy"])),
        ("치명오검", summary["criticalErrorRows"]),
    )
    return """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MES Shadow Audit</title>
<style>
:root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f5f7f8; }
body { margin: 0; padding: 24px; }
main { max-width: 1280px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 8px; }
p { margin: 0; line-height: 1.5; }
.meta { color: #53616d; font-size: 13px; }
.banner { margin-top: 18px; padding: 10px 12px; border-left: 4px solid #bd5a00; background: #fff4dc; color: #673500; font-weight: 650; }
.banner.live { border-left-color: #157347; background: #e8f5ed; color: #104f31; }
.kpis { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-top: 18px; }
.kpi { min-width: 0; border: 1px solid #d8e0e5; background: #fff; padding: 12px; }
.kpi dt { color: #5f6c76; font-size: 12px; }
.kpi dd { margin: 5px 0 0; font-size: 20px; font-weight: 700; overflow-wrap: anywhere; }
.status { display: inline-block; margin-top: 12px; padding: 4px 8px; border: 1px solid #c7d0d7; font-size: 13px; font-weight: 650; }
.table-wrap { overflow-x: auto; border: 1px solid #d8e0e5; background: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 9px; text-align: left; border-bottom: 1px solid #e5eaed; vertical-align: top; overflow-wrap: anywhere; }
th { background: #f0f4f6; color: #40505e; font-weight: 650; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
.pass { color: #0c6c42; font-weight: 650; }
.fail { color: #a13f17; font-weight: 650; }
.integrity { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; font-size: 13px; }
.integrity dt { color: #5f6c76; }
.integrity dd { margin: 0; overflow-wrap: anywhere; }
@media (max-width: 780px) { body { padding: 14px; } .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
</head>
<body><main>
""" + f"""
<h1>MES Shadow 감사 보고서</h1>
<p class="meta">세대 {_escape(summary['receipt']['generationId'])} | 봉인 {_escape(summary['receipt']['sealedAtUtc'])} | {_escape(summary['asOfDate'])} 기준</p>
<p class="banner{' ' if is_dry_run else ' live'}">{_escape(evidence)}</p>
<dl class="kpis">{''.join(_kpi_html(label, value) for label, value in kpis)}</dl>
<p class="status">승격 판정: {_escape(status)} | 범위 정합성: {_escape('일치' if result.reconciled else '불일치')}</p>
<p class="meta">완결 월은 감사 기준 월의 직전 달까지이며, 해당 월의 봉인 행 전체에 라벨이 있어야 합니다.</p>
<h2>전체 지표</h2>
{_metrics_table({'전체': result.overall}, 'scope')}
<h2>고객사별 지표</h2>
{_metrics_table(result.company_metrics, 'customer')}
<h2>월별 지표</h2>
{_metrics_table(result.monthly_metrics, 'month')}
<h2>승격 게이트</h2>
{_gates_table(result)}
<h2>치명오검 상세</h2>
{_critical_table(result.critical_errors)}
<h2>Receipt / Integrity</h2>
{_integrity_html(summary['receipt'])}
</main></body></html>
"""


def _kpi_html(label: str, value: object) -> str:
    return f"<div class=\"kpi\"><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"


def _metrics_table(metrics: Mapping[str, Metrics], first_heading: str) -> str:
    headers = [(first_heading, first_heading), *((key, label) for key, label, _ in _METRIC_COLUMNS)]
    rows = []
    for name, metric in sorted(metrics.items()):
        payload = _metric_payload(metric)
        cells = [name] + [
            _percent(payload[key]) if key.endswith("Rate") or key.endswith("Recall") or key == "autoAccuracy" else payload[key]
            for key, _, _ in _METRIC_COLUMNS
        ]
        rows.append(cells)
    return _table(headers, rows)


def _gates_table(result: AuditResult) -> str:
    rows = [
        (name, check.observed, check.comparator, check.threshold, "통과" if check.passed else "미통과")
        for name, check in sorted(result.promotion_gates.checks.items())
    ]
    return _table(
        [("name", "게이트"), ("observed", "관측값"), ("comparator", "조건"), ("threshold", "기준"), ("status", "상태")],
        rows,
        status_column=4,
    )


def _critical_table(rows: tuple[Mapping[str, Any], ...]) -> str:
    values = [[row.get(source, "") for _, source in _CRITICAL_COLUMNS] for row in rows]
    return _table(list(_CRITICAL_COLUMNS), values)


def _integrity_html(receipt: Mapping[str, Any]) -> str:
    fields = (
        ("무결성 수준", receipt.get("integrityLevel")),
        ("receipt SHA-256", receipt.get("receiptSha256")),
        ("prediction commitment SHA-256", receipt.get("predictionCommitmentSha256")),
        ("봉인 행 수", receipt.get("rowCount")),
    )
    return "<dl class=\"integrity\">" + "".join(
        f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in fields
    ) + "</dl>"


def _table(
    headers: list[tuple[str, str]],
    rows: list[object],
    *,
    status_column: int | None = None,
) -> str:
    head = "".join(f"<th>{_escape(label)}</th>" for _, label in headers)
    body = []
    for row in rows:
        values = list(row)
        cells = []
        for index, value in enumerate(values):
            class_name = ""
            if index == status_column:
                class_name = " class=\"pass\"" if value == "통과" else " class=\"fail\""
            cells.append(f"<td{class_name}>{_escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    if not body:
        body.append(f"<tr><td colspan=\"{len(headers)}\">없음</td></tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _percent(value: object) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def _escape(value: object) -> str:
    return html.escape(str(value if value is not None else "-"), quote=True)

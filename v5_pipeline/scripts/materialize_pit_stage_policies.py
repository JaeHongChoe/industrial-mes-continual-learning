from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_lib.point_in_time_labels import LabelAvailabilityIndex
from src import data_contract as data_runtime
from src.prior_supported_process_recovery import PriorSupportedProcessRecovery
from src.prior_supported_repair_guard import PriorSupportedRepairGuard
from src.shadow_validation.feature_bridge import load_frozen_history
from src.shadow_validation.integrity import canonical_json_sha256


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PACKAGE_ROOT / "frozen/history/history_rows.jsonl.gz"
GENERATION_PATH = PACKAGE_ROOT / "frozen/generation.json"
SEMANTIC_SOURCE_PATH = PACKAGE_ROOT / "build/pit_semantic_policy_source.json"
TRAINING_END = "2025-12-31"
CUTOFF_EXCLUSIVE = "2026-01-01"

EXPECTED_FIT_ROW_IDS_SHA256 = (
    "4d1ea5ecfb06eb49c428cd083820ba3f79c91b2eba7bf126b808370623f3b4ea"
)
EXPECTED_GUARD_EVIDENCE_SHA256 = (
    "37925f513c6a36e4560dac32ed736456f9082c11e192e8b7f5cec0f4adc4e49c"
)
EXPECTED_RECOVERY_MODEL_SHA256 = (
    "cce424e5e13eee3994dcb6f2610aef3129c409fbf7c45e4b25653709a520eb79"
)
EXPECTED_RECOVERY_SIGNATURES = 12_413
EXPECTED_SEMANTIC_CONFIG_SHA256 = (
    "baba3b7b3023b53fa19df1201a59e0d89993814b640edc35f5ffcbd03cce2f15"
)


def build_guard_policy(
    guard: PriorSupportedRepairGuard,
    *,
    source_dataset_id: str,
) -> dict[str, Any]:
    evidence = {
        company: {
            signal_id: {
                "customerKey": item.customer_key,
                "signalId": item.signal_id,
                "support": item.support,
                "processRows": item.process_rows,
                "shipRows": item.ship_rows,
                "repairRows": item.repair_rows,
                "riskRows": item.risk_rows,
                "riskRate": item.risk_rate,
                "enabled": item.enabled,
            }
            for signal_id, item in sorted(by_signal.items())
        }
        for company, by_signal in sorted(guard.evidence_by_company.items())
    }
    unsigned: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "pit_prior_supported_repair_guard_policy",
        "sourceDatasetId": source_dataset_id,
        "selectionLabelsEndInclusive": guard.training_end,
        "auditInputsIncluded": False,
        "thresholds": {
            "signalIds": list(guard.params.signal_ids),
            "minRows": guard.params.min_rows,
            "minRiskRows": guard.params.min_risk_rows,
            "minRiskRate": guard.params.min_risk_rate,
        },
        "trainingRows": {"rowIdsSha256": guard.fit_row_ids_sha256},
        "enabledByCompany": {
            company: list(values)
            for company, values in sorted(guard.enabled_by_company.items())
        },
        "evidenceByCompany": evidence,
    }
    return _seal(unsigned)


def build_recovery_policy(
    recovery: PriorSupportedProcessRecovery,
    *,
    source_dataset_id: str,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "pit_prior_pure_process_recovery_policy",
        "sourceDatasetId": source_dataset_id,
        "selectionLabelsEndInclusive": recovery.training_end,
        "auditInputsIncluded": False,
        "recovery": recovery.to_dict(),
    }
    return _seal(unsigned)


def build_semantic_policy(
    config: Mapping[str, Any],
    *,
    source_dataset_id: str,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "pit_semantic_review_veto_policy",
        "sourceDatasetId": source_dataset_id,
        "selectionLabelsEndInclusive": TRAINING_END,
        "auditInputsIncluded": False,
        "semanticVeto": _json_ready(config),
    }
    return _seal(unsigned)


def materialize_policies(*, output_dir: Path) -> dict[str, Any]:
    generation = json.loads(GENERATION_PATH.read_text(encoding="utf-8"))
    source_dataset_id = str(generation["sourceDatasetId"])
    label_index = LabelAvailabilityIndex.from_rows(
        _iter_jsonl(HISTORY_PATH),
        source_dataset_id=source_dataset_id,
        parse_date_value=data_runtime._parse_date_value,
    )
    all_rows, _ = load_frozen_history(
        HISTORY_PATH,
        source_dataset_id=source_dataset_id,
    )
    eligible_labels = label_index.eligible_labels(CUTOFF_EXCLUSIVE)
    history_rows = tuple(row for row in all_rows if row.row_id in eligible_labels)
    history_labels = {row.row_id: eligible_labels[row.row_id] for row in history_rows}

    guard = PriorSupportedRepairGuard.fit(
        history_rows,
        history_labels,
        training_end=TRAINING_END,
        signal_ids=("electrical_short", "fastener_functional_damage"),
        min_rows=20,
        min_risk_rows=5,
        min_risk_rate=0.30,
    )
    recovery = PriorSupportedProcessRecovery.fit(
        history_rows,
        history_labels,
        training_end=TRAINING_END,
        scopes=("company_detail_counts", "product_detail_counts"),
        min_support=5,
        min_months=2,
        min_recent_support=3,
        recent_window_months=12,
    )
    semantic_source = json.loads(SEMANTIC_SOURCE_PATH.read_text(encoding="utf-8"))
    semantic_config = semantic_source["policy"]["semanticVeto"]

    recovery_payload = recovery.to_dict()
    evidence_payload = {
        company: {
            signal_id: asdict(item)
            for signal_id, item in sorted(values.items())
        }
        for company, values in sorted(guard.evidence_by_company.items())
    }
    recovery_signature_count = sum(
        len(values) for values in recovery.signatures_by_scope.values()
    )
    checks = {
        "fitRowIdsSha256": guard.fit_row_ids_sha256,
        "guardEvidenceSha256": canonical_json_sha256(evidence_payload),
        "recoveryModelSha256": canonical_json_sha256(recovery_payload),
        "recoverySignatureCount": recovery_signature_count,
        "semanticConfigSha256": canonical_json_sha256(semantic_config),
    }
    expected = {
        "fitRowIdsSha256": EXPECTED_FIT_ROW_IDS_SHA256,
        "guardEvidenceSha256": EXPECTED_GUARD_EVIDENCE_SHA256,
        "recoveryModelSha256": EXPECTED_RECOVERY_MODEL_SHA256,
        "recoverySignatureCount": EXPECTED_RECOVERY_SIGNATURES,
        "semanticConfigSha256": EXPECTED_SEMANTIC_CONFIG_SHA256,
    }
    if checks != expected:
        raise ValueError(
            "PIT stage policy reproduction differs from the selected run: "
            + json.dumps({"observed": checks, "expected": expected}, ensure_ascii=False)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "repairGuard": _write_json(
            output_dir / "repair_guard.json",
            build_guard_policy(guard, source_dataset_id=source_dataset_id),
        ),
        "processRecovery": _write_json(
            output_dir / "process_recovery.json",
            build_recovery_policy(recovery, source_dataset_id=source_dataset_id),
        ),
        "semanticVeto": _write_json(
            output_dir / "semantic_veto.json",
            build_semantic_policy(semantic_config, source_dataset_id=source_dataset_id),
        ),
    }
    receipt = _seal(
        {
            "schemaVersion": 1,
            "kind": "pit_stage_policy_materialization_receipt",
            "sourceDatasetId": source_dataset_id,
            "historyCutoffExclusive": CUTOFF_EXCLUSIVE,
            "eligibleHistoryRows": len(history_rows),
            "auditInputsIncluded": False,
            "checks": checks,
            "artifacts": artifacts,
        }
    )
    _write_json(output_dir / "materialization_receipt.json", receipt)
    return receipt


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_ready(unsigned)
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {"path": str(path), "sha256": canonical_json_sha256(payload)}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the selected PIT stage policies.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "build/pit_stage_policies",
    )
    args = parser.parse_args()
    result = materialize_policies(output_dir=args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

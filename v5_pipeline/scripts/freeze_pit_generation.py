from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shadow_validation.integrity import sha256_file, verify_manifest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"
BUILD_MODELS = PACKAGE_ROOT / "build/pit_models"
BUILD_POLICIES = PACKAGE_ROOT / "build/pit_stage_policies"
PIT_EXPERIMENT = Path(
    "artifacts/mes_rule_miner_v5_upgrade_experiments_20260714"
)
PIT_SELECTED_RUN = (
    PIT_EXPERIMENT
    / "outputs/point_in_time_stage_process_atomic_write_baseline_process_20260716"
)
GENERATION_ID = "mes-pit-stable-20260716-g1"


def build_generation_config(
    *,
    danger_model_sha256_by_seed: Mapping[int, str],
    model_export_receipt_sha256: str,
    stage_policy_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "generationId": GENERATION_ID,
        "displayName": "PIT stable 회고 검증",
        "sourceDatasetId": "mes_all_customer_corrected_loop93",
        "historyEndInclusive": "2025-12-31",
        "labelAvailabilityCutoffExclusive": "2026-01-01",
        "prospectiveStartDate": "2026-08-01",
        "evaluationStatus": "retrospective_only",
        "promotionEligible": False,
        "futureShadowRequired": True,
        "dangerThresholds": {"shipment": 0.025, "repair": 0.001},
        "dangerModelSha256BySeed": {
            str(seed): digest
            for seed, digest in sorted(danger_model_sha256_by_seed.items())
        },
        "modelExportReceiptSha256": model_export_receipt_sha256,
        "stagePolicyReceiptSha256": stage_policy_receipt_sha256,
        "recovery": {
            "scopes": ["company_detail_counts", "product_detail_counts"],
            "minSupport": 5,
            "minMonths": 2,
            "minRecentSupport": 3,
            "recentWindowMonths": 12,
            "allowDangerOverride": True,
        },
        "selected2026Metrics": {
            "evaluationRows": 140_477,
            "automaticRows": 127_563,
            "automaticRate": 0.9080703602725001,
            "processRecall": 0.9360703627587619,
            "automaticAccuracy": 0.9986986822197659,
            "shipmentToProcess": 46,
            "repairToProcess": 35,
            "dangerTotal": 81,
            "reviewRows": 12_914,
        },
        "pipeline": [
            "loop93_clean_baseline",
            "hierarchical_company_line",
            "v4_green_review_fill_guard",
            "pit_three_seed_danger_veto",
            "pit_prior_supported_repair_guard",
            "pit_semantic_review_veto",
            "pit_prior_pure_process_recovery",
        ],
    }


def freeze_pit_generation() -> dict[str, Any]:
    model_receipt_path = BUILD_MODELS / "export_receipt.json"
    stage_receipt_path = BUILD_POLICIES / "materialization_receipt.json"
    model_receipt = json.loads(model_receipt_path.read_text(encoding="utf-8"))
    stage_receipt = json.loads(stage_receipt_path.read_text(encoding="utf-8"))
    if model_receipt.get("kind") != "pit_runtime_model_export":
        raise ValueError("PIT model export receipt kind mismatch")
    if stage_receipt.get("kind") != "pit_stage_policy_materialization_receipt":
        raise ValueError("PIT stage policy receipt kind mismatch")

    model_hashes: dict[int, str] = {}
    for seed in (17, 31, 47):
        source = BUILD_MODELS / f"seed_{seed}.pkl"
        contract = BUILD_MODELS / f"seed_{seed}.pkl.contract.json"
        _copy(source, f"models/pit_danger/seed_{seed}.pkl")
        _copy(contract, f"models/pit_danger/seed_{seed}.pkl.contract.json")
        model_hashes[seed] = sha256_file(source)
        if model_hashes[seed] != model_receipt["models"][str(seed)]["modelSha256"]:
            raise ValueError(f"PIT model export hash mismatch: seed={seed}")

    _copy(model_receipt_path, "lineage/pit_model_export_receipt.json")
    _copy(stage_receipt_path, "lineage/pit_stage_policy_receipt.json")
    for name in ("repair_guard.json", "process_recovery.json", "semantic_veto.json"):
        _copy(BUILD_POLICIES / name, f"policies/pit_{name}")
    for name in (
        "selected_policy.json",
        "december_gate.json",
        "predictions.json",
        "audit_report.json",
        "company_metrics.csv",
    ):
        _copy(PIT_SELECTED_RUN / name, f"parity/pit_selected_{name}")

    config = build_generation_config(
        danger_model_sha256_by_seed=model_hashes,
        model_export_receipt_sha256=sha256_file(model_receipt_path),
        stage_policy_receipt_sha256=sha256_file(stage_receipt_path),
    )
    _write_json(FROZEN_ROOT / "generation.json", config)

    artifact_paths = sorted(
        path
        for path in FROZEN_ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    runtime_paths = sorted(
        path
        for root in (
            PACKAGE_ROOT / "src",
            PACKAGE_ROOT / "scripts",
            PACKAGE_ROOT / "experiment_lib",
        )
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".mjs"}
        and "__pycache__" not in path.parts
    )
    artifacts = [
        {
            "relativePath": path.relative_to(PACKAGE_ROOT).as_posix(),
            "sizeBytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (*artifact_paths, *runtime_paths)
    ]
    manifest = {**config, "artifacts": artifacts, "artifactCount": len(artifacts)}
    _write_json(FROZEN_ROOT / "manifest.json", manifest)
    return verify_manifest(
        PACKAGE_ROOT,
        FROZEN_ROOT / "manifest.json",
        expected_generation=GENERATION_ID,
    )


def _copy(source: Path, relative: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = FROZEN_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    return target


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    result = freeze_pit_generation()
    print(
        f"Frozen {result['verifiedArtifactCount']} artifacts for "
        f"{result['generationId']} ({result['manifestSha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

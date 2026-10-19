from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"
GENERATION_ID = "mes-balanced-20260713-g1"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.append(str(PACKAGE_ROOT / "src"))

ROLLING_ROOT = Path("artifacts/mes_rule_miner_rolling_upgrade_20260713")
HIERARCHY_ROOT = Path("artifacts/mes_rule_miner_hierarchical_frontier_20260711")
TRUE_FRONTIER_ROOT = Path("artifacts/mes_rule_miner_all_customer_true_frontier_20260709")
CLEAN_ROOT = Path("artifacts/mes_rule_miner_clean_optimization_20260710")

LOOP93_DIR = TRUE_FRONTIER_ROOT / "outputs/company_policy_search_20260709/loop93_current_policy_crossfit"
HIERARCHY_RUN_DIR = HIERARCHY_ROOT / "outputs/postfit_company_hs_grid1_final_refit_all_run4"
HIERARCHY_EVIDENCE = HIERARCHY_ROOT / "outputs/algebra_company_hs_grid1/scope_evidence_algebra_2025.json"
HIERARCHY_BUDGET = HIERARCHY_ROOT / "outputs/frontier_company_hs_grid1/budgeted_frontier_2025.json"
V4_GREEN_DIR = HIERARCHY_ROOT / "outputs/v4_derived_hybrid_strict_release"
FINAL_DIR = ROLLING_ROOT / "outputs/rolling_upgrade/run_20260713_all_actions_final"
DANGER_MODELS = (
    FINAL_DIR
    / "models/final/lightgbm__core__gray_downweight_075__all_actions"
)
DANGER_SELECTION_RECEIPT = FINAL_DIR / "selection_receipt.json"


def _imports() -> tuple[Any, ...]:
    sys.path.insert(0, str(HIERARCHY_ROOT))
    from scripts.materialize_hierarchical_evidence import (  # noqa: PLC0415
        DEFAULT_RUN_FILES,
        DEFAULT_SOURCE_ROOT,
        load_candidate_profiles,
    )
    from scripts.materialize_policy_algebra import (  # noqa: PLC0415
        algebra_policies_from_registry,
    )
    from scripts.run_hierarchical_lockbox_audit import (  # noqa: PLC0415
        _candidate_run_files,
        _load_routes,
        _required_base_candidates,
    )

    return (
        DEFAULT_RUN_FILES,
        DEFAULT_SOURCE_ROOT,
        load_candidate_profiles,
        algebra_policies_from_registry,
        _candidate_run_files,
        _load_routes,
        _required_base_candidates,
    )


def _copy(source: Path, relative: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = FROZEN_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.chmod(0o444)
    temporary.replace(target)
    return target


def _gzip_copy(source: Path, relative: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = FROZEN_ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with source.open("rb") as input_handle, temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            compresslevel=6,
            mtime=0,
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    temporary.replace(target)
    return target


def _copy_artifact_triplet(manifest_path: Path, destination: str) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_manifest = _copy(manifest_path, f"models/{destination}/{manifest_path.name}")
    related_files = [str(manifest["artifactFile"])]
    schema_file = manifest.get("featureSchemaFile")
    if schema_file:
        related_files.append(str(schema_file))
    else:
        inferred_schema = manifest_path.with_name(
            manifest_path.name.removesuffix(".fitted_artifact.json") + ".feature_schema.json"
        )
        if inferred_schema.is_file():
            related_files.append(inferred_schema.name)
    for filename in related_files:
        source = manifest_path.parent / filename
        _copy(source, f"models/{destination}/{source.name}")
    return copied_manifest


def _profile_payloads() -> list[dict[str, Any]]:
    (
        default_run_files,
        default_source_root,
        load_candidate_profiles,
        algebra_policies_from_registry,
        candidate_run_files,
        load_routes,
        required_base_candidates,
    ) = _imports()
    evidence = json.loads(HIERARCHY_EVIDENCE.read_text(encoding="utf-8"))
    budget = json.loads(HIERARCHY_BUDGET.read_text(encoding="utf-8"))
    company_routes, line_routes = load_routes(budget)
    algebra = algebra_policies_from_registry(evidence["candidates"])
    required = required_base_candidates(
        {*company_routes.values(), *line_routes.values()},
        algebra,
    )
    run_files = candidate_run_files(
        evidence,
        tuple(default_source_root / name for name in default_run_files),
    )
    profiles = {
        profile.candidate_id: profile
        for profile in load_candidate_profiles(run_files)
        if profile.candidate_id in required
    }
    if set(profiles) != required:
        raise RuntimeError(f"missing hierarchy profiles: {sorted(required - set(profiles))}")

    runtime_manifest_by_source: dict[tuple[str, str | None], Path] = {}
    for profile in profiles.values():
        source_stem = profile.fitted_manifest_path.name.removesuffix(".fitted_artifact.json")
        key = (source_stem, profile.scope_company)
        if key in runtime_manifest_by_source:
            continue
        if profile.scope_company in {"customer_a", "customer_b"}:
            final_name = f"{source_stem}_{profile.scope_company}_final.fitted_artifact.json"
            source_manifest = HIERARCHY_RUN_DIR / "scoped_final_refit_artifacts" / final_name
            destination = f"hierarchy_final/{profile.scope_company.lower()}_{source_stem}"
        else:
            source_manifest = profile.fitted_manifest_path
            destination = f"hierarchy_global/{source_stem}"
        runtime_manifest_by_source[key] = _copy_artifact_triplet(source_manifest, destination)

    output: list[dict[str, Any]] = []
    for candidate_id, profile in sorted(profiles.items()):
        source_stem = profile.fitted_manifest_path.name.removesuffix(".fitted_artifact.json")
        runtime_manifest = runtime_manifest_by_source[(source_stem, profile.scope_company)]
        output.append(
            {
                "candidateId": candidate_id,
                "family": profile.family,
                "baseModelId": profile.base_model_id,
                "runtimeManifest": runtime_manifest.relative_to(FROZEN_ROOT).as_posix(),
                "runtimeManifestSha256": _sha256(runtime_manifest),
                "thresholds": dict(profile.thresholds),
                "thresholdEnabled": dict(profile.threshold_enabled),
                "thresholdProfile": profile.threshold_profile,
                "scopeCompany": profile.scope_company,
                "trainEnd": profile.train_end,
                "selectorMonths": list(profile.selector_months),
            }
        )
    return output


def _sha256(path: Path) -> str:
    from shadow_validation.integrity import sha256_file

    return sha256_file(path)


def _write_json(relative: str, value: Any) -> Path:
    path = FROZEN_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o444)
    temporary.replace(path)
    return path


def _copy_tree_files(source: Path, destination: str, names: Iterable[str]) -> None:
    for name in names:
        _copy(source / name, f"{destination}/{name}")


def verify_danger_model_sources(
    model_dir: str | Path,
    selection_receipt_path: str | Path,
) -> dict[int, str]:
    """Fail closed if same-named models came from a different experiment run."""
    source = Path(model_dir)
    receipt_path = Path(selection_receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    final_models = receipt.get("finalModels")
    if not isinstance(final_models, list):
        raise ValueError(f"selection receipt has no finalModels: {receipt_path}")
    expected = {
        int(item["seed"]): str(item["sha256"])
        for item in final_models
        if isinstance(item, dict) and "seed" in item and "sha256" in item
    }
    required = {17, 31, 47}
    if set(expected) != required:
        raise ValueError(
            "selection receipt danger seed set mismatch: "
            f"expected={sorted(required)}, observed={sorted(expected)}"
        )

    verified: dict[int, str] = {}
    for seed in sorted(required):
        model_path = source / f"seed_{seed}.pkl"
        contract_path = model_path.with_suffix(model_path.suffix + ".contract.json")
        if not model_path.is_file() or not contract_path.is_file():
            raise FileNotFoundError(f"danger model or contract missing: seed={seed}")
        actual = _sha256(model_path)
        if actual != expected[seed]:
            raise ValueError(
                "danger model selection receipt SHA mismatch: "
                f"seed={seed}, actual={actual}, expected={expected[seed]}"
            )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if str(contract.get("modelSha256") or "") != actual:
            raise ValueError(f"danger model runtime contract SHA mismatch: seed={seed}")
        verified[seed] = actual
    return verified


def freeze_generation() -> dict[str, Any]:
    FROZEN_ROOT.mkdir(parents=True, exist_ok=True)

    _copy(TRUE_FRONTIER_ROOT / "rule_miner.browser.js", "loop93/rule_miner.browser.js")
    _copy(
        LOOP93_DIR / "loop93_current_policy_crossfit_frozen_policy.json",
        "loop93/frozen_policy.json",
    )
    _gzip_copy(
        LOOP93_DIR / "loop93_current_policy_crossfit_history_rows.jsonl",
        "history/history_rows.jsonl.gz",
    )
    _gzip_copy(
        LOOP93_DIR / "loop93_current_policy_crossfit_2026_audit_rows.jsonl",
        "parity/audit_2026_rows.jsonl.gz",
    )

    _copy(HIERARCHY_EVIDENCE, "hierarchy/evidence.json")
    _copy(HIERARCHY_BUDGET, "hierarchy/budgeted_policy.json")
    _copy(HIERARCHY_RUN_DIR / "frozen_policy.json", "hierarchy/frozen_policy.json")
    _copy(
        HIERARCHY_RUN_DIR / "hierarchical_lockbox_audit_report.json",
        "hierarchy/lockbox_audit_report.json",
    )
    _copy(
        HIERARCHY_RUN_DIR / "sealed_audit_predictions.json",
        "parity/current_hierarchy_predictions.json",
    )
    _write_json("hierarchy/profiles.json", {"profiles": _profile_payloads()})

    _copy_tree_files(
        V4_GREEN_DIR,
        "v4_green",
        (
            "green_frozen_policy.json",
            "green_freeze_receipt.json",
            "green_sealed_shadow_audit_predictions.json",
        ),
    )
    _copy_tree_files(
        V4_GREEN_DIR / "final_models",
        "models/v4_green/gray_auxiliary",
        (
            "gray_auxiliary.pkl",
            "gray_auxiliary.pkl.contract.json",
            "gray_auxiliary.threshold.json",
            "gray_auxiliary.threshold.json.contract.json",
        ),
    )

    danger_model_sha256 = verify_danger_model_sources(
        DANGER_MODELS,
        DANGER_SELECTION_RECEIPT,
    )
    for seed in (17, 31, 47):
        _copy(DANGER_MODELS / f"seed_{seed}.pkl", f"models/danger/seed_{seed}.pkl")
        _copy(
            DANGER_MODELS / f"seed_{seed}.pkl.contract.json",
            f"models/danger/seed_{seed}.pkl.contract.json",
        )
    _copy(
        ROLLING_ROOT
        / "outputs/rolling_upgrade/run_20260713_all_actions_final/sealed_future_predictions.json",
        "parity/danger_predictions.json",
    )

    repair_dir = FINAL_DIR / "prior_supported_repair_guard"
    recovery_dir = FINAL_DIR / "prior_supported_process_recovery"
    _copy(repair_dir / "policy.json", "policies/repair_guard.json")
    _copy(recovery_dir / "policy.json", "policies/process_recovery.json")
    _copy(recovery_dir / "predictions.json", "parity/final_predictions.json")
    _copy(recovery_dir / "audit.json", "parity/final_audit.json")

    config = {
        "schemaVersion": 1,
        "generationId": GENERATION_ID,
        "sourceDatasetId": "mes_all_customer_corrected_loop93",
        "prospectiveStartDate": "2026-08-01",
        "historyEndInclusive": "2025-12-31",
        "v4PriorCutoff": "2025-11-30",
        "dangerPriorCutoff": "2025-12-31",
        "dangerThresholds": {"shipment": 0.025, "repair": 0.030},
        "dangerModelSha256BySeed": {
            str(seed): digest for seed, digest in sorted(danger_model_sha256.items())
        },
        "dangerSelectionReceiptSha256": _sha256(DANGER_SELECTION_RECEIPT),
        "pipeline": [
            "loop93_clean_baseline",
            "hierarchical_company_line",
            "v4_green_review_fill_guard",
            "three_seed_danger_veto",
            "prior_supported_repair_guard",
            "prior_supported_process_recovery",
        ],
    }
    _write_json("generation.json", config)

    artifact_paths = sorted(
        path
        for path in FROZEN_ROOT.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    runtime_paths = sorted(
        path
        for root in (PACKAGE_ROOT / "src", PACKAGE_ROOT / "scripts")
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".mjs"}
        and "__pycache__" not in path.parts
    )
    artifacts = []
    for path in (*artifact_paths, *runtime_paths):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        artifacts.append(
            {
                "relativePath": relative,
                "sizeBytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        **config,
        "artifacts": artifacts,
        "artifactCount": len(artifacts),
    }
    manifest_path = FROZEN_ROOT / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.chmod(0o444)
    temporary_manifest.replace(manifest_path)

    from shadow_validation.integrity import verify_manifest

    return verify_manifest(
        PACKAGE_ROOT,
        manifest_path,
        expected_generation=GENERATION_ID,
    )


def main() -> int:
    result = freeze_generation()
    print(
        f"Frozen {result['verifiedArtifactCount']} artifacts for "
        f"{result['generationId']} ({result['manifestSha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

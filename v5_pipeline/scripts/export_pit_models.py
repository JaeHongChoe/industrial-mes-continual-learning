from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(
    "artifacts/mes_rule_miner_v5_upgrade_experiments_20260714"
)
TRAINING_RUNNER = EXPERIMENT_ROOT / "scripts/run_point_in_time_2026.py"
DECISION_SOURCE = EXPERIMENT_ROOT / "outputs/point_in_time_decision_source_20260716"
SELECTION_ROOT = (
    EXPERIMENT_ROOT
    / "outputs/point_in_time_selection_fit_lineage_baseline_process_20260716"
)
ROLLING_ROOT = Path("artifacts/mes_rule_miner_rolling_upgrade_20260713")
REFERENCE_RUN = (
    EXPERIMENT_ROOT
    / "outputs/point_in_time_2026_fit_lineage_baseline_process_20260716_v2"
)
EXPECTED_SCORES_SHA256 = "c8f7f452c2e21e4ea4176948b1e47475556bcab5f6ee551fac45497d50c0ba45"
EXPECTED_PREDICTIONS_SHA256 = "813c9f4aab202f91a236090f948200c2269c945c4ba23a3b39e6f694470930e4"
EXPECTED_FIT_RECEIPTS_SHA256 = "94eda0d938450406abef67aa1045f661b83b8f9102a146a68b0fb407d793db09"


def compare_score_archives(
    reference_path: str | Path,
    observed_path: str | Path,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify score parity without requiring byte-identical float serialization.

    Retraining the same LightGBM model can differ by a few machine-epsilon units
    across processes even when the fitted receipts and final decisions are
    identical. Row identity and month identity therefore remain exact gates,
    while the two probability arrays use a deliberately tiny numeric tolerance.
    """
    expected_keys = {"row_ids", "months", "shipment", "repair"}
    with np.load(Path(reference_path), allow_pickle=False) as reference, np.load(
        Path(observed_path), allow_pickle=False
    ) as observed:
        if set(reference.files) != expected_keys or set(observed.files) != expected_keys:
            raise ValueError("PIT score archive schema differs from the selected run")

        row_ids_equal = np.array_equal(reference["row_ids"], observed["row_ids"])
        months_equal = np.array_equal(reference["months"], observed["months"])
        if not row_ids_equal or not months_equal:
            raise ValueError("PIT score row or month universe differs from the selected run")

        differences: dict[str, float] = {}
        for name in ("shipment", "repair"):
            expected = np.asarray(reference[name], dtype=float)
            actual = np.asarray(observed[name], dtype=float)
            if expected.shape != actual.shape:
                raise ValueError(f"PIT {name} score shape differs from the selected run")
            delta = np.abs(expected - actual)
            maximum = float(np.max(delta)) if delta.size else 0.0
            if not np.isfinite(maximum) or maximum > float(tolerance):
                raise ValueError(
                    f"PIT score numeric drift exceeds tolerance for {name}: "
                    f"{maximum:.17g} > {float(tolerance):.17g}"
                )
            differences[name] = maximum

    return {
        "rowIdsEqual": row_ids_equal,
        "monthsEqual": months_equal,
        "tolerance": float(tolerance),
        "maxShipmentDifference": differences["shipment"],
        "maxRepairDifference": differences["repair"],
    }


def state_from_training_model(
    model: Any,
    *,
    fit_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract fitted inference state without retaining labels or runtime modules."""
    state = {
        "schemaVersion": 1,
        "kind": "pit_danger_model_state",
        "rawEncoder": model._raw_encoder,
        "domainScaler": model._domain_scaler,
        "temporalScaler": model._temporal_scaler,
        "domainSchemaSha256": str(model._domain_schema_sha256),
        "temporalSchemaSha256": str(model._temporal_schema_sha256),
        "calibrationCutoff": str(model._calibration_cutoff),
        "heads": dict(model._heads),
        "calibrators": dict(model._calibrators),
        "fitReceipt": dict(fit_receipt or model._fit_receipt),
        "featureNames": tuple(model._feature_names),
    }
    if any(state[name] is None for name in ("rawEncoder", "domainScaler")):
        raise ValueError("PIT training model is not fitted")
    return state


def write_model_state(
    output_dir: str | Path,
    *,
    seed: int,
    state: Mapping[str, Any],
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / f"seed_{int(seed)}.pkl"
    temporary = model_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(dict(state), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(model_path)
    digest = _sha256(model_path)
    contract = {
        "schemaVersion": 1,
        "kind": "pit_danger_model_contract",
        "randomState": int(seed),
        "modelSha256": digest,
        "fitReceipt": dict(state["fitReceipt"]),
        "domainSchemaSha256": str(state["domainSchemaSha256"]),
        "temporalSchemaSha256": str(state["temporalSchemaSha256"]),
        "calibrationCutoff": str(state["calibrationCutoff"]),
    }
    contract_path = model_path.with_suffix(".pkl.contract.json")
    contract_temporary = contract_path.with_suffix(".json.tmp")
    contract_temporary.write_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    contract_temporary.replace(contract_path)
    return model_path, contract_path


def export_models(*, work_dir: Path, output_dir: Path) -> dict[str, Any]:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"PIT export work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = _load_training_runner()
    captured: list[Any] = []
    original_fit = runner.PointInTimeDangerVeto.fit

    def capturing_fit(model: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_fit(model, *args, **kwargs)
        captured.append(model)
        return result

    runner.PointInTimeDangerVeto.fit = capturing_fit
    try:
        runner.run_apply(
            config_path=(DECISION_SOURCE / "config.json").resolve(),
            source_receipt_path=(DECISION_SOURCE / "receipt.json").resolve(),
            upgrade_path=(ROLLING_ROOT / "rolling_upgrade_all_actions_final_config.json").resolve(),
            selection_path=(SELECTION_ROOT / "selected_policy.json").resolve(),
            december_gate_path=(SELECTION_ROOT / "december_gate.json").resolve(),
            output_dir=work_dir.resolve(),
            seeds=(17, 31, 47),
            lockbox_vault_path=(DECISION_SOURCE / "lockbox_labels.json").resolve(),
        )
    finally:
        runner.PointInTimeDangerVeto.fit = original_fit

    scores_path = work_dir / "future_scores.npz"
    predictions_path = work_dir / "predictions.json"
    fit_receipts_path = work_dir / "fit_receipts.json"
    if _sha256(fit_receipts_path) != EXPECTED_FIT_RECEIPTS_SHA256:
        raise ValueError("exported PIT fit receipts differ from the selected run")
    score_parity = compare_score_archives(
        REFERENCE_RUN / "future_scores.npz",
        scores_path,
    )
    expected_predictions = json.loads(
        (REFERENCE_RUN / "predictions.json").read_text(encoding="utf-8")
    )
    observed_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if observed_predictions.get("predictions") != expected_predictions.get("predictions"):
        raise ValueError("exported PIT prediction map differs from the selected run")
    if observed_predictions.get("predictionMapSha256") != expected_predictions.get(
        "predictionMapSha256"
    ):
        raise ValueError("exported PIT prediction map receipt differs from the selected run")
    if len(captured) != 3:
        raise ValueError(f"expected three captured PIT models, observed {len(captured)}")

    receipts = json.loads(fit_receipts_path.read_text(encoding="utf-8"))
    models: dict[str, Any] = {}
    for seed, model in zip((17, 31, 47), captured, strict=True):
        state = state_from_training_model(
            model,
            fit_receipt=receipts[f"seed_{seed}"],
        )
        model_path, contract_path = write_model_state(
            output_dir,
            seed=seed,
            state=state,
        )
        models[str(seed)] = {
            "model": str(model_path),
            "modelSha256": _sha256(model_path),
            "contract": str(contract_path),
            "contractSha256": _sha256(contract_path),
        }
    receipt = {
        "schemaVersion": 1,
        "kind": "pit_runtime_model_export",
        "sourceRunnerSha256": _sha256(TRAINING_RUNNER),
        "selectedPolicySha256": _sha256(SELECTION_ROOT / "selected_policy.json"),
        "referenceFutureScoresSha256": EXPECTED_SCORES_SHA256,
        "reproducedFutureScoresSha256": _sha256(scores_path),
        "scoreParity": score_parity,
        "referencePredictionsSha256": EXPECTED_PREDICTIONS_SHA256,
        "reproducedPredictionsSha256": _sha256(predictions_path),
        "predictionMapSha256": observed_predictions["predictionMapSha256"],
        "fitReceiptsSha256": EXPECTED_FIT_RECEIPTS_SHA256,
        "models": models,
    }
    receipt_path = output_dir / "export_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def _load_training_runner() -> Any:
    if str(EXPERIMENT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_ROOT))
    spec = importlib.util.spec_from_file_location("pit_training_runner_frozen", TRAINING_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PIT training runner: {TRAINING_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the selected PIT model state.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PACKAGE_ROOT / "build/pit_model_export_run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "build/pit_models",
    )
    args = parser.parse_args()
    result = export_models(
        work_dir=args.work_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

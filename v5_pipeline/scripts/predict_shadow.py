from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from shadow_validation.input_contract import load_input_files  # noqa: E402
from shadow_validation.integrity import sha256_file  # noqa: E402
from shadow_validation.predictor import FrozenPredictor  # noqa: E402
from shadow_validation.receipt import (  # noqa: E402
    PredictionReceiptError,
    seal_prediction_run,
)


def _prediction_records(batch: Any, bundle: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = bundle.by_id
    predictions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for row in batch.rows:
        prediction = asdict(by_id[row.row_id])
        predictions.append(
            {
                "rowId": row.row_id,
                "sourceFile": row.source_file,
                "sourceRow": row.source_row,
                "customer": row.raw_customer,
                "productKey": row.product_key,
                "date": row.date,
                "runScope": row.run_scope,
                "generationId": bundle.generation_id,
                "loop93Prediction": prediction["loop93_prediction"],
                "hierarchyPrediction": prediction["hierarchy_prediction"],
                "v4GreenPrediction": prediction["v4_green_prediction"],
                "dangerVetoPrediction": prediction["danger_veto_prediction"],
                "repairGuardPrediction": prediction["repair_guard_prediction"],
                "semanticVetoPrediction": prediction["semantic_veto_prediction"],
                "finalPrediction": prediction["final_prediction"],
                "shipmentProbability": prediction["shipment_probability"],
                "repairProbability": prediction["repair_probability"],
                "protected": prediction["protected"],
                "appliedStage": prediction["applied_stage"],
                "reason": prediction["reason"],
            }
        )
        features.append({"rowId": row.row_id, "features": dict(row.feature_payload)})
    return predictions, features


def run_prediction(
    input_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    run_name: str,
    mode: str,
) -> dict[str, Any]:
    resolved_inputs = [Path(path).expanduser().resolve() for path in input_paths]
    frozen_manifest = PACKAGE_ROOT / "frozen/manifest.json"
    frozen_generation = PACKAGE_ROOT / "frozen/generation.json"
    watched_paths = tuple(resolved_inputs) + (frozen_manifest, frozen_generation)
    initial_state = _input_state(watched_paths)
    batch = load_input_files(input_paths, mode=mode)
    predictor = FrozenPredictor(PACKAGE_ROOT)
    prediction_bundle = predictor.predict(batch)
    if _input_state(watched_paths) != initial_state:
        raise PredictionReceiptError("입력 파일 또는 동결 기준이 예측 중 변경되었습니다")
    predictions, features = _prediction_records(batch, prediction_bundle)
    input_artifacts: dict[str, Path] = {
        f"input:{index}:{path.name}": path
        for index, path in enumerate(sorted(resolved_inputs), start=1)
    }
    input_artifacts["frozenManifest"] = frozen_manifest
    input_artifacts["frozenGeneration"] = frozen_generation
    run_log = (
        f"generationId={prediction_bundle.generation_id}\n"
        f"mode={mode}\n"
        f"rowCount={len(predictions)}\n"
    )
    return seal_prediction_run(
        Path(output_root).expanduser().resolve() / run_name,
        predictions=predictions,
        feature_rows=features,
        input_artifacts=input_artifacts,
        generation_id=prediction_bundle.generation_id,
        mode=mode,
        run_log=run_log,
    )


def _input_state(paths: Sequence[Path]) -> tuple[tuple[str, int, int, str], ...]:
    state: list[tuple[str, int, int, str]] = []
    for path in paths:
        if not path.is_file():
            raise PredictionReceiptError(f"입력 파일이 없습니다: {path}")
        stat = path.stat()
        state.append((str(path), stat.st_size, stat.st_mtime_ns, sha256_file(path)))
    return tuple(state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen MES shadow predictor")
    parser.add_argument("inputs", nargs="+", help="XLSX or CSV intake files")
    parser.add_argument("--output-root", default=PACKAGE_ROOT / "shadow_runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("prospective", "dry_run"), default="prospective")
    args = parser.parse_args(argv)
    try:
        receipt = run_prediction(
            args.inputs,
            output_root=args.output_root,
            run_name=args.run_name,
            mode=args.mode,
        )
    except Exception as error:
        print(f"예측 실패: {error}", file=sys.stderr)
        return 2
    print(receipt["receiptSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

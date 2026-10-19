from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_contract import ACTION_LABELS, ALLOWED_RAW_FEATURES, AuditLabelsLockedError, FeatureRow
from src.experiment import prepare_experiment
from src.features import FeatureSpec
from src.freeze import PolicySpec
from src.metrics import Metrics, OutcomeRow, compute_metrics
from src.privileged_training import gray_zone_sample_weights, load_privileged_training_metadata
from src.runtime_contract import RuntimeContract, validate_runtime_contract
from src.v4_derived_policy import (
    HierarchicalRoutes,
    ModelRiskGuard,
    apply_model_risk_guard,
    compose_v4_derived_policy,
    derive_protection_signatures,
    derive_recovery_signatures,
    fit_model_risk_guard,
)
from src.v4_hybrid_candidate import (
    HybridThresholdFit,
    V4HybridCandidate,
    V4HybridScoreRow,
    apply_hybrid_thresholds,
    fit_hybrid_thresholds,
)
from src.select import LockboxResult


ABLATION_IDS = (
    "current_hierarchy",
    "v4_domain_features",
    "v4_three_head",
    "v4_gray_weighted",
    "v4_gray_auxiliary",
    "v4_process_recovery",
    "v4_cold_start",
    "v4_full_fallback",
    "v4_protection",
    "v4_green_selected",
    "v4_amber_selected",
)


@dataclass(frozen=True, slots=True)
class RiskEnvelope:
    name: str
    max_ship_to_process: int
    max_repair_to_process: int
    max_risk_to_process: int
    min_auto_accuracy: float


GREEN_ENVELOPE = RiskEnvelope("green", 65, 35, 100, 0.994)
AMBER_ENVELOPE = RiskEnvelope("amber", 80, 45, 120, 0.993)


def validate_ablation_order(records: Sequence[Mapping[str, Any]]) -> None:
    observed = tuple(str(item.get("candidateId") or "") for item in records)
    if observed != ABLATION_IDS:
        raise ValueError(f"ablation order differs from preregistration: {observed}")


def validate_chronology_contract(chronology: Any) -> None:
    if not chronology.train or max(row.month for row in chronology.train) != "2025-07":
        raise ValueError("training chronology must end in 2025-07")
    if {row.month for row in chronology.probability_calibration} != {"2025-08"}:
        raise ValueError("probability calibration must contain only 2025-08")
    if {row.month for row in chronology.threshold_fit} != {"2025-09"}:
        raise ValueError("threshold fitting must contain only 2025-09")
    if {row.month for row in chronology.selector} != {"2025-10", "2025-11"}:
        raise ValueError("selector must contain exactly 2025-10 and 2025-11")


def select_frontier_candidate(
    records: Sequence[Mapping[str, Any]],
    envelope: RiskEnvelope,
) -> dict[str, Any]:
    accepted = [dict(item) for item in records if _within_envelope(_metrics(item), envelope)]
    if not accepted:
        raise ValueError(f"no candidate satisfies the {envelope.name} risk envelope")
    accepted.sort(key=_frontier_sort_key)
    return accepted[0]


def reject_only_lockbox(
    selected: Mapping[str, Any],
    lockbox_metrics: Mapping[str, Any],
    envelope: RiskEnvelope,
) -> dict[str, Any]:
    accepted = _within_envelope(lockbox_metrics, envelope)
    return {
        "candidateId": str(selected["candidateId"]),
        "accepted": accepted,
        "status": "accepted_without_retuning" if accepted else "rejected_without_replacement",
    }


def assert_pre_audit_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("auditInputsIncluded") is not False:
        raise ValueError("pre-audit artifact must state auditInputsIncluded=false")
    forbidden_keys = {
        "auditmetrics",
        "auditlabels",
        "auditpredictions",
        "2026metrics",
        "2026labels",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                token = "".join(character for character in str(key).lower() if character.isalnum())
                if token in forbidden_keys:
                    raise ValueError(f"audit-derived field is forbidden before audit: {path}{key}")
                visit(item, f"{path}{key}.")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}{index}.")
        elif isinstance(value, str) and "2026" in value:
            raise ValueError(f"2026 material is forbidden before audit: {path[:-1]}")

    visit(payload, "")


def canonical_artifact_bytes(payload: Mapping[str, Any]) -> bytes:
    assert_pre_audit_artifact(payload)
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metrics")
    if not isinstance(value, Mapping):
        raise ValueError("candidate record is missing metrics")
    return value


def _within_envelope(metrics: Mapping[str, Any], envelope: RiskEnvelope) -> bool:
    accuracy = metrics.get("auto_accuracy")
    return (
        int(metrics.get("ship_to_process") or 0) <= envelope.max_ship_to_process
        and int(metrics.get("repair_to_process") or 0) <= envelope.max_repair_to_process
        and int(metrics.get("risk_to_process") or 0) <= envelope.max_risk_to_process
        and accuracy is not None
        and float(accuracy) >= envelope.min_auto_accuracy
    )


def _frontier_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _metrics(record)
    return (
        -int(metrics.get("correct_process_rows") or 0),
        -int(metrics.get("correct_auto_rows") or 0),
        -int(metrics.get("auto_rows") or 0),
        int(metrics.get("process_to_ng") or 0),
        int(record.get("complexityRank") or 0),
        str(record.get("candidateId") or ""),
    )


def _scaled_envelope(
    envelope: RiskEnvelope,
    evaluation_rows: int,
    reference_rows: int,
) -> RiskEnvelope:
    if evaluation_rows < 1 or reference_rows < 1:
        raise ValueError("risk-envelope row counts must be positive")
    return RiskEnvelope(
        name=f"{envelope.name}_scaled",
        max_ship_to_process=max(
            1,
            int(np.ceil(envelope.max_ship_to_process * evaluation_rows / reference_rows)),
        ),
        max_repair_to_process=max(
            1,
            int(np.ceil(envelope.max_repair_to_process * evaluation_rows / reference_rows)),
        ),
        max_risk_to_process=max(
            1,
            int(np.ceil(envelope.max_risk_to_process * evaluation_rows / reference_rows)),
        ),
        min_auto_accuracy=envelope.min_auto_accuracy,
    )


def _validate_model_checkpoint(
    candidate: V4HybridCandidate,
    *,
    source_dataset_id: str,
    history_sha256: str,
    gray_auxiliary: bool,
    model_family: str,
    random_state: int,
    training_rows_sha256: str,
    calibration_rows_sha256: str,
    sample_weight_sha256: str,
) -> None:
    receipt = candidate.fit_receipt
    if (
        receipt.get("sourceDatasetId") != source_dataset_id
        or receipt.get("historySha256") != history_sha256
        or receipt.get("modelFamily") != model_family
        or bool(receipt.get("grayAuxiliaryHead")) is not gray_auxiliary
        or receipt.get("domainScaling") != "maxabs_fit_on_training_oof"
        or receipt.get("inferenceWaitFeature") is not False
        or int(receipt.get("randomState", getattr(candidate, "random_state", -1)))
        != int(random_state)
        or receipt.get("trainingRowsSha256") != training_rows_sha256
        or receipt.get("calibrationRowsSha256") != calibration_rows_sha256
        or receipt.get("sampleWeightSha256") != sample_weight_sha256
    ):
        raise ValueError("fitted model checkpoint does not match the current experiment contract")


def _row_ids_sha256(rows: Sequence[FeatureRow]) -> str:
    payload = json.dumps(
        tuple(sorted(row.row_id for row in rows)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()


def _sample_weight_sha256(
    sample_weights: Sequence[float] | np.ndarray | None,
    row_count: int,
) -> str:
    weights = (
        np.ones(row_count, dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if weights.shape != (row_count,):
        raise ValueError("sample-weight checkpoint contract is misaligned")
    return hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()


def _score_checkpoint_contract(
    *,
    purpose: str,
    model_fit_receipt_sha256: str,
    model_artifact_sha256: str,
    rows: Sequence[FeatureRow],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "v4_hybrid_score_checkpoint_contract",
        "purpose": purpose,
        "modelFitReceiptSha256": model_fit_receipt_sha256,
        "modelArtifactSha256": model_artifact_sha256,
        "rowUniverseSha256": _row_ids_sha256(rows),
    }


def _contract_sidecar(path: Path) -> Path:
    return Path(f"{path}.contract.json")


def _contract_bytes(contract: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(contract),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_binding_bytes(
    *,
    contract: Mapping[str, Any],
    artifact_sha256: str,
) -> bytes:
    return _contract_bytes(
        {
            "schemaVersion": 1,
            "artifactSha256": artifact_sha256,
            "contract": dict(contract),
        }
    )


def _write_model_checkpoint(path: Path, candidate: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(candidate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    contract = {
        "kind": "v4_hybrid_model_checkpoint",
        "modelFitReceiptSha256": str(candidate.fit_receipt["receiptSha256"]),
        "runtime": dict(RuntimeContract.capture().to_payload()),
    }
    _atomic_write(
        _contract_sidecar(path),
        _artifact_binding_bytes(
            contract=contract,
            artifact_sha256=_sha256_file(path),
        ),
    )


def _read_model_checkpoint(path: Path) -> Any:
    sidecar = _contract_sidecar(path)
    if not sidecar.is_file():
        raise ValueError(f"model checkpoint contract is missing: {path}")
    binding = json.loads(sidecar.read_text(encoding="utf-8"))
    if binding.get("artifactSha256") != _sha256_file(path):
        raise ValueError(f"model checkpoint artifact SHA mismatch: {path}")
    contract = binding.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"model checkpoint contract mismatch: {path}")
    validate_runtime_contract(contract.get("runtime"))
    with path.open("rb") as handle:
        candidate = pickle.load(handle)
    if candidate.fit_receipt.get("receiptSha256") != contract.get(
        "modelFitReceiptSha256"
    ):
        raise ValueError(f"model checkpoint fit receipt mismatch: {path}")
    return candidate


def _write_score_npz(
    path: Path,
    scores: Sequence[V4HybridScoreRow],
    *,
    contract: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_id=np.asarray([row.row_id for row in scores]),
            month=np.asarray([row.month for row in scores]),
            process=np.asarray([row.process_probability for row in scores], dtype=np.float64),
            shipment=np.asarray([row.shipment_probability for row in scores], dtype=np.float64),
            repair=np.asarray([row.repair_probability for row in scores], dtype=np.float64),
            gray=np.asarray([row.gray_probability for row in scores], dtype=np.float64),
        )
    temporary.replace(path)
    _atomic_write(
        _contract_sidecar(path),
        _artifact_binding_bytes(
            contract=contract,
            artifact_sha256=_sha256_file(path),
        ),
    )


def _read_score_npz(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> tuple[V4HybridScoreRow, ...]:
    sidecar = _contract_sidecar(path)
    if not sidecar.is_file():
        raise ValueError(f"score checkpoint contract mismatch: {path}")
    binding = json.loads(sidecar.read_text(encoding="utf-8"))
    if binding.get("contract") != dict(expected_contract):
        raise ValueError(f"score checkpoint contract mismatch: {path}")
    if binding.get("artifactSha256") != _sha256_file(path):
        raise ValueError(f"score checkpoint artifact SHA mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {"row_id", "month", "process", "shipment", "repair", "gray"}
        if set(payload.files) != required:
            raise ValueError(f"score checkpoint schema mismatch: {path}")
        lengths = {len(payload[name]) for name in required}
        if len(lengths) != 1:
            raise ValueError(f"score checkpoint arrays are misaligned: {path}")
        return tuple(
            V4HybridScoreRow(
                row_id=str(row_id),
                month=str(month),
                process_probability=float(process),
                shipment_probability=float(shipment),
                repair_probability=float(repair),
                gray_probability=float(gray),
            )
            for row_id, month, process, shipment, repair, gray in zip(
                payload["row_id"],
                payload["month"],
                payload["process"],
                payload["shipment"],
                payload["repair"],
                payload["gray"],
                strict=True,
            )
        )


def _threshold_checkpoint_contract(
    *,
    purpose: str,
    model_fit_receipt_sha256: str,
    model_artifact_sha256: str,
    threshold_rows: Sequence[FeatureRow],
    envelope: RiskEnvelope,
    reference_rows: int,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "v4_hybrid_threshold_checkpoint_contract",
        "purpose": purpose,
        "modelFitReceiptSha256": model_fit_receipt_sha256,
        "modelArtifactSha256": model_artifact_sha256,
        "thresholdRowUniverseSha256": _row_ids_sha256(threshold_rows),
        "envelope": asdict(envelope),
        "referenceRows": int(reference_rows),
    }


def _write_threshold_checkpoint(
    path: Path,
    threshold: HybridThresholdFit,
    *,
    contract: Mapping[str, Any],
) -> None:
    _atomic_write(
        path,
        (
            json.dumps(asdict(threshold), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8"),
    )
    _atomic_write(
        _contract_sidecar(path),
        _artifact_binding_bytes(
            contract=contract,
            artifact_sha256=_sha256_file(path),
        ),
    )


def _read_threshold_checkpoint(
    path: Path,
    *,
    expected_contract: Mapping[str, Any],
) -> HybridThresholdFit:
    sidecar = _contract_sidecar(path)
    if not sidecar.is_file():
        raise ValueError(f"threshold checkpoint contract mismatch: {path}")
    binding = json.loads(sidecar.read_text(encoding="utf-8"))
    if binding.get("contract") != dict(expected_contract):
        raise ValueError(f"threshold checkpoint contract mismatch: {path}")
    if binding.get("artifactSha256") != _sha256_file(path):
        raise ValueError(f"threshold checkpoint artifact SHA mismatch: {path}")
    return HybridThresholdFit(**json.loads(path.read_text(encoding="utf-8")))


def _fit_selector_threshold(
    scores: Sequence[Any],
    labels: Mapping[str, str],
    *,
    envelope: RiskEnvelope,
    reference_rows: int,
) -> HybridThresholdFit:
    months = {str(row.month) for row in scores}
    if len(months) != 1:
        raise ValueError("threshold score rows must belong to one month")
    selection_month = next(iter(months))
    scaled = _scaled_envelope(envelope, len(scores), reference_rows)
    gray_vetoes = (
        (1.0,)
        if not scores or max(float(row.gray_probability) for row in scores) <= 0.0
        else (0.5, 0.75, 0.9, 1.0)
    )
    common = {
        "process_thresholds": (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995),
        "risk_vetoes": (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4),
        "ng_thresholds": (0.98, 0.99, 0.995, 0.999, 1.0),
        "gray_vetoes": gray_vetoes,
        "selection_month": selection_month,
        "max_ship_to_process": scaled.max_ship_to_process,
        "max_repair_to_process": scaled.max_repair_to_process,
        "max_risk_to_process": scaled.max_risk_to_process,
        "min_auto_accuracy": scaled.min_auto_accuracy,
    }
    try:
        return fit_hybrid_thresholds(scores, labels, **common)
    except ValueError:
        relaxed = _scaled_envelope(AMBER_ENVELOPE, len(scores), reference_rows)
        return fit_hybrid_thresholds(
            scores,
            labels,
            **{
                **common,
                "max_ship_to_process": relaxed.max_ship_to_process,
                "max_repair_to_process": relaxed.max_repair_to_process,
                "max_risk_to_process": relaxed.max_risk_to_process,
                "min_auto_accuracy": relaxed.min_auto_accuracy,
            },
        )


def _load_prediction_jsonl_gz(path: Path, expected_ids: set[str]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    with path.open("rb") as raw_handle:
        with gzip.open(raw_handle, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                row_id = str(item.get("row_id") or "")
                if row_id in predictions:
                    raise ValueError(f"duplicate baseline prediction at line {line_number}: {row_id}")
                predictions[row_id] = str(item.get("predicted") or "")
    if set(predictions) != expected_ids:
        raise ValueError(
            "current hierarchy selector row universe mismatch: "
            f"missing={len(expected_ids - set(predictions))} extra={len(set(predictions) - expected_ids)}"
        )
    return predictions


def _fill_reviews(
    baseline: Mapping[str, str],
    candidate: Mapping[str, str],
) -> dict[str, str]:
    if set(baseline) != set(candidate):
        raise ValueError("review-fill candidate must match the baseline row universe")
    return {
        row_id: (
            str(baseline[row_id])
            if str(baseline[row_id]) in ACTION_LABELS
            else str(candidate[row_id])
        )
        for row_id in baseline
    }


def _apply_low_support_global_fallback(
    rows: Sequence[Any],
    baseline: Mapping[str, str],
    global_predictions: Mapping[str, str],
    *,
    support_rows: Sequence[Any],
    min_company_rows: int,
) -> dict[str, str]:
    if min_company_rows < 1:
        raise ValueError("global fallback support must be positive")
    row_ids = {row.row_id for row in rows}
    if row_ids != set(baseline) or row_ids != set(global_predictions):
        raise ValueError("global fallback inputs must share one row universe")
    support: dict[str, int] = {}
    for row in support_rows:
        customer = str(row.customer_key)
        support[customer] = support.get(customer, 0) + 1
    output: dict[str, str] = {}
    for row in rows:
        current = str(baseline[row.row_id])
        fallback = str(global_predictions[row.row_id])
        if (
            current not in ACTION_LABELS
            and support.get(str(row.customer_key), 0) < min_company_rows
            and fallback in ACTION_LABELS
        ):
            current = fallback
        output[row.row_id] = current
    return output


def _restrict_recovery_to_cold_products(
    rows: Sequence[Any],
    baseline_predictions: Mapping[str, str],
    recovered_predictions: Mapping[str, str],
    *,
    prior_history_rows: Sequence[Any],
) -> dict[str, str]:
    row_ids = {row.row_id for row in rows}
    if row_ids != set(baseline_predictions) or row_ids != set(recovered_predictions):
        raise ValueError("cold-start recovery inputs must share one row universe")
    known_products = {str(row.product_key) for row in prior_history_rows}
    output: dict[str, str] = {}
    months = sorted({str(row.month) for row in rows})
    for month in months:
        month_rows = tuple(row for row in rows if str(row.month) == month)
        for row in month_rows:
            output[row.row_id] = (
                str(recovered_predictions[row.row_id])
                if str(row.product_key) not in known_products
                else str(baseline_predictions[row.row_id])
            )
        known_products.update(str(row.product_key) for row in month_rows)
    return output


def _apply_frozen_recipe(
    rows: Sequence[FeatureRow],
    *,
    baseline_predictions: Mapping[str, str],
    raw_model_predictions: Mapping[str, Mapping[str, str]],
    model_scores: Mapping[str, Sequence[V4HybridScoreRow]],
    recipe: Mapping[str, Any],
    recovery_signatures: Sequence[Any],
    protection_signatures: Sequence[Any],
    support_rows: Sequence[FeatureRow],
    cold_start_history_rows: Sequence[FeatureRow],
) -> dict[str, str]:
    variant = str(recipe.get("baseVariant") or "current")
    predictions = dict(baseline_predictions)
    if variant != "current" and recipe.get("predictionBase") != "current":
        if variant not in raw_model_predictions:
            raise ValueError("frozen recipe is missing its base model predictions")
        predictions = _fill_reviews(predictions, raw_model_predictions[variant])
    layers = set(str(item) for item in recipe.get("layers", ()))
    routes = HierarchicalRoutes(global_candidate="__review__")
    if "process_recovery" in layers:
        predictions = compose_v4_derived_policy(
            rows,
            baseline_predictions=predictions,
            candidate_predictions={},
            routes=routes,
            recovery_signatures=tuple(
                item
                for item in recovery_signatures
                if item.scope in {"product_detail_counts", "product_type"}
            ),
        )
    if "cold_start" in layers:
        recovered = compose_v4_derived_policy(
            rows,
            baseline_predictions=predictions,
            candidate_predictions={},
            routes=routes,
            recovery_signatures=tuple(
                item for item in recovery_signatures if item.scope == "detail_counts"
            ),
        )
        predictions = _restrict_recovery_to_cold_products(
            rows,
            predictions,
            recovered,
            prior_history_rows=cold_start_history_rows,
        )
    if "global_fallback" in layers:
        fallback_variant = str(recipe.get("globalFallbackVariant") or "")
        if fallback_variant not in raw_model_predictions:
            raise ValueError("frozen recipe is missing global fallback predictions")
        predictions = _apply_low_support_global_fallback(
            rows,
            predictions,
            raw_model_predictions[fallback_variant],
            support_rows=support_rows,
            min_company_rows=int(recipe.get("globalFallbackMinCompanyRows") or 50),
        )
    if "protection" in layers:
        predictions = compose_v4_derived_policy(
            rows,
            baseline_predictions=predictions,
            candidate_predictions={},
            routes=routes,
            protection_signatures=protection_signatures,
        )
    if "model_risk_guard" in layers:
        raw_guard = recipe.get("guardThresholds")
        if not isinstance(raw_guard, Mapping) or variant not in model_scores:
            raise ValueError("frozen model-risk guard is missing thresholds or scores")
        predictions = apply_model_risk_guard(
            predictions,
            model_scores[variant],
            ModelRiskGuard(**raw_guard),
        )
    return predictions


def _bind_reject_only_result(
    experiment: Any,
    *,
    registry_sha256: str,
    companies: Sequence[str],
    selected_candidate: str,
    accepted: bool,
) -> tuple[LockboxResult, str]:
    assignments = {
        company: selected_candidate if accepted else "review_all"
        for company in sorted(companies)
    }
    rejection_reasons = {
        company: () if accepted else ("lockbox_risk_envelope_rejected",)
        for company in sorted(companies)
    }
    lockbox_universe_sha256 = _row_ids_sha256(experiment.bundle.lockbox)
    assignment_payload = {
        "schemaVersion": 1,
        "registrySha256": registry_sha256,
        "lockboxUniverseSha256": lockbox_universe_sha256,
        "assignmentMode": "strict",
        "assignments": assignments,
        "rejections": {
            company: list(rejection_reasons[company]) for company in sorted(companies)
        },
    }
    assignment_sha256 = hashlib.sha256(
        (
            json.dumps(
                assignment_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    result = LockboxResult(
        assignment_mode="strict",
        assignments=assignments,
        rejection_reasons=rejection_reasons,
        lockbox_universe_sha256=lockbox_universe_sha256,
        assignment_sha256=assignment_sha256,
    )
    return result, experiment.bind_lockbox_result(result)


def _records_for_maps(
    prediction_maps: Mapping[str, Mapping[str, str]],
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    *,
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    november_rows = tuple(row for row in rows if row.month == "2025-11")
    for complexity, candidate_id in enumerate(candidate_ids):
        predictions = prediction_maps[candidate_id]
        if set(predictions) != {row.row_id for row in rows}:
            raise ValueError(f"candidate row universe mismatch: {candidate_id}")
        selector_metrics = _compute_prediction_metrics(rows, labels, predictions)
        selection_metrics = _compute_prediction_metrics(november_rows, labels, predictions)
        output.append(
            {
                "candidateId": candidate_id,
                "complexityRank": complexity,
                "metrics": _metric_payload(selection_metrics),
                "selectorMetrics": _metric_payload(selector_metrics),
                "selectionPeriod": "2025-11",
            }
        )
    return output


def _compute_prediction_metrics(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    predictions: Mapping[str, str],
) -> Metrics:
    return compute_metrics(
        OutcomeRow(
            row_id=row.row_id,
            customer_key=row.customer_key,
            month=row.month,
            episode_id=row.episode_id,
            actual=labels[row.row_id],
            predicted=str(predictions.get(row.row_id) or ""),
            raw_customer=row.raw_customer,
            legacy_customer_key=row.legacy_customer_key,
            product_key=row.product_key,
        )
        for row in rows
    )


def _metric_payload(metrics: Metrics) -> dict[str, Any]:
    return {
        "evaluation_rows": metrics.evaluation_rows,
        "auto_rows": metrics.auto_rows,
        "review_rows": metrics.review_rows,
        "automation_rate": metrics.automation_rate,
        "correct_auto_rows": metrics.correct_auto_rows,
        "auto_accuracy": metrics.auto_accuracy,
        "actual_process_rows": metrics.actual_process_rows,
        "correct_process_rows": metrics.correct_process_rows,
        "correct_process_recovery": metrics.correct_process_recovery,
        "actual_process_automation_rate": metrics.actual_process_automation_rate,
        "ship_to_process": metrics.ship_to_process,
        "repair_to_process": metrics.repair_to_process,
        "risk_to_process": metrics.risk_to_process,
        "process_to_ship": metrics.process_to_ship,
        "process_to_repair": metrics.process_to_repair,
        "process_to_ng": metrics.process_to_ng,
    }


def _assert_label_vaults_locked(experiment: Any) -> None:
    for opener in (experiment.bundle.audit_labels, experiment.bundle.lockbox_labels):
        try:
            opener()
        except AuditLabelsLockedError:
            continue
        raise ValueError("audit and lockbox labels must remain sealed during selector execution")


def _write_ablation_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "candidate_id",
        "selected_source",
        "evaluation_rows",
        "auto_rows",
        "automation_rate",
        "auto_accuracy",
        "actual_process_automation_rate",
        "ship_to_process",
        "repair_to_process",
        "risk_to_process",
        "process_to_ng",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = _metrics(record)
            recipe = record.get("recipe") if isinstance(record.get("recipe"), Mapping) else {}
            writer.writerow(
                {
                    "candidate_id": record["candidateId"],
                    "selected_source": recipe.get("selectedSourceCandidateId", ""),
                    "evaluation_rows": metrics["evaluation_rows"],
                    "auto_rows": metrics["auto_rows"],
                    "automation_rate": metrics["automation_rate"],
                    "auto_accuracy": metrics["auto_accuracy"],
                    "actual_process_automation_rate": metrics[
                        "actual_process_automation_rate"
                    ],
                    "ship_to_process": metrics["ship_to_process"],
                    "repair_to_process": metrics["repair_to_process"],
                    "risk_to_process": metrics["risk_to_process"],
                    "process_to_ng": metrics["process_to_ng"],
                }
            )
    temporary.replace(path)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prediction_json(path: Path, expected_ids: set[str]) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError(f"prediction artifact is missing a map: {path}")
    normalized = {str(row_id): str(label or "") for row_id, label in predictions.items()}
    if set(normalized) != expected_ids:
        raise ValueError(
            f"prediction artifact row universe mismatch at {path}: "
            f"missing={len(expected_ids - set(normalized))} extra={len(set(normalized) - expected_ids)}"
        )
    return normalized


def _threshold_numeric_payload(threshold: HybridThresholdFit | None) -> dict[str, float]:
    if threshold is None:
        return {}
    return {
        "processThreshold": float(threshold.process_threshold),
        "shipmentVeto": float(threshold.shipment_veto),
        "repairVeto": float(threshold.repair_veto),
        "shipmentAutoThreshold": float(threshold.shipment_auto_threshold),
        "repairAutoThreshold": float(threshold.repair_auto_threshold),
        "grayVeto": float(threshold.gray_veto),
    }


def _finalize_pre_audit_frontiers(
    config_path: Path,
    output_dir: Path,
    *,
    experiment: Any,
    config: Mapping[str, Any],
    common_spec: FeatureSpec,
    selected_records: Mapping[str, Mapping[str, Any]],
    policy_recipes: Mapping[str, Mapping[str, Any]],
    envelopes: Mapping[str, RiskEnvelope],
    model_family: str,
) -> dict[str, Any]:
    final_model_dir = output_dir / "final_models"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    chronology = experiment.chronology
    needed_variants: set[str] = set()
    for record in selected_records.values():
        recipe = policy_recipes[str(record["candidateId"])]
        needed_variants.add(str(recipe.get("baseVariant") or "current"))
        if recipe.get("globalFallbackVariant"):
            needed_variants.add(str(recipe["globalFallbackVariant"]))
    needed_variants.discard("current")
    metadata = load_privileged_training_metadata(
        config["historyRows"],
        experiment.bundle.metadata.source_dataset_id,
        end_date="2025-08-31",
    )
    final_models: dict[str, V4HybridCandidate] = {}
    final_thresholds: dict[str, HybridThresholdFit] = {}
    selector_model_dir = output_dir / "models"
    selector_settings = {
        "unweighted": {"grayWeight": None, "grayAuxiliary": False, "seed": 17},
        "gray_weighted": {"grayWeight": 0.5, "grayAuxiliary": False, "seed": 31},
        "gray_auxiliary": {"grayWeight": None, "grayAuxiliary": True, "seed": 47},
    }
    for variant in sorted(needed_variants):
        settings = selector_settings[variant]
        source_model_path = selector_model_dir / f"selector_{variant}.pkl"
        source_threshold_path = selector_model_dir / f"selector_{variant}.threshold.json"
        model_path = final_model_dir / f"{variant}.pkl"
        threshold_path = final_model_dir / f"{variant}.threshold.json"
        gray_weight = settings["grayWeight"]
        weights = (
            gray_zone_sample_weights(
                chronology.train,
                metadata,
                gray_weight=float(gray_weight),
            )
            if gray_weight is not None
            else None
        )
        candidate = _read_model_checkpoint(source_model_path)
        candidate.score_batch_size = 500
        _validate_model_checkpoint(
            candidate,
            source_dataset_id=experiment.bundle.metadata.source_dataset_id,
            history_sha256=experiment.bundle.metadata.source_hashes["historyRows"],
            gray_auxiliary=bool(settings["grayAuxiliary"]),
            model_family=model_family,
            random_state=int(settings["seed"]),
            training_rows_sha256=_row_ids_sha256(chronology.train),
            calibration_rows_sha256=_row_ids_sha256(
                chronology.probability_calibration
            ),
            sample_weight_sha256=_sample_weight_sha256(
                weights,
                len(chronology.train),
            ),
        )
        threshold_contract = _threshold_checkpoint_contract(
            purpose=f"selector_threshold::{variant}",
            model_fit_receipt_sha256=str(candidate.fit_receipt["receiptSha256"]),
            model_artifact_sha256=_sha256_file(source_model_path),
            threshold_rows=chronology.threshold_fit,
            envelope=GREEN_ENVELOPE,
            reference_rows=experiment.bundle.metadata.expected_audit_action_rows,
        )
        threshold = _read_threshold_checkpoint(
            source_threshold_path,
            expected_contract=threshold_contract,
        )
        _atomic_write(model_path, source_model_path.read_bytes())
        _atomic_write(
            _contract_sidecar(model_path),
            _contract_sidecar(source_model_path).read_bytes(),
        )
        _atomic_write(threshold_path, source_threshold_path.read_bytes())
        _atomic_write(
            _contract_sidecar(threshold_path),
            _contract_sidecar(source_threshold_path).read_bytes(),
        )
        final_models[variant] = candidate
        final_thresholds[variant] = threshold

    lockbox_ids = {row.row_id for row in experiment.bundle.lockbox}
    baseline_lockbox_path = (
        Path(__file__).resolve().parents[1]
        / "outputs/postfit_company_hs_grid1_final_refit_all_run4/sealed_lockbox_predictions.json"
    )
    baseline_lockbox = _load_prediction_json(baseline_lockbox_path, lockbox_ids)
    lockbox_scores: dict[str, tuple[V4HybridScoreRow, ...]] = {}
    lockbox_raw_maps: dict[str, dict[str, str]] = {}
    prior_rows = (
        *chronology.train,
        *chronology.probability_calibration,
        *chronology.threshold_fit,
        *chronology.selector,
    )
    prior_labels = {
        **chronology.train_labels,
        **chronology.probability_calibration_labels,
        **chronology.threshold_fit_labels,
        **chronology.selector_labels,
    }
    for variant, candidate in final_models.items():
        print(f"[final:{variant}] scoring reject-only lockbox", flush=True)
        scores = candidate.score_rows(
            experiment.bundle.lockbox,
            prior_history_rows=prior_rows,
            prior_history_labels=prior_labels,
            prior_history_cutoff="2025-11-30",
        )
        labels = apply_hybrid_thresholds(scores, final_thresholds[variant])
        lockbox_raw_maps[variant] = labels
        lockbox_scores[variant] = scores

    audit_ids = {row.row_id for row in experiment.bundle.audit}
    baseline_audit_path = (
        Path(__file__).resolve().parents[1]
        / "outputs/postfit_company_hs_grid1_final_refit_all_run4/sealed_audit_predictions.json"
    )
    baseline_audit = _load_prediction_json(baseline_audit_path, audit_ids)
    audit_scores: dict[str, tuple[V4HybridScoreRow, ...]] = {}
    audit_raw_maps: dict[str, dict[str, str]] = {}
    for variant, candidate in final_models.items():
        print(f"[final:{variant}] scoring sealed retrospective audit", flush=True)
        scores = candidate.score_rows(
            experiment.bundle.audit,
            prior_history_rows=prior_rows,
            prior_history_labels=prior_labels,
            prior_history_cutoff="2025-11-30",
        )
        audit_scores[variant] = scores
        audit_raw_maps[variant] = apply_hybrid_thresholds(
            scores,
            final_thresholds[variant],
        )

    all_selector_rows = tuple(chronology.selector)
    all_selector_labels = chronology.selector_labels
    final_recovery = derive_recovery_signatures(
        all_selector_rows,
        all_selector_labels,
        min_support=10,
        min_process_rate=1.0,
    )
    final_protection = derive_protection_signatures(
        all_selector_rows,
        all_selector_labels,
        min_support=20,
        min_ng=2,
        min_ng_rate=0.1,
    )
    results: dict[str, Any] = {}
    for tier in ("green", "amber"):
        frontier_path = output_dir / f"{tier}_frontier.json"
        if frontier_path.exists():
            raise ValueError(
                "a frozen frontier already exists; use a new output directory "
                "instead of rebuilding against mutable checkpoints"
            )
        record = selected_records[tier]
        alias = str(record["candidateId"])
        recipe = policy_recipes[alias]
        variant = str(recipe.get("baseVariant") or "current")
        predictions = _apply_frozen_recipe(
            experiment.bundle.lockbox,
            baseline_predictions=baseline_lockbox,
            raw_model_predictions=lockbox_raw_maps,
            model_scores=lockbox_scores,
            recipe=recipe,
            recovery_signatures=final_recovery,
            protection_signatures=final_protection,
            support_rows=all_selector_rows,
            cold_start_history_rows=prior_rows,
        )
        frontier_experiment = prepare_experiment(
            config_path,
            output_dir / f"{tier}_state",
        )
        model_path = final_model_dir / f"{variant}.pkl" if variant != "current" else None
        threshold = final_thresholds.get(variant)
        used_variants = {variant} - {"current"}
        if recipe.get("globalFallbackVariant"):
            used_variants.add(str(recipe["globalFallbackVariant"]))
        model_artifacts = {
            used_variant: {
                "modelSha256": _sha256_file(
                    final_model_dir / f"{used_variant}.pkl"
                ),
                "modelFitReceiptSha256": str(
                    final_models[used_variant].fit_receipt["receiptSha256"]
                ),
                "thresholdFileSha256": _sha256_file(
                    final_model_dir / f"{used_variant}.threshold.json"
                ),
                "thresholds": _threshold_numeric_payload(
                    final_thresholds[used_variant]
                ),
            }
            for used_variant in sorted(used_variants)
        }
        registry_payload = {
            "tier": tier,
            "recipe": dict(recipe),
            "modelArtifacts": model_artifacts,
            "selectionMonths": ["2025-10", "2025-11"],
        }
        registry_sha = hashlib.sha256(
            (
                json.dumps(
                    registry_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        companies = sorted(
            {
                row.customer_key
                for row in (*frontier_experiment.bundle.selector, *frontier_experiment.bundle.lockbox)
            }
        )
        spec = PolicySpec(
            policy_name=f"v4_derived_hybrid_{tier}",
            source_dataset_id=frontier_experiment.bundle.metadata.source_dataset_id,
            history_sha256=frontier_experiment.bundle.metadata.source_hashes["historyRows"],
            selection_months=("2025-10", "2025-11"),
            feature_names=tuple(ALLOWED_RAW_FEATURES),
            model_params={
                "candidateRegistrySha256": registry_sha,
                "selectedCandidate": str(recipe.get("selectedSourceCandidateId") or alias),
                "modelSha256": _sha256_file(model_path) if model_path is not None else "",
                "modelFitReceiptSha256": (
                    str(final_models[variant].fit_receipt["receiptSha256"])
                    if variant in final_models
                    else ""
                ),
                "modelArtifacts": model_artifacts,
                "recipe": dict(recipe),
                "assignmentMode": "strict",
            },
            thresholds=_threshold_numeric_payload(threshold),
            company_assignments={company: alias for company in companies},
        )
        receipt = frontier_experiment.freeze_policy(spec)
        prediction_sha = frontier_experiment.seal_lockbox_predictions(predictions)
        lockbox_labels = frontier_experiment.open_lockbox(receipt)
        action_rows = tuple(
            row
            for row in frontier_experiment.bundle.lockbox
            if lockbox_labels.get(row.row_id) in ACTION_LABELS
        )
        metrics = _compute_prediction_metrics(action_rows, lockbox_labels, predictions)
        scaled_envelope = _scaled_envelope(
            envelopes[tier],
            len(action_rows),
            experiment.bundle.metadata.expected_audit_action_rows,
        )
        disposition = reject_only_lockbox(record, _metric_payload(metrics), scaled_envelope)
        lockbox_result, final_policy_sha = _bind_reject_only_result(
            frontier_experiment,
            registry_sha256=registry_sha,
            companies=companies,
            selected_candidate=alias,
            accepted=bool(disposition["accepted"]),
        )
        shadow_audit_predictions = _apply_frozen_recipe(
            frontier_experiment.bundle.audit,
            baseline_predictions=baseline_audit,
            raw_model_predictions=audit_raw_maps,
            model_scores=audit_scores,
            recipe=recipe,
            recovery_signatures=final_recovery,
            protection_signatures=final_protection,
            support_rows=all_selector_rows,
            cold_start_history_rows=prior_rows,
        )
        operating_audit_predictions = (
            shadow_audit_predictions
            if disposition["accepted"]
            else {row_id: "REVIEW" for row_id in sorted(audit_ids)}
        )
        audit_prediction_sha = frontier_experiment.seal_audit_predictions(
            operating_audit_predictions,
            final_policy_sha256=final_policy_sha,
        )
        _atomic_write(output_dir / f"{tier}_frozen_policy.json", receipt.policy_bytes)
        _atomic_write(
            output_dir / f"{tier}_freeze_receipt.json",
            (
                json.dumps(
                    {
                        "policySha256": receipt.sha256,
                        "bindingSha256": receipt.binding_sha256,
                        "lockboxPredictionSha256": prediction_sha,
                        "lockboxAssignmentSha256": lockbox_result.assignment_sha256,
                        "finalPolicySha256": final_policy_sha,
                        "auditPredictionSha256": audit_prediction_sha,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        sealed_lockbox = {
            "schemaVersion": 1,
            "kind": "v4_derived_hybrid_sealed_lockbox_predictions",
            "auditInputsIncluded": False,
            "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
            "historySha256": experiment.bundle.metadata.source_hashes["historyRows"],
            "selectionMonths": ["2025-10", "2025-11"],
            "policySha256": receipt.sha256,
            "predictionSha256": prediction_sha,
            "predictions": dict(sorted(predictions.items())),
        }
        _atomic_write(
            output_dir / f"{tier}_sealed_lockbox_predictions.json",
            canonical_artifact_bytes(sealed_lockbox),
        )
        shadow_payload = {
            "schemaVersion": 1,
            "kind": "v4_derived_hybrid_sealed_shadow_audit_predictions",
            "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
            "basePolicySha256": receipt.sha256,
            "lockboxDisposition": disposition,
            "expectedRowCount": len(audit_ids),
            "predictionMapSha256": hashlib.sha256(
                (
                    json.dumps(
                        dict(sorted(shadow_audit_predictions.items())),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest(),
            "predictions": dict(sorted(shadow_audit_predictions.items())),
        }
        _atomic_write(
            output_dir / f"{tier}_sealed_shadow_audit_predictions.json",
            (
                json.dumps(
                    shadow_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )
        operating_payload = {
            "schemaVersion": 2,
            "kind": "v4_derived_hybrid_sealed_operating_audit_predictions",
            "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
            "basePolicySha256": receipt.sha256,
            "finalPolicySha256": final_policy_sha,
            "lockboxPredictionSha256": prediction_sha,
            "lockboxAssignmentSha256": lockbox_result.assignment_sha256,
            "predictionSealSha256": audit_prediction_sha,
            "expectedRowCount": len(audit_ids),
            "predictions": dict(sorted(operating_audit_predictions.items())),
        }
        _atomic_write(
            output_dir / f"{tier}_sealed_operating_audit_predictions.json",
            (
                json.dumps(
                    operating_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        )
        frontier_payload = {
            "schemaVersion": 1,
            "kind": "v4_derived_hybrid_frontier_lockbox_result",
            "auditInputsIncluded": False,
            "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
            "historySha256": experiment.bundle.metadata.source_hashes["historyRows"],
            "selectionMonths": ["2025-10", "2025-11"],
            "tier": tier,
            "recipe": dict(recipe),
            "policySha256": receipt.sha256,
            "policyBindingSha256": receipt.binding_sha256,
            "lockboxPredictionSha256": prediction_sha,
            "lockboxAssignmentSha256": lockbox_result.assignment_sha256,
            "finalPolicySha256": final_policy_sha,
            "auditPredictionSha256": audit_prediction_sha,
            "shadowAuditPredictionArtifactSha256": _sha256_file(
                output_dir / f"{tier}_sealed_shadow_audit_predictions.json"
            ),
            "operatingAuditPredictionArtifactSha256": _sha256_file(
                output_dir / f"{tier}_sealed_operating_audit_predictions.json"
            ),
            "lockboxMetrics": _metric_payload(metrics),
            "lockboxEnvelope": asdict(scaled_envelope),
            "disposition": disposition,
            "promotionBoundary": "requires_future_shadow",
        }
        _atomic_write(frontier_path, canonical_artifact_bytes(frontier_payload))
        results[tier] = frontier_payload
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the leakage-controlled V4-derived frontier")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", default="outputs/v4_derived_hybrid")
    parser.add_argument("--model-family", choices=("linear", "lightgbm"), default="linear")
    parser.add_argument("--dry-contract", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_contract:
        print(json.dumps({"ablations": ABLATION_IDS}, ensure_ascii=False))
        return 0
    run_frontier(args.config, args.output_dir, model_family=args.model_family)
    return 0


def run_frontier(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    model_family: str = "linear",
) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    resolved_output = Path(output_dir).resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    model_dir = resolved_output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(resolved_config.read_text(encoding="utf-8"))
    experiment = prepare_experiment(resolved_config, resolved_output / "prepared")
    chronology = experiment.chronology
    validate_chronology_contract(chronology)
    _assert_label_vaults_locked(experiment)

    selector_rows = tuple(chronology.selector)
    selector_labels = chronology.selector_labels
    selector_ids = {row.row_id for row in selector_rows}
    baseline_path = Path(__file__).resolve().parents[1] / (
        "outputs/frontier_company_hs_grid1/budgeted_selector_decisions.jsonl.gz"
    )
    baseline_predictions = _load_prediction_jsonl_gz(baseline_path, selector_ids)
    all_maps: dict[str, dict[str, str]] = {
        "current_hierarchy": baseline_predictions,
    }
    policy_recipes: dict[str, dict[str, Any]] = {
        "current_hierarchy": {"baseVariant": "current", "layers": []},
    }

    metadata = load_privileged_training_metadata(
        config["historyRows"],
        experiment.bundle.metadata.source_dataset_id,
        end_date="2025-08-31",
    )
    common_spec = FeatureSpec(
        include_customer_key=True,
        include_product_key=False,
        word_min_df=5,
        char_min_df=5,
        word_max_features=30_000,
        char_max_features=60_000,
        max_total_features=120_000,
    )
    variant_config = {
        "unweighted": {"grayWeight": None, "grayAuxiliary": False, "seed": 17},
        "gray_weighted": {"grayWeight": 0.5, "grayAuxiliary": False, "seed": 31},
        "gray_auxiliary": {"grayWeight": None, "grayAuxiliary": True, "seed": 47},
    }
    variant_predictions: dict[str, dict[str, str]] = {}
    variant_thresholds: dict[str, HybridThresholdFit] = {}
    variant_receipts: dict[str, Mapping[str, Any]] = {}
    for variant, settings in variant_config.items():
        model_path = model_dir / f"selector_{variant}.pkl"
        threshold_path = model_dir / f"selector_{variant}.threshold.json"
        threshold_score_path = model_dir / f"selector_{variant}.threshold_scores.npz"
        selector_score_path = model_dir / f"selector_{variant}.selector_scores.npz"
        gray_weight = settings["grayWeight"]
        weights = (
            gray_zone_sample_weights(
                chronology.train,
                metadata,
                gray_weight=float(gray_weight),
            )
            if gray_weight is not None
            else None
        )
        gray_labels = (
            {row.row_id: metadata[row.row_id].wait_nonblank for row in chronology.train}
            if settings["grayAuxiliary"]
            else None
        )
        calibration_gray = (
            {
                row.row_id: metadata[row.row_id].wait_nonblank
                for row in chronology.probability_calibration
            }
            if settings["grayAuxiliary"]
            else None
        )
        if model_path.exists():
            print(f"[{variant}] loading fitted checkpoint", flush=True)
            candidate = _read_model_checkpoint(model_path)
            _validate_model_checkpoint(
                candidate,
                source_dataset_id=experiment.bundle.metadata.source_dataset_id,
                history_sha256=experiment.bundle.metadata.source_hashes["historyRows"],
                gray_auxiliary=bool(settings["grayAuxiliary"]),
                model_family=model_family,
                random_state=int(settings["seed"]),
                training_rows_sha256=_row_ids_sha256(chronology.train),
                calibration_rows_sha256=_row_ids_sha256(
                    chronology.probability_calibration
                ),
                sample_weight_sha256=_sample_weight_sha256(
                    weights,
                    len(chronology.train),
                ),
            )
            candidate.score_batch_size = 500
        else:
            print(f"[{variant}] fitting model", flush=True)
            candidate = V4HybridCandidate(
                model_family=model_family,
                random_state=int(settings["seed"]),
                feature_spec=common_spec,
                validation_config=config,
                score_batch_size=500,
            )
            candidate.fit(
                chronology.train,
                chronology.train_labels,
                calibration_rows=chronology.probability_calibration,
                calibration_labels=chronology.probability_calibration_labels,
                source_dataset_id=experiment.bundle.metadata.source_dataset_id,
                history_sha256=experiment.bundle.metadata.source_hashes["historyRows"],
                sample_weights=weights,
                gray_labels=gray_labels,
                calibration_gray_labels=calibration_gray,
            )
            _write_model_checkpoint(model_path, candidate)
            print(f"[{variant}] model checkpoint saved", flush=True)
        variant_receipts[variant] = dict(candidate.fit_receipt)
        threshold_score_contract = _score_checkpoint_contract(
            purpose=f"selector_threshold::{variant}",
            model_fit_receipt_sha256=str(candidate.fit_receipt["receiptSha256"]),
            model_artifact_sha256=_sha256_file(model_path),
            rows=chronology.threshold_fit,
        )
        selector_score_contract = _score_checkpoint_contract(
            purpose=f"selector_scoring::{variant}",
            model_fit_receipt_sha256=str(candidate.fit_receipt["receiptSha256"]),
            model_artifact_sha256=_sha256_file(model_path),
            rows=selector_rows,
        )
        threshold_contract = _threshold_checkpoint_contract(
            purpose=f"selector_threshold::{variant}",
            model_fit_receipt_sha256=str(candidate.fit_receipt["receiptSha256"]),
            model_artifact_sha256=_sha256_file(model_path),
            threshold_rows=chronology.threshold_fit,
            envelope=GREEN_ENVELOPE,
            reference_rows=experiment.bundle.metadata.expected_audit_action_rows,
        )
        if threshold_path.exists():
            threshold = _read_threshold_checkpoint(
                threshold_path,
                expected_contract=threshold_contract,
            )
            print(f"[{variant}] loading threshold checkpoint", flush=True)
        else:
            if threshold_score_path.exists():
                threshold_scores = _read_score_npz(
                    threshold_score_path,
                    expected_contract=threshold_score_contract,
                )
            else:
                print(f"[{variant}] scoring threshold month", flush=True)
                threshold_scores = candidate.score_rows(chronology.threshold_fit)
                _write_score_npz(
                    threshold_score_path,
                    threshold_scores,
                    contract=threshold_score_contract,
                )
            del candidate
            gc.collect()
            print(f"[{variant}] fitting threshold grid", flush=True)
            threshold = _fit_selector_threshold(
                threshold_scores,
                chronology.threshold_fit_labels,
                envelope=GREEN_ENVELOPE,
                reference_rows=experiment.bundle.metadata.expected_audit_action_rows,
            )
            _write_threshold_checkpoint(
                threshold_path,
                threshold,
                contract=threshold_contract,
            )
            print(f"[{variant}] threshold checkpoint saved", flush=True)
        variant_thresholds[variant] = threshold
        if selector_score_path.exists():
            selector_scores = _read_score_npz(
                selector_score_path,
                expected_contract=selector_score_contract,
            )
            print(f"[{variant}] loading selector-score checkpoint", flush=True)
        else:
            candidate = _read_model_checkpoint(model_path)
            candidate.score_batch_size = 500
            prior_rows = (
                *chronology.train,
                *chronology.probability_calibration,
                *chronology.threshold_fit,
            )
            prior_labels = {
                **chronology.train_labels,
                **chronology.probability_calibration_labels,
                **chronology.threshold_fit_labels,
            }
            print(f"[{variant}] scoring selector months", flush=True)
            selector_scores = candidate.score_rows(
                selector_rows,
                prior_history_rows=prior_rows,
                prior_history_labels=prior_labels,
                prior_history_cutoff="2025-09-30",
            )
            _write_score_npz(
                selector_score_path,
                selector_scores,
                contract=selector_score_contract,
            )
            del candidate
            gc.collect()
            print(f"[{variant}] selector-score checkpoint saved", flush=True)
        variant_predictions[variant] = apply_hybrid_thresholds(selector_scores, threshold)
        del selector_scores, weights, gray_labels, calibration_gray
        gc.collect()

    unweighted = variant_predictions["unweighted"]
    domain_threshold = HybridThresholdFit(
        process_threshold=variant_thresholds["unweighted"].process_threshold,
        shipment_veto=1.0,
        repair_veto=1.0,
        shipment_auto_threshold=1.0,
        repair_auto_threshold=1.0,
        gray_veto=1.0,
        selection_month="2025-09",
    )
    domain_scores = _read_score_npz(
        model_dir / "selector_unweighted.selector_scores.npz",
        expected_contract=_score_checkpoint_contract(
            purpose="selector_scoring::unweighted",
            model_fit_receipt_sha256=str(
                variant_receipts["unweighted"]["receiptSha256"]
            ),
            model_artifact_sha256=_sha256_file(
                model_dir / "selector_unweighted.pkl"
            ),
            rows=selector_rows,
        ),
    )
    domain_predictions = apply_hybrid_thresholds(domain_scores, domain_threshold)
    del domain_scores
    gc.collect()

    all_maps["v4_domain_features"] = _fill_reviews(
        baseline_predictions,
        domain_predictions,
    )
    all_maps["v4_three_head"] = _fill_reviews(baseline_predictions, unweighted)
    all_maps["v4_gray_weighted"] = _fill_reviews(
        baseline_predictions,
        variant_predictions["gray_weighted"],
    )
    all_maps["v4_gray_auxiliary"] = _fill_reviews(
        baseline_predictions,
        variant_predictions["gray_auxiliary"],
    )
    policy_recipes.update(
        {
            "v4_domain_features": {"baseVariant": "unweighted", "layers": ["process_only"]},
            "v4_three_head": {"baseVariant": "unweighted", "layers": ["three_head"]},
            "v4_gray_weighted": {"baseVariant": "gray_weighted", "layers": ["three_head"]},
            "v4_gray_auxiliary": {"baseVariant": "gray_auxiliary", "layers": ["three_head"]},
        }
    )

    pre_layer_records = _records_for_maps(
        all_maps,
        selector_rows,
        selector_labels,
        candidate_ids=ABLATION_IDS[:5],
    )
    scaled_green = _scaled_envelope(
        GREEN_ENVELOPE,
        sum(row.month == "2025-11" for row in selector_rows),
        experiment.bundle.metadata.expected_audit_action_rows,
    )
    try:
        base_choice = select_frontier_candidate(pre_layer_records, scaled_green)
    except ValueError:
        base_choice = select_frontier_candidate(
            pre_layer_records,
            _scaled_envelope(
                AMBER_ENVELOPE,
                sum(row.month == "2025-11" for row in selector_rows),
                experiment.bundle.metadata.expected_audit_action_rows,
            ),
        )
    base_id = str(base_choice["candidateId"])
    base_map = all_maps[base_id]
    october_rows = tuple(row for row in selector_rows if row.month == "2025-10")
    recovery = derive_recovery_signatures(
        october_rows,
        selector_labels,
        min_support=10,
        min_process_rate=1.0,
    )
    product_recovery = tuple(
        item for item in recovery if item.scope in {"product_detail_counts", "product_type"}
    )
    cold_recovery = tuple(item for item in recovery if item.scope == "detail_counts")
    protection = derive_protection_signatures(
        october_rows,
        selector_labels,
        min_support=20,
        min_ng=2,
        min_ng_rate=0.1,
    )
    no_candidate_routes = HierarchicalRoutes(global_candidate="__review__")
    all_maps["v4_process_recovery"] = compose_v4_derived_policy(
        selector_rows,
        baseline_predictions=base_map,
        candidate_predictions={},
        routes=no_candidate_routes,
        recovery_signatures=product_recovery,
    )
    cold_recovered = compose_v4_derived_policy(
        selector_rows,
        baseline_predictions=all_maps["v4_process_recovery"],
        candidate_predictions={},
        routes=no_candidate_routes,
        recovery_signatures=cold_recovery,
    )
    all_maps["v4_cold_start"] = _restrict_recovery_to_cold_products(
        selector_rows,
        all_maps["v4_process_recovery"],
        cold_recovered,
        prior_history_rows=(
            *chronology.train,
            *chronology.probability_calibration,
            *chronology.threshold_fit,
        ),
    )
    all_maps["v4_full_fallback"] = _apply_low_support_global_fallback(
        selector_rows,
        all_maps["v4_cold_start"],
        variant_predictions["unweighted"],
        support_rows=october_rows,
        min_company_rows=50,
    )
    policy_recipes["v4_process_recovery"] = {
        "baseVariant": policy_recipes[base_id]["baseVariant"],
        "baseCandidateId": base_id,
        "layers": ["process_recovery"],
    }
    policy_recipes["v4_cold_start"] = {
        **policy_recipes["v4_process_recovery"],
        "layers": ["process_recovery", "cold_start"],
    }
    policy_recipes["v4_full_fallback"] = {
        **policy_recipes["v4_process_recovery"],
        "layers": ["process_recovery", "cold_start", "global_fallback"],
        "globalFallbackVariant": "unweighted",
        "globalFallbackMinCompanyRows": 50,
    }

    scaled_amber = _scaled_envelope(
        AMBER_ENVELOPE,
        sum(row.month == "2025-11" for row in selector_rows),
        experiment.bundle.metadata.expected_audit_action_rows,
    )
    october_ids = {row.row_id for row in october_rows}
    guarded_by_tier: dict[str, list[tuple[dict[str, Any], dict[str, str], dict[str, Any]]]] = {
        "green": [],
        "amber": [],
    }
    guard_sources = {
        "unweighted": "v4_three_head",
        "gray_weighted": "v4_gray_weighted",
        "gray_auxiliary": "v4_gray_auxiliary",
    }
    for variant, fill_candidate_id in guard_sources.items():
        full_scores = _read_score_npz(
            model_dir / f"selector_{variant}.selector_scores.npz",
            expected_contract=_score_checkpoint_contract(
                purpose=f"selector_scoring::{variant}",
                model_fit_receipt_sha256=str(
                    variant_receipts[variant]["receiptSha256"]
                ),
                model_artifact_sha256=_sha256_file(
                    model_dir / f"selector_{variant}.pkl"
                ),
                rows=selector_rows,
            ),
        )
        october_scores = tuple(row for row in full_scores if row.row_id in october_ids)
        for prediction_base, source_map in (
            ("current", baseline_predictions),
            ("review_fill", all_maps[fill_candidate_id]),
        ):
            october_base = {row_id: source_map[row_id] for row_id in october_ids}
            for tier, envelope in (("green", GREEN_ENVELOPE), ("amber", AMBER_ENVELOPE)):
                scaled_fit = _scaled_envelope(
                    envelope,
                    len(october_rows),
                    experiment.bundle.metadata.expected_audit_action_rows,
                )
                try:
                    guard = fit_model_risk_guard(
                        october_rows,
                        october_base,
                        october_scores,
                        selector_labels,
                        max_ship_to_process=scaled_fit.max_ship_to_process,
                        max_repair_to_process=scaled_fit.max_repair_to_process,
                        max_risk_to_process=scaled_fit.max_risk_to_process,
                        min_auto_accuracy=scaled_fit.min_auto_accuracy,
                    )
                except ValueError:
                    continue
                guarded_map = apply_model_risk_guard(source_map, full_scores, guard)
                internal_id = f"guarded::{tier}::{variant}::{prediction_base}"
                internal_record = _records_for_maps(
                    {internal_id: guarded_map},
                    selector_rows,
                    selector_labels,
                    candidate_ids=(internal_id,),
                )[0]
                recipe = {
                    "baseVariant": variant,
                    "baseCandidateId": (
                        "current_hierarchy" if prediction_base == "current" else fill_candidate_id
                    ),
                    "predictionBase": prediction_base,
                    "layers": ["model_risk_guard"],
                    "guardThresholds": asdict(guard),
                }
                guarded_by_tier[tier].append((internal_record, guarded_map, recipe))

    if not guarded_by_tier["green"] or not guarded_by_tier["amber"]:
        raise ValueError("model risk-guard search produced no Green or Amber candidate")
    best_guarded: dict[str, tuple[dict[str, Any], dict[str, str], dict[str, Any]]] = {}
    for tier, envelope in (("green", scaled_green), ("amber", scaled_amber)):
        choices = [item[0] for item in guarded_by_tier[tier]]
        selected = select_frontier_candidate(choices, envelope)
        best_guarded[tier] = next(
            item for item in guarded_by_tier[tier] if item[0]["candidateId"] == selected["candidateId"]
        )

    all_maps["v4_protection"] = compose_v4_derived_policy(
        selector_rows,
        baseline_predictions=all_maps["v4_full_fallback"],
        candidate_predictions={},
        routes=no_candidate_routes,
        protection_signatures=protection,
    )
    policy_recipes["v4_protection"] = {
        **policy_recipes["v4_full_fallback"],
        "layers": [
            "process_recovery",
            "cold_start",
            "global_fallback",
            "protection",
        ],
    }
    records = _records_for_maps(
        all_maps,
        selector_rows,
        selector_labels,
        candidate_ids=ABLATION_IDS[:9],
    )
    green_guard_record, green_guard_map, green_guard_recipe = best_guarded["green"]
    green_pool = [*records, green_guard_record]
    green_choice = select_frontier_candidate(green_pool, scaled_green)
    green_source_id = str(green_choice["candidateId"])
    if green_source_id == green_guard_record["candidateId"]:
        all_maps["v4_green_selected"] = dict(green_guard_map)
        policy_recipes["v4_green_selected"] = {
            **green_guard_recipe,
            "selectedSourceCandidateId": green_source_id,
        }
    else:
        all_maps["v4_green_selected"] = dict(all_maps[green_source_id])
        policy_recipes["v4_green_selected"] = {
            **policy_recipes[green_source_id],
            "selectedSourceCandidateId": green_source_id,
        }
    amber_guard_record, amber_guard_map, amber_guard_recipe = best_guarded["amber"]
    amber_pool = [*records, amber_guard_record]
    amber_choice = select_frontier_candidate(amber_pool, scaled_amber)
    amber_source_id = str(amber_choice["candidateId"])
    if amber_source_id == amber_guard_record["candidateId"]:
        all_maps["v4_amber_selected"] = dict(amber_guard_map)
        policy_recipes["v4_amber_selected"] = {
            **amber_guard_recipe,
            "selectedSourceCandidateId": amber_source_id,
        }
    else:
        all_maps["v4_amber_selected"] = dict(all_maps[amber_source_id])
        policy_recipes["v4_amber_selected"] = {
            **policy_recipes[amber_source_id],
            "selectedSourceCandidateId": amber_source_id,
        }
    records = _records_for_maps(
        all_maps,
        selector_rows,
        selector_labels,
        candidate_ids=ABLATION_IDS,
    )
    for record in records:
        record["recipe"] = policy_recipes[str(record["candidateId"])]
    validate_ablation_order(records)

    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "v4_derived_hybrid_frontier",
        "auditInputsIncluded": False,
        "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
        "historySha256": experiment.bundle.metadata.source_hashes["historyRows"],
        "selectionMonths": ["2025-10", "2025-11"],
        "chronology": {
            "trainingEndInclusive": "2025-07-31",
            "probabilityCalibrationMonth": "2025-08",
            "thresholdFitMonth": "2025-09",
            "selectorMonths": ["2025-10", "2025-11"],
        },
        "riskEnvelopes": {
            "green": asdict(GREEN_ENVELOPE),
            "amber": asdict(AMBER_ENVELOPE),
            "greenSelectorScaled": asdict(scaled_green),
            "amberSelectorScaled": asdict(scaled_amber),
        },
        "models": {
            variant: {
                "fitReceipt": dict(variant_receipts[variant]),
                "thresholds": asdict(variant_thresholds[variant]),
                "artifact": str((model_dir / f"selector_{variant}.pkl").relative_to(resolved_output)),
                "artifactSha256": _sha256_file(model_dir / f"selector_{variant}.pkl"),
            }
            for variant in sorted(variant_config)
        },
        "signatureCounts": {
            "processRecovery": len(product_recovery),
            "coldStart": len(cold_recovery),
            "protection": len(protection),
        },
        "ablations": records,
        "greenSelectedSource": policy_recipes["v4_green_selected"]["selectedSourceCandidateId"],
        "amberSelectedSource": policy_recipes["v4_amber_selected"]["selectedSourceCandidateId"],
        "status": "selected_before_reject_only_lockbox",
    }
    matrix_bytes = canonical_artifact_bytes(payload)
    _atomic_write(resolved_output / "ablation_matrix.json", matrix_bytes)
    _write_ablation_csv(resolved_output / "ablation_matrix.csv", records)
    selected_payload = {
        "schemaVersion": 1,
        "kind": "v4_derived_hybrid_selected_policy",
        "auditInputsIncluded": False,
        "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
        "historySha256": experiment.bundle.metadata.source_hashes["historyRows"],
        "selectionMonths": ["2025-10", "2025-11"],
        "ablationMatrixSha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "green": policy_recipes["v4_green_selected"],
        "amber": policy_recipes["v4_amber_selected"],
    }
    _atomic_write(
        resolved_output / "selected_policy.json",
        canonical_artifact_bytes(selected_payload),
    )
    frontier_results = _finalize_pre_audit_frontiers(
        resolved_config,
        resolved_output,
        experiment=experiment,
        config=config,
        common_spec=common_spec,
        selected_records={
            "green": next(item for item in records if item["candidateId"] == "v4_green_selected"),
            "amber": next(item for item in records if item["candidateId"] == "v4_amber_selected"),
        },
        policy_recipes=policy_recipes,
        envelopes={"green": GREEN_ENVELOPE, "amber": AMBER_ENVELOPE},
        model_family=model_family,
    )
    pre_audit_manifest = {
        "schemaVersion": 1,
        "kind": "v4_derived_hybrid_pre_audit_manifest",
        "auditInputsIncluded": False,
        "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
        "historySha256": experiment.bundle.metadata.source_hashes["historyRows"],
        "selectionMonths": ["2025-10", "2025-11"],
        "ablationMatrixSha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "frontiers": frontier_results,
    }
    _atomic_write(
        resolved_output / "pre_audit_manifest.json",
        canonical_artifact_bytes(pre_audit_manifest),
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())

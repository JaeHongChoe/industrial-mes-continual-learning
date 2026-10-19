from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import date
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .candidates import ACTION_ORDER, CandidatePrediction, ScoreRow
from .data_contract import ACTION_LABELS, IMMUTABLE_SELECTOR_MONTHS


_POLICY_PRECISION = {
    "calibration_zero_unsafe": 0.98,
    "calibration_precision_0.995": 0.995,
    "calibration_precision_0.990": 0.990,
}
THRESHOLD_ALGORITHM_VERSION = "top_one_threshold_fit_v2"
THRESHOLD_FIT_PHASE = "pre_selector_calibration"
THRESHOLD_METRIC_SEMANTICS = "top_one_empirical_precision"
THRESHOLD_SELECTOR_RESPONSIBILITY = "company_month_ucb"
_CALIBRATED_STATES = frozenset(
    {"calibrated", "cross_fit_calibrated", "cross_fitted_calibrated"}
)
_FIRST_SELECTOR_DATE = date.fromisoformat(f"{min(IMMUTABLE_SELECTOR_MONTHS)}-01")


@dataclass(frozen=True, slots=True)
class CalibrationMetric:
    automatic: int
    correct: int
    unsafe: int
    precision: float | None


@dataclass(frozen=True, slots=True)
class ThresholdFit:
    policy: str
    min_support: int
    thresholds: Mapping[str, float]
    enabled: Mapping[str, bool]
    calibration_metrics: Mapping[str, CalibrationMetric]
    calibration_rows_sha256: str
    source_dataset_id: str
    history_sha256: str
    phase: str
    calibration_start_date: str
    calibration_end_date: str
    calibration_row_universe_sha256: str
    fitted_artifact_sha256: str
    feature_schema_sha256: str
    score_family: str
    calibrated_model_provenance: Mapping[str, Any]
    calibrated_model_provenance_sha256: str
    class_order: tuple[str, ...]
    algorithm_version: str
    metric_semantics: str
    selector_responsibility: str
    sha256: str


def fit_thresholds(
    score_rows: Iterable[ScoreRow],
    labels: Mapping[str, str],
    *,
    policy: str,
    min_support: int = 10,
    source_dataset_id: str,
    history_sha256: str,
    phase: str,
    row_dates: Mapping[str, str],
    fitted_artifact_sha256: str,
    feature_schema_sha256: str,
) -> ThresholdFit:
    if policy not in _POLICY_PRECISION:
        raise ValueError(f"unknown threshold policy: {policy}")
    if min_support < 1:
        raise ValueError("min_support must be positive")
    materialized = tuple(sorted(score_rows, key=lambda row: row.row_id))
    row_ids = [row.row_id for row in materialized]
    if not materialized or len(row_ids) != len(set(row_ids)) or set(row_ids) != set(labels):
        raise ValueError("calibration score/label identity mismatch or duplicate")
    if any(label not in ACTION_LABELS for label in labels.values()):
        raise ValueError("calibration labels contain an unknown action")
    for row in materialized:
        _validate_scores(row)
    (
        calibration_start_date,
        calibration_end_date,
        calibration_row_universe_sha256,
        score_family,
        calibrated_model_provenance,
        calibrated_model_provenance_sha256,
    ) = _validate_fit_binding(
        materialized,
        source_dataset_id=source_dataset_id,
        history_sha256=history_sha256,
        phase=phase,
        row_dates=row_dates,
        fitted_artifact_sha256=fitted_artifact_sha256,
        feature_schema_sha256=feature_schema_sha256,
    )

    minimum_precision = _POLICY_PRECISION[policy]
    thresholds: dict[str, float] = {}
    enabled: dict[str, bool] = {}
    metrics: dict[str, CalibrationMetric] = {}
    for action in ACTION_ORDER:
        winning = [
            row
            for row in materialized
            if max(ACTION_ORDER, key=lambda candidate: row.scores[candidate]) == action
        ]
        valid: list[tuple[tuple[int, int, float], float, CalibrationMetric]] = []
        for threshold in sorted({float(row.scores[action]) for row in winning}):
            selected = [row for row in winning if float(row.scores[action]) >= threshold]
            if len(selected) < min_support:
                continue
            correct = sum(labels[row.row_id] == action for row in selected)
            unsafe = (
                sum(labels[row.row_id] in {"RETURN", "REPAIR"} for row in selected)
                if action == "PROCESS"
                else 0
            )
            precision = correct / len(selected)
            if action == "PROCESS":
                passes = unsafe == 0
            else:
                passes = precision >= minimum_precision
            if passes:
                metric = CalibrationMetric(len(selected), correct, unsafe, precision)
                valid.append(((len(selected), correct, -threshold), threshold, metric))
        if valid:
            _, threshold, metric = max(valid, key=lambda item: item[0])
            thresholds[action] = threshold
            enabled[action] = True
            metrics[action] = metric
        else:
            thresholds[action] = 1.0
            enabled[action] = False
            metrics[action] = CalibrationMetric(0, 0, 0, None)

    calibration_rows_sha256 = _sha256_json(
        [
            {
                "rowId": row.row_id,
                "date": row_dates[row.row_id],
                "family": row.family,
                "scores": {action: float(row.scores[action]) for action in ACTION_ORDER},
                "actual": labels[row.row_id],
                "calibratedModelProvenanceSha256": calibrated_model_provenance_sha256,
            }
            for row in materialized
        ]
    )
    fitted = ThresholdFit(
        policy=policy,
        min_support=min_support,
        thresholds=MappingProxyType(thresholds),
        enabled=MappingProxyType(enabled),
        calibration_metrics=MappingProxyType(metrics),
        calibration_rows_sha256=calibration_rows_sha256,
        source_dataset_id=source_dataset_id,
        history_sha256=history_sha256,
        phase=phase,
        calibration_start_date=calibration_start_date,
        calibration_end_date=calibration_end_date,
        calibration_row_universe_sha256=calibration_row_universe_sha256,
        fitted_artifact_sha256=fitted_artifact_sha256,
        feature_schema_sha256=feature_schema_sha256,
        score_family=score_family,
        calibrated_model_provenance=_freeze_json(calibrated_model_provenance),
        calibrated_model_provenance_sha256=calibrated_model_provenance_sha256,
        class_order=tuple(ACTION_ORDER),
        algorithm_version=THRESHOLD_ALGORITHM_VERSION,
        metric_semantics=THRESHOLD_METRIC_SEMANTICS,
        selector_responsibility=THRESHOLD_SELECTOR_RESPONSIBILITY,
        sha256="",
    )
    return replace(fitted, sha256=_sha256_json(_threshold_fit_payload(fitted)))


def threshold_fit_receipt(fitted: ThresholdFit) -> Mapping[str, Any]:
    validate_threshold_fit(fitted)
    return MappingProxyType({**_threshold_fit_payload(fitted), "sha256": fitted.sha256})


def validate_threshold_fit(fitted: ThresholdFit) -> None:
    if not isinstance(fitted, ThresholdFit):
        raise TypeError("threshold fit receipt must be a ThresholdFit")
    if set(fitted.thresholds) != set(ACTION_ORDER) or set(fitted.enabled) != set(ACTION_ORDER):
        raise ValueError("threshold fit receipt has an invalid action contract")
    if any(type(value) is not bool for value in fitted.enabled.values()):
        raise ValueError("threshold fit receipt enabled state must contain booleans")
    if tuple(fitted.class_order) != tuple(ACTION_ORDER):
        raise ValueError("threshold fit receipt class order does not match the scoring contract")
    if fitted.algorithm_version != THRESHOLD_ALGORITHM_VERSION:
        raise ValueError("threshold fit receipt algorithm version is not supported")
    if (
        fitted.metric_semantics != THRESHOLD_METRIC_SEMANTICS
        or fitted.selector_responsibility != THRESHOLD_SELECTOR_RESPONSIBILITY
    ):
        raise ValueError("threshold fit receipt overstates its empirical metric contract")
    if fitted.phase != THRESHOLD_FIT_PHASE:
        raise ValueError("threshold fit receipt phase is not pre-selector calibration")
    start = _parse_iso_date(fitted.calibration_start_date, "threshold fit start date")
    end = _parse_iso_date(fitted.calibration_end_date, "threshold fit end date")
    if start > end or end >= _FIRST_SELECTOR_DATE:
        raise ValueError("threshold fit receipt chronology is outside pre-selector calibration")
    for name, value in (
        ("history SHA-256", fitted.history_sha256),
        ("row-universe SHA-256", fitted.calibration_row_universe_sha256),
        ("calibration-row SHA-256", fitted.calibration_rows_sha256),
        ("fitted artifact SHA-256", fitted.fitted_artifact_sha256),
        ("feature schema SHA-256", fitted.feature_schema_sha256),
        ("model provenance SHA-256", fitted.calibrated_model_provenance_sha256),
        ("threshold fit SHA-256", fitted.sha256),
    ):
        _require_sha256(value, name)
    if not fitted.source_dataset_id.strip() or not fitted.score_family.strip():
        raise ValueError("threshold fit receipt requires source and score family")
    provenance = _plain_json(fitted.calibrated_model_provenance)
    if _sha256_json(provenance) != fitted.calibrated_model_provenance_sha256:
        raise ValueError("threshold fit receipt calibrated model provenance digest mismatch")
    _validate_calibrated_provenance(
        provenance,
        threshold_start=start,
    )
    if _sha256_json(_threshold_fit_payload(fitted)) != fitted.sha256:
        raise ValueError("threshold fit receipt digest mismatch")


def apply_threshold_fit(
    score_rows: Iterable[ScoreRow],
    fitted: ThresholdFit,
) -> tuple[CandidatePrediction, ...]:
    validate_threshold_fit(fitted)
    return apply_threshold_policy(
        score_rows,
        thresholds=fitted.thresholds,
        enabled=fitted.enabled,
        policy=fitted.policy,
        threshold_fit_sha256=fitted.sha256,
    )


def apply_threshold_policy(
    score_rows: Iterable[ScoreRow],
    *,
    thresholds: Mapping[str, float],
    enabled: Mapping[str, bool],
    policy: str,
    threshold_fit_sha256: str | None = None,
) -> tuple[CandidatePrediction, ...]:
    if set(thresholds) != set(ACTION_ORDER) or set(enabled) != set(ACTION_ORDER):
        raise ValueError("thresholds and enabled state must contain exactly the three actions")
    if any(type(value) is not bool for value in enabled.values()):
        raise ValueError("threshold enabled state must contain booleans")
    if any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in thresholds.values()
    ):
        raise ValueError("thresholds must be finite probabilities")
    output: list[CandidatePrediction] = []
    for row in score_rows:
        _validate_scores(row)
        winner = max(ACTION_ORDER, key=lambda action: row.scores[action])
        score = float(row.scores[winner])
        threshold = float(thresholds[winner])
        automatic = enabled[winner] and score >= threshold
        evidence = {
            **dict(row.evidence),
            "abstained": not automatic,
            "winningClass": winner,
            "threshold": threshold,
            "classEnabled": enabled[winner],
            "thresholdPolicy": policy,
            "scoreVector": {
                action: float(row.scores[action]) for action in ACTION_ORDER
            },
        }
        if threshold_fit_sha256 is not None:
            evidence["thresholdFitSha256"] = threshold_fit_sha256
        output.append(
            CandidatePrediction(
                row_id=row.row_id,
                label=winner if automatic else "REVIEW",
                score=score,
                family=row.family,
                evidence=MappingProxyType(evidence),
            )
        )
    return tuple(output)


def _validate_fit_binding(
    rows: tuple[ScoreRow, ...],
    *,
    source_dataset_id: str,
    history_sha256: str,
    phase: str,
    row_dates: Mapping[str, str],
    fitted_artifact_sha256: str,
    feature_schema_sha256: str,
) -> tuple[str, str, str, str, Mapping[str, Any], str]:
    if not source_dataset_id.strip():
        raise ValueError("source_dataset_id must not be empty")
    _require_sha256(history_sha256, "history SHA-256")
    _require_sha256(fitted_artifact_sha256, "fitted artifact SHA-256")
    _require_sha256(feature_schema_sha256, "feature schema SHA-256")
    if phase != THRESHOLD_FIT_PHASE:
        raise ValueError(f"threshold fit phase must be {THRESHOLD_FIT_PHASE}")
    row_ids = {row.row_id for row in rows}
    if set(row_dates) != row_ids:
        raise ValueError("threshold fit row-date identity mismatch")
    parsed_dates = {
        row_id: _parse_iso_date(value, f"threshold fit row date for {row_id}")
        for row_id, value in row_dates.items()
    }
    start = min(parsed_dates.values())
    end = max(parsed_dates.values())
    if end >= _FIRST_SELECTOR_DATE:
        raise ValueError("threshold fit rows must remain in pre-selector calibration history")

    families = {row.family for row in rows}
    provenance_values = []
    for row in rows:
        provenance = row.evidence.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("threshold fitting requires calibrated model provenance on every score row")
        provenance_values.append(_plain_json(provenance))
    provenance_digests = {_sha256_json(value) for value in provenance_values}
    if len(families) != 1 or len(provenance_digests) != 1:
        raise ValueError("mixed calibrated model provenance is not allowed in one threshold fit")
    score_family = next(iter(families))
    if not score_family.strip():
        raise ValueError("threshold score family must not be empty")
    calibrated_model_provenance = provenance_values[0]
    _validate_calibrated_provenance(calibrated_model_provenance, threshold_start=start)
    provenance_sha256 = next(iter(provenance_digests))
    row_universe_sha256 = _sha256_json(
        [
            {"rowId": row.row_id, "date": parsed_dates[row.row_id].isoformat()}
            for row in rows
        ]
    )
    return (
        start.isoformat(),
        end.isoformat(),
        row_universe_sha256,
        score_family,
        calibrated_model_provenance,
        provenance_sha256,
    )


def _validate_calibrated_provenance(
    provenance: Mapping[str, Any],
    *,
    threshold_start: date,
) -> None:
    if tuple(provenance.get("classOrder") or ()) != tuple(ACTION_ORDER):
        raise ValueError("calibrated model provenance class order does not match score order")
    _require_sha256(str(provenance.get("modelSha256") or ""), "model SHA-256")
    _require_sha256(
        str(provenance.get("featureSchemaSha256") or ""),
        "model feature schema SHA-256",
    )
    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("state") not in _CALIBRATED_STATES:
        raise ValueError("threshold fitting requires calibrated model provenance")
    if not str(calibration.get("method") or "").strip():
        raise ValueError("calibrated model provenance requires a calibration method")
    _require_sha256(
        str(provenance.get("calibratorSha256") or ""),
        "calibrator SHA-256",
    )
    _require_sha256(
        str(calibration.get("rowSha256") or ""),
        "model calibration row SHA-256",
    )
    cutoff = _parse_iso_date(
        str(calibration.get("cutoff") or ""),
        "model calibration cutoff",
    )
    if cutoff >= threshold_start:
        raise ValueError("threshold rows must be strictly after the model calibration cutoff")


def _threshold_fit_payload(fitted: ThresholdFit) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "kind": "threshold_fit",
        "sourceDatasetId": fitted.source_dataset_id,
        "historySha256": fitted.history_sha256,
        "phase": fitted.phase,
        "dateInterval": {
            "start": fitted.calibration_start_date,
            "end": fitted.calibration_end_date,
        },
        "rowUniverseSha256": fitted.calibration_row_universe_sha256,
        "calibrationRowsSha256": fitted.calibration_rows_sha256,
        "scoreFamily": fitted.score_family,
        "fittedArtifactSha256": fitted.fitted_artifact_sha256,
        "featureSchemaSha256": fitted.feature_schema_sha256,
        "calibratedModelProvenance": _plain_json(fitted.calibrated_model_provenance),
        "calibratedModelProvenanceSha256": fitted.calibrated_model_provenance_sha256,
        "classOrder": list(fitted.class_order),
        "algorithmVersion": fitted.algorithm_version,
        "selectionSemantics": "top_one_argmax_then_class_threshold",
        "metricSemantics": fitted.metric_semantics,
        "statisticalGuarantee": "none",
        "selectorResponsibility": fitted.selector_responsibility,
        "policy": fitted.policy,
        "minSupport": fitted.min_support,
        "thresholds": dict(fitted.thresholds),
        "enabled": dict(fitted.enabled),
        "metrics": {
            action: {
                "automatic": fitted.calibration_metrics[action].automatic,
                "correct": fitted.calibration_metrics[action].correct,
                "unsafe": fitted.calibration_metrics[action].unsafe,
                "precision": fitted.calibration_metrics[action].precision,
            }
            for action in ACTION_ORDER
        },
    }


def _validate_scores(row: ScoreRow) -> None:
    if set(row.scores) != set(ACTION_ORDER):
        raise ValueError(f"invalid score action set for {row.row_id}")
    values = [float(row.scores[action]) for action in ACTION_ORDER]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError(f"invalid probability for {row.row_id}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"probabilities do not sum to one for {row.row_id}")


def _sha256_json(value: object) -> str:
    data = (
        json.dumps(
            _plain_json(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _parse_iso_date(value: str, name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an ISO date") from error
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be an ISO date")
    return parsed


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"unsupported calibrated model provenance value: {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value

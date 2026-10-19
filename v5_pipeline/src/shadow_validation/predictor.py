from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiment_lib.point_in_time_domain_features import (
    PointInTimeDomainFeatureTransformer,
)
from experiment_lib.point_in_time_temporal import (
    build_point_in_time_temporal_matrix,
)
from experiment_lib.review_recovery_veto import semantic_review_veto

from src.baseline_risk_veto import (
    DangerScoreRow,
    apply_danger_veto,
)
from src.data_contract import FeatureRow
from src.hierarchical_veto import matched_semantic_groups
from src.prior_supported_repair_guard import matched_repair_signal_ids
from src.runtime_contract import validate_runtime_contract
from src.temporal_features import TemporalFeatureBuilder

from .feature_bridge import (
    feature_rows_from_input,
    load_frozen_history,
    load_label_availability_index,
)
from .hierarchy_runtime import HierarchyPredictor
from .input_contract import InputBatch
from .integrity import verify_manifest
from .loop93 import Loop93Prediction, Loop93WorkerClient, run_loop93_baseline
from .pit_runtime import PitDangerModel
from .policy_loaders import (
    load_process_recovery,
    load_repair_guard,
    load_semantic_veto,
)
from .runtime_state import (
    current_cache_fingerprint,
    load_or_build_frozen_history_state,
    runtime_cache_tag,
)
from .v4_baseline import V4GreenPredictor


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"


def _log_predictor_timing(
    stage: str,
    started: float,
    **counts: int,
) -> None:
    payload: dict[str, Any] = {
        "component": "predictor",
        "stage": stage,
        "seconds": round(time.perf_counter() - started, 6),
    }
    payload.update(counts)
    print(
        "[TIMING] "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


@dataclass(frozen=True, slots=True)
class FinalLayerResult:
    danger_veto_predictions: Mapping[str, str]
    repair_guard_predictions: Mapping[str, str]
    semantic_veto_predictions: Mapping[str, str]
    final_predictions: Mapping[str, str]
    danger_demoted_ids: tuple[str, ...]
    repair_guard_demoted_ids: tuple[str, ...]
    semantic_demoted_ids: tuple[str, ...]
    recovered_ids: tuple[str, ...]
    danger_override_ids: tuple[str, ...]
    protected_row_ids: tuple[str, ...]
    repair_reasons: Mapping[str, tuple[Mapping[str, Any], ...]]
    recovery_reasons: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True, slots=True)
class PredictionRow:
    row_id: str
    loop93_prediction: str
    hierarchy_prediction: str
    v4_green_prediction: str
    danger_veto_prediction: str
    repair_guard_prediction: str
    semantic_veto_prediction: str
    final_prediction: str
    shipment_probability: float | None
    repair_probability: float | None
    protected: bool
    applied_stage: str
    reason: str


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    generation_id: str
    rows: tuple[PredictionRow, ...]
    by_id: Mapping[str, PredictionRow]


def average_danger_scores(
    scores_by_seed: Sequence[Sequence[DangerScoreRow]],
) -> tuple[DangerScoreRow, ...]:
    if not scores_by_seed:
        raise ValueError("danger ensemble requires at least one seed")
    indexed = [{row.row_id: row for row in values} for values in scores_by_seed]
    row_ids = set(indexed[0])
    if any(len(mapping) != len(values) for mapping, values in zip(indexed, scores_by_seed)):
        raise ValueError("danger ensemble contains duplicate row IDs")
    if any(set(mapping) != row_ids for mapping in indexed[1:]):
        raise ValueError("danger ensemble row universes differ")
    return tuple(
        DangerScoreRow(
            row_id=row_id,
            month=indexed[0][row_id].month,
            shipment_probability=float(
                np.mean([mapping[row_id].shipment_probability for mapping in indexed])
            ),
            repair_probability=float(
                np.mean([mapping[row_id].repair_probability for mapping in indexed])
            ),
        )
        for row_id in sorted(row_ids)
    )


def index_danger_scores(
    scores: Iterable[DangerScoreRow],
) -> Mapping[str, DangerScoreRow]:
    indexed: dict[str, DangerScoreRow] = {}
    for score in scores:
        if score.row_id in indexed:
            raise ValueError(f"duplicate danger score row ID: {score.row_id}")
        indexed[score.row_id] = score
    return MappingProxyType(indexed)


def pit_semantic_protected_ids(
    rows: Sequence[Any],
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    result = semantic_review_veto(
        rows,
        repair_signal_ids=tuple(str(value) for value in config["repairSignalIds"]),
        repair_matcher=matched_repair_signal_ids,
        semantic_group_ids=tuple(
            str(value) for value in config.get("frozenSemanticGroupIds", ())
        ),
        semantic_group_matcher=matched_semantic_groups,
        include_legacy_v4_recovery_risk=bool(config.get("legacyV4RecoveryRisk")),
    )
    return tuple(sorted(result.vetoed_row_ids))


def score_danger_models_from_frozen_state(
    models: Sequence[Any],
    rows: Sequence[FeatureRow],
    *,
    temporal: Any,
) -> tuple[tuple[DangerScoreRow, ...], ...]:
    """Score with each model's sealed domain-history snapshot.

    The fitted danger models already contain the exact July-December 2025
    domain state used by the selected run. Supplying a broader history here
    silently changes boundary probabilities even though the model bytes and
    thresholds are unchanged. Temporal features are built separately from the
    full frozen pre-2026 history and are therefore still passed explicitly.
    """
    return tuple(
        model.score_rows(rows, temporal=temporal)
        for model in models
    )


def apply_final_policy_layers(
    rows: Sequence[Any],
    green_predictions: Mapping[str, str],
    danger_scores: Sequence[DangerScoreRow],
    *,
    repair_guard: Any,
    recovery: Any,
    shipment_threshold: float,
    repair_threshold: float,
) -> FinalLayerResult:
    danger = apply_danger_veto(
        green_predictions,
        danger_scores,
        shipment_threshold=shipment_threshold,
        repair_threshold=repair_threshold,
    )
    danger_demoted = tuple(
        sorted(
            row_id
            for row_id, value in green_predictions.items()
            if value == "PROCESS" and danger[row_id] != "PROCESS"
        )
    )
    guarded = repair_guard.apply(rows, danger)
    protected = tuple(sorted(set(danger_demoted) | set(guarded.demoted_ids)))
    recovered = recovery.apply(
        rows,
        guarded.predictions,
        protected_row_ids=protected,
    )
    return FinalLayerResult(
        danger_veto_predictions=MappingProxyType(dict(sorted(danger.items()))),
        repair_guard_predictions=MappingProxyType(
            dict(sorted(guarded.predictions.items()))
        ),
        semantic_veto_predictions=MappingProxyType(
            dict(sorted(guarded.predictions.items()))
        ),
        final_predictions=MappingProxyType(
            dict(sorted(recovered.predictions.items()))
        ),
        danger_demoted_ids=danger_demoted,
        repair_guard_demoted_ids=tuple(sorted(guarded.demoted_ids)),
        semantic_demoted_ids=(),
        recovered_ids=tuple(sorted(recovered.recovered_ids)),
        danger_override_ids=(),
        protected_row_ids=protected,
        repair_reasons=guarded.reasons_by_row,
        recovery_reasons=recovered.reasons_by_row,
    )


def apply_pit_final_policy_layers(
    rows: Sequence[Any],
    baseline_predictions: Mapping[str, str],
    danger_scores: Sequence[DangerScoreRow],
    *,
    repair_guard: Any,
    semantic_protected_row_ids: Sequence[str],
    recovery: Any,
    shipment_threshold: float,
    repair_threshold: float,
    allow_danger_override: bool,
) -> FinalLayerResult:
    """Apply the selected PIT policy without reopening protected risk rows.

    The selected stable policy permits exact prior-pure signatures to reopen a
    row demoted by probability alone. Repair guard matches are actively
    demoted. Semantic matches are recovery guards only: they keep an existing
    review row closed but never demote an already-approved process row. Existing
    shipment and repair actions are carried through unchanged by every layer.
    """
    row_ids = {str(row.row_id) for row in rows}
    if set(baseline_predictions) != row_ids:
        raise ValueError("PIT baseline prediction universe mismatch")
    semantic_ids = tuple(sorted(str(value) for value in semantic_protected_row_ids))
    if len(set(semantic_ids)) != len(semantic_ids) or not set(semantic_ids) <= row_ids:
        raise ValueError("PIT semantic protected row universe is invalid")

    danger = apply_danger_veto(
        baseline_predictions,
        danger_scores,
        shipment_threshold=shipment_threshold,
        repair_threshold=repair_threshold,
    )
    danger_demoted = tuple(
        sorted(
            row_id
            for row_id, value in baseline_predictions.items()
            if value == "PROCESS" and danger[row_id] != "PROCESS"
        )
    )
    # Match the selected PIT replay exactly: probe repair risk against the
    # pre-veto process candidates, then carry those protected IDs into the
    # already danger-vetoed prediction map. Probing ``danger`` directly would
    # hide rows matched by both layers and allow recovery to reopen them.
    guard_probe = repair_guard.apply(rows, baseline_predictions)
    guarded_predictions = dict(danger)
    for row_id in guard_probe.demoted_ids:
        guarded_predictions[row_id] = ""
    # Match ``apply_point_in_time_stage_pipeline`` exactly. Semantic IDs are
    # protection metadata for the recovery layer; they are not an active veto
    # over the baseline action. The earlier integration demoted every matched
    # process row here and incorrectly sent tens of thousands of rows to review.
    semantic = dict(guarded_predictions)

    protected_ids = set(guard_probe.demoted_ids) | set(semantic_ids)
    if not allow_danger_override:
        protected_ids.update(danger_demoted)
    protected = tuple(sorted(protected_ids))
    recovered = recovery.apply(
        rows,
        MappingProxyType(dict(sorted(semantic.items()))),
        protected_row_ids=protected,
    )
    recovered_ids = tuple(sorted(recovered.recovered_ids))
    danger_override_ids = tuple(sorted(set(recovered_ids) & set(danger_demoted)))
    return FinalLayerResult(
        danger_veto_predictions=MappingProxyType(dict(sorted(danger.items()))),
        repair_guard_predictions=MappingProxyType(
            dict(sorted(guarded_predictions.items()))
        ),
        semantic_veto_predictions=MappingProxyType(dict(sorted(semantic.items()))),
        final_predictions=MappingProxyType(
            dict(sorted(recovered.predictions.items()))
        ),
        danger_demoted_ids=danger_demoted,
        repair_guard_demoted_ids=tuple(sorted(guard_probe.demoted_ids)),
        semantic_demoted_ids=(),
        recovered_ids=recovered_ids,
        danger_override_ids=danger_override_ids,
        protected_row_ids=protected,
        repair_reasons=guard_probe.reasons_by_row,
        recovery_reasons=recovered.reasons_by_row,
    )


class FrozenPredictor:
    def __init__(self, package_root: str | Path = PACKAGE_ROOT) -> None:
        self._closed = False
        self.package_root = Path(package_root).resolve()
        self.frozen_root = self.package_root / "frozen"
        verified = verify_manifest(
            self.package_root,
            self.frozen_root / "manifest.json",
            expected_generation="mes-pit-stable-20260716-g1",
        )
        self.generation_id = str(verified["generationId"])
        generation = json.loads(
            (self.frozen_root / "generation.json").read_text(encoding="utf-8")
        )
        self.source_dataset_id = str(generation["sourceDatasetId"])
        self.danger_thresholds = MappingProxyType(
            {
                "shipment": float(generation["dangerThresholds"]["shipment"]),
                "repair": float(generation["dangerThresholds"]["repair"]),
            }
        )
        self.allow_danger_override = bool(generation["recovery"]["allowDangerOverride"])
        self.history_path = self.frozen_root / "history/history_rows.jsonl.gz"
        self.loop93_worker = Loop93WorkerClient(
            engine_path=self.frozen_root / "loop93/rule_miner.browser.js",
            policy_path=self.frozen_root / "loop93/frozen_policy.json",
            history_path=self.history_path,
            script_path=self.package_root / "scripts/run_frozen_baseline.mjs",
        )
        cache_fingerprint = current_cache_fingerprint(
            generation_id=self.generation_id,
            manifest_sha256=str(verified["manifestSha256"]),
        )
        self.runtime_cache_fingerprint = cache_fingerprint
        history_state, self.runtime_cache_status = load_or_build_frozen_history_state(
            history_path=self.history_path,
            source_dataset_id=self.source_dataset_id,
            cutoff_exclusive=str(generation["labelAvailabilityCutoffExclusive"]),
            cache_path=(
                self.package_root
                / ".runtime_cache"
                / f"predictor-history-v1-{runtime_cache_tag()}.bin"
            ),
            fingerprint=cache_fingerprint,
        )
        self.history_rows = history_state.history_rows
        self.history_labels = history_state.history_labels
        self.label_index = history_state.label_index
        self.pit_domain = history_state.pit_domain
        self.hierarchy = HierarchyPredictor(self.frozen_root)
        self.v4_green = V4GreenPredictor(self.frozen_root)
        self.repair_guard = load_repair_guard(
            self.frozen_root / "policies/pit_repair_guard.json"
        )
        self.recovery = load_process_recovery(
            self.frozen_root / "policies/pit_process_recovery.json"
        )
        self.semantic_veto = load_semantic_veto(
            self.frozen_root / "policies/pit_semantic_veto.json"
        )
        self.danger_models = tuple(
            _load_pit_danger_model(
                self.frozen_root / f"models/pit_danger/seed_{seed}.pkl"
            )
            for seed in (17, 31, 47)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.loop93_worker.close()

    def predict(self, batch: InputBatch) -> PredictionBundle:
        started = time.perf_counter()
        feature_rows = feature_rows_from_input(batch.rows)
        _log_predictor_timing("feature_rows", started, rows=len(feature_rows))

        started = time.perf_counter()
        loop93 = run_loop93_baseline(batch.rows, worker=self.loop93_worker)
        loop93_map = {row_id: item.prediction for row_id, item in loop93.items()}
        _log_predictor_timing("loop93", started, rows=len(loop93))

        started = time.perf_counter()
        hierarchy = self.hierarchy.predict(feature_rows, loop93_map)
        _log_predictor_timing("hierarchy", started, rows=len(feature_rows))

        started = time.perf_counter()
        green = self.v4_green.predict(
            feature_rows,
            hierarchy.predictions,
            prior_history_rows=self.history_rows,
            prior_history_labels=self.history_labels,
        )
        _log_predictor_timing("v4_green", started, rows=len(feature_rows))

        started = time.perf_counter()
        process_rows = tuple(
            row
            for row in feature_rows
            if green.predictions[row.row_id] == "PROCESS"
        )
        _log_predictor_timing(
            "process_rows",
            started,
            rows=len(feature_rows),
            process_rows=len(process_rows),
        )

        started = time.perf_counter()
        temporal = build_point_in_time_temporal_matrix(
            process_rows,
            all_observed_rows=(*self.history_rows, *feature_rows),
            label_index=self.label_index,
            label_cutoff_exclusive="2026-01-01",
        )
        _log_predictor_timing("temporal", started, process_rows=len(process_rows))

        started = time.perf_counter()
        domain = self.pit_domain.transform_frozen(process_rows)
        _log_predictor_timing("domain", started, process_rows=len(process_rows))

        seed_scores_list: list[tuple[DangerScoreRow, ...]] = []
        for seed, model in zip((17, 31, 47), self.danger_models, strict=True):
            started = time.perf_counter()
            scores = model.score_rows(
                process_rows,
                temporal=temporal,
                domain=domain,
                domain_schema_sha256=self.pit_domain.schema_sha256,
            )
            seed_scores_list.append(scores)
            _log_predictor_timing(
                f"danger_seed_{seed}",
                started,
                process_rows=len(process_rows),
            )
        seed_scores = tuple(seed_scores_list)

        started = time.perf_counter()
        danger_scores = average_danger_scores(seed_scores)
        danger_scores_by_id = index_danger_scores(danger_scores)
        _log_predictor_timing(
            "danger_aggregate",
            started,
            process_rows=len(process_rows),
        )

        started = time.perf_counter()
        semantic_ids = pit_semantic_protected_ids(feature_rows, self.semantic_veto)
        _log_predictor_timing("semantic", started, rows=len(feature_rows))

        started = time.perf_counter()
        final = apply_pit_final_policy_layers(
            feature_rows,
            green.predictions,
            danger_scores,
            repair_guard=self.repair_guard,
            semantic_protected_row_ids=semantic_ids,
            recovery=self.recovery,
            shipment_threshold=self.danger_thresholds["shipment"],
            repair_threshold=self.danger_thresholds["repair"],
            allow_danger_override=self.allow_danger_override,
        )
        _log_predictor_timing("final_policy", started, rows=len(feature_rows))

        started = time.perf_counter()
        rows = tuple(
            _prediction_row(
                row,
                loop93[row.row_id],
                hierarchy.predictions[row.row_id],
                green.predictions[row.row_id],
                final,
                danger_scores_by_id,
            )
            for row in feature_rows
        )
        bundle = PredictionBundle(
            generation_id=self.generation_id,
            rows=rows,
            by_id=MappingProxyType({row.row_id: row for row in rows}),
        )
        _log_predictor_timing("prediction_bundle", started, rows=len(rows))
        return bundle


def _prediction_row(
    row: FeatureRow,
    loop93: Loop93Prediction,
    hierarchy: str,
    green: str,
    final: FinalLayerResult,
    danger_scores: Mapping[str, DangerScoreRow],
) -> PredictionRow:
    row_id = row.row_id
    score = danger_scores.get(row_id)
    if row_id in final.recovered_ids:
        stage = "pit_prior_pure_process_recovery"
        reason = (
            "PIT 위험 보류 후 과거 무반례 공투 시그니처 재승인"
            if row_id in final.danger_override_ids
            else "과거 무반례 공투 시그니처 회수"
        )
    elif row_id in final.semantic_demoted_ids:
        stage, reason = "pit_semantic_review_veto", "수리·출고 의미 위험 신호로 검토 전환"
    elif row_id in final.repair_guard_demoted_ids:
        stage, reason = "pit_prior_supported_repair_guard", "과거 수리 위험 신호로 검토 전환"
    elif row_id in final.danger_demoted_ids:
        stage, reason = "pit_three_seed_danger_veto", "PIT 출고/수리 위험 확률로 검토 전환"
    elif final.final_predictions[row_id] != green:
        stage, reason = "final_policy", "최종 동결 정책 적용"
    elif green != hierarchy:
        stage, reason = "v4_green", "계층 리뷰를 V4 Green으로 판정"
    elif hierarchy != loop93.prediction:
        stage, reason = "hierarchy", "회사/라인 계층 정책 적용"
    else:
        stage, reason = "loop93", loop93.reason
    return PredictionRow(
        row_id=row_id,
        loop93_prediction=loop93.prediction,
        hierarchy_prediction=hierarchy,
        v4_green_prediction=green,
        danger_veto_prediction=final.danger_veto_predictions[row_id],
        repair_guard_prediction=final.repair_guard_predictions[row_id],
        semantic_veto_prediction=final.semantic_veto_predictions[row_id],
        final_prediction=final.final_predictions[row_id],
        shipment_probability=score.shipment_probability if score else None,
        repair_probability=score.repair_probability if score else None,
        protected=row_id in final.protected_row_ids,
        applied_stage=stage,
        reason=reason,
    )


def _load_danger_model(path: Path) -> Any:
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(contract.get("modelSha256") or ""):
        raise ValueError(f"danger model SHA mismatch: {path}")
    validate_runtime_contract(contract.get("runtime"))
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if model.fit_receipt != contract.get("fitReceipt"):
        raise ValueError(f"danger model fit receipt mismatch: {path}")
    return model


def _load_pit_danger_model(path: Path) -> PitDangerModel:
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != str(contract.get("modelSha256") or ""):
        raise ValueError(f"PIT danger model SHA mismatch: {path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    model = PitDangerModel.from_state(state)
    if dict(model.fit_receipt) != contract.get("fitReceipt"):
        raise ValueError(f"PIT danger model fit receipt mismatch: {path}")
    return model

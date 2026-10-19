from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.binary_candidates import load_binary_process_artifact
from src.candidates import ScoreRow, load_candidate_artifact
from src.company_scope import prediction_or_review, select_company_rows
from src.data_contract import ACTION_LABELS, FeatureRow
from src.experiment import prepare_experiment
from src.hierarchical_evidence import PredictionDecision, build_scope_evidence
from src.hierarchy import line_collision_report, normalize_line_name
from src.thresholds import apply_threshold_policy


DEFAULT_SOURCE_ROOT = Path(
    "artifacts/mes_rule_miner_clean_optimization_20260710/outputs/runs"
)
DEFAULT_RUN_FILES = (
    "selector_frontier_linear.json",
    "selector_frontier_lightgbm.json",
    "binary_process_frontier_proj.json",
)
DEFAULT_LOOP93_REPLAY_SOURCE = Path(
    "artifacts/mes_rule_miner_clean_optimization_20260710/"
    "outputs/materialized/selector.jsonl"
)
LOOP93_CANDIDATE_ID = "loop93_current_policy_crossfit"


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    candidate_id: str
    family: str
    base_model_id: str
    fitted_manifest_path: Path
    fitted_manifest_sha256: str
    thresholds: Mapping[str, float]
    threshold_enabled: Mapping[str, bool]
    threshold_profile: str
    train_end: str
    selector_months: tuple[str, ...]
    source_dataset_id: str
    source_run: Path
    scope_company: str | None = None


def parse_candidate_payload(
    payload: Mapping[str, Any], source_run: str | Path
) -> tuple[CandidateProfile, ...]:
    source_path = Path(source_run).resolve()
    chronology = payload.get("chronology")
    candidates = payload.get("candidates")
    if not isinstance(chronology, Mapping) or not isinstance(candidates, Mapping):
        raise ValueError(f"candidate run is missing chronology/candidates: {source_path}")
    train_end = str(chronology.get("trainEndInclusive", ""))
    selector_months = tuple(str(item) for item in chronology.get("selectorMonths", ()))
    if train_end != "2025-07-31" or selector_months != ("2025-10", "2025-11"):
        raise ValueError(f"candidate run chronology is not frozen: {source_path}")
    source_dataset_id = str(payload.get("sourceDatasetId", ""))
    if not source_dataset_id:
        raise ValueError(f"candidate run has no source dataset id: {source_path}")

    family = (
        "binary_process"
        if payload.get("kind") == "binary_process_selector_frontier"
        else "multiclass"
    )
    profiles: list[CandidateProfile] = []
    for candidate_id, candidate in sorted(candidates.items()):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"invalid candidate payload: {candidate_id}")
        artifact = candidate.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError(f"candidate has no fitted artifact: {candidate_id}")
        manifest_path = Path(str(artifact.get("fittedManifestPath", ""))).resolve()
        manifest_sha256 = str(artifact.get("fittedManifestSha256", ""))
        if not str(manifest_path) or len(manifest_sha256) != 64:
            raise ValueError(f"candidate artifact binding is incomplete: {candidate_id}")

        if family == "binary_process":
            thresholds = {"PROCESS": float(candidate["threshold"])}
            threshold_enabled = {"PROCESS": bool(candidate["thresholdEnabled"])}
        else:
            thresholds_raw = candidate.get("thresholds")
            enabled_raw = candidate.get("thresholdEnabled")
            if not isinstance(thresholds_raw, Mapping) or not isinstance(
                enabled_raw, Mapping
            ):
                raise ValueError(f"candidate threshold policy is incomplete: {candidate_id}")
            if set(thresholds_raw) != ACTION_LABELS or set(enabled_raw) != ACTION_LABELS:
                raise ValueError(f"candidate threshold actions do not match: {candidate_id}")
            thresholds = {key: float(value) for key, value in thresholds_raw.items()}
            threshold_enabled = {key: bool(value) for key, value in enabled_raw.items()}

        profiles.append(
            CandidateProfile(
                candidate_id=str(candidate_id),
                family=family,
                base_model_id=str(candidate.get("baseModelId", "")),
                fitted_manifest_path=manifest_path,
                fitted_manifest_sha256=manifest_sha256,
                thresholds=thresholds,
                threshold_enabled=threshold_enabled,
                threshold_profile=str(candidate.get("thresholdProfile", "")),
                train_end=train_end,
                selector_months=selector_months,
                source_dataset_id=source_dataset_id,
                source_run=source_path,
                scope_company=(
                    str(candidate.get("scopeCompany")).strip()
                    if candidate.get("scopeCompany") is not None
                    else None
                ),
            )
        )
    return tuple(profiles)


def load_candidate_profiles(run_files: Iterable[str | Path]) -> tuple[CandidateProfile, ...]:
    profiles: list[CandidateProfile] = []
    source_dataset_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for run_file in run_files:
        path = Path(run_file).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        parsed = parse_candidate_payload(payload, path)
        for profile in parsed:
            if profile.candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate id: {profile.candidate_id}")
            candidate_ids.add(profile.candidate_id)
            source_dataset_ids.add(profile.source_dataset_id)
            profiles.append(profile)
    if len(source_dataset_ids) != 1:
        raise ValueError("candidate runs do not share one source dataset")
    return tuple(profiles)


def apply_candidate_profile(
    score_rows: Iterable[ScoreRow], profile: CandidateProfile
) -> dict[str, str]:
    materialized = tuple(score_rows)
    if profile.family == "binary_process":
        threshold = float(profile.thresholds["PROCESS"])
        enabled = bool(profile.threshold_enabled["PROCESS"])
        return {
            row.row_id: (
                "PROCESS"
                if enabled and float(row.scores["PROCESS"]) >= threshold
                else "REVIEW"
            )
            for row in materialized
        }
    predictions = apply_threshold_policy(
        materialized,
        thresholds=profile.thresholds,
        enabled=profile.threshold_enabled,
        policy=profile.threshold_profile,
    )
    return {item.row_id: item.label for item in predictions}


def score_frozen_model(
    profile: CandidateProfile, rows: Sequence[FeatureRow]
) -> tuple[ScoreRow, ...]:
    actual_sha256 = _sha256_file(profile.fitted_manifest_path)
    if actual_sha256 != profile.fitted_manifest_sha256:
        raise ValueError(f"fitted manifest digest mismatch: {profile.candidate_id}")
    if profile.family == "binary_process":
        model = load_binary_process_artifact(profile.fitted_manifest_path)
        return model.score_selector(rows)
    model = load_candidate_artifact(profile.fitted_manifest_path)
    return model.predict_scores(rows)


def materialize_profile_group(
    profiles: Sequence[CandidateProfile],
    rows: Sequence[Any],
    labels: Mapping[str, str],
    output_dir: str | Path,
    *,
    score_once: Callable[
        [CandidateProfile, Sequence[Any]], tuple[ScoreRow, ...]
    ] = score_frozen_model,
) -> dict[str, dict[str, Any]]:
    if not profiles:
        return {}
    manifests = {profile.fitted_manifest_path for profile in profiles}
    families = {profile.family for profile in profiles}
    scope_companies = {profile.scope_company for profile in profiles}
    if len(manifests) != 1 or len(families) != 1 or len(scope_companies) != 1:
        raise ValueError("profile group must share one fitted model and company scope")
    scope_company = profiles[0].scope_company
    scoring_rows = select_company_rows(rows, scope_company)
    score_rows = score_once(profiles[0], scoring_rows)
    score_ids = {item.row_id for item in score_rows}
    row_ids = {item.row_id for item in rows}
    scoring_ids = {item.row_id for item in scoring_rows}
    if score_ids != scoring_ids or set(labels) != row_ids:
        raise ValueError("score, label, and selector row identities do not match")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        predictions = apply_candidate_profile(score_rows, profile)
        decisions = tuple(
            PredictionDecision(
                candidate_id=profile.candidate_id,
                row_id=row.row_id,
                episode_id=row.episode_id,
                company=row.customer_key,
                raw_customer=row.raw_customer,
                line=normalize_line_name(row.features.get("line", "")),
                month=row.month,
                actual=labels[row.row_id],
                predicted=prediction_or_review(predictions, row.row_id),
                train_end=profile.train_end,
            )
            for row in rows
        )
        decision_path = destination / f"{_safe_name(profile.candidate_id)}.jsonl.gz"
        _write_gzip_jsonl_atomic(decision_path, (item.as_dict() for item in decisions))
        evidence = build_scope_evidence(decisions)
        global_rows = [item for item in evidence if item.scope_id == "global"]
        auto_rows = sum(item.auto_rows for item in global_rows)
        correct_rows = sum(item.correct_auto_rows for item in global_rows)
        result[profile.candidate_id] = {
            "candidateId": profile.candidate_id,
            "baseModelId": profile.base_model_id,
            "family": profile.family,
            "thresholdProfile": profile.threshold_profile,
            "scopeCompany": profile.scope_company,
            "decisionPath": str(decision_path),
            "decisionSha256": _sha256_file(decision_path),
            "rows": len(decisions),
            "autoRows": auto_rows,
            "correctAutoRows": correct_rows,
            "scopeEvidence": [asdict(item) for item in evidence],
        }
    return result


def materialize_loop93_replay(
    rows: Sequence[Any],
    labels: Mapping[str, str],
    replay_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(replay_dir).resolve()
    source_files = (
        (source,)
        if source.is_file()
        else tuple(sorted(source.glob("*_predicted_rows.jsonl")))
    )
    if not source_files:
        raise ValueError(f"Loop93 replay files not found: {source}")
    rows_by_source = {int(row.source_row): row for row in rows}
    if len(rows_by_source) != len(rows):
        raise ValueError("selector source row identity is not unique")
    replay_records: dict[int, Mapping[str, Any]] = {}
    receipts: list[dict[str, str]] = []
    for path in source_files:
        receipts.append({"path": str(path), "sha256": _sha256_file(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("month") not in {"2025-10", "2025-11"}:
                    continue
                source_row = int(record["source_row"])
                if source_row not in rows_by_source:
                    raise ValueError(
                        f"Loop93 replay contains an unknown selector row: {source_row}"
                    )
                if source_row in replay_records:
                    raise ValueError(f"duplicate Loop93 selector row: {source_row}")
                replay_records[source_row] = record
    if set(replay_records) != set(rows_by_source):
        missing = sorted(set(rows_by_source) - set(replay_records))
        raise ValueError(
            f"Loop93 replay selector coverage mismatch: missing {len(missing)} rows"
        )

    decisions: list[PredictionDecision] = []
    for row in rows:
        record = replay_records[int(row.source_row)]
        replay_actual = str(record.get("actual", "")).strip()
        actual = labels[row.row_id]
        if replay_actual and replay_actual != actual:
            raise ValueError(f"Loop93 replay label mismatch: {row.row_id}")
        if str(record.get("month", "")) != row.month:
            raise ValueError(f"Loop93 replay month mismatch: {row.row_id}")
        predicted_raw = str(record.get("predicted", "")).strip()
        predicted = predicted_raw if predicted_raw in ACTION_LABELS else "REVIEW"
        train_end = str(record.get("predictionTrainEndDate", ""))
        decisions.append(
            PredictionDecision(
                candidate_id=LOOP93_CANDIDATE_ID,
                row_id=row.row_id,
                episode_id=row.episode_id,
                company=row.customer_key,
                raw_customer=row.raw_customer,
                line=normalize_line_name(row.features.get("line", "")),
                month=row.month,
                actual=actual,
                predicted=predicted,
                train_end=train_end,
            )
        )
    evidence = build_scope_evidence(decisions)
    global_rows = [item for item in evidence if item.scope_id == "global"]
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    decision_path = destination / f"{LOOP93_CANDIDATE_ID}.jsonl.gz"
    _write_gzip_jsonl_atomic(decision_path, (item.as_dict() for item in decisions))
    return {
        "candidateId": LOOP93_CANDIDATE_ID,
        "baseModelId": LOOP93_CANDIDATE_ID,
        "family": "rule_replay",
        "thresholdProfile": "historical_oof_policy",
        "decisionPath": str(decision_path),
        "decisionSha256": _sha256_file(decision_path),
        "rows": len(decisions),
        "autoRows": sum(item.auto_rows for item in global_rows),
        "correctAutoRows": sum(item.correct_auto_rows for item in global_rows),
        "scopeEvidence": [asdict(item) for item in evidence],
        "sourceReplay": str(source),
        "sourceFiles": receipts,
    }


def materialize_hierarchical_evidence(
    *,
    config_path: str | Path,
    run_files: Sequence[str | Path],
    output_dir: str | Path,
    loop93_replay_dir: str | Path = DEFAULT_LOOP93_REPLAY_SOURCE,
) -> Path:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    experiment = prepare_experiment(
        config_path, output_root / "chronology_probe"
    )
    rows = experiment.bundle.selector
    labels = experiment.bundle.development_labels("selector")
    profiles = load_candidate_profiles(run_files)
    if profiles[0].source_dataset_id != experiment.bundle.metadata.source_dataset_id:
        raise ValueError("candidate and prepared dataset identity mismatch")

    groups: dict[tuple[Path, str | None], list[CandidateProfile]] = {}
    for profile in profiles:
        groups.setdefault(
            (profile.fitted_manifest_path, profile.scope_company), []
        ).append(profile)
    candidate_results: dict[str, dict[str, Any]] = {}
    decision_dir = output_root / "selector_decisions"
    for index, grouped_profiles in enumerate(groups.values(), start=1):
        print(
            f"[{index}/{len(groups)}] scoring {grouped_profiles[0].base_model_id} "
            f"for {len(grouped_profiles)} profiles",
            flush=True,
        )
        candidate_results.update(
            materialize_profile_group(grouped_profiles, rows, labels, decision_dir)
        )

    print("[baseline] materializing Loop93 OOF replay", flush=True)
    loop93_result = materialize_loop93_replay(
        rows,
        labels,
        loop93_replay_dir,
        decision_dir,
    )

    scope_evidence: list[dict[str, Any]] = []
    candidate_registry: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        result = candidate_results[profile.candidate_id]
        scope_evidence.extend(result.pop("scopeEvidence"))
        candidate_registry[profile.candidate_id] = {
            **result,
            "sourceRun": str(profile.source_run),
            "sourceRunSha256": _sha256_file(profile.source_run),
            "fittedManifestPath": str(profile.fitted_manifest_path),
            "fittedManifestSha256": profile.fitted_manifest_sha256,
            "thresholds": dict(profile.thresholds),
            "thresholdEnabled": dict(profile.threshold_enabled),
            "scopeCompany": profile.scope_company,
        }
    scope_evidence.extend(loop93_result.pop("scopeEvidence"))
    candidate_registry[LOOP93_CANDIDATE_ID] = loop93_result

    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "hierarchical_selector_evidence",
        "status": "retrospective_selection_only",
        "promotion": "requires_december_reject_and_future_shadow",
        "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
        "chronology": {
            "trainEndInclusive": "2025-07-31",
            "probabilityCalibrationMonth": "2025-08",
            "thresholdFitMonth": "2025-09",
            "selectorMonths": ["2025-10", "2025-11"],
            "note": "This is frozen October-November selector evidence, not expanding-fold OOF.",
        },
        "rowCounts": {
            "selector": len(rows),
            "candidates": len(profiles) + 1,
            "fittedModelsScored": len(groups),
            "decisionRows": len(rows) * (len(profiles) + 1),
        },
        "lineNormalizationCollisions": line_collision_report(rows),
        "candidates": candidate_registry,
        "scopeEvidence": sorted(
            scope_evidence,
            key=lambda item: (
                item["candidate_id"],
                item["scope_id"],
                item["month"],
            ),
        ),
    }
    payload["payloadSha256"] = _sha256_json(payload)
    report_path = output_root / "scope_evidence_2025.json"
    _write_json_atomic(report_path, payload)
    return report_path


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_gzip_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as raw:
        temporary = Path(raw.name)
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for row in rows:
                    text.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    text.write("\n")
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--source-root", default=str(DEFAULT_SOURCE_ROOT)
    )
    parser.add_argument(
        "--run-file",
        action="append",
        default=[],
        help="Additional frozen candidate run JSON; may be supplied repeatedly.",
    )
    parser.add_argument("--output-dir", default="outputs/evidence")
    parser.add_argument(
        "--loop93-replay-dir", default=str(DEFAULT_LOOP93_REPLAY_SOURCE)
    )
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    run_files = (
        *(source_root / name for name in DEFAULT_RUN_FILES),
        *(Path(value).resolve() for value in args.run_file),
    )
    report_path = materialize_hierarchical_evidence(
        config_path=args.config,
        run_files=run_files,
        output_dir=args.output_dir,
        loop93_replay_dir=args.loop93_replay_dir,
    )
    print(report_path)


if __name__ == "__main__":
    main()

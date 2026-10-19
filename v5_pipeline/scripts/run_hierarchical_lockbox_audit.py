from __future__ import annotations

import argparse
import hashlib
import json
import sys
import re
import unicodedata
from collections import defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.materialize_hierarchical_evidence import (
    DEFAULT_RUN_FILES,
    DEFAULT_SOURCE_ROOT,
    CandidateProfile,
    _sha256_file,
    _write_json_atomic,
    apply_candidate_profile,
    load_candidate_profiles,
)
from scripts.materialize_policy_algebra import algebra_policies_from_registry
from src.binary_candidates import load_binary_process_artifact
from src.binary_final_refit import refit_binary_process_candidate
from src.candidates import load_candidate_artifact
from src.company_scope import prediction_or_review, select_company_rows
from src.company_scope import validate_route_company
from src.data_contract import ACTION_LABELS, ALLOWED_RAW_FEATURES, FeatureRow
from src.experiment import prepare_experiment
from src.freeze import PolicySpec
from src.hierarchical_select import REVIEW_ALL_CANDIDATE_ID
from src.hierarchy import canonical_line_key
from src.metrics import OutcomeRow, compute_metrics
from src.postfit_policy import resolve_postfit_prediction
from src.select import LockboxResult


DEFAULT_MATERIALIZED_ROOT = Path(
    "artifacts/mes_rule_miner_clean_optimization_20260710/"
    "outputs/materialized"
)


def run_hierarchical_lockbox_audit(
    *,
    config_path: str | Path,
    evidence_path: str | Path,
    budgeted_policy_path: str | Path,
    output_dir: str | Path,
    materialized_root: str | Path = DEFAULT_MATERIALIZED_ROOT,
    veto_mode: str = "company_product",
) -> Path:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_dir = output_root / "state"
    if (state_dir / "freeze_state.json").exists():
        raise RuntimeError("output state already exists; use a fresh output directory")

    evidence_file = Path(evidence_path).resolve()
    evidence_payload = json.loads(evidence_file.read_text(encoding="utf-8"))
    policy_file = Path(budgeted_policy_path).resolve()
    policy_payload = json.loads(policy_file.read_text(encoding="utf-8"))
    _validate_policy_evidence_parent(policy_payload, evidence_file)
    company_routes, line_routes = _load_routes(policy_payload)
    _validate_route_scopes(
        company_routes,
        line_routes,
        evidence_payload.get("candidates", {}),
    )
    algebra = algebra_policies_from_registry(evidence_payload["candidates"])
    required_base_ids = _required_base_candidates(
        {*company_routes.values(), *line_routes.values()}, algebra
    )
    profile_run_files = _candidate_run_files(
        evidence_payload,
        tuple(DEFAULT_SOURCE_ROOT / name for name in DEFAULT_RUN_FILES),
    )
    profiles = {
        item.candidate_id: item
        for item in load_candidate_profiles(
            profile_run_files
        )
        if item.candidate_id in required_base_ids
    }
    if set(profiles) != required_base_ids:
        raise ValueError(
            f"required postfit candidate profiles are missing: {sorted(required_base_ids - set(profiles))}"
        )
    _validate_profile_receipts(profiles, evidence_payload)

    experiment = prepare_experiment(config_path, state_dir)
    materialized = Path(materialized_root).resolve()
    baseline_receipt = _materialized_baseline_receipt(
        materialized,
        source_dataset_id=experiment.bundle.metadata.source_dataset_id,
        expected_rows={
            "selector": len(experiment.bundle.selector),
            "lockbox": len(experiment.bundle.lockbox),
            "audit": len(experiment.bundle.audit),
        },
    )
    candidate_registry_sha256 = _sha256_file(evidence_file)
    assignment_mode = "hierarchical_budgeted_company_line_v1"
    spec = PolicySpec(
        policy_name="hierarchical_budgeted_company_line",
        source_dataset_id=experiment.bundle.metadata.source_dataset_id,
        history_sha256=experiment.bundle.metadata.source_hashes["historyRows"],
        selection_months=("2025-10", "2025-11"),
        feature_names=tuple(ALLOWED_RAW_FEATURES),
        model_params={
            "candidateRegistrySha256": candidate_registry_sha256,
            "assignmentMode": assignment_mode,
            "selectorPolicySha256": _sha256_file(policy_file),
            "lineAssignments": {
                f"{company}|{line}": candidate
                for (company, line), candidate in sorted(line_routes.items())
            },
            "transitionCaps": policy_payload["transitionCaps"],
            "processVetoRule": {
                "mode": "process_prediction_to_review_only",
                "trigger": "any_lockbox_ship_or_repair_to_process",
                "scope": veto_mode,
            },
            "scopedModelRefit": {
                "multiclass": "train_through_2025_09_calibrate_2025_10",
                "threshold": "preserve_frozen_selector_probability_threshold",
                "binary": "train_through_2025_09_calibrate_2025_10",
            },
            "baselineMaterialization": baseline_receipt,
        },
        thresholds={},
        company_assignments=company_routes,
    )
    receipt = experiment.freeze_policy(spec)
    (output_root / "frozen_policy.json").write_bytes(receipt.policy_bytes)
    _write_json_atomic(
        output_root / "frozen_policy_receipt.json",
        {
            "policySha256": receipt.sha256,
            "bindingSha256": receipt.binding_sha256,
            "candidateRegistrySha256": candidate_registry_sha256,
        },
    )

    final_partitions = experiment.final_refit_partitions(receipt)
    refit_models, refit_model_receipts = _refit_scoped_models(
        profiles,
        final_partitions,
        output_root / "scoped_final_refit_artifacts",
    )

    lockbox_predictions = _score_phase(
        experiment.bundle.lockbox,
        baseline_path=materialized / "lockbox.jsonl",
        profiles=profiles,
        company_routes=company_routes,
        line_routes=line_routes,
        algebra=algebra,
        model_overrides=refit_models,
    )
    lockbox_artifact = output_root / "sealed_lockbox_predictions.json"
    _write_prediction_artifact(
        lockbox_artifact,
        phase="lockbox",
        policy_sha256=receipt.sha256,
        predictions=lockbox_predictions,
    )
    lockbox_prediction_sha256 = experiment.seal_lockbox_predictions(
        lockbox_predictions
    )
    lockbox_labels = experiment.open_lockbox(receipt)
    lockbox_outcomes = _outcomes(
        experiment.bundle.lockbox,
        lockbox_labels,
        lockbox_predictions,
    )
    lockbox_initial_metrics = compute_metrics(lockbox_outcomes)
    rejected_scopes = _dangerous_scopes(
        lockbox_outcomes,
        experiment.bundle.lockbox,
        mode=veto_mode,
    )
    lockbox_final_predictions = _apply_process_veto(
        experiment.bundle.lockbox,
        lockbox_predictions,
        rejected_scopes,
        mode=veto_mode,
    )
    lockbox_final_metrics = compute_metrics(
        _outcomes(
            experiment.bundle.lockbox,
            lockbox_labels,
            lockbox_final_predictions,
        )
    )

    rejections_by_company: dict[str, list[str]] = defaultdict(list)
    for scope in sorted(rejected_scopes):
        company = scope[0]
        rejections_by_company[company].append(
            f"{veto_mode}_process_veto:{'|'.join(scope[1:])}"
        )
    rejection_reasons = {
        company: tuple(rejections_by_company.get(company, ()))
        for company in sorted(company_routes)
    }
    lockbox_universe_sha256 = hashlib.sha256(
        "\n".join(sorted(lockbox_predictions)).encode("utf-8")
    ).hexdigest()
    assignment_payload = {
        "schemaVersion": 1,
        "registrySha256": candidate_registry_sha256,
        "lockboxUniverseSha256": lockbox_universe_sha256,
        "assignmentMode": assignment_mode,
        "assignments": dict(sorted(company_routes.items())),
        "rejections": {
            company: list(rejection_reasons[company])
            for company in sorted(rejection_reasons)
        },
    }
    assignment_sha256 = hashlib.sha256(
        _stable_json_bytes(assignment_payload)
    ).hexdigest()
    lockbox_result = LockboxResult(
        assignment_mode=assignment_mode,
        assignments=MappingProxyType(dict(sorted(company_routes.items()))),
        rejection_reasons=MappingProxyType(rejection_reasons),
        lockbox_universe_sha256=lockbox_universe_sha256,
        assignment_sha256=assignment_sha256,
    )
    final_policy_sha256 = experiment.bind_lockbox_result(lockbox_result)

    audit_initial_predictions = _score_phase(
        experiment.bundle.audit,
        baseline_path=materialized / "audit.jsonl",
        profiles=profiles,
        company_routes=company_routes,
        line_routes=line_routes,
        algebra=algebra,
        model_overrides=refit_models,
    )
    audit_initial_artifact = output_root / "sealed_audit_initial_predictions.json"
    _write_prediction_artifact(
        audit_initial_artifact,
        phase="final_evaluation_before_reject_only_veto",
        policy_sha256=final_policy_sha256,
        predictions=audit_initial_predictions,
    )
    audit_predictions = _apply_process_veto(
        experiment.bundle.audit,
        audit_initial_predictions,
        rejected_scopes,
        mode=veto_mode,
    )
    audit_artifact = output_root / "sealed_audit_predictions.json"
    _write_prediction_artifact(
        audit_artifact,
        phase="final_evaluation",
        policy_sha256=final_policy_sha256,
        predictions=audit_predictions,
    )
    audit_prediction_sha256 = experiment.seal_audit_predictions(
        audit_predictions,
        final_policy_sha256=final_policy_sha256,
    )
    audit_report = experiment.open_audit_and_materialize(receipt)

    report = {
        "schemaVersion": 1,
        "kind": "hierarchical_lockbox_and_fixed_denominator_evaluation",
        "status": "retrospective_only_requires_future_shadow",
        "sourceDatasetId": experiment.bundle.metadata.source_dataset_id,
        "fixedEvaluationRows": experiment.bundle.metadata.expected_audit_action_rows,
        "nonActionRowsExcluded": experiment.bundle.metadata.expected_audit_non_action_rows,
        "selectorPolicyPath": str(policy_file),
        "selectorPolicySha256": _sha256_file(policy_file),
        "frozenPolicySha256": receipt.sha256,
        "frozenPolicyBindingSha256": receipt.binding_sha256,
        "lockboxPredictionArtifact": str(lockbox_artifact),
        "lockboxPredictionArtifactSha256": _sha256_file(lockbox_artifact),
        "lockboxPredictionSha256": lockbox_prediction_sha256,
        "lockbox": {
            "rows": len(experiment.bundle.lockbox),
            "initialMetrics": _plain(lockbox_initial_metrics),
            "vetoMode": veto_mode,
            "rejectedProcessScopes": [
                list(scope) for scope in sorted(rejected_scopes)
            ],
            "finalMetrics": _plain(lockbox_final_metrics),
            "assignmentSha256": assignment_sha256,
        },
        "finalPolicySha256": final_policy_sha256,
        "auditPredictionArtifact": str(audit_artifact),
        "auditPredictionArtifactSha256": _sha256_file(audit_artifact),
        "auditPredictionSha256": audit_prediction_sha256,
        "auditInitialPredictionArtifact": str(audit_initial_artifact),
        "auditInitialPredictionArtifactSha256": _sha256_file(audit_initial_artifact),
        "scopedFinalRefit": {
            "chronology": {
                "trainEndInclusive": "2025-09-30",
                "probabilityCalibrationMonth": "2025-10",
                "thresholdPolicy": "frozen_selector_probability_threshold",
            },
            "models": refit_model_receipts,
        },
        "globalMetrics": _plain(audit_report.global_metrics),
        "companyMetrics": {
            company: _plain(metrics)
            for company, metrics in sorted(audit_report.company_metrics.items())
        },
        "monthMetrics": {
            month: _plain(metrics)
            for month, metrics in sorted(audit_report.month_metrics.items())
        },
        "reportSha256": audit_report.sha256,
    }
    report_path = output_root / "hierarchical_lockbox_audit_report.json"
    _write_json_atomic(report_path, report)
    return report_path


def _score_phase(
    rows: Sequence[FeatureRow],
    *,
    baseline_path: Path,
    profiles: Mapping[str, CandidateProfile],
    company_routes: Mapping[str, str],
    line_routes: Mapping[tuple[str, str], str],
    algebra: Mapping[str, Any],
    model_overrides: Mapping[tuple[Path, str | None], Any] | None = None,
) -> dict[str, str]:
    row_ids = {row.row_id for row in rows}
    baseline = _load_baseline_predictions(baseline_path, row_ids)
    groups: dict[tuple[Path, str | None], list[CandidateProfile]] = defaultdict(list)
    for profile in profiles.values():
        groups[(profile.fitted_manifest_path, profile.scope_company)].append(profile)
    candidate_predictions: dict[str, dict[str, str]] = {}
    for grouped_profiles in groups.values():
        exemplar = grouped_profiles[0]
        scoring_rows = select_company_rows(rows, exemplar.scope_company)
        group_key = (exemplar.fitted_manifest_path, exemplar.scope_company)
        model = (model_overrides or {}).get(group_key)
        if model is None and exemplar.family == "binary_process":
            model = load_binary_process_artifact(exemplar.fitted_manifest_path)
        elif model is None:
            model = load_candidate_artifact(exemplar.fitted_manifest_path)
        if exemplar.family == "binary_process":
            scores = _score_binary_in_chunks(model, scoring_rows)
        else:
            scores = _score_multiclass_in_chunks(model, scoring_rows)
        for profile in grouped_profiles:
            candidate_predictions[profile.candidate_id] = apply_candidate_profile(
                scores, profile
            )

    predictions: dict[str, str] = {}
    for row in rows:
        line = canonical_line_key(row.features.get("line", ""))
        route = line_routes.get(
            (row.customer_key, line),
            company_routes.get(row.customer_key, REVIEW_ALL_CANDIDATE_ID),
        )
        base_values = {
            candidate_id: prediction_or_review(values, row.row_id)
            for candidate_id, values in candidate_predictions.items()
        }
        predictions[row.row_id] = resolve_postfit_prediction(
            route,
            baseline[row.row_id],
            base_values,
            algebra,
        )
    return predictions


def _scoped_labeled_partition(
    rows: Sequence[Any], labels: Mapping[str, str], company: str
) -> tuple[tuple[Any, ...], dict[str, str]]:
    selected = select_company_rows(rows, company)
    return selected, {row.row_id: labels[row.row_id] for row in selected}


def _refit_scoped_models(
    profiles: Mapping[str, CandidateProfile],
    partitions: Any,
    output_dir: Path,
) -> tuple[dict[tuple[Path, str | None], Any], dict[str, Any]]:
    """Final-refit company-local models before lockbox labels open."""

    groups: dict[tuple[Path, str], list[CandidateProfile]] = defaultdict(list)
    for profile in profiles.values():
        if profile.scope_company:
            groups[(profile.fitted_manifest_path, profile.scope_company)].append(profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[tuple[Path, str | None], Any] = {}
    receipts: dict[str, Any] = {}
    for (manifest_path, company), grouped in sorted(
        groups.items(), key=lambda item: (item[0][1], str(item[0][0]))
    ):
        train_rows, train_labels = _scoped_labeled_partition(
            partitions.train, partitions.train_labels, company
        )
        calibration_rows, calibration_labels = _scoped_labeled_partition(
            partitions.probability_calibration,
            partitions.probability_calibration_labels,
            company,
        )
        if set(calibration_labels.values()) != set(ACTION_LABELS):
            raise ValueError(
                f"scoped final calibration lacks all action classes: {company}"
            )
        max_score_rows = max(
            len(select_company_rows(partitions.lockbox_scoring, company)),
            len(select_company_rows(partitions.audit_scoring, company)),
            1,
        )
        exemplar = grouped[0]
        if exemplar.family == "binary_process":
            template = load_binary_process_artifact(manifest_path)
            template.max_score_rows = max_score_rows
            model = refit_binary_process_candidate(
                template,
                SimpleNamespace(
                    train=train_rows,
                    probability_calibration=calibration_rows,
                    train_labels=train_labels,
                    probability_calibration_labels=calibration_labels,
                ),
                source_dataset_id=template.source_dataset_id,
                history_sha256=template.history_sha256,
            )
        else:
            model = load_candidate_artifact(manifest_path)
            model.max_score_rows = max_score_rows
            model.fit(
                train_rows,
                train_labels,
                calibration_rows=calibration_rows,
                calibration_labels=calibration_labels,
            )
        stem = manifest_path.name.removesuffix(".fitted_artifact.json")
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{stem}_{company}_final")
        persisted = model.persist_artifacts(output_dir, stem=stem)
        final_manifest = Path(persisted["fitted_artifact"]).resolve()
        models[(manifest_path, company)] = model
        receipts[f"{company}|{manifest_path.name}"] = {
            "company": company,
            "sourceCandidates": sorted(item.candidate_id for item in grouped),
            "trainingRows": len(train_rows),
            "calibrationRows": len(calibration_rows),
            "fittedManifestPath": str(final_manifest),
            "fittedManifestSha256": _sha256_file(final_manifest),
        }
    return models, receipts


def _candidate_run_files(
    evidence_payload: Mapping[str, Any], defaults: Sequence[Path]
) -> tuple[Path, ...]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for value in defaults:
        path = Path(value).resolve()
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    candidates = evidence_payload.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("hierarchical evidence candidate registry is missing")
    for candidate in candidates.values():
        if not isinstance(candidate, Mapping):
            continue
        source_run = candidate.get("sourceRun")
        if not source_run:
            continue
        path = Path(str(source_run)).resolve()
        if path not in seen:
            ordered.append(path)
            seen.add(path)
    return tuple(ordered)


def _validate_policy_evidence_parent(
    policy_payload: Mapping[str, Any], evidence_file: Path
) -> None:
    expected = str(policy_payload.get("sourceEvidenceSha256") or "")
    if not expected:
        raise ValueError("selector policy source evidence digest is missing")
    actual = _sha256_file(evidence_file)
    if expected != actual:
        raise ValueError(
            f"selector policy source evidence digest mismatch: {expected} != {actual}"
        )


def _materialized_baseline_receipt(
    materialized_root: Path,
    *,
    source_dataset_id: str,
    expected_rows: Mapping[str, int],
) -> dict[str, Any]:
    """Validate and bind every external baseline prediction artifact.

    The existing materialization intentionally fails closed for 48 development
    rows that have no executable Loop93 replay.  An incomplete manifest is
    accepted only when every declared missing selector/lockbox row is present
    in the JSONL as ``fallback_unproven`` and does not carry an action label.
    """

    root = Path(materialized_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("materialized baseline manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("source_dataset_id") or "") != source_dataset_id:
        raise ValueError("materialized baseline source dataset mismatch")
    files = manifest.get("files")
    splits = manifest.get("splits")
    if not isinstance(files, Mapping) or not isinstance(splits, Mapping):
        raise ValueError("materialized baseline manifest is malformed")
    receipt_files: dict[str, Any] = {}
    for split in ("selector", "lockbox", "audit"):
        declared = files.get(split)
        diagnostics = splits.get(split)
        if not isinstance(declared, Mapping) or not isinstance(diagnostics, Mapping):
            raise ValueError(f"materialized baseline split is missing: {split}")
        split_path = root / str(declared.get("path") or f"{split}.jsonl")
        actual_sha = _sha256_file(split_path)
        if actual_sha != str(declared.get("sha256") or ""):
            raise ValueError(f"materialized split digest mismatch: {split}")
        declared_rows = int(declared.get("rows", -1))
        if declared_rows != int(expected_rows[split]):
            raise ValueError(
                f"materialized split row count mismatch: {split} "
                f"{declared_rows} != {expected_rows[split]}"
            )
        receipt_files[split] = {
            "path": str(split_path),
            "rows": declared_rows,
            "sha256": actual_sha,
            "missingRows": int(diagnostics.get("missing_count", 0)),
        }

    unresolved = int(
        (manifest.get("fallback") or {}).get("unresolved_selector_lockbox_rows", 0)
    )
    if not bool(manifest.get("materialization_complete", False)):
        declared_missing = sum(
            int((splits.get(split) or {}).get("missing_count", 0))
            for split in ("selector", "lockbox")
        )
        if unresolved != declared_missing or unresolved < 1:
            raise ValueError("incomplete materialization fallback count mismatch")
        for split in ("selector", "lockbox"):
            missing_ids = set((splits.get(split) or {}).get("missing", ()))
            if not missing_ids:
                continue
            observed: set[str] = set()
            with (root / str(files[split]["path"])).open(
                "r", encoding="utf-8"
            ) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_id = str(row.get("row_id") or "")
                    if row_id not in missing_ids:
                        continue
                    if row.get("prediction_status") != "fallback_unproven":
                        raise ValueError(
                            f"unresolved baseline row is not fail closed: {row_id}"
                        )
                    if str(row.get("predicted") or "").strip() in ACTION_LABELS:
                        raise ValueError(
                            f"unresolved baseline row carries an action: {row_id}"
                        )
                    observed.add(row_id)
            if observed != missing_ids:
                raise ValueError(
                    f"unresolved baseline row identity mismatch: {split}"
                )
    return {
        "manifestPath": str(manifest_path),
        "manifestSha256": _sha256_file(manifest_path),
        "declaredManifestSha256": str(manifest.get("manifest_sha256") or ""),
        "materializationComplete": bool(
            manifest.get("materialization_complete", False)
        ),
        "unresolvedFailClosedRows": unresolved,
        "files": receipt_files,
    }


def _validate_route_scopes(
    company_routes: Mapping[str, str],
    line_routes: Mapping[tuple[str, str], str],
    candidate_registry: Mapping[str, Mapping[str, Any]],
) -> None:
    for company, candidate_id in company_routes.items():
        if candidate_id != REVIEW_ALL_CANDIDATE_ID:
            validate_route_company(candidate_registry, candidate_id, company)
    for (company, _line), candidate_id in line_routes.items():
        if candidate_id != REVIEW_ALL_CANDIDATE_ID:
            validate_route_company(candidate_registry, candidate_id, company)


def _validate_profile_receipts(
    profiles: Mapping[str, CandidateProfile], evidence_payload: Mapping[str, Any]
) -> None:
    candidates = evidence_payload.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("hierarchical evidence candidate registry is missing")
    for candidate_id, profile in profiles.items():
        receipt = candidates.get(candidate_id)
        if not isinstance(receipt, Mapping):
            raise ValueError(f"candidate evidence receipt is missing: {candidate_id}")
        expected_run_sha = str(receipt.get("sourceRunSha256") or "")
        expected_manifest_sha = str(receipt.get("fittedManifestSha256") or "")
        expected_scope = receipt.get("scopeCompany")
        if expected_run_sha != _sha256_file(profile.source_run):
            raise ValueError(f"candidate source run digest mismatch: {candidate_id}")
        if expected_manifest_sha != profile.fitted_manifest_sha256:
            raise ValueError(f"candidate fitted manifest digest mismatch: {candidate_id}")
        if expected_scope != profile.scope_company:
            raise ValueError(f"candidate company scope mismatch: {candidate_id}")


def _score_multiclass_in_chunks(
    model: Any,
    rows: Sequence[Any],
    *,
    chunk_size: int = 40_000,
) -> tuple[Any, ...]:
    model_limit = int(getattr(model, "max_score_rows", chunk_size))
    effective_chunk_size = min(chunk_size, model_limit)
    if effective_chunk_size < 1:
        raise ValueError("chunk size must be positive")
    output: list[Any] = []
    for start in range(0, len(rows), effective_chunk_size):
        output.extend(
            model.predict_scores(rows[start : start + effective_chunk_size])
        )
    return tuple(output)


def _score_binary_in_chunks(
    model: Any,
    rows: Sequence[Any],
    *,
    chunk_size: int = 40_000,
) -> tuple[Any, ...]:
    model_limit = int(getattr(model, "max_score_rows", chunk_size))
    effective_chunk_size = min(chunk_size, model_limit)
    if effective_chunk_size < 1:
        raise ValueError("chunk size must be positive")
    output: list[Any] = []
    for start in range(0, len(rows), effective_chunk_size):
        output.extend(model.score_postfit(rows[start : start + effective_chunk_size]))
    return tuple(output)


def _load_routes(
    payload: Mapping[str, Any],
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    companies = {
        str(company): str(choice["candidate_id"])
        for company, choice in payload["companyChoices"].items()
    }
    lines: dict[tuple[str, str], str] = {}
    for key, choice in payload["lineChoices"].items():
        company, line = key.split("|", 1)
        lines[(company, line)] = str(choice["candidate_id"])
    return companies, lines


def _required_base_candidates(
    route_ids: set[str], algebra: Mapping[str, Any]
) -> set[str]:
    required: set[str] = set()
    for candidate_id in route_ids:
        if candidate_id == REVIEW_ALL_CANDIDATE_ID:
            continue
        policy = algebra.get(candidate_id)
        if policy is None:
            if candidate_id != "loop93_current_policy_crossfit":
                required.add(candidate_id)
            continue
        required.update(policy.guard_candidates)
        required.update(policy.recovery_candidates)
    return required


def _load_baseline_predictions(path: Path, expected_ids: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id not in expected_ids:
                continue
            if row_id in output:
                raise ValueError(f"duplicate baseline phase row: {row_id}")
            predicted = str(row.get("predicted", "")).strip()
            output[row_id] = predicted if predicted in ACTION_LABELS else "REVIEW"
    if set(output) != expected_ids:
        raise ValueError(
            f"baseline phase identity mismatch: {len(output)} != {len(expected_ids)}"
        )
    return output


def _outcomes(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    predictions: Mapping[str, str],
) -> tuple[OutcomeRow, ...]:
    return tuple(
        OutcomeRow(
            row_id=row.row_id,
            customer_key=row.customer_key,
            raw_customer=row.raw_customer,
            legacy_customer_key=row.legacy_customer_key,
            month=row.month,
            product_key=row.product_key,
            episode_id=row.episode_id,
            actual=labels[row.row_id],
            predicted=predictions[row.row_id],
        )
        for row in rows
    )


def _dangerous_scopes(
    outcomes: Sequence[OutcomeRow],
    rows: Sequence[FeatureRow],
    *,
    mode: str,
) -> frozenset[tuple[str, ...]]:
    row_map = {row.row_id: row for row in rows}
    return frozenset(
        _veto_scope(row_map[outcome.row_id], mode)
        for outcome in outcomes
        if outcome.actual in {"RETURN", "REPAIR"}
        and outcome.predicted == "PROCESS"
    )


def _apply_process_veto(
    rows: Sequence[FeatureRow],
    predictions: Mapping[str, str],
    rejected_scopes: frozenset[tuple[str, ...]],
    *,
    mode: str,
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        predicted = predictions[row.row_id]
        output[row.row_id] = (
            "REVIEW"
            if predicted == "PROCESS"
            and _veto_scope(row, mode) in rejected_scopes
            else predicted
        )
    return output


def _veto_scope(row: Any, mode: str) -> tuple[str, ...]:
    if mode == "company_line":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
        )
    if mode == "company_product":
        return str(row.customer_key), str(row.product_key)
    if mode == "company_line_product":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
            str(row.product_key),
        )
    if mode == "company_line_type_phenomenon":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
            _normalize_scope_value(row.features.get("type", "")),
            _normalize_scope_value(row.features.get("symptom", "")),
        )
    if mode == "company_line_type":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
            _normalize_scope_value(row.features.get("type", "")),
        )
    if mode == "company_line_phenomenon":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
            _normalize_scope_value(row.features.get("symptom", "")),
        )
    if mode == "company_type_phenomenon":
        return (
            str(row.customer_key),
            _normalize_scope_value(row.features.get("type", "")),
            _normalize_scope_value(row.features.get("symptom", "")),
        )
    if mode == "company_phenomenon":
        return (
            str(row.customer_key),
            _normalize_scope_value(row.features.get("symptom", "")),
        )
    if mode == "company_line_detail":
        return (
            str(row.customer_key),
            canonical_line_key(row.features.get("line", "")),
            _normalize_scope_value(row.features.get("detail", "")),
        )
    if mode == "company_product_type_phenomenon":
        return (
            str(row.customer_key),
            str(row.product_key),
            _normalize_scope_value(row.features.get("type", "")),
            _normalize_scope_value(row.features.get("symptom", "")),
        )
    raise ValueError(f"unsupported veto mode: {mode}")


def _normalize_scope_value(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")).strip(),
    ).upper()


def _write_prediction_artifact(
    path: Path,
    *,
    phase: str,
    policy_sha256: str,
    predictions: Mapping[str, str],
) -> None:
    _write_json_atomic(
        path,
        {
            "schemaVersion": 1,
            "kind": "sealed_phase_predictions",
            "phase": phase,
            "policySha256": policy_sha256,
            "rows": len(predictions),
            "predictions": dict(sorted(predictions.items())),
        },
    )


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--evidence", default="outputs/algebra/scope_evidence_algebra_2025.json"
    )
    parser.add_argument(
        "--policy", default="outputs/frontier_budgeted/budgeted_frontier_2025.json"
    )
    parser.add_argument("--output-dir", default="outputs/postfit_audit_run1")
    parser.add_argument("--materialized-root", default=str(DEFAULT_MATERIALIZED_ROOT))
    parser.add_argument(
        "--veto-mode",
        choices=(
            "company_line",
            "company_product",
            "company_line_product",
            "company_line_type_phenomenon",
            "company_line_type",
            "company_line_phenomenon",
            "company_type_phenomenon",
            "company_phenomenon",
            "company_line_detail",
            "company_product_type_phenomenon",
        ),
        default="company_product",
    )
    args = parser.parse_args()
    print(
        run_hierarchical_lockbox_audit(
            config_path=args.config,
            evidence_path=args.evidence,
            budgeted_policy_path=args.policy,
            output_dir=args.output_dir,
            materialized_root=args.materialized_root,
            veto_mode=args.veto_mode,
        )
    )


if __name__ == "__main__":
    main()

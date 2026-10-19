from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from src.prior_supported_process_recovery import (
    PriorSupportedProcessRecovery,
    PriorSupportedProcessRecoveryParams,
    ProcessSignature,
)
from src.prior_supported_repair_guard import (
    PriorSupportEvidence,
    PriorSupportedRepairGuard,
    PriorSupportedRepairGuardParams,
)

from .integrity import canonical_json_sha256


def load_repair_guard(path: str | Path) -> PriorSupportedRepairGuard:
    policy = _verified_policy(path)
    evidence: dict[str, Mapping[str, PriorSupportEvidence]] = {}
    for company, by_signal in policy["evidenceByCompany"].items():
        restored = {
            str(signal_id): PriorSupportEvidence(
                customer_key=str(raw["customerKey"]),
                signal_id=str(raw["signalId"]),
                support=int(raw["support"]),
                process_rows=int(raw["processRows"]),
                ship_rows=int(raw["shipRows"]),
                repair_rows=int(raw["repairRows"]),
                risk_rows=int(raw["riskRows"]),
                risk_rate=float(raw["riskRate"]),
                enabled=bool(raw["enabled"]),
            )
            for signal_id, raw in by_signal.items()
        }
        evidence[str(company)] = MappingProxyType(dict(sorted(restored.items())))
    thresholds = policy["thresholds"]
    return PriorSupportedRepairGuard(
        training_end=str(policy["selectionLabelsEndInclusive"]),
        params=PriorSupportedRepairGuardParams(
            signal_ids=tuple(str(value) for value in thresholds["signalIds"]),
            min_rows=int(thresholds["minRows"]),
            min_risk_rows=int(thresholds["minRiskRows"]),
            min_risk_rate=float(thresholds["minRiskRate"]),
        ),
        fit_row_ids_sha256=str(policy["trainingRows"]["rowIdsSha256"]),
        enabled_by_company=MappingProxyType(
            {
                str(company): tuple(str(value) for value in values)
                for company, values in sorted(policy["enabledByCompany"].items())
            }
        ),
        evidence_by_company=MappingProxyType(dict(sorted(evidence.items()))),
    )


def load_process_recovery(path: str | Path) -> PriorSupportedProcessRecovery:
    policy = _verified_policy(path)
    raw = policy["recovery"]
    params = raw["params"]
    signatures: dict[str, tuple[ProcessSignature, ...]] = {}
    for scope, values in raw["signatures_by_scope"].items():
        signatures[str(scope)] = tuple(_restore_signature(value) for value in values)
    return PriorSupportedProcessRecovery(
        training_end=str(raw["training_end"]),
        params=PriorSupportedProcessRecoveryParams(
            scopes=tuple(str(value) for value in params["scopes"]),
            min_support=int(params["min_support"]),
            min_months=int(params["min_months"]),
            min_recent_support=int(params["min_recent_support"]),
            recent_window_months=int(params["recent_window_months"]),
        ),
        fit_row_ids_sha256=str(raw["fit_row_ids_sha256"]),
        signatures_by_scope=MappingProxyType(signatures),
    )


def load_semantic_veto(path: str | Path) -> Mapping[str, Any]:
    policy = _verified_policy(path)
    raw = policy.get("semanticVeto")
    if not isinstance(raw, Mapping):
        raise ValueError(f"sealed semantic veto policy is malformed: {path}")
    return MappingProxyType(dict(raw))


def _restore_signature(value: Mapping[str, Any]) -> ProcessSignature:
    return ProcessSignature(
        signature_id=str(value["signature_id"]),
        scope=str(value["scope"]),
        key=tuple(str(item) for item in value["key"]),
        support=int(value["support"]),
        process_rows=int(value["process_rows"]),
        months=tuple(str(item) for item in value["months"]),
        recent_support=int(value["recent_support"]),
        training_end=str(value["training_end"]),
    )


def _verified_policy(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.get("sha256") or "")
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    if not expected or canonical_json_sha256(unsigned) != expected:
        raise ValueError(f"sealed policy SHA mismatch: {path}")
    if payload.get("auditInputsIncluded") is not False:
        raise ValueError(f"sealed policy contains audit inputs: {path}")
    return MappingProxyType(payload)

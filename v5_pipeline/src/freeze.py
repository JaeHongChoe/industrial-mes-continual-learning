from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .data_contract import (
    ACTION_LABELS,
    DataContractError,
    DatasetBundle,
    IMMUTABLE_LOCKBOX_MONTHS,
    IMMUTABLE_SELECTOR_MONTHS,
    parse_source_date,
    stable_row_id,
    validate_requested_features,
)


class FreezeGateError(DataContractError):
    """Raised when audit outcomes are opened before a valid deterministic freeze."""


@dataclass(frozen=True)
class PolicySpec:
    policy_name: str
    source_dataset_id: str
    history_sha256: str
    selection_months: tuple[str, ...]
    feature_names: tuple[str, ...]
    model_params: Mapping[str, Any]
    thresholds: Mapping[str, float]
    company_assignments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FreezeReceipt:
    policy_bytes: bytes
    sha256: str
    binding_sha256: str = ""

    def with_sha256(self, sha256: str) -> "FreezeReceipt":
        return FreezeReceipt(
            policy_bytes=self.policy_bytes,
            sha256=sha256,
            binding_sha256=self.binding_sha256,
        )


def freeze_policy(spec: PolicySpec) -> FreezeReceipt:
    if not spec.policy_name.strip():
        raise FreezeGateError("policy_name must not be empty")
    if not spec.history_sha256 or len(spec.history_sha256) != 64:
        raise FreezeGateError("history SHA-256 must be a 64-character digest")
    if not spec.selection_months:
        raise FreezeGateError("selection months must not be empty")
    payload = {
        "schemaVersion": 1,
        "policyName": spec.policy_name,
        "sourceDatasetId": spec.source_dataset_id,
        "historySha256": spec.history_sha256,
        "selectionMonths": list(spec.selection_months),
        "featureNames": list(spec.feature_names),
        "modelParams": dict(spec.model_params),
        "thresholds": dict(spec.thresholds),
        "companyAssignments": dict(spec.company_assignments),
        "auditInputsIncluded": False,
    }
    policy_bytes = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return FreezeReceipt(policy_bytes=policy_bytes, sha256=_sha256_bytes(policy_bytes))


class FreezeCoordinator:
    def __init__(
        self,
        config_path: str | Path,
        bundle: DatasetBundle,
        state_path: str | Path,
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._bundle = bundle
        self._state_path = Path(state_path).resolve()

    def freeze(self, spec: PolicySpec) -> FreezeReceipt:
        if self._state_path.exists():
            phase = str(_read_state(self._state_path).get("phase") or "unknown")
            raise FreezeGateError(f"cannot freeze after run entered {phase} audit phase or was already frozen")
        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        if spec.source_dataset_id != self._bundle.metadata.source_dataset_id:
            raise FreezeGateError("policy source dataset does not match prepared bundle")
        if spec.history_sha256 != self._bundle.metadata.source_hashes["historyRows"]:
            raise FreezeGateError("policy history SHA-256 does not match prepared bundle")
        if tuple(spec.selection_months) != tuple(IMMUTABLE_SELECTOR_MONTHS):
            raise FreezeGateError("policy selection months violate immutable selector chronology")
        try:
            validate_requested_features(spec.feature_names, config)
        except DataContractError as error:
            raise FreezeGateError(str(error)) from error
        _reject_audit_derived_payload(spec.model_params, "model_params")
        _reject_audit_derived_payload(spec.thresholds, "thresholds")
        _reject_audit_derived_payload(spec.company_assignments, "company_assignments")

        receipt = freeze_policy(spec)
        selector_ids_sha256 = _sha256_bytes(
            "\n".join(row.row_id for row in self._bundle.selector).encode("utf-8")
        )
        binding_payload = {
            "sourceDatasetId": self._bundle.metadata.source_dataset_id,
            "historySha256": self._bundle.metadata.source_hashes["historyRows"],
            "selectionMonths": list(IMMUTABLE_SELECTOR_MONTHS),
            "selectorRowIdsSha256": selector_ids_sha256,
            "policySha256": receipt.sha256,
        }
        binding_sha256 = _sha256_bytes(_stable_json_bytes(binding_payload))
        bound = FreezeReceipt(
            policy_bytes=receipt.policy_bytes,
            sha256=receipt.sha256,
            binding_sha256=binding_sha256,
        )
        _write_state(
            self._state_path,
            {
                "schemaVersion": 1,
                "phase": "frozen",
                "policySha256": bound.sha256,
                "bindingSha256": bound.binding_sha256,
                **binding_payload,
            },
        )
        return bound


class LockboxLabelVault:
    """Open December outcomes once, only after selector policy bytes are frozen."""

    def __init__(
        self,
        config_path: str | Path,
        bundle: DatasetBundle,
        state_path: str | Path,
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._bundle = bundle
        self._state_path = Path(state_path).resolve()

    def open(self, receipt: FreezeReceipt | None) -> Mapping[str, str]:
        state = _validate_vault_receipt(
            receipt,
            self._bundle,
            self._state_path,
            required_phase="frozen",
            vault_name="lockbox",
        )
        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        history_path = Path(config["historyRows"]).resolve()
        observed_history_sha = _sha256_file(history_path)
        if observed_history_sha != self._bundle.metadata.source_hashes["historyRows"]:
            raise FreezeGateError("history source changed before lockbox opening")
        configured_history_sha = str(config.get("historyRowsSha256") or "").lower()
        if configured_history_sha and configured_history_sha != observed_history_sha:
            raise FreezeGateError("history source does not match the configured immutable hash")

        expected_ids = {row.row_id for row in self._bundle.lockbox}
        source_dataset_id = self._bundle.metadata.source_dataset_id
        labels: dict[str, str] = {}
        with history_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise FreezeGateError(f"invalid history JSONL at line {line_number}") from error
                observed_date = parse_source_date(raw)
                if not observed_date or observed_date[:7] not in IMMUTABLE_LOCKBOX_MONTHS:
                    continue
                row_id = stable_row_id(source_dataset_id, raw.get("__sourceRowNumber"))
                if row_id not in expected_ids:
                    raise FreezeGateError(f"unexpected lockbox row after freeze: {row_id}")
                if row_id in labels:
                    raise FreezeGateError(f"duplicate lockbox label row: {row_id}")
                labels[row_id] = str(raw.get("action_result") or "").strip()
        if set(labels) != expected_ids:
            missing = len(expected_ids - set(labels))
            extra = len(set(labels) - expected_ids)
            raise FreezeGateError(
                f"lockbox label universe mismatch after freeze: missing={missing}, extra={extra}"
            )
        _write_state(
            self._state_path,
            {
                **state,
                "phase": "lockbox_opened",
                "lockboxRows": len(labels),
                "lockboxActionRows": sum(label in ACTION_LABELS for label in labels.values()),
                "lockboxSourceSha256": observed_history_sha,
            },
        )
        return MappingProxyType(labels)


class AuditLabelVault:
    def __init__(
        self,
        config_path: str | Path,
        bundle: DatasetBundle,
        state_path: str | Path,
    ) -> None:
        self._config_path = Path(config_path).resolve()
        self._bundle = bundle
        self._state_path = Path(state_path).resolve()

    def open(self, receipt: FreezeReceipt | None) -> Mapping[str, str]:
        state = _validate_vault_receipt(
            receipt,
            self._bundle,
            self._state_path,
            required_phase="lockbox_opened",
            vault_name="audit",
        )

        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        audit_path = Path(config["auditRows"]).resolve()
        observed_audit_sha = _sha256_file(audit_path)
        prepared_audit_sha = self._bundle.metadata.source_hashes["auditRows"]
        if observed_audit_sha != prepared_audit_sha:
            raise FreezeGateError("audit source changed after dataset preparation")
        configured_audit_sha = str(config.get("auditRowsSha256") or "").lower()
        if configured_audit_sha and configured_audit_sha != observed_audit_sha:
            raise FreezeGateError("audit source does not match the configured immutable hash")

        _write_state(self._state_path, {**state, "phase": "audit_opening"})

        source_dataset_id = str(config.get("sourceDatasetId") or "mes_loop93_corrected_source")
        audit_start = str(config["auditStart"])
        expected_ids = {row.row_id for row in self._bundle.audit}
        labels: dict[str, str] = {}
        with audit_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise FreezeGateError(f"invalid audit JSONL at line {line_number}") from error
                observed_date = parse_source_date(raw)
                if not observed_date or observed_date < audit_start:
                    continue
                actual = str(raw.get("action_result") or "").strip()
                row_id = stable_row_id(source_dataset_id, raw.get("__sourceRowNumber"))
                if row_id not in expected_ids:
                    raise FreezeGateError(f"unexpected eligible audit row after freeze: {row_id}")
                if row_id in labels:
                    raise FreezeGateError(f"duplicate audit label row: {row_id}")
                labels[row_id] = actual
        if set(labels) != expected_ids:
            missing = len(expected_ids - set(labels))
            extra = len(set(labels) - expected_ids)
            raise FreezeGateError(f"audit label universe mismatch after freeze: missing={missing}, extra={extra}")
        action_rows = sum(label in ACTION_LABELS for label in labels.values())
        non_action_rows = len(labels) - action_rows
        if action_rows != self._bundle.metadata.expected_audit_action_rows:
            raise FreezeGateError(
                f"audit action denominator mismatch after freeze: expected "
                f"{self._bundle.metadata.expected_audit_action_rows}, observed {action_rows}"
            )
        if non_action_rows != self._bundle.metadata.expected_audit_non_action_rows:
            raise FreezeGateError(
                f"audit non-action denominator mismatch after freeze: expected "
                f"{self._bundle.metadata.expected_audit_non_action_rows}, observed {non_action_rows}"
            )
        _write_state(
            self._state_path,
            {
                **state,
                "phase": "audit_opened",
                "auditRows": len(labels),
                "auditActionRows": action_rows,
                "auditNonActionRows": non_action_rows,
                "auditSourceSha256": observed_audit_sha,
            },
        )
        return MappingProxyType(labels)


def _validate_vault_receipt(
    receipt: FreezeReceipt | None,
    bundle: DatasetBundle,
    state_path: Path,
    *,
    required_phase: str,
    vault_name: str,
) -> dict[str, Any]:
    if receipt is None:
        raise FreezeGateError("a valid frozen policy receipt is required")
    observed_receipt_sha = _sha256_bytes(receipt.policy_bytes)
    if observed_receipt_sha != receipt.sha256:
        raise FreezeGateError(
            f"receipt SHA-256 mismatch: expected {receipt.sha256}, observed {observed_receipt_sha}"
        )
    if not state_path.exists():
        raise FreezeGateError(f"freeze state is missing; {vault_name} labels cannot be opened")
    state = _read_state(state_path)
    if state.get("phase") != required_phase:
        if required_phase == "lockbox_opened" and state.get("phase") == "frozen":
            raise FreezeGateError("lockbox labels must be opened before audit labels")
        raise FreezeGateError(
            f"{vault_name} label vault was already opened or is in {state.get('phase')} phase"
        )
    if receipt.binding_sha256 != state.get("bindingSha256"):
        raise FreezeGateError("receipt binding does not match the frozen run")
    if receipt.sha256 != state.get("policySha256"):
        raise FreezeGateError("receipt policy does not match the frozen run")
    try:
        policy = json.loads(receipt.policy_bytes)
    except json.JSONDecodeError as error:
        raise FreezeGateError("frozen policy bytes are not valid JSON") from error
    if policy.get("sourceDatasetId") != bundle.metadata.source_dataset_id:
        raise FreezeGateError("frozen policy dataset binding is invalid")
    if policy.get("historySha256") != bundle.metadata.source_hashes["historyRows"]:
        raise FreezeGateError("frozen policy history binding is invalid")
    if tuple(policy.get("selectionMonths") or ()) != tuple(IMMUTABLE_SELECTOR_MONTHS):
        raise FreezeGateError("frozen policy selector chronology is invalid")
    return state


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_stable_json_bytes(value))
    temporary.replace(path)


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeGateError(f"invalid freeze state: {path}") from error
    if not isinstance(value, dict):
        raise FreezeGateError("freeze state must be a JSON object")
    return value


def _reject_audit_derived_payload(value: Any, path: str) -> None:
    forbidden_tokens = ("audit", "2026", "actual", "action_result", "on_hold", "customer_comment")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            canonical = str(key).replace("_", "").replace("-", "").lower()
            if any(token in canonical for token in forbidden_tokens):
                raise FreezeGateError(f"audit-derived policy field is forbidden: {path}.{key}")
            _reject_audit_derived_payload(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_audit_derived_payload(nested, f"{path}[{index}]")
    elif isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        raise FreezeGateError(f"non-finite policy value is forbidden: {path}")

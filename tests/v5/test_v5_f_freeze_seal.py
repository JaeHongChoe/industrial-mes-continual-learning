"""Stage F: policy freeze receipts and shadow-validation integrity helpers."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.freeze import FreezeGateError, PolicySpec, _reject_audit_derived_payload, freeze_policy
from src.shadow_validation.integrity import (
    IntegrityError,
    canonical_json_sha256,
    sha256_file,
    verify_manifest,
)
from src.shadow_validation.metrics import MetricsError, normalize_action_label
from src.shadow_validation.receipt import PredictionReceiptError, _reject_forbidden_fields

_HISTORY_SHA = hashlib.sha256(b"synthetic-history").hexdigest()


def _spec(**overrides):
    values = dict(
        policy_name="synthetic_policy",
        source_dataset_id="synthetic_ds",
        history_sha256=_HISTORY_SHA,
        selection_months=("2025-10", "2025-11"),
        feature_names=("type", "symptom", "detail"),
        model_params={"family": "ovr_linear", "seed": 17},
        thresholds={"PROCESS": 0.9, "RETURN": 0.95, "REPAIR": 0.95},
        company_assignments={"customer_a": "cand_safe", "customer_b": "cand_line"},
    )
    values.update(overrides)
    return PolicySpec(**values)


def test_freeze_policy_receipt_is_deterministic_and_self_verifying():
    receipt = freeze_policy(_spec())
    assert receipt.sha256 == hashlib.sha256(receipt.policy_bytes).hexdigest()
    assert receipt.binding_sha256 == ""
    payload = json.loads(receipt.policy_bytes)
    assert payload["auditInputsIncluded"] is False
    assert payload["selectionMonths"] == ["2025-10", "2025-11"]

    reordered = _spec(
        thresholds={"REPAIR": 0.95, "RETURN": 0.95, "PROCESS": 0.9},
        company_assignments={"customer_b": "cand_line", "customer_a": "cand_safe"},
    )
    assert freeze_policy(reordered).sha256 == receipt.sha256
    changed = _spec(thresholds={"PROCESS": 0.91, "RETURN": 0.95, "REPAIR": 0.95})
    assert freeze_policy(changed).sha256 != receipt.sha256


@pytest.mark.parametrize(
    "overrides",
    [{"policy_name": "  "}, {"history_sha256": "abc"}, {"selection_months": ()}],
)
def test_freeze_policy_rejects_invalid_specs(overrides):
    with pytest.raises(FreezeGateError):
        freeze_policy(_spec(**overrides))


def test_audit_derived_payload_guard_rejects_obvious_tokens():
    with pytest.raises(FreezeGateError):
        _reject_audit_derived_payload({"audit_rate": 0.1}, "model_params")
    with pytest.raises(FreezeGateError):
        _reject_audit_derived_payload({"nested": {"actual_counts": 1}}, "model_params")
    with pytest.raises(FreezeGateError):
        _reject_audit_derived_payload({"x": float("nan")}, "thresholds")
    _reject_audit_derived_payload({"seed": 17, "family": "ovr_linear"}, "model_params")


@pytest.mark.parametrize("key", ["action_result", "on_hold", "customer_comment"])
def test_audit_derived_payload_guard_rejects_postdecision_field_names(key):
    with pytest.raises(FreezeGateError):
        _reject_audit_derived_payload({key: 1}, "model_params")


def test_receipt_forbidden_field_guard_canonicalizes_correctly():
    # Contrast with the freeze guard above: receipt.py canonicalizes both sides.
    with pytest.raises(PredictionReceiptError):
        _reject_forbidden_fields({"meta": {"Action Result": "PROCESS"}})
    _reject_forbidden_fields({"meta": {"detail": "ok"}})


def test_canonical_json_sha256_is_key_order_invariant():
    a = canonical_json_sha256({"b": [1, 2], "a": {"y": 1, "x": "ünïcode"}})
    b = canonical_json_sha256({"a": {"x": "ünïcode", "y": 1}, "b": [1, 2]})
    assert a == b and len(a) == 64
    assert a != canonical_json_sha256({"a": {"x": "ünïcode", "y": 2}, "b": [1, 2]})


def _write_manifest(root, artifact_name="policy.json", content=b'{"k":1}\n', **extra):
    artifact = root / artifact_name
    artifact.write_bytes(content)
    manifest = {
        "schemaVersion": 1,
        "generationId": "synthetic-gen-1",
        "artifacts": [
            {
                "relativePath": artifact_name,
                "sizeBytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ],
        **extra,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, path


def test_verify_manifest_accepts_intact_and_detects_tampering(tmp_path):
    artifact, manifest = _write_manifest(tmp_path)
    verified = verify_manifest(tmp_path, manifest, expected_generation="synthetic-gen-1")
    assert verified["verifiedArtifactCount"] == 1
    assert verified["manifestSha256"] == sha256_file(manifest)

    with pytest.raises(IntegrityError):
        verify_manifest(tmp_path, manifest, expected_generation="other-gen")

    artifact.write_bytes(b'{"k":2}\n')  # same size, different bytes
    with pytest.raises(IntegrityError):
        verify_manifest(tmp_path, manifest, expected_generation="synthetic-gen-1")


def test_verify_manifest_rejects_path_escape(tmp_path):
    inner = tmp_path / "pkg"
    inner.mkdir()
    (tmp_path / "outside.json").write_bytes(b"{}")
    manifest = inner / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "generationId": "synthetic-gen-1",
                "artifacts": [
                    {"relativePath": "../outside.json", "sizeBytes": 2, "sha256": "0" * 64}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError):
        verify_manifest(inner, manifest, expected_generation="synthetic-gen-1")


def test_shadow_metrics_normalizes_process_repair_and_review():
    assert normalize_action_label("PROCESS") == "PROCESS"
    assert normalize_action_label(" REPAIR ") == "REPAIR"
    assert normalize_action_label("", allow_review=True) == "REVIEW"
    with pytest.raises(MetricsError):
        normalize_action_label("", allow_review=False)


def test_shadow_metrics_normalizes_return_label():
    assert normalize_action_label("RETURN") == "RETURN"


def test_audit_treats_unconfirmed_label_as_missing():
    # 미확정 라벨은 감사 대상에서 빠져야 한다 (canonical_field_name 은 소문자로 정규화)
    from src.shadow_validation import audit
    from src.shadow_validation.input_contract import InputRow

    fields = set(InputRow.__dataclass_fields__)
    kwargs = {name: None for name in fields}
    kwargs.update(raw={"action_result": "UNCONFIRMED"})
    row = InputRow(**{k: v for k, v in kwargs.items() if k in fields})
    assert audit._label_action(row) is None

"""Raw-first Loop93 replay bridge with sealed evaluation labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .data_contract import ALLOWED_RAW_FEATURES, FeatureRow, prepare_dataset


AUTHORITATIVE_SOURCE_FILE = "mes_export.xlsx"
SPLITS = ("selector", "lockbox", "audit")
PREDICTION_FIELDS = (
    "predicted",
    "recommendation",
    "confidence",
    "ruleUnit",
    "reason",
    "scope",
    "foldName",
    "foldRole",
    "selectorMonths",
    "selectionEvidenceEndDate",
    "predictionTrainEndDate",
    "selectedVariant",
    "appliedPolicy",
    "residualExperimentEvidenceEligible",
    "policyFamilyParity",
    "selectedWithoutTargetLabels",
)
REPLAY_PARITY_FIELDS = {
    "date": "date",
    "month": "month",
    "customer": "raw_customer",
    "customerKey": "legacy_customer_key",
    "productKey": "product_key",
    "intakeCount": "intake_count",
    "serviceCount": "service_count",
    "treatmentCount": "treatment_count",
    "usageHours": "usage_hours",
    "cumUsageHours": "cumulative_usage_hours",
}
PORTABLE_FALSY_NUMERIC_FIELDS = frozenset(
    {"intakeCount", "serviceCount", "treatmentCount", "usageHours", "cumUsageHours"}
)


class MaterializationError(RuntimeError):
    """Raised when a replay cannot be safely joined to the contract universe."""


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    records_by_split: Mapping[str, tuple[Mapping[str, Any], ...]]
    manifest: Mapping[str, Any]


FallbackPredictor = Callable[[FeatureRow], Mapping[str, Any]]


def materialize_bundle(
    bundle: Any,
    replay_rows_by_split: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    fallback_predictor: FallbackPredictor | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MaterializationResult:
    """Join replay predictions to the immutable contract base by source row only.

    A missing selector or lockbox replay is only predicted through an explicitly
    supplied executable fallback. The frozen Loop93 JSON records the fallback
    choice but not executable rules, so callers must otherwise receive a
    fail-closed record and an incomplete manifest.
    """

    records_by_split: dict[str, tuple[Mapping[str, Any], ...]] = {}
    split_manifest: dict[str, dict[str, Any]] = {}
    complete = True
    for split in SPLITS:
        base_rows = tuple(getattr(bundle, split, ()))
        replay_by_id = _index_replay_rows(replay_rows_by_split.get(split, ()), bundle.metadata.source_dataset_id)
        base_by_id = _index_base_rows(base_rows)
        extras = sorted(set(replay_by_id) - set(base_by_id))
        if extras:
            raise MaterializationError(f"extra replay rows for {split}: {extras[:5]}")
        missing = sorted(set(base_by_id) - set(replay_by_id))
        records: list[Mapping[str, Any]] = []
        not_scored: list[str] = []
        for row in base_rows:
            replay = replay_by_id.get(row.row_id)
            if replay is not None:
                _validate_metadata_parity(row, replay)
                record = _record_from_row(row, replay, prediction_status="replayed")
                if split == "audit":
                    record["scoring_status"] = "scored"
            elif split == "audit":
                record = _record_from_row(row, {}, prediction_status="not_scored")
                record["scoring_status"] = "not_scored"
                not_scored.append(row.row_id)
            elif fallback_predictor is not None:
                fallback = dict(fallback_predictor(row))
                record = _record_from_row(row, fallback, prediction_status="fallback")
            else:
                record = _record_from_row(row, {}, prediction_status="fallback_unproven")
                complete = False
            records.append(record)
        records_by_split[split] = tuple(records)
        split_manifest[split] = {
            "base_rows": len(base_rows),
            "replay_rows": len(replay_by_id),
            "joined_rows": len(base_rows) - len(missing),
            "missing": missing,
            "missing_count": len(missing),
            "extras": extras,
            "extra_count": len(extras),
            "not_scored": not_scored,
            "not_scored_count": len(not_scored),
        }
    metadata = bundle.metadata
    manifest = {
        "schema_version": 1,
        "kind": "loop93_raw_first_materialization",
        "source_dataset_id": metadata.source_dataset_id,
        "source_hashes": dict(sorted(metadata.source_hashes.items())),
        "split_date_ranges": {name: list(metadata.split_date_ranges[name]) for name in SPLITS},
        "identity_diagnostics": {
            "audit_raw_customers": metadata.audit_raw_customers,
            "audit_canonical_customers": metadata.audit_customers,
            "audit_legacy_customers": metadata.audit_legacy_customers,
            "raw_vs_legacy_delta": metadata.audit_raw_customers - metadata.audit_legacy_customers,
        },
        "audit_scope": {
            "date_range": list(metadata.split_date_ranges["audit"]),
            "calendar_year_complete": False,
            "expected_action_rows": metadata.expected_audit_action_rows,
            "expected_not_scored_rows": metadata.expected_audit_non_action_rows,
        },
        "splits": split_manifest,
        "materialization_complete": complete,
        "fallback": {
            "executable_proof_supplied": fallback_predictor is not None,
            "unresolved_selector_lockbox_rows": sum(
                split_manifest[name]["missing_count"] for name in ("selector", "lockbox")
            ) if fallback_predictor is None else 0,
        },
        "provenance": _stable_value(provenance or {}),
    }
    return MaterializationResult(records_by_split=records_by_split, manifest=manifest)


def materialize_current(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build current artifacts from ``prepare_dataset`` without opening raw workbooks."""

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle = prepare_dataset(config_path)
    directories = _current_replay_directories(config)
    replay_rows = {name: _read_replay_directory(directory) for name, directory in directories.items()}
    provenance = {
        "config_sha256": _sha256_file(config_path),
        "frozen_policy": _file_pin(config.get("frozenPolicy")),
        "replay_directories": {name: _directory_pin(directory) for name, directory in directories.items()},
        "fallback_proof": {
            "status": "unresolved",
            "reason": "frozen policy declares baseline_cutoff30 but does not serialize executable rule state",
        },
    }
    result = materialize_bundle(bundle, replay_rows, provenance=provenance)
    return write_materialization(result, output_dir)


def inspect_current_replay_coverage(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Return authoritative current coverage diagnostics without exposing labels."""

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle = prepare_dataset(config_path)
    directories = _current_replay_directories(config)
    diagnostics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        base_rows = tuple(getattr(bundle, split))
        replay_by_id = _index_replay_rows(
            _read_replay_directory(directories[split]), bundle.metadata.source_dataset_id
        )
        base_by_id = _index_base_rows(base_rows)
        extras = sorted(set(replay_by_id) - set(base_by_id))
        missing = sorted(set(base_by_id) - set(replay_by_id))
        if extras:
            raise MaterializationError(f"extra replay rows for {split}: {extras[:5]}")
        diagnostics[split] = {
            "base_rows": len(base_rows),
            "replay_rows": len(replay_by_id),
            "missing_count": len(missing),
            "extra_count": len(extras),
            "not_scored_count": len(missing) if split == "audit" else 0,
            "date_range": list(bundle.metadata.split_date_ranges[split]),
        }
    return diagnostics


def write_materialization(result: MaterializationResult, output_dir: str | Path) -> dict[str, Any]:
    """Write stable JSONL files and a self-describing deterministic manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        payload = "".join(_canonical_json(record) + "\n" for record in result.records_by_split[split])
        path.write_text(payload, encoding="utf-8")
        files[split] = {"path": path.name, "rows": len(result.records_by_split[split]), "sha256": _sha256_bytes(payload.encode("utf-8"))}
    manifest = {**_stable_value(result.manifest), "files": files}
    manifest_sha256 = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
    written_manifest = {**manifest, "manifest_sha256": manifest_sha256}
    (output / "manifest.json").write_text(_canonical_json(written_manifest) + "\n", encoding="utf-8")
    return {"manifest_sha256": manifest_sha256, "files": files, "materialization_complete": manifest["materialization_complete"]}


def _record_from_row(
    row: FeatureRow, replay: Mapping[str, Any], *, prediction_status: str
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "row_id": row.row_id,
        "source_row": row.source_row,
        "date": row.date,
        "month": row.month,
        "raw_customer": row.raw_customer,
        "customer_key": row.customer_key,
        "legacy_customer_key": row.legacy_customer_key,
        "product_key": row.product_key,
        "episode_id": row.episode_id,
        "episode_basis": row.episode_basis,
        "features": {name: row.features.get(name, "") for name in ALLOWED_RAW_FEATURES},
        "prediction_status": prediction_status,
    }
    for name in PREDICTION_FIELDS:
        if name in replay:
            record[name] = replay[name]
    return record


def _index_base_rows(rows: Iterable[FeatureRow]) -> dict[str, FeatureRow]:
    indexed: dict[str, FeatureRow] = {}
    for row in rows:
        if row.row_id in indexed:
            raise MaterializationError(f"duplicate authoritative row id: {row.row_id}")
        indexed[row.row_id] = row
    return indexed


def _index_replay_rows(
    rows: Iterable[Mapping[str, Any]], source_dataset_id: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("source_file") or "") != AUTHORITATIVE_SOURCE_FILE:
            raise MaterializationError(f"unexpected replay source_file: {row.get('source_file')!r}")
        source_row = _positive_source_row(row.get("source_row"))
        row_id = f"{source_dataset_id}:{source_row}"
        if row_id in indexed:
            raise MaterializationError(f"duplicate replay source row: {row_id}")
        indexed[row_id] = row
    return indexed


def _validate_metadata_parity(row: FeatureRow, replay: Mapping[str, Any]) -> None:
    for replay_name, contract_name in REPLAY_PARITY_FIELDS.items():
        expected = getattr(row, contract_name) if hasattr(row, contract_name) else row.features.get(contract_name, "")
        observed_value = _portable_parity_value(replay_name, replay.get(replay_name))
        expected_value = _portable_parity_value(replay_name, expected)
        if observed_value != expected_value:
            raise MaterializationError(
                f"replay metadata mismatch for {row.row_id}: {replay_name} "
                f"expected={expected!r} observed={replay.get(replay_name)!r}"
            )


def _current_replay_directories(config: Mapping[str, Any]) -> dict[str, Path]:
    source_root = Path(str(config["sourceRoot"])).resolve()
    policy_root = source_root / "outputs/company_policy_search_20260709"
    directories = {
        "selector": policy_root / "loop97_clean_risk_bounds/prepared_selector_oct_nov",
        "lockbox": policy_root / "loop97_clean_risk_bounds/prepared_lockbox_dec",
        "audit": policy_root / "replays/loop93_current_policy_crossfit_2026_rows",
    }
    for name, path in directories.items():
        if not path.is_dir():
            raise MaterializationError(f"{name} replay directory is missing: {path}")
    return directories


def _read_replay_directory(directory: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for path in sorted(directory.glob("*_predicted_rows.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MaterializationError(f"invalid replay JSONL at {path}:{line_number}") from error
                if not isinstance(value, dict):
                    raise MaterializationError(f"replay row is not an object at {path}:{line_number}")
                rows.append(value)
    return tuple(rows)


def _directory_pin(directory: Path) -> dict[str, Any]:
    files = []
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*_predicted_rows.jsonl")):
        content_hash = _sha256_file(path)
        files.append({"name": path.name, "sha256": content_hash, "size_bytes": path.stat().st_size})
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return {"file_count": len(files), "sha256": digest.hexdigest(), "files": files}


def _file_pin(value: Any) -> dict[str, Any]:
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise MaterializationError(f"required provenance file is missing: {path}")
    return {"name": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _positive_source_row(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as error:
        raise MaterializationError(f"invalid replay source_row: {value!r}") from error
    if number < 1:
        raise MaterializationError(f"invalid replay source_row: {value!r}")
    return number


def _normalized_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).split())


def _portable_parity_value(replay_name: str, value: Any) -> str:
    # Loop93 portableRow serialized these five numeric fields with `value || ""`.
    if replay_name in PORTABLE_FALSY_NUMERIC_FIELDS and isinstance(value, (int, float)) and not value:
        return ""
    normalized = _normalized_value(value)
    if replay_name == "productKey":
        # data_contract compacts and uppercases Clean코드/Part코드 before product_key construction.
        return "".join(normalized.split()).upper()
    return normalized


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_stable_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

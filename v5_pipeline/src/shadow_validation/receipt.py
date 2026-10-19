from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .input_contract import POST_DECISION_FIELDS, canonical_field_name
from .integrity import sha256_file


PREDICTIONS_FILE = "predictions.csv"
FEATURES_FILE = "features.jsonl.gz"
RECEIPT_FILE = "prediction_receipt.json"
RUN_LOG_FILE = "run.log"
OUTPUT_FILES = frozenset({PREDICTIONS_FILE, FEATURES_FILE, RUN_LOG_FILE})
REQUIRED_INPUT_ARTIFACTS = frozenset({"frozenManifest", "frozenGeneration"})
_FORBIDDEN_FIELDS = frozenset(canonical_field_name(name) for name in POST_DECISION_FIELDS)


class PredictionReceiptError(RuntimeError):
    """Raised when a sealed prediction run is incomplete or has been altered."""


def seal_prediction_run(
    run_dir: str | Path,
    *,
    predictions: Iterable[Mapping[str, Any]],
    feature_rows: Iterable[Mapping[str, Any]],
    input_artifacts: Mapping[str, str | Path],
    generation_id: str,
    mode: str,
    run_log: str,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write a complete prediction run and atomically publish its receipt."""

    target = Path(run_dir).expanduser().resolve()
    if not generation_id.strip():
        raise PredictionReceiptError("generation_id is required")
    if mode not in {"prospective", "dry_run"}:
        raise PredictionReceiptError(f"unsupported prediction mode: {mode}")
    if target.exists():
        raise PredictionReceiptError(f"prediction run already exists: {target}")
    sealed_at = _sealed_timestamp(sealed_at_utc)

    prepared_predictions = _prepare_predictions(predictions)
    prepared_features, feature_hashes = _prepare_features(feature_rows)
    sorted_row_ids = sorted(prepared_predictions)
    if sorted(prepared_features) != sorted_row_ids:
        raise PredictionReceiptError("prediction and feature row universes differ")
    prepared_inputs = _prepare_input_hashes(input_artifacts)
    missing_inputs = REQUIRED_INPUT_ARTIFACTS - set(prepared_inputs)
    if missing_inputs:
        raise PredictionReceiptError(
            "required frozen input artifacts are missing: " + ", ".join(sorted(missing_inputs))
        )
    commitment = {
        "generationId": generation_id,
        "mode": mode,
        "sealedAtUtc": sealed_at,
        "sortedRowIds": sorted_row_ids,
        "featureHashes": {row_id: feature_hashes[row_id] for row_id in sorted_row_ids},
        "inputHashes": prepared_inputs,
    }
    commitment_sha = _canonical_sha256(commitment)
    for row in prepared_predictions.values():
        if "predictionReceiptSha256" in row:
            raise PredictionReceiptError("predictionReceiptSha256 is a reserved field")
        row["predictionReceiptSha256"] = commitment_sha

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise PredictionReceiptError(f"prediction run is already being created: {target}") from error
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        if target.exists():
            raise PredictionReceiptError(f"prediction run already exists: {target}")
        temporary.mkdir()
        _write_predictions(temporary / PREDICTIONS_FILE, prepared_predictions)
        _write_features(temporary / FEATURES_FILE, prepared_features)
        _write_text(temporary / RUN_LOG_FILE, run_log)
        output_hashes = {
            name: sha256_file(temporary / name)
            for name in sorted(OUTPUT_FILES)
        }
        receipt = {
            "schemaVersion": 2,
            "generationId": generation_id,
            "mode": mode,
            "sealedAtUtc": sealed_at,
            "integrityLevel": "local_tamper_evident",
            "externalTimestampAuthority": None,
            "rowCount": len(sorted_row_ids),
            "sortedRowIds": sorted_row_ids,
            "featureHashes": {row_id: feature_hashes[row_id] for row_id in sorted_row_ids},
            "inputHashes": prepared_inputs,
            "outputHashes": output_hashes,
            "predictionCommitmentSha256": commitment_sha,
        }
        receipt["receiptSha256"] = _canonical_sha256(receipt)
        _write_json(temporary / RECEIPT_FILE, receipt)
        _fsync_directory(temporary)
        os.rename(temporary, target)
        _fsync_directory(target.parent)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def verify_prediction_receipt(run_dir: str | Path) -> dict[str, Any]:
    """Verify a published run before outcomes or labels are opened."""

    root = Path(run_dir).expanduser().resolve()
    receipt_path = root / RECEIPT_FILE
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionReceiptError("prediction receipt is unreadable") from error
    if not isinstance(receipt, dict):
        raise PredictionReceiptError("prediction receipt must be a JSON object")

    expected_receipt_hash = str(receipt.get("receiptSha256") or "")
    body = {key: value for key, value in receipt.items() if key != "receiptSha256"}
    if expected_receipt_hash != _canonical_sha256(body):
        raise PredictionReceiptError("prediction receipt hash mismatch")

    if receipt.get("integrityLevel") != "local_tamper_evident":
        raise PredictionReceiptError("prediction receipt integrity level is invalid")
    _sealed_timestamp(str(receipt.get("sealedAtUtc") or ""))

    sorted_row_ids = _receipt_row_ids(receipt)
    input_hashes = receipt.get("inputHashes")
    if not isinstance(input_hashes, Mapping) or not REQUIRED_INPUT_ARTIFACTS.issubset(
        input_hashes
    ):
        raise PredictionReceiptError("prediction receipt is missing frozen inputs")
    _verify_input_hashes(input_hashes)
    _verify_output_hashes(root, receipt.get("outputHashes"))
    expected_commitment = _canonical_sha256(
        {
            "generationId": receipt.get("generationId"),
            "mode": receipt.get("mode"),
            "sealedAtUtc": receipt.get("sealedAtUtc"),
            "sortedRowIds": sorted_row_ids,
            "featureHashes": receipt.get("featureHashes"),
            "inputHashes": input_hashes,
        }
    )
    if expected_commitment != receipt.get("predictionCommitmentSha256"):
        raise PredictionReceiptError("prediction commitment hash mismatch")
    _verify_prediction_rows(
        root / PREDICTIONS_FILE,
        sorted_row_ids,
        expected_commitment,
    )
    _verify_feature_rows(
        root / FEATURES_FILE,
        sorted_row_ids,
        receipt.get("featureHashes"),
    )
    return receipt


def _prepare_predictions(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise PredictionReceiptError("prediction rows must be mappings")
        _reject_forbidden_fields(raw)
        row_id = str(raw.get("rowId") or "")
        if not row_id or row_id in output:
            raise PredictionReceiptError("prediction rows require unique rowId values")
        output[row_id] = {str(key): value for key, value in raw.items()}
    if not output:
        raise PredictionReceiptError("prediction run has no rows")
    return output


def _prepare_features(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    output: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise PredictionReceiptError("feature rows must be mappings")
        row_id = str(raw.get("rowId") or "")
        features = raw.get("features")
        if not row_id or row_id in output or not isinstance(features, Mapping):
            raise PredictionReceiptError("feature rows require unique rowId and feature mappings")
        _reject_forbidden_fields(features)
        normalized = {str(key): value for key, value in features.items()}
        output[row_id] = {"rowId": row_id, "features": normalized}
        hashes[row_id] = _canonical_sha256(normalized)
    if not output:
        raise PredictionReceiptError("prediction run has no feature rows")
    return output, hashes


def _prepare_input_hashes(
    input_artifacts: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    if not input_artifacts:
        raise PredictionReceiptError("input artifacts are required")
    output: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(input_artifacts.items()):
        key = str(name)
        path = Path(raw_path).expanduser().resolve()
        if not key or key in output or not path.is_file():
            raise PredictionReceiptError("input artifacts must have unique names and existing files")
        stat = path.stat()
        output[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sizeBytes": stat.st_size,
            "modifiedTimeNs": stat.st_mtime_ns,
        }
    return output


def _write_predictions(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    fieldnames = ["rowId"] + sorted(
        {key for row in rows.values() for key in row if key != "rowId"}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_id in sorted(rows):
            writer.writerow({key: _csv_value(rows[row_id].get(key)) for key in fieldnames})
        _fsync_file_handle(handle)


def _write_features(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> None:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as handle:
            for row_id in sorted(rows):
                handle.write(_canonical_json_bytes(rows[row_id]) + b"\n")
        _fsync_file_handle(raw_handle)


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        _fsync_file_handle(handle)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, _canonical_json_bytes(value).decode("utf-8") + "\n")


def _verify_input_hashes(raw: Any) -> None:
    if not isinstance(raw, Mapping) or not raw:
        raise PredictionReceiptError("prediction receipt has no input hashes")
    for name, entry in raw.items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise PredictionReceiptError("prediction receipt input hashes are malformed")
        path = Path(str(entry.get("path") or ""))
        expected = str(entry.get("sha256") or "")
        if not path.is_file():
            raise PredictionReceiptError(f"input artifact hash mismatch: {name}")
        stat = path.stat()
        if (
            sha256_file(path) != expected
            or stat.st_size != entry.get("sizeBytes")
            or stat.st_mtime_ns != entry.get("modifiedTimeNs")
        ):
            raise PredictionReceiptError(f"input artifact hash mismatch: {name}")


def _verify_output_hashes(root: Path, raw: Any) -> None:
    if not isinstance(raw, Mapping) or set(raw) != OUTPUT_FILES:
        raise PredictionReceiptError("prediction receipt output hashes are malformed")
    for name in sorted(OUTPUT_FILES):
        path = root / name
        if not path.is_file() or sha256_file(path) != str(raw[name] or ""):
            raise PredictionReceiptError(f"output artifact hash mismatch: {name}")


def _verify_prediction_rows(
    path: Path,
    expected_ids: list[str],
    expected_commitment: str,
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise PredictionReceiptError("predictions CSV has no header")
            _reject_forbidden_fields({name: "" for name in reader.fieldnames})
            rows = list(reader)
            row_ids = [str(row.get("rowId") or "") for row in rows]
    except (OSError, csv.Error) as error:
        raise PredictionReceiptError("predictions CSV is unreadable") from error
    if row_ids != expected_ids:
        raise PredictionReceiptError("prediction CSV row IDs differ from receipt")
    if any(
        str(row.get("predictionReceiptSha256") or "") != expected_commitment
        for row in rows
    ):
        raise PredictionReceiptError("prediction CSV commitment differs from receipt")


def _verify_feature_rows(path: Path, expected_ids: list[str], raw_hashes: Any) -> None:
    if not isinstance(raw_hashes, Mapping):
        raise PredictionReceiptError("prediction receipt feature hashes are malformed")
    observed_ids: list[str] = []
    observed_hashes: dict[str, str] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping) or not isinstance(raw.get("features"), Mapping):
                    raise PredictionReceiptError("sealed feature row is malformed")
                row_id = str(raw.get("rowId") or "")
                if not row_id or row_id in observed_hashes:
                    raise PredictionReceiptError("sealed feature row IDs are invalid")
                features = raw["features"]
                _reject_forbidden_fields(features)
                observed_ids.append(row_id)
                observed_hashes[row_id] = _canonical_sha256(features)
    except (OSError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise PredictionReceiptError("sealed features are unreadable") from error
    expected_hashes = {str(key): str(value) for key, value in raw_hashes.items()}
    if observed_ids != expected_ids or observed_hashes != expected_hashes:
        raise PredictionReceiptError("sealed features differ from receipt")


def _receipt_row_ids(receipt: Mapping[str, Any]) -> list[str]:
    raw = receipt.get("sortedRowIds")
    if not isinstance(raw, list) or not raw or any(not isinstance(value, str) for value in raw):
        raise PredictionReceiptError("prediction receipt row IDs are malformed")
    if raw != sorted(raw) or len(raw) != len(set(raw)):
        raise PredictionReceiptError("prediction receipt row IDs are not sorted and unique")
    if receipt.get("rowCount") != len(raw):
        raise PredictionReceiptError("prediction receipt row count differs from row IDs")
    return raw


def _reject_forbidden_fields(value: Mapping[str, Any]) -> None:
    for key, nested in value.items():
        if canonical_field_name(key) in _FORBIDDEN_FIELDS:
            raise PredictionReceiptError(f"forbidden postdecision field: {key}")
        if isinstance(nested, Mapping):
            _reject_forbidden_fields(nested)
        elif isinstance(nested, (list, tuple)):
            for item in nested:
                if isinstance(item, Mapping):
                    _reject_forbidden_fields(item)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PredictionReceiptError("sealed payload is not JSON serializable") from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sealed_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise PredictionReceiptError("sealed_at_utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise PredictionReceiptError("sealed_at_utc must be UTC")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    return _canonical_json_bytes(value).decode("utf-8")


def _fsync_file_handle(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

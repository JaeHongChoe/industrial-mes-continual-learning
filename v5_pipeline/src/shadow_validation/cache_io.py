from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = b"MES_DERIVED_CACHE_V1\n"
MAX_HEADER_BYTES = 64 * 1024
COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CacheFingerprint:
    schema_version: int
    generation_id: str
    manifest_sha256: str
    python_version: str
    runtime_versions: tuple[tuple[str, str], ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "schemaVersion": int(self.schema_version),
            "generationId": str(self.generation_id),
            "manifestSha256": str(self.manifest_sha256),
            "pythonVersion": str(self.python_version),
            "runtimeVersions": [
                [str(name), str(version)]
                for name, version in self.runtime_versions
            ],
        }


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_cache(
    path: Path,
    expected: CacheFingerprint,
    *,
    expected_kind: str,
) -> object | None:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            if handle.readline(len(MAGIC) + 1) != MAGIC:
                return None
            header_line = handle.readline(MAX_HEADER_BYTES + 1)
            if (
                not header_line.endswith(b"\n")
                or len(header_line) > MAX_HEADER_BYTES
            ):
                return None
            header = json.loads(header_line)
            if not isinstance(header, dict):
                return None
            if header.get("kind") != expected_kind:
                return None
            if header.get("fingerprint") != expected.as_record():
                return None
            expected_digest = str(header.get("payloadSha256") or "")
            if len(expected_digest) != 64:
                return None

            payload_offset = handle.tell()
            observed_digest = _sha256_stream(handle)
            if not hmac.compare_digest(observed_digest, expected_digest):
                return None

            handle.seek(payload_offset)
            payload = pickle.load(handle)
            if handle.read(1):
                return None
            return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError, pickle.PickleError, EOFError):
        return None


def store_cache(
    path: Path,
    fingerprint: CacheFingerprint,
    *,
    kind: str,
    payload: object,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_path: Path | None = None
    final_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".payload.tmp",
            dir=destination.parent,
            delete=False,
        ) as payload_handle:
            payload_path = Path(payload_handle.name)
            pickle.dump(payload, payload_handle, protocol=pickle.HIGHEST_PROTOCOL)
            payload_handle.flush()
            os.fsync(payload_handle.fileno())
            payload_handle.seek(0)
            payload_sha256 = _sha256_stream(payload_handle)

        header = {
            "kind": str(kind),
            "fingerprint": fingerprint.as_record(),
            "payloadSha256": payload_sha256,
        }
        header_bytes = json.dumps(
            header,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"

        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".cache.tmp",
            dir=destination.parent,
            delete=False,
        ) as final_handle:
            final_path = Path(final_handle.name)
            final_handle.write(MAGIC)
            final_handle.write(header_bytes)
            with payload_path.open("rb") as payload_handle:
                shutil.copyfileobj(
                    payload_handle,
                    final_handle,
                    length=COPY_CHUNK_BYTES,
                )
            final_handle.flush()
            os.fsync(final_handle.fileno())

        os.replace(final_path, destination)
        final_path = None
    finally:
        if payload_path is not None:
            payload_path.unlink(missing_ok=True)
        if final_path is not None:
            final_path.unlink(missing_ok=True)

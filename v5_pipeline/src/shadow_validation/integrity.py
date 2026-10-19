from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class IntegrityError(RuntimeError):
    """Raised when a frozen or sealed artifact no longer matches its receipt."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    expected_generation: str,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"동결 manifest를 읽을 수 없습니다: {manifest_path}") from error
    if payload.get("schemaVersion") != 1:
        raise IntegrityError("지원하지 않는 동결 manifest schemaVersion입니다")
    if str(payload.get("generationId") or "") != expected_generation:
        raise IntegrityError("동결 세대 ID가 실행 세대와 다릅니다")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise IntegrityError("동결 manifest에 artifacts가 없습니다")
    seen: set[str] = set()
    for entry in artifacts:
        relative = str(entry.get("relativePath") or "")
        if not relative or relative in seen:
            raise IntegrityError("동결 artifact 상대 경로가 비어 있거나 중복되었습니다")
        seen.add(relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise IntegrityError(f"동결 artifact relative 경로가 패키지를 벗어납니다: {relative}") from error
        if not candidate.is_file():
            raise IntegrityError(f"동결 artifact가 없습니다: {relative}")
        expected_size = entry.get("sizeBytes")
        if not isinstance(expected_size, int) or candidate.stat().st_size != expected_size:
            raise IntegrityError(f"동결 artifact size 불일치: {relative}")
        expected_sha = str(entry.get("sha256") or "").lower()
        if not _SHA256.fullmatch(expected_sha):
            raise IntegrityError(f"동결 artifact SHA-256 형식 오류: {relative}")
        observed_sha = sha256_file(candidate)
        if observed_sha != expected_sha:
            raise IntegrityError(f"동결 artifact SHA-256 불일치: {relative}")

    return {
        **payload,
        "verifiedArtifactCount": len(artifacts),
        "manifestSha256": sha256_file(manifest_path),
    }


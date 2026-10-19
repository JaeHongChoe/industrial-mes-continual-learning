from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .feature_bridge import sanitized_raw_rows
from .input_contract import InputRow


import shutil

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"


def find_node_binary(binary: str = "node") -> str:
    if binary and binary != "node" and shutil.which(binary):
        return binary
    found = shutil.which("node")
    if found:
        return found
    venv_node = PACKAGE_ROOT.parents[1] / ".venv/bin/node"
    if venv_node.is_file() and os.access(venv_node, os.X_OK):
        return str(venv_node)
    for path in (
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return binary or "node"


@dataclass(frozen=True, slots=True)
class Loop93Prediction:
    row_id: str
    prediction: str
    selected_variant: str
    rule_unit: str
    reason: str
    confidence: str


def _predictions_from_items(
    items: Any,
    *,
    expected_row_ids: set[str],
) -> Mapping[str, Loop93Prediction]:
    if not isinstance(items, list):
        raise RuntimeError("Loop93 worker predictions are not a list")
    output: dict[str, Loop93Prediction] = {}
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("Loop93 worker prediction is not an object")
        row_id = str(item.get("rowId") or "")
        if not row_id or row_id in output:
            raise RuntimeError(f"duplicate Loop93 output row: {row_id}")
        output[row_id] = Loop93Prediction(
            row_id=row_id,
            prediction=str(item.get("prediction") or ""),
            selected_variant=str(item.get("selectedVariant") or ""),
            rule_unit=str(item.get("ruleUnit") or ""),
            reason=str(item.get("reason") or ""),
            confidence=str(item.get("confidence") or ""),
        )
    if set(output) != expected_row_ids:
        raise RuntimeError(
            "Loop93 row universe mismatch: "
            f"expected={len(expected_row_ids)} observed={len(output)}"
        )
    return output


class Loop93WorkerClient:
    def __init__(
        self,
        *,
        engine_path: Path,
        policy_path: Path,
        history_path: Path,
        script_path: Path,
        node_binary: str = "node",
        timeout_seconds: float = 300.0,
        process_factory: Callable[..., Any] = subprocess.Popen,
        fallback: Callable[
            [Sequence[InputRow]], Mapping[str, Loop93Prediction]
        ] | None = None,
    ) -> None:
        self.engine_path = Path(engine_path).resolve()
        self.policy_path = Path(policy_path).resolve()
        self.history_path = Path(history_path).resolve()
        self.script_path = Path(script_path).resolve()
        self.node_binary = find_node_binary(node_binary)
        self.timeout_seconds = float(timeout_seconds)
        self._process_factory = process_factory
        self._fallback = fallback or (
            lambda rows: _run_loop93_one_shot(
                rows,
                history_path=self.history_path,
                node_binary=self.node_binary,
            )
        )
        self._process: Any | None = None
        self._lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=50)

    def _start_process(self) -> Any:
        process = self._process_factory(
            (
                self.node_binary,
                str(self.script_path),
                "--engine",
                str(self.engine_path),
                "--policy",
                str(self.policy_path),
                "--history",
                str(self.history_path),
                "--serve",
                "1",
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "NODE_OPTIONS": os.environ.get(
                    "NODE_OPTIONS",
                    "--max-old-space-size=12288",
                ),
            },
        )
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Loop93 worker pipes did not initialize")
        self._process = process
        if process.stderr is not None:
            threading.Thread(
                target=self._drain_stderr,
                args=(process,),
                daemon=True,
            ).start()
        return process

    def _drain_stderr(self, process: Any) -> None:
        try:
            for line in process.stderr:
                if line.strip():
                    self._stderr_tail.append(line.rstrip())
        except (OSError, ValueError):
            return

    def _ensure_process(self) -> Any:
        if self._process is None or self._process.poll() is not None:
            return self._start_process()
        return self._process

    def _read_response_line(self, process: Any) -> str:
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            raise TimeoutError(
                f"Loop93 worker response exceeded {self.timeout_seconds:g}s"
            )
        line = process.stdout.readline()
        if not line:
            detail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                "Loop93 worker exited before responding"
                + (f": {detail}" if detail else "")
            )
        return line

    def _request_once(
        self,
        rows: Sequence[InputRow],
    ) -> Mapping[str, Loop93Prediction]:
        process = self._ensure_process()
        request_id = uuid.uuid4().hex
        request = {
            "requestId": request_id,
            "rows": sanitized_raw_rows(rows),
        }
        process.stdin.write(
            json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        process.stdin.flush()
        try:
            response = json.loads(self._read_response_line(process))
        except json.JSONDecodeError as error:
            raise RuntimeError("Loop93 worker returned malformed JSON") from error
        if not isinstance(response, dict):
            raise RuntimeError("Loop93 worker response is not an object")
        if str(response.get("requestId") or "") != request_id:
            raise RuntimeError("Loop93 worker response request ID mismatch")
        if response.get("ok") is not True:
            raise RuntimeError(
                str(response.get("error") or "Loop93 worker request failed")
            )
        return _predictions_from_items(
            response.get("predictions"),
            expected_row_ids={row.row_id for row in rows},
        )

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def predict(
        self,
        rows: Sequence[InputRow],
    ) -> Mapping[str, Loop93Prediction]:
        if not rows:
            return {}
        with self._lock:
            for _attempt in range(2):
                try:
                    return self._request_once(rows)
                except (OSError, RuntimeError, TimeoutError, ValueError):
                    self._stop_process()
            return self._fallback(rows)

    def close(self) -> None:
        with self._lock:
            self._stop_process()


def _run_loop93_one_shot(
    rows: Sequence[InputRow],
    *,
    history_path: str | Path | None = None,
    node_binary: str = "node",
) -> Mapping[str, Loop93Prediction]:
    if not rows:
        return {}
    resolved_node = find_node_binary(node_binary)
    history = Path(history_path or FROZEN_ROOT / "history/history_rows.jsonl.gz").resolve()
    script = PACKAGE_ROOT / "scripts/run_frozen_baseline.mjs"
    engine = FROZEN_ROOT / "loop93/rule_miner.browser.js"
    policy = FROZEN_ROOT / "loop93/frozen_policy.json"
    with tempfile.TemporaryDirectory(prefix="mes-shadow-loop93-") as temporary:
        temp_root = Path(temporary)
        input_path = temp_root / "future.jsonl"
        output_path = temp_root / "predictions.jsonl"
        input_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                for item in sanitized_raw_rows(rows)
            ),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment.setdefault("NODE_OPTIONS", "--max-old-space-size=12288")
        completed = subprocess.run(
            (
                resolved_node,
                str(script),
                "--engine",
                str(engine),
                "--policy",
                str(policy),
                "--history",
                str(history),
                "--future",
                str(input_path),
                "--output",
                str(output_path),
            ),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Loop93 frozen adapter failed: {detail}")
        output_items: list[dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                output_items.append(item)
    return _predictions_from_items(
        output_items,
        expected_row_ids={row.row_id for row in rows},
    )


def run_loop93_baseline(
    rows: Sequence[InputRow],
    *,
    history_path: str | Path | None = None,
    node_binary: str = "node",
    worker: Loop93WorkerClient | None = None,
) -> Mapping[str, Loop93Prediction]:
    if worker is not None:
        return worker.predict(rows)
    return _run_loop93_one_shot(
        rows,
        history_path=history_path,
        node_binary=node_binary,
    )

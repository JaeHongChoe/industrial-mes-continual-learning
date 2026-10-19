"""Shared setup for the v5_pipeline synthetic smoke tests.

Import-path rule: the package root ``v5_pipeline`` goes FIRST on ``sys.path`` so
that ``src.*``, ``src.shadow_validation.*`` and ``experiment_lib.*`` resolve, and
``v5_pipeline/src`` is APPENDED LAST. Inserting ``src`` first would let
``src/select.py`` shadow the stdlib ``select`` module and break scipy.

This conftest lives in ``tests/v5`` so it is only loaded when that directory is
collected; the legacy ``mes_cl`` tests in ``tests/`` do not depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Do not leave __pycache__ folders inside v5_pipeline/.
sys.dont_write_bytecode = True

_V5_ROOT = Path(__file__).resolve().parents[2] / "v5_pipeline"
_V5_SRC = _V5_ROOT / "src"

if str(_V5_ROOT) in sys.path:
    sys.path.remove(str(_V5_ROOT))
sys.path.insert(0, str(_V5_ROOT))
if str(_V5_SRC) not in sys.path:
    sys.path.append(str(_V5_SRC))

SOURCE_DATASET_ID = "synthetic_ds"


def build_raw(
    n: int,
    day: str,
    *,
    customer: str = "customer_a",
    line: str = "L1",
    process_code: str = "PC01",
    part_code: str = "PT01",
    type_: str = "",
    symptom: str = "",
    detail: str = "",
    serial: str | None = None,
    label: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "__sourceRowNumber": n,
        "occurrence_date": day,
        "registered_date": day,
        "customer": customer,
        "line": line,
        "process_code": process_code,
        "part_code": part_code,
        "type": type_,
        "symptom": symptom,
        "detail": detail,
        "serial": serial if serial is not None else f"SN{n:04d}",
    }
    if label is not None:
        raw["action_result"] = label
    raw.update(extra)
    return raw


@pytest.fixture
def make_row():
    """Factory: build a FeatureRow through the real raw -> FeatureRow contract."""

    from src.data_contract import _to_feature_row

    def _make(n: int, day: str, **kwargs: Any):
        return _to_feature_row(build_raw(n, day, **kwargs), SOURCE_DATASET_ID)

    return _make


@pytest.fixture
def raw_factory():
    return build_raw

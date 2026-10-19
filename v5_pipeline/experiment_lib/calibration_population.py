from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SUPPORTED_CALIBRATION_POPULATIONS = frozenset(
    {"baseline_process", "all_actions"}
)


def require_calibration_population(value: object) -> str:
    population = str(value)
    if population not in SUPPORTED_CALIBRATION_POPULATIONS:
        raise ValueError(
            "calibration population must be baseline_process or all_actions"
        )
    return population


def calibration_population_from_policy(policy: Mapping[str, Any]) -> str:
    if "calibrationPopulation" not in policy:
        raise ValueError("selected policy is missing calibrationPopulation")
    return require_calibration_population(policy["calibrationPopulation"])


def validate_calibration_receipts(
    receipts: Mapping[str, Any],
    *,
    expected: str,
    expected_keys: Iterable[str] | None = None,
) -> None:
    population = require_calibration_population(expected)
    if not receipts:
        raise ValueError("fit receipts cannot be empty")
    if expected_keys is not None:
        required = {str(key) for key in expected_keys}
        observed = set(receipts)
        if observed != required:
            missing = ",".join(sorted(required - observed)) or "<none>"
            extra = ",".join(sorted(observed - required)) or "<none>"
            raise ValueError(
                "fit receipt key universe differs: "
                f"missing={missing}; extra={extra}"
            )
    for key, receipt in receipts.items():
        if not isinstance(receipt, Mapping):
            raise ValueError(f"fit receipt must be a mapping: {key}")
        if receipt.get("calibrationPopulation") != population:
            raise ValueError(
                f"fit receipt calibration population differs: {key}"
            )

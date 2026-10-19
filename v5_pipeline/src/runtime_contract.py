from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


class RuntimeContractError(RuntimeError):
    """Raised before deserialization when a persisted model runtime does not match."""


REQUIRED_PACKAGES = ("lightgbm", "numpy", "scikit-learn", "scipy")


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    python: str
    packages: Mapping[str, str]

    @classmethod
    def capture(cls) -> "RuntimeContract":
        return cls(
            python=f"{sys.version_info.major}.{sys.version_info.minor}",
            packages=MappingProxyType(
                {name: importlib.metadata.version(name) for name in REQUIRED_PACKAGES}
            ),
        )

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "python": self.python,
                "packages": dict(sorted(self.packages.items())),
            }
        )


def validate_runtime_contract(expected: Mapping[str, Any]) -> None:
    if not isinstance(expected, Mapping):
        raise RuntimeContractError("runtime contract is missing or malformed")
    expected_python = str(expected.get("python") or "")
    expected_packages = expected.get("packages")
    if not isinstance(expected_packages, Mapping):
        raise RuntimeContractError("runtime package contract is missing or malformed")
    if set(expected_packages) != set(REQUIRED_PACKAGES):
        raise RuntimeContractError(
            "runtime package contract does not match required packages: "
            f"expected={sorted(REQUIRED_PACKAGES)} observed={sorted(expected_packages)}"
        )

    actual = RuntimeContract.capture()
    mismatches: dict[str, tuple[str, str]] = {}
    if expected_python != actual.python:
        mismatches["python"] = (expected_python, actual.python)
    for name in REQUIRED_PACKAGES:
        expected_version = str(expected_packages[name])
        actual_version = str(actual.packages[name])
        if expected_version != actual_version:
            mismatches[name] = (expected_version, actual_version)
    if mismatches:
        raise RuntimeContractError(f"runtime dependency mismatch: {mismatches}")

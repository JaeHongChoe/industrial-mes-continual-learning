from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_BLANK_LINES = frozenset({"", "-", "N/A", "NA", "NONE", "NULL", "미기재", "(BLANK)"})
_DASH_RE = re.compile(r"[_‐‑‒–—−]+")
_SPACE_RE = re.compile(r"\s+")
_DELIMITER_SPACE_RE = re.compile(r"\s*-\s*")


def normalize_line_name(value: Any) -> str:
    """Normalize formatting noise without discarding the source line identity."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _DASH_RE.sub("-", text)
    text = _SPACE_RE.sub(" ", text).strip()
    text = _DELIMITER_SPACE_RE.sub("-", text)
    return "" if text.upper() in _BLANK_LINES else text


def canonical_line_key(value: Any) -> str:
    """Return the deterministic line key used inside a company boundary."""

    return normalize_line_name(value).upper()


@dataclass(frozen=True, slots=True)
class ScopeKey:
    company: str
    line: str = ""

    def __post_init__(self) -> None:
        company = _SPACE_RE.sub(
            " ", unicodedata.normalize("NFKC", str(self.company or ""))
        ).strip()
        if not company:
            raise ValueError("company scope cannot be empty")
        object.__setattr__(self, "company", company)
        object.__setattr__(self, "line", canonical_line_key(self.line))


def scope_chain(key: ScopeKey) -> tuple[str, ...]:
    company_scope = f"company:{key.company}"
    if not key.line:
        return company_scope, "global"
    return f"company_line:{key.company}:{key.line}", company_scope, "global"


def line_collision_report(rows: Iterable[Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    """List raw line labels that collapse to one key within the same company."""

    labels: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        company = str(getattr(row, "raw_customer", "") or "").strip()
        features = getattr(row, "features", {})
        raw_line = features.get("line", "") if isinstance(features, Mapping) else ""
        source_label = _SPACE_RE.sub(
            " ", unicodedata.normalize("NFKC", str(raw_line or ""))
        ).strip()
        normalized = normalize_line_name(source_label)
        canonical = canonical_line_key(normalized)
        if company and canonical:
            labels[company][canonical].add(source_label)

    return {
        company: {
            canonical: tuple(sorted(raw_values))
            for canonical, raw_values in sorted(company_lines.items())
            if len(raw_values) > 1
        }
        for company, company_lines in sorted(labels.items())
        if any(len(raw_values) > 1 for raw_values in company_lines.values())
    }

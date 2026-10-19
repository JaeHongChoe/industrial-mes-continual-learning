"""Frozen, demotion-only semantic guards for intake-time fields.

The groups in this module are intentionally small and auditable.  They were
ported from semantic guards that pre-date this hierarchical experiment.  A
group can only turn an automatic ``PROCESS`` prediction into ``REVIEW``; it can
never create or change an automatic action label.  Company activation is kept
outside the group definitions so that activation can be selected on 2025 data
without editing regular expressions after the 2026 audit is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data_contract import FeatureRow


@dataclass(frozen=True, slots=True)
class SemanticGroup:
    group_id: str
    description: str
    pattern: str
    fields: tuple[str, ...] = ("type", "symptom", "detail")


@dataclass(frozen=True, slots=True)
class SemanticVetoResult:
    predictions: Mapping[str, str]
    demoted_rows: tuple[str, ...]
    matched_groups_by_row: Mapping[str, tuple[str, ...]]


# 의미 그룹 정의. group_id 와 구조는 원본과 같고, 패턴 어휘는 공개용 예시로 바꿨다.
_GROUPS = (
    SemanticGroup(
        "direct_repair_request",
        "수리 요청 또는 repair 지시 직접 문구",
        r"(?:\bREPAIR\b|수\s*리\s*(?:요청|필요|진행|대상|처리)|재\s*작업|분해\s*(?:불가|필요))",
    ),
    SemanticGroup(
        "count_limit",
        "사용·처리 횟수 기준 도달 또는 초과 문구",
        r"(?:(?:사용|처리|누적).{0,18}(?:횟수|회수).{0,24}"
        r"(?:초과|기준|이상|도달)|(?:횟수|회수)\s*초과|기준\s*(?:횟수|회수))",
    ),
    SemanticGroup(
        "product_info_or_line_change",
        "제품/SN/코드 불일치 또는 라인 전환 문구",
        r"(?:제품\s*정보|제품\s*(?:상이|불일치|다름)|코드\s*불일치|"
        r"SN\s*(?:상이|불일치|혼입)|S\s*/\s*N\s*(?:상이|불일치|혼입)|라인\s*(?:변경|전환))",
    ),
    SemanticGroup(
        "surface_peel_or_etching",
        "표면 처리 맥락과 박리/얼룩이 함께 나타나는 문구",
        r"(?:(?:SURFACE|표면).{0,80}(?:PEEL|박리|STAIN|얼룩)|"
        r"(?:PEEL|박리|STAIN|얼룩).{0,80}(?:SURFACE|표면))",
    ),
    SemanticGroup(
        "leak_or_hole_repair",
        "Leak 또는 hole 수리 맥락",
        r"(?:\bLEAK\b|누설|(?:HOLE|홀).{0,40}(?:REPAIR|수리|변형|CRACK))",
    ),
    SemanticGroup(
        "structural_crack_or_breakage",
        "갈라짐, 크랙, 파손 등 구조 손상 직접 문구",
        r"(?:CRACK|갈라짐|균열|파손|BROKEN|BREAKAGE)",
    ),
    SemanticGroup(
        "fastener_or_accessory_repair",
        "체결부품 또는 부속품의 수리성 결함 문구",
        r"(?:(?:SCREW|나사|볼트|부속품).{0,40}(?:파손|누락|헐거움|수리|교체))",
    ),
)

FROZEN_SEMANTIC_GROUPS: Mapping[str, SemanticGroup] = MappingProxyType(
    {group.group_id: group for group in _GROUPS}
)
_COMPILED = {
    group_id: re.compile(group.pattern, re.IGNORECASE)
    for group_id, group in FROZEN_SEMANTIC_GROUPS.items()
}


def normalize_semantic_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[_‐‑‒–—−]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def matched_semantic_groups(row: FeatureRow) -> tuple[str, ...]:
    matched: list[str] = []
    for group_id, group in FROZEN_SEMANTIC_GROUPS.items():
        text = " | ".join(
            normalize_semantic_text(row.features.get(field, ""))
            for field in group.fields
        )
        if _COMPILED[group_id].search(text):
            matched.append(group_id)
    return tuple(matched)


def apply_semantic_veto(
    rows: Sequence[FeatureRow],
    predictions: Mapping[str, str],
    enabled_by_company: Mapping[str, Sequence[str]],
) -> SemanticVetoResult:
    expected = {row.row_id for row in rows}
    if set(predictions) != expected:
        raise ValueError("semantic veto prediction universe mismatch")
    enabled = _validated_policy(enabled_by_company)
    output: dict[str, str] = {}
    demoted: list[str] = []
    matched_by_row: dict[str, tuple[str, ...]] = {}
    for row in rows:
        predicted = predictions[row.row_id]
        active = enabled.get(row.customer_key, frozenset())
        matched = tuple(
            group_id
            for group_id in matched_semantic_groups(row)
            if group_id in active
        )
        if predicted == "PROCESS" and matched:
            output[row.row_id] = "REVIEW"
            demoted.append(row.row_id)
            matched_by_row[row.row_id] = matched
        else:
            output[row.row_id] = predicted
    return SemanticVetoResult(
        predictions=MappingProxyType(output),
        demoted_rows=tuple(demoted),
        matched_groups_by_row=MappingProxyType(matched_by_row),
    )


def semantic_policy_sha256(
    enabled_by_company: Mapping[str, Sequence[str]],
) -> str:
    enabled = _validated_policy(enabled_by_company)
    payload = {
        "schemaVersion": 1,
        "kind": "frozen_demote_only_semantic_veto",
        "groups": {
            group_id: {
                "description": group.description,
                "fields": list(group.fields),
                "pattern": group.pattern,
            }
            for group_id, group in sorted(FROZEN_SEMANTIC_GROUPS.items())
        },
        "enabledByCompany": {
            company: sorted(group_ids)
            for company, group_ids in sorted(enabled.items())
        },
    }
    body = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _validated_policy(
    enabled_by_company: Mapping[str, Sequence[str]],
) -> dict[str, frozenset[str]]:
    output: dict[str, frozenset[str]] = {}
    known = set(FROZEN_SEMANTIC_GROUPS)
    for company, raw_ids in enabled_by_company.items():
        if not str(company).strip():
            raise ValueError("semantic veto company cannot be blank")
        group_ids = frozenset(str(group_id) for group_id in raw_ids)
        unknown = sorted(group_ids - known)
        if unknown:
            raise ValueError(f"unknown semantic veto group: {unknown}")
        output[str(company)] = group_ids
    return output

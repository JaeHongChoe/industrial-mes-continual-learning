"""History-conditioned semantic veto using development-period labels only."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .data_contract import ACTION_LABELS, FeatureRow
from .hierarchical_veto import FROZEN_SEMANTIC_GROUPS, matched_semantic_groups, normalize_semantic_text
from .hierarchy import canonical_line_key


SCOPE_MODES = (
    "company_product_group",
    "company_line_group",
    "company_type_phenomenon_group",
    "company_line_type_phenomenon_group",
)


@dataclass(frozen=True, slots=True)
class SemanticRiskParams:
    scope_mode: str
    min_rows: int
    min_risk_rows: int
    min_risk_rate: float
    active_groups: tuple[str, ...] = tuple(FROZEN_SEMANTIC_GROUPS)

    def __post_init__(self) -> None:
        if self.scope_mode not in SCOPE_MODES:
            raise ValueError(f"unsupported semantic risk scope: {self.scope_mode}")
        if self.min_rows < 1 or self.min_risk_rows < 1:
            raise ValueError("semantic risk support thresholds must be positive")
        if not 0.0 <= self.min_risk_rate <= 1.0:
            raise ValueError("semantic risk rate must be between zero and one")
        unknown = sorted(set(self.active_groups) - set(FROZEN_SEMANTIC_GROUPS))
        if unknown:
            raise ValueError(f"unknown semantic veto group: {unknown}")


@dataclass(frozen=True, slots=True)
class HistoricalSemanticVetoResult:
    predictions: Mapping[str, str]
    demoted_rows: tuple[str, ...]
    reasons_by_row: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True, slots=True)
class HistoricalSemanticRiskIndex:
    counts_by_mode: Mapping[str, Mapping[tuple[str, ...], Mapping[str, int]]]
    training_rows: int

    @classmethod
    def fit(
        cls,
        rows: Sequence[FeatureRow],
        labels: Mapping[str, str],
    ) -> "HistoricalSemanticRiskIndex":
        expected = {row.row_id for row in rows}
        if set(labels) != expected:
            raise ValueError("historical semantic label universe mismatch")
        mutable: dict[str, dict[tuple[str, ...], Counter[str]]] = {
            mode: defaultdict(Counter) for mode in SCOPE_MODES
        }
        for row in rows:
            label = labels[row.row_id]
            if label not in ACTION_LABELS:
                raise ValueError(f"unsupported historical semantic label: {label}")
            for group_id in matched_semantic_groups(row):
                for mode in SCOPE_MODES:
                    mutable[mode][_signature(row, group_id, mode)][label] += 1
        frozen = {
            mode: MappingProxyType(
                {
                    signature: MappingProxyType(dict(counter))
                    for signature, counter in sorted(mode_counts.items())
                }
            )
            for mode, mode_counts in mutable.items()
        }
        return cls(MappingProxyType(frozen), len(rows))

    def risk_evidence(
        self,
        row: FeatureRow,
        params: SemanticRiskParams,
    ) -> tuple[Mapping[str, Any], ...]:
        active = set(params.active_groups)
        evidence: list[Mapping[str, Any]] = []
        counts = self.counts_by_mode[params.scope_mode]
        for group_id in matched_semantic_groups(row):
            if group_id not in active:
                continue
            signature = _signature(row, group_id, params.scope_mode)
            label_counts = counts.get(signature, {})
            total = sum(int(value) for value in label_counts.values())
            risk_rows = int(label_counts.get("RETURN", 0)) + int(
                label_counts.get("REPAIR", 0)
            )
            risk_rate = risk_rows / total if total else 0.0
            if (
                total >= params.min_rows
                and risk_rows >= params.min_risk_rows
                and risk_rate >= params.min_risk_rate
            ):
                evidence.append(
                    MappingProxyType(
                        {
                            "groupId": group_id,
                            "scopeMode": params.scope_mode,
                            "knownRows": total,
                            "riskRows": risk_rows,
                            "riskRate": risk_rate,
                            "shipRows": int(label_counts.get("RETURN", 0)),
                            "repairRows": int(label_counts.get("REPAIR", 0)),
                        }
                    )
                )
        return tuple(evidence)

    def should_veto(self, row: FeatureRow, params: SemanticRiskParams) -> bool:
        return bool(self.risk_evidence(row, params))


def apply_historical_semantic_veto(
    rows: Sequence[FeatureRow],
    predictions: Mapping[str, str],
    index: HistoricalSemanticRiskIndex,
    params_by_company: Mapping[str, SemanticRiskParams],
) -> HistoricalSemanticVetoResult:
    expected = {row.row_id for row in rows}
    if set(predictions) != expected:
        raise ValueError("historical semantic prediction universe mismatch")
    output: dict[str, str] = {}
    demoted: list[str] = []
    reasons: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for row in rows:
        predicted = predictions[row.row_id]
        params = params_by_company.get(row.customer_key)
        evidence = (
            index.risk_evidence(row, params)
            if predicted == "PROCESS" and params is not None
            else ()
        )
        if evidence:
            output[row.row_id] = "REVIEW"
            demoted.append(row.row_id)
            reasons[row.row_id] = evidence
        else:
            output[row.row_id] = predicted
    return HistoricalSemanticVetoResult(
        predictions=MappingProxyType(output),
        demoted_rows=tuple(demoted),
        reasons_by_row=MappingProxyType(reasons),
    )


def _signature(
    row: FeatureRow,
    group_id: str,
    mode: str,
) -> tuple[str, ...]:
    company = str(row.customer_key)
    if mode == "company_product_group":
        return company, str(row.product_key), group_id
    if mode == "company_line_group":
        return company, canonical_line_key(row.features.get("line", "")), group_id
    kind = normalize_semantic_text(row.features.get("type", ""))
    phenomenon = normalize_semantic_text(row.features.get("symptom", ""))
    if mode == "company_type_phenomenon_group":
        return company, kind, phenomenon, group_id
    if mode == "company_line_type_phenomenon_group":
        return (
            company,
            canonical_line_key(row.features.get("line", "")),
            kind,
            phenomenon,
            group_id,
        )
    raise ValueError(f"unsupported semantic risk scope: {mode}")

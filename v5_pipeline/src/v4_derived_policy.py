from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data_contract import ACTION_LABELS, FeatureRow


_RECOVERY_SCOPES = ("product_detail_counts", "detail_counts", "product_type")
_PROTECTION_SCOPES = ("product_type", "product_detail_counts")


@dataclass(frozen=True, slots=True)
class DerivedSignature:
    signature_id: str
    scope: str
    key: tuple[str, ...]
    support: int
    class_counts: Mapping[str, int]
    months: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True, slots=True)
class HierarchicalRoutes:
    company_line: Mapping[str, str] = field(default_factory=dict)
    company: Mapping[str, str] = field(default_factory=dict)
    archetype: Mapping[str, str] = field(default_factory=dict)
    company_archetype: Mapping[str, str] = field(default_factory=dict)
    global_candidate: str = "review"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    candidate_id: str
    level: str


@dataclass(frozen=True, slots=True)
class ModelRiskGuard:
    shipment_threshold: float
    repair_threshold: float
    gray_threshold: float
    selection_month: str

    def __post_init__(self) -> None:
        for name in ("shipment_threshold", "repair_threshold", "gray_threshold"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be between zero and one")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if len(self.selection_month) != 7 or self.selection_month[4] != "-":
            raise ValueError("selection_month must use YYYY-MM")


def fit_model_risk_guard(
    rows: Iterable[FeatureRow],
    baseline_predictions: Mapping[str, str],
    scores: Iterable[Any],
    labels: Mapping[str, str],
    *,
    shipment_thresholds: Sequence[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0),
    repair_thresholds: Sequence[float] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0),
    gray_thresholds: Sequence[float] = (0.5, 0.75, 0.9, 1.0),
    max_ship_to_process: int = 0,
    max_repair_to_process: int = 0,
    max_risk_to_process: int = 0,
    min_auto_accuracy: float = 0.0,
    allow_december_for_january_launch: bool = False,
) -> ModelRiskGuard:
    materialized = tuple(rows)
    score_rows = tuple(scores)
    row_ids = tuple(row.row_id for row in materialized)
    if not materialized or len(row_ids) != len(set(row_ids)):
        raise ValueError("risk-guard selection rows must be non-empty and unique")
    if set(row_ids) != set(baseline_predictions) or set(row_ids) != {row.row_id for row in score_rows}:
        raise ValueError("risk-guard rows, baseline, and scores must share one row universe")
    months = {row.month for row in materialized} | {row.month for row in score_rows}
    permitted_months = {"2025-10", "2025-11"}
    if allow_december_for_january_launch:
        permitted_months.add("2025-12")
    if len(months) != 1 or next(iter(months)) not in permitted_months:
        if months == {"2025-12"}:
            raise ValueError(
                "December guard fitting requires explicit January-launch permission"
            )
        raise ValueError("risk-guard fitting requires one permitted 2025 selection month")
    selection_month = next(iter(months))
    score_by_id = {row.row_id: row for row in score_rows}
    actual = np.asarray([labels.get(row_id) for row_id in row_ids], dtype=object)
    if any(value not in ACTION_LABELS for value in actual):
        raise ValueError("risk-guard fitting requires an action label for every row")
    predicted = np.asarray([_label(baseline_predictions[row_id]) for row_id in row_ids], dtype=object)
    process_base = predicted == "PROCESS"
    auto_base = np.isin(predicted, tuple(ACTION_LABELS))
    correct_base = predicted == actual
    shipment_probability = np.asarray(
        [float(score_by_id[row_id].shipment_probability) for row_id in row_ids]
    )
    repair_probability = np.asarray(
        [float(score_by_id[row_id].repair_probability) for row_id in row_ids]
    )
    gray_probability = np.asarray(
        [float(score_by_id[row_id].gray_probability) for row_id in row_ids]
    )
    best: tuple[tuple[int, int, int, int, int], ModelRiskGuard] | None = None
    for shipment_threshold in shipment_thresholds:
        for repair_threshold in repair_thresholds:
            for gray_threshold in gray_thresholds:
                fitted = ModelRiskGuard(
                    float(shipment_threshold),
                    float(repair_threshold),
                    float(gray_threshold),
                    selection_month,
                )
                demoted = process_base & (
                    (shipment_probability > fitted.shipment_threshold)
                    | (repair_probability > fitted.repair_threshold)
                    | (gray_probability > fitted.gray_threshold)
                )
                final_process = process_base & ~demoted
                ship_to_process = int(np.count_nonzero((actual == "RETURN") & final_process))
                repair_to_process = int(np.count_nonzero((actual == "REPAIR") & final_process))
                risk = ship_to_process + repair_to_process
                if (
                    ship_to_process > max_ship_to_process
                    or repair_to_process > max_repair_to_process
                    or risk > max_risk_to_process
                ):
                    continue
                auto = int(np.count_nonzero(auto_base) - np.count_nonzero(demoted))
                correct = int(
                    np.count_nonzero(correct_base)
                    - np.count_nonzero(demoted & (actual == "PROCESS"))
                )
                if auto == 0 or correct / auto < float(min_auto_accuracy):
                    continue
                correct_process = int(np.count_nonzero((actual == "PROCESS") & final_process))
                process_to_ng = int(
                    np.count_nonzero(
                        (actual == "PROCESS")
                        & np.isin(predicted, ("RETURN", "REPAIR"))
                    )
                )
                score = (correct_process, correct, auto, -process_to_ng, -risk)
                if best is None or score > best[0]:
                    best = score, fitted
    if best is None:
        raise ValueError("no model risk guard satisfies the supplied budget")
    return best[1]


def apply_model_risk_guard(
    predictions: Mapping[str, str],
    scores: Iterable[Any],
    guard: ModelRiskGuard,
) -> dict[str, str]:
    score_rows = tuple(scores)
    score_by_id = {row.row_id: row for row in score_rows}
    if len(score_by_id) != len(score_rows) or set(score_by_id) != set(predictions):
        raise ValueError("risk-guard scores must match the prediction row universe")
    output: dict[str, str] = {}
    for row_id, raw_label in predictions.items():
        label = _label(raw_label)
        score = score_by_id[row_id]
        if label == "PROCESS" and (
            float(score.shipment_probability) > guard.shipment_threshold
            or float(score.repair_probability) > guard.repair_threshold
            or float(score.gray_probability) > guard.gray_threshold
        ):
            label = ""
        output[row_id] = label
    return output


def derive_recovery_signatures(
    rows: Iterable[FeatureRow],
    labels: Mapping[str, str],
    *,
    min_support: int = 5,
    min_process_rate: float = 1.0,
) -> tuple[DerivedSignature, ...]:
    materialized = tuple(rows)
    _validate_derivation_rows(materialized, labels)
    if min_support < 1:
        raise ValueError("min_support must be positive")
    if not 0 <= min_process_rate <= 1:
        raise ValueError("min_process_rate must be between zero and one")
    return _derive_signatures(
        materialized,
        labels,
        scopes=_RECOVERY_SCOPES,
        predicate=lambda counts, support: (
            support >= min_support
            and counts["PROCESS"] / support >= min_process_rate
            and counts["RETURN"] == 0
            and counts["REPAIR"] == 0
        ),
        family="recovery",
    )


def derive_protection_signatures(
    rows: Iterable[FeatureRow],
    labels: Mapping[str, str],
    *,
    min_support: int = 5,
    min_ng: int = 1,
    min_ng_rate: float = 0.0,
) -> tuple[DerivedSignature, ...]:
    materialized = tuple(rows)
    _validate_derivation_rows(materialized, labels)
    if min_support < 1 or min_ng < 1:
        raise ValueError("protection support and NG counts must be positive")
    if not 0 <= min_ng_rate <= 1:
        raise ValueError("min_ng_rate must be between zero and one")
    return _derive_signatures(
        materialized,
        labels,
        scopes=_PROTECTION_SCOPES,
        predicate=lambda counts, support: (
            support >= min_support
            and counts["RETURN"] + counts["REPAIR"] >= min_ng
            and (counts["RETURN"] + counts["REPAIR"]) / support >= min_ng_rate
        ),
        family="protection",
    )


def route_hierarchical_policy(
    row: FeatureRow,
    routes: HierarchicalRoutes,
) -> RouteDecision:
    line = _route_token(row.features.get("line"))
    company_line_key = f"{row.customer_key}|{line}"
    if company_line_key in routes.company_line:
        return RouteDecision(str(routes.company_line[company_line_key]), "company_line")
    if row.customer_key in routes.company:
        return RouteDecision(str(routes.company[row.customer_key]), "company")
    archetype_id = routes.company_archetype.get(row.customer_key)
    if archetype_id is not None and archetype_id in routes.archetype:
        return RouteDecision(str(routes.archetype[archetype_id]), "archetype")
    return RouteDecision(str(routes.global_candidate), "global")


def compose_v4_derived_policy(
    rows: Iterable[FeatureRow],
    *,
    baseline_predictions: Mapping[str, str],
    candidate_predictions: Mapping[str, Mapping[str, str]],
    routes: HierarchicalRoutes,
    recovery_signatures: Sequence[DerivedSignature] = (),
    protection_signatures: Sequence[DerivedSignature] = (),
) -> dict[str, str]:
    output: dict[str, str] = {}
    for row in rows:
        if row.row_id in output:
            raise ValueError(f"duplicate policy row: {row.row_id}")
        route = route_hierarchical_policy(row, routes)
        candidate_map = candidate_predictions.get(route.candidate_id, {})
        candidate = _label(candidate_map.get(row.row_id))
        predicted = candidate or _label(baseline_predictions.get(row.row_id))
        if not predicted and _matches_any(row, recovery_signatures):
            predicted = "PROCESS"
        if predicted == "PROCESS" and _matches_any(row, protection_signatures):
            predicted = ""
        output[row.row_id] = predicted
    return output


def _derive_signatures(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
    *,
    scopes: Sequence[str],
    predicate,
    family: str,
) -> tuple[DerivedSignature, ...]:
    counts: dict[str, dict[tuple[str, ...], Counter[str]]] = {
        scope: defaultdict(Counter) for scope in scopes
    }
    months: dict[str, dict[tuple[str, ...], set[str]]] = {
        scope: defaultdict(set) for scope in scopes
    }
    for row in rows:
        for scope in scopes:
            key = _signature_key(row, scope)
            counts[scope][key][labels[row.row_id]] += 1
            months[scope][key].add(row.month)
    source_sha = _selection_source_sha256(rows, labels)
    output: list[DerivedSignature] = []
    for scope in scopes:
        for key, observed in sorted(counts[scope].items()):
            support = sum(observed.values())
            if not predicate(observed, support):
                continue
            basis = {
                "family": family,
                "scope": scope,
                "key": key,
                "support": support,
                "classCounts": dict(sorted(observed.items())),
                "months": sorted(months[scope][key]),
                "sourceSha256": source_sha,
            }
            output.append(
                DerivedSignature(
                    signature_id=hashlib.sha256(_canonical_json_bytes(basis)).hexdigest(),
                    scope=scope,
                    key=key,
                    support=support,
                    class_counts=MappingProxyType(dict(sorted(observed.items()))),
                    months=tuple(sorted(months[scope][key])),
                    source_sha256=source_sha,
                )
            )
    return tuple(sorted(output, key=lambda item: (item.scope, item.key)))


def _validate_derivation_rows(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
) -> None:
    seen: set[str] = set()
    for row in rows:
        if row.row_id in seen:
            raise ValueError(f"duplicate signature derivation row: {row.row_id}")
        seen.add(row.row_id)
        if row.month not in {"2025-10", "2025-11"}:
            raise ValueError("signature derivation requires 2025 selector months only")
        if labels.get(row.row_id) not in ACTION_LABELS:
            raise ValueError(f"missing signature derivation label: {row.row_id}")


def _matches_any(row: FeatureRow, signatures: Sequence[DerivedSignature]) -> bool:
    return any(_signature_key(row, item.scope) == item.key for item in signatures)


def _signature_key(row: FeatureRow, scope: str) -> tuple[str, ...]:
    type_value = _normalized(row.features.get("type"))
    detail = _normalized(row.features.get("detail"))
    counts = (
        _count_token(row.features.get("intake_count")),
        _count_token(row.features.get("service_count")),
        _count_token(row.features.get("treatment_count")),
    )
    if scope == "product_detail_counts":
        return row.product_key, detail, *counts
    if scope == "detail_counts":
        return detail, *counts
    if scope == "product_type":
        return row.product_key, type_value
    if scope == "type_detail":
        return type_value, detail
    raise ValueError(f"unsupported signature scope: {scope}")


def _selection_source_sha256(
    rows: Sequence[FeatureRow],
    labels: Mapping[str, str],
) -> str:
    payload = [
        {
            "rowId": row.row_id,
            "month": row.month,
            "label": labels[row.row_id],
        }
        for row in sorted(rows, key=lambda item: item.row_id)
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _count_token(value: Any) -> str:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return "__missing__"
    return f"{number:g}" if math.isfinite(number) else "__missing__"


def _normalized(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).lower().split())


def _route_token(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in ACTION_LABELS else ""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


ACTION_LABELS = frozenset({"PROCESS", "RETURN", "REPAIR"})
NG_LABELS = frozenset({"RETURN", "REPAIR"})
SUPPORTED_MODES = frozenset({"frozen_history", "prequential_replay"})
SUPPORTED_OVERLAY_MODES = frozenset({"ng_only", "all_actions"})


def apply_portable_policy(
    rows: Sequence[Mapping[str, Any]],
    resolved_history_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    *,
    mode: str = "frozen_history",
) -> list[dict[str, Any]]:
    """Apply the declarative V4 layer without reading the current row's label.

    `frozen_history` is the safe batch-inference mode. `prequential_replay`
    reproduces the historical V4 walk-forward audit: a row is scored first and
    only then may its resolved label update state for later rows.
    """

    _validate_policy(policy)
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported portable policy mode: {mode}")
    _assert_disjoint_runtime_rows(
        rows,
        resolved_history_rows,
        str(policy["sourceDatasetId"]),
    )

    output = [_copy_runtime_row(row) for row in rows]
    history = tuple(dict(row) for row in resolved_history_rows)
    for layer in policy["layers"]:
        kind = layer["kind"]
        if kind == "signature_set":
            _apply_signature_layer(output, layer)
        elif kind == "walk_forward_veto":
            _apply_walk_forward_layer(output, history, layer, mode=mode)
        else:  # pragma: no cover - protected by validation
            raise ValueError(f"unsupported portable policy layer kind: {kind}")
    return output


def prediction_map(
    rows: Iterable[Mapping[str, Any]],
    source_dataset_id: str,
) -> dict[str, str]:
    if not isinstance(source_dataset_id, str) or not source_dataset_id:
        raise ValueError("source dataset ID must be a non-empty string")
    output: dict[str, str] = {}
    for row in rows:
        source_row = _positive_source_row(row.get("source_row"))
        row_id = f"{source_dataset_id}:{source_row}"
        if row_id in output:
            raise ValueError(f"duplicate portable policy row ID: {row_id}")
        output[row_id] = _label(row.get("predicted"))
    return dict(sorted(output.items()))


def compose_monotone_overlay(
    baseline_predictions: Mapping[str, Any],
    candidate_predictions: Mapping[str, Any],
    *,
    mode: str = "ng_only",
    allowed_customers: set[str] | frozenset[str] | None = None,
    row_context: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Preserve every V4 automatic decision and fill eligible review rows."""

    if mode not in SUPPORTED_OVERLAY_MODES:
        raise ValueError(f"unsupported monotone overlay mode: {mode}")
    missing = sorted(set(baseline_predictions) - set(candidate_predictions))
    if missing:
        raise ValueError(f"candidate predictions are missing baseline rows: {missing[:3]}")
    output: dict[str, str] = {}
    for row_id in sorted(baseline_predictions):
        baseline = _label(baseline_predictions[row_id])
        candidate = _label(candidate_predictions[row_id])
        customer_allowed = True
        if allowed_customers is not None:
            context = row_context.get(row_id) if row_context is not None else None
            if not isinstance(context, Mapping):
                raise ValueError(f"row context is required for customer-guarded overlay: {row_id}")
            customer = str(context.get("rawCustomer") or context.get("__customerKey") or "")
            customer_allowed = customer in allowed_customers
        if baseline:
            output[row_id] = baseline
        elif customer_allowed and mode == "ng_only" and candidate in NG_LABELS:
            output[row_id] = candidate
        elif customer_allowed and mode == "all_actions" and candidate in ACTION_LABELS:
            output[row_id] = candidate
        else:
            output[row_id] = ""
    return output


def _validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schemaVersion") != 1:
        raise ValueError("portable V4 policy schemaVersion must be 1")
    if policy.get("kind") != "mes_v4_portable_policy":
        raise ValueError("portable V4 policy kind is invalid")
    if not isinstance(policy.get("sourceDatasetId"), str) or not policy["sourceDatasetId"]:
        raise ValueError("portable V4 policy sourceDatasetId is required")
    layers = policy.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("portable V4 policy layers must be a non-empty list")
    seen: set[str] = set()
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("portable V4 policy layer must be an object")
        layer_id = layer.get("id")
        if not isinstance(layer_id, str) or not layer_id or layer_id in seen:
            raise ValueError("portable V4 policy layer IDs must be unique")
        seen.add(layer_id)
        kind = layer.get("kind")
        action = layer.get("action")
        if kind == "signature_set":
            if action not in {"review_to_process", "ng_to_review"}:
                raise ValueError(f"invalid signature action for {layer_id}")
            signatures = layer.get("signatures")
            if not isinstance(signatures, list) or not signatures:
                raise ValueError(f"signature layer {layer_id} requires signatures")
        elif kind == "walk_forward_veto":
            if action != "process_to_review":
                raise ValueError(f"invalid walk-forward action for {layer_id}")
            if layer.get("keyMode") not in {
                "prod_type",
                "prod_detail",
                "prod_detail_counts",
                "type_detail",
                "detail_only",
            }:
                raise ValueError(f"invalid walk-forward key mode for {layer_id}")
            for field in ("minHist", "minNg", "maxNgRatio", "activeNgDays"):
                value = layer.get(field, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f"invalid {field} for {layer_id}")
        else:
            raise ValueError(f"unsupported portable policy layer kind: {kind}")


def _assert_disjoint_runtime_rows(
    rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    source_dataset_id: str,
) -> None:
    current_ids = _runtime_id_set(rows, "current", source_dataset_id)
    history_ids = _runtime_id_set(history_rows, "resolved history", source_dataset_id)
    overlap = sorted(current_ids & history_ids)
    if overlap:
        raise ValueError(f"current row overlaps resolved history: {overlap[:3]}")


def _runtime_id_set(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    source_dataset_id: str,
) -> set[str]:
    output: set[str] = set()
    for row in rows:
        source_row = _positive_source_row(row.get("source_row"))
        row_id = f"{source_dataset_id}:{source_row}"
        if row_id in output:
            raise ValueError(f"duplicate {name} runtime row: {row_id}")
        output.add(row_id)
    return output


def _copy_runtime_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = deepcopy(dict(row))
    changes = copied.get("policyChanges")
    copied["policyChanges"] = list(changes) if isinstance(changes, list) else []
    copied["predicted"] = _label(copied.get("predicted"))
    return copied


def _apply_signature_layer(
    rows: list[dict[str, Any]],
    layer: Mapping[str, Any],
) -> None:
    action = layer["action"]
    signatures = layer["signatures"]
    for row in rows:
        predicted = _label(row.get("predicted"))
        if action == "review_to_process" and predicted:
            continue
        if action == "ng_to_review" and predicted not in NG_LABELS:
            continue
        if not any(_signature_matches(row, signature) for signature in signatures):
            continue
        row["predicted"] = "PROCESS" if action == "review_to_process" else ""
        _record_change(row, str(layer["id"]))


def _signature_matches(row: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    customer = str(row.get("__customerKey") or "")
    if signature.get("customer") is not None and customer != str(signature["customer"]):
        return False
    predicted_label = signature.get("predictedLabel")
    if predicted_label is not None and _label(row.get("predicted")) != predicted_label:
        return False

    product_keys = signature.get("productKeys")
    if product_keys is not None:
        if not isinstance(product_keys, list) or str(row.get("productKey") or "") not in {
            str(value) for value in product_keys
        }:
            return False

    fields = {
        "productKey": "productKey",
        "type": "type",
        "phenomenon": "phenomenon",
        "detailFamily": "detailFamily",
        "serviceCount": "serviceCount",
        "intakeCount": "intakeCount",
    }
    for signature_field, row_field in fields.items():
        if signature_field not in signature:
            continue
        if _clean_part(row.get(row_field), counts=signature_field in {"serviceCount", "intakeCount"}) != _clean_part(
            signature[signature_field],
            counts=signature_field in {"serviceCount", "intakeCount"},
        ):
            return False

    includes_all = signature.get("detailFamilyIncludesAll")
    if includes_all is not None:
        detail = str(row.get("detailFamily") or "")
        if not isinstance(includes_all, list) or not all(
            str(fragment) in detail for fragment in includes_all
        ):
            return False
    return True


def _apply_walk_forward_layer(
    rows: list[dict[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    layer: Mapping[str, Any],
    *,
    mode: str,
) -> None:
    state: dict[str, dict[str, Any]] = {}
    for row in history_rows:
        _update_walk_state(state, row, layer)

    ordered = sorted(enumerate(rows), key=lambda item: _sort_key(item[1], item[0]))
    for _, row in ordered:
        key = _walk_key(row, str(layer["keyMode"]))
        item = state.get(key, {"total": 0, "ng": 0, "process": 0, "lastNg": None})
        total_known = item["ng"] + item["process"]
        ng_ratio = item["ng"] / total_known if total_known else 0.0
        active = _active_ng_window(row, item["lastNg"], float(layer.get("activeNgDays") or 0))
        should_veto = (
            _label(row.get("predicted")) == "PROCESS"
            and _customer_pass(row, layer.get("customers"))
            and active
            and item["total"] >= int(layer.get("minHist") or 0)
            and item["ng"] >= int(layer.get("minNg") or 0)
            and (item["process"] == 0 or ng_ratio >= float(layer.get("maxNgRatio") or 0))
        )
        if should_veto:
            row["predicted"] = ""
            _record_change(row, str(layer["id"]))
        if mode == "prequential_replay":
            _update_walk_state(state, row, layer)


def _update_walk_state(
    state: dict[str, dict[str, Any]],
    row: Mapping[str, Any],
    layer: Mapping[str, Any],
) -> None:
    actual = _label(row.get("actual"))
    if actual not in ACTION_LABELS:
        return
    key = _walk_key(row, str(layer["keyMode"]))
    item = state.setdefault(key, {"total": 0, "ng": 0, "process": 0, "lastNg": None})
    item["total"] += 1
    if actual == "PROCESS":
        item["process"] += 1
    else:
        item["ng"] += 1
        current = _row_date(row)
        if current is not None and (item["lastNg"] is None or current > item["lastNg"]):
            item["lastNg"] = current


def _walk_key(row: Mapping[str, Any], mode: str) -> str:
    customer = row.get("__customerKey")
    columns = {
        "prod_type": (customer, row.get("productKey"), row.get("type"), row.get("phenomenon")),
        "prod_detail": (
            customer,
            row.get("productKey"),
            row.get("type"),
            row.get("phenomenon"),
            row.get("detailFamily"),
        ),
        "prod_detail_counts": (
            customer,
            row.get("productKey"),
            row.get("type"),
            row.get("phenomenon"),
            row.get("detailFamily"),
            row.get("serviceCount"),
            row.get("intakeCount"),
        ),
        "type_detail": (customer, row.get("type"), row.get("phenomenon"), row.get("detailFamily")),
        "detail_only": (customer, row.get("detailFamily")),
    }
    return " | ".join(_clean_part(value) for value in columns[mode])


def _customer_pass(row: Mapping[str, Any], customers: Any) -> bool:
    if customers is None:
        return True
    if not isinstance(customers, list):
        raise ValueError("walk-forward customers must be a list or null")
    return str(row.get("__customerKey") or "") in {str(value) for value in customers}


def _active_ng_window(row: Mapping[str, Any], last_ng: date | None, active_days: float) -> bool:
    if active_days <= 0:
        return True
    current = _row_date(row)
    if current is None or last_ng is None:
        return False
    delta = (current - last_ng).days
    return 0 <= delta <= active_days


def _row_date(row: Mapping[str, Any]) -> date | None:
    text = str(row.get("date") or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _sort_key(row: Mapping[str, Any], index: int) -> tuple[str, int, int]:
    try:
        source_row = int(float(row.get("source_row") or 0))
    except (TypeError, ValueError):
        source_row = 0
    return str(row.get("date") or ""), source_row, index


def _record_change(row: dict[str, Any], layer_id: str) -> None:
    changes = row.setdefault("policyChanges", [])
    if layer_id not in changes:
        changes.append(layer_id)


def _label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in ACTION_LABELS else ""


def _clean_part(value: Any, *, counts: bool = False) -> str:
    text = str(value if value is not None else "").strip()
    if counts:
        return text
    return text or "-"


def _positive_source_row(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError("portable policy source_row must be a positive integer") from error
    if number < 1 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise ValueError("portable policy source_row must be a positive integer")
    return number

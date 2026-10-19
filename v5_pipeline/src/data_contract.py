from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


ACTION_LABELS = frozenset({"PROCESS", "RETURN", "REPAIR"})
# 반출일 is created after the intake decision and must never place a row in a
# training/evaluation window. Missing intake dates therefore fail closed.
DATE_COLUMNS = ("occurrence_date", "registered_date")
IMMUTABLE_SELECTOR_MONTHS = ("2025-10", "2025-11")
IMMUTABLE_LOCKBOX_MONTHS = ("2025-12",)
IMMUTABLE_AUDIT_START = "2026-01-01"
IMMUTABLE_EVALUATION_STATUS = {
    "selector": "retrospective_selection",
    "lockbox": "retrospective_reject_only",
    "audit": "retrospective_audit_only",
    "promotion": "requires_future_shadow",
}

# Only intake-time fields are copied into FeatureRow. The raw artifact contains many
# workflow and outcome columns, but an explicit allow-list is safer than attempting
# to enumerate every future alias that must be removed.
ALLOWED_RAW_FEATURES = (
    "customer",
    "line",
    "customer_equipment",
    "EMS Circle",
    "product_category",
    "process_code",
    "part_code",
    "part_name",
    "serial",
    "P/R",
    "Vendor Name",
    "Bulk",
    "occurrence_category",
    "type",
    "symptom",
    "detail",
    "occurrence_process",
    "service_type",
    "material",
    "Area",
    "Maker",
    "customer_process",
    "Chamber",
    "intake_count",
    "service_count",
    "treatment_count",
    "usage_hours",
    "cumulative_usage_hours",
    "usage_cycles",
    "quantity",
)


class DataContractError(RuntimeError):
    """Raised when immutable dataset expectations do not hold."""


class FeatureSchemaError(DataContractError):
    """Raised when a candidate requests a forbidden or unknown feature."""


class AuditLabelsLockedError(DataContractError):
    """Raised when audit outcomes are requested before the freeze gate."""


@dataclass(frozen=True, slots=True)
class FeatureRow:
    row_id: str
    source_row: int
    date: str
    month: str
    raw_customer: str
    customer_key: str
    legacy_customer_key: str
    product_key: str
    episode_id: str
    episode_basis: str
    features: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    audit_raw_rows: int
    expected_audit_action_rows: int
    expected_audit_non_action_rows: int
    audit_customers: int
    audit_raw_customers: int
    audit_legacy_customers: int
    raw_to_canonical_customer_map: Mapping[str, tuple[str, ...]]
    ambiguous_raw_customer_mappings: Mapping[str, tuple[str, ...]]
    raw_to_legacy_policy_map: Mapping[str, tuple[str, ...]]
    ambiguous_raw_legacy_mappings: Mapping[str, tuple[str, ...]]
    canonical_customer_collisions: Mapping[str, tuple[str, ...]]
    legacy_customer_collisions: Mapping[str, tuple[str, ...]]
    audit_label_counts: Mapping[str, None]
    split_counts: Mapping[str, int]
    split_date_ranges: Mapping[str, tuple[str, str]]
    source_hashes: Mapping[str, str]
    source_dataset_id: str


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    history: tuple[FeatureRow, ...]
    selector: tuple[FeatureRow, ...]
    lockbox: tuple[FeatureRow, ...]
    audit: tuple[FeatureRow, ...]
    metadata: DatasetMetadata
    audit_labels_locked: bool = True
    _development_labels: Mapping[str, Mapping[str, str]] = field(repr=False, compare=False, default_factory=dict)

    def audit_labels(self) -> Mapping[str, str]:
        raise AuditLabelsLockedError("audit labels are unavailable until a frozen policy token is verified")

    def lockbox_labels(self) -> Mapping[str, str]:
        raise AuditLabelsLockedError("lockbox labels are reject-only and unavailable until policy freeze")

    def development_labels(self, split: str) -> Mapping[str, str]:
        if split not in {"history", "selector"}:
            raise KeyError(f"development labels are not available for split: {split}")
        return self._development_labels[split]

    def all_feature_splits(self) -> tuple[tuple[FeatureRow, ...], ...]:
        return self.history, self.selector, self.lockbox, self.audit


def canonical_field_name(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return "".join(char for char in normalized if char.isalnum())


def validate_requested_features(names: Iterable[str], config: Mapping[str, Any]) -> tuple[str, ...]:
    forbidden = {
        canonical_field_name(name)
        for name in config.get("forbiddenFeatureNames", ())
        if canonical_field_name(name)
    }
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.get("forbiddenFeaturePatterns", ())]
    allowed = {canonical_field_name(name): name for name in ALLOWED_RAW_FEATURES}
    validated: list[str] = []
    for name in names:
        canonical = canonical_field_name(name)
        forbidden_hit = any(token == canonical or token in canonical for token in forbidden)
        pattern_hit = any(pattern.search(canonical) for pattern in patterns)
        if forbidden_hit or pattern_hit:
            raise FeatureSchemaError(f"forbidden feature alias: {name}")
        if canonical not in allowed:
            raise FeatureSchemaError(f"feature is not in the intake-time allow-list: {name}")
        validated.append(allowed[canonical])
    return tuple(validated)


def parse_source_date(row: Mapping[str, Any]) -> str:
    for column in DATE_COLUMNS:
        parsed = _parse_date_value(row.get(column))
        if parsed:
            return parsed
    return ""


def stable_row_id(source_dataset_id: str, source_row_number: Any) -> str:
    try:
        number = int(float(source_row_number))
    except (TypeError, ValueError) as error:
        raise DataContractError(f"invalid __sourceRowNumber: {source_row_number!r}") from error
    if number < 1:
        raise DataContractError(f"invalid __sourceRowNumber: {source_row_number!r}")
    return f"{source_dataset_id}:{number}"


def prepare_dataset(config_path: str | Path) -> DatasetBundle:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    history_path = Path(config["historyRows"]).resolve()
    audit_path = Path(config["auditRows"]).resolve()
    source_dataset_id = str(config.get("sourceDatasetId") or "mes_loop93_corrected_source")

    _validate_immutable_config(config)
    _validate_prepare_provenance(config)
    source_hashes = {
        "historyRows": _sha256_file(history_path),
        "auditRows": _sha256_file(audit_path),
    }
    for source_name, expected_key in (
        ("historyRows", "historyRowsSha256"),
        ("auditRows", "auditRowsSha256"),
    ):
        expected = str(config[expected_key]).lower()
        if source_hashes[source_name] != expected:
            raise DataContractError(
                f"{source_name} SHA-256 mismatch: expected {expected}, observed {source_hashes[source_name]}"
            )

    validate_requested_features(ALLOWED_RAW_FEATURES, config)
    selector_months = frozenset(IMMUTABLE_SELECTOR_MONTHS)
    lockbox_months = frozenset(IMMUTABLE_LOCKBOX_MONTHS)
    first_selector_month = min(selector_months)
    audit_start = IMMUTABLE_AUDIT_START

    split_rows: dict[str, list[FeatureRow]] = {name: [] for name in ("history", "selector", "lockbox", "audit")}
    development_labels: dict[str, dict[str, str]] = {name: {} for name in ("history", "selector")}
    seen_ids: set[str] = set()

    for raw in _iter_jsonl(history_path):
        observed_date = parse_source_date(raw)
        if not observed_date:
            continue
        month = observed_date[:7]
        if month in selector_months:
            split = "selector"
        elif month in lockbox_months:
            split = "lockbox"
        elif month < first_selector_month:
            split = "history"
        else:
            continue
        if split != "lockbox":
            actual = _actual_label(raw)
            if actual not in ACTION_LABELS:
                continue
        feature_row = _to_feature_row(raw, source_dataset_id)
        _append_unique(split_rows[split], feature_row, seen_ids)
        if split in development_labels:
            development_labels[split][feature_row.row_id] = actual

    audit_raw_rows = 0
    audit_customers: set[str] = set()
    audit_legacy_customers: set[str] = set()
    audit_raw_customers: set[str] = set()
    raw_to_canonical: dict[str, set[str]] = {}
    raw_to_legacy: dict[str, set[str]] = {}
    for raw in _iter_jsonl(audit_path):
        audit_raw_rows += 1
        observed_date = parse_source_date(raw)
        if not observed_date or observed_date < audit_start:
            continue
        feature_row = _to_feature_row(raw, source_dataset_id)
        _append_unique(split_rows["audit"], feature_row, seen_ids)
        audit_customers.add(feature_row.customer_key)
        audit_legacy_customers.add(feature_row.legacy_customer_key)
        audit_raw_customers.add(feature_row.raw_customer)
        raw_to_canonical.setdefault(feature_row.raw_customer, set()).add(feature_row.customer_key)
        raw_to_legacy.setdefault(feature_row.raw_customer, set()).add(feature_row.legacy_customer_key)

    expected_raw_rows = int(config["expectedAuditRawRows"])
    expected_action_rows = int(config["expectedAuditActionRows"])
    expected_non_action_rows = int(config["expectedAuditNonActionRows"])
    expected_customers = int(config["expectedAuditCustomers"])
    expected_legacy_customers = int(config["expectedAuditLegacyCustomers"])
    expected_raw_customers = int(config["expectedAuditRawCustomers"])
    if audit_raw_rows != expected_raw_rows or len(split_rows["audit"]) != expected_raw_rows:
        raise DataContractError(
            f"audit raw universe mismatch: expected {expected_raw_rows}, "
            f"read {audit_raw_rows}, prepared {len(split_rows['audit'])}"
        )
    if len(audit_customers) != expected_customers:
        raise DataContractError(
            f"audit customer universe mismatch: expected {expected_customers}, observed {len(audit_customers)}"
        )
    if len(audit_legacy_customers) != expected_legacy_customers:
        raise DataContractError(
            f"audit legacy customer universe mismatch: expected {expected_legacy_customers}, "
            f"observed {len(audit_legacy_customers)}"
        )
    if len(audit_raw_customers) != expected_raw_customers:
        raise DataContractError(
            f"audit raw-customer universe mismatch: expected {expected_raw_customers}, "
            f"observed {len(audit_raw_customers)}"
        )

    for rows in split_rows.values():
        rows.sort(key=lambda row: (row.date, row.source_row))
    by_canonical: dict[str, list[str]] = {}
    for raw_customer, canonical_values in raw_to_canonical.items():
        for canonical in canonical_values:
            by_canonical.setdefault(canonical, []).append(raw_customer)
    collisions = {
        canonical: tuple(sorted(raw_values))
        for canonical, raw_values in by_canonical.items()
        if len(raw_values) > 1
    }
    by_legacy: dict[str, list[str]] = {}
    for raw_customer, legacy_values in raw_to_legacy.items():
        for legacy in legacy_values:
            by_legacy.setdefault(legacy, []).append(raw_customer)
    legacy_collisions = {
        legacy: tuple(sorted(raw_values))
        for legacy, raw_values in by_legacy.items()
        if len(raw_values) > 1
    }

    frozen_splits = {name: tuple(rows) for name, rows in split_rows.items()}
    split_counts = MappingProxyType({name: len(rows) for name, rows in frozen_splits.items()})
    split_date_ranges = MappingProxyType(
        {name: _date_range(rows) for name, rows in frozen_splits.items()}
    )
    metadata = DatasetMetadata(
        audit_raw_rows=audit_raw_rows,
        expected_audit_action_rows=expected_action_rows,
        expected_audit_non_action_rows=expected_non_action_rows,
        audit_customers=len(audit_customers),
        audit_raw_customers=len(audit_raw_customers),
        audit_legacy_customers=len(audit_legacy_customers),
        raw_to_canonical_customer_map=MappingProxyType(
            {
                raw_customer: tuple(sorted(canonical_values))
                for raw_customer, canonical_values in sorted(raw_to_canonical.items())
            }
        ),
        ambiguous_raw_customer_mappings=MappingProxyType(
            {
                raw_customer: tuple(sorted(canonical_values))
                for raw_customer, canonical_values in sorted(raw_to_canonical.items())
                if len(canonical_values) > 1
            }
        ),
        raw_to_legacy_policy_map=MappingProxyType(
            {
                raw_customer: tuple(sorted(legacy_values))
                for raw_customer, legacy_values in sorted(raw_to_legacy.items())
            }
        ),
        ambiguous_raw_legacy_mappings=MappingProxyType(
            {
                raw_customer: tuple(sorted(legacy_values))
                for raw_customer, legacy_values in sorted(raw_to_legacy.items())
                if len(legacy_values) > 1
            }
        ),
        canonical_customer_collisions=MappingProxyType(dict(sorted(collisions.items()))),
        legacy_customer_collisions=MappingProxyType(dict(sorted(legacy_collisions.items()))),
        audit_label_counts=MappingProxyType({label: None for label in sorted(ACTION_LABELS)}),
        split_counts=split_counts,
        split_date_ranges=split_date_ranges,
        source_hashes=MappingProxyType(source_hashes),
        source_dataset_id=source_dataset_id,
    )
    frozen_labels = MappingProxyType(
        {split: MappingProxyType(labels) for split, labels in development_labels.items()}
    )
    return DatasetBundle(
        history=frozen_splits["history"],
        selector=frozen_splits["selector"],
        lockbox=frozen_splits["lockbox"],
        audit=frozen_splits["audit"],
        metadata=metadata,
        _development_labels=frozen_labels,
    )


def write_dataset_manifest(
    bundle: DatasetBundle,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_path = Path(output_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "sourceDatasetId": bundle.metadata.source_dataset_id,
        "configPath": str(config_path),
        "configSha256": _sha256_file(config_path),
        "sourceSha256": dict(bundle.metadata.source_hashes),
        "prepareManifestSha256": str(config["prepareManifestSha256"]),
        "prepareScriptSha256": str(config["prepareScriptSha256"]),
        "rawWorkbookSha256": str(config["rawWorkbookSha256"]),
        "selectorMonths": list(config["selectorMonths"]),
        "lockboxMonths": list(config["lockboxMonths"]),
        "auditStart": str(config["auditStart"]),
        "evaluationStatus": dict(config.get("evaluationStatus", {})),
        "splitCounts": dict(bundle.metadata.split_counts),
        "splitDateRanges": {
            split: list(date_range)
            for split, date_range in bundle.metadata.split_date_ranges.items()
        },
        "auditRawRows": bundle.metadata.audit_raw_rows,
        "expectedAuditActionRows": bundle.metadata.expected_audit_action_rows,
        "expectedAuditNonActionRows": bundle.metadata.expected_audit_non_action_rows,
        "auditRawCustomers": bundle.metadata.audit_raw_customers,
        "auditCustomersCanonical": bundle.metadata.audit_customers,
        "auditCustomersLegacyPolicy": bundle.metadata.audit_legacy_customers,
        "rawToCanonicalCustomerMap": {
            key: list(values)
            for key, values in bundle.metadata.raw_to_canonical_customer_map.items()
        },
        "ambiguousRawCustomerMappings": {
            key: list(values)
            for key, values in bundle.metadata.ambiguous_raw_customer_mappings.items()
        },
        "rawToLegacyPolicyMap": {
            key: list(values)
            for key, values in bundle.metadata.raw_to_legacy_policy_map.items()
        },
        "ambiguousRawLegacyMappings": {
            key: list(values)
            for key, values in bundle.metadata.ambiguous_raw_legacy_mappings.items()
        },
        "canonicalCustomerCollisions": {
            key: list(values)
            for key, values in bundle.metadata.canonical_customer_collisions.items()
        },
        "legacyCustomerCollisions": {
            key: list(values)
            for key, values in bundle.metadata.legacy_customer_collisions.items()
        },
        "auditLabelsExposedDuringPrepare": False,
        "auditMembershipDependsOnLabels": False,
        "lockboxLabelsExposedDuringPrepare": False,
        "rowIdentityBasis": "sourceDatasetId:__sourceRowNumber",
        "featurePolicy": "closed intake-time allow-list",
        "sourcePinsRequired": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _to_feature_row(raw: Mapping[str, Any], source_dataset_id: str) -> FeatureRow:
    observed_date = parse_source_date(raw)
    source_row = _source_row_number(raw)
    row_id = stable_row_id(source_dataset_id, source_row)
    raw_customer = _raw_customer(raw)
    customer_key = canonical_customer_key(raw)
    legacy_customer_key = legacy_policy_customer_key(raw)
    clean = _clean_token(raw.get("process_code"))
    part = _clean_token(raw.get("part_code"))
    product_key = f"{clean}-{part}" if clean and part else clean or part or "UNCLASSIFIED"
    episode_id, episode_basis = _episode_identity(
        raw,
        customer_key=customer_key,
        product_key=product_key,
        observed_date=observed_date,
        source_row=source_row,
    )
    features = {
        name: raw.get(name, "")
        for name in ALLOWED_RAW_FEATURES
    }
    return FeatureRow(
        row_id=row_id,
        source_row=source_row,
        date=observed_date,
        month=observed_date[:7],
        raw_customer=raw_customer,
        customer_key=customer_key,
        legacy_customer_key=legacy_customer_key,
        product_key=product_key,
        episode_id=episode_id,
        episode_basis=episode_basis,
        features=MappingProxyType(features),
    )


def _actual_label(raw: Mapping[str, Any]) -> str:
    return str(raw.get("action_result") or "").strip()


def _source_row_number(raw: Mapping[str, Any]) -> int:
    value = raw.get("__sourceRowNumber")
    try:
        number = int(float(value))
    except (TypeError, ValueError) as error:
        raise DataContractError(f"invalid __sourceRowNumber: {value!r}") from error
    if number < 1:
        raise DataContractError(f"invalid __sourceRowNumber: {value!r}")
    return number


def _append_unique(target: list[FeatureRow], row: FeatureRow, seen_ids: set[str]) -> None:
    if row.row_id in seen_ids:
        raise DataContractError(f"duplicate stable row id: {row.row_id}")
    seen_ids.add(row.row_id)
    target.append(row)


# 고객사 표기 정규화 규칙: (정규식, 정규화 키). 원본은 현장 고객사 표기를 몇 개의 키로 묶었지만,
# 공개본에서는 비워 둔다. 필요하면 실행 전에 이 튜플을 채운다.
CUSTOMER_ALIASES: tuple[tuple[str, str], ...] = ()
# 과거 정책용 키는 고객사·라인·설비 문자열을 함께 보고 판단한다.
LEGACY_CUSTOMER_ALIASES: tuple[tuple[str, str], ...] = ()


def canonical_customer_key(row: Mapping[str, Any]) -> str:
    customer = _raw_customer(row)
    text = customer.lower()
    for pattern, key in CUSTOMER_ALIASES:
        if re.search(pattern, text):
            return key
    return customer


def legacy_policy_customer_key(row: Mapping[str, Any]) -> str:
    customer = _raw_customer(row)
    line = str(row.get("line") or "").strip()
    equipment = str(row.get("customer_equipment") or "").strip()
    text = f"{customer} {line} {equipment}".lower()
    for pattern, key in LEGACY_CUSTOMER_ALIASES:
        if re.search(pattern, text):
            return key
    if customer:
        return re.sub(r"\s+", " ", customer).strip()
    fallback = line or equipment
    if fallback:
        cleaned_fallback = re.sub(r"\s+", " ", fallback).strip()
        return f"UNCLASSIFIED:{cleaned_fallback}"
    return "Other"


def _raw_customer(row: Mapping[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(row.get("customer") or "")).strip() or "UNCLASSIFIED"


def _episode_identity(
    row: Mapping[str, Any],
    *,
    customer_key: str,
    product_key: str,
    observed_date: str,
    source_row: int,
) -> tuple[str, str]:
    identifier = ""
    basis = ""
    for field_name in ("barcode", "document_no", "customer_serial", "serial"):
        value = unicodedata.normalize("NFKC", str(row.get(field_name) or "")).strip()
        if value:
            identifier = value
            basis = field_name
            break
    if not identifier:
        identifier = str(source_row)
        basis = "__sourceRowNumber"
    payload = "\x1f".join((customer_key, identifier, product_key, observed_date)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), basis


def _validate_immutable_config(config: Mapping[str, Any]) -> None:
    for key in (
        "historyRowsSha256",
        "auditRowsSha256",
        "prepareManifestSha256",
        "prepareScriptSha256",
        "rawWorkbookSha256",
    ):
        value = str(config.get(key) or "").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise DataContractError(f"{key} is required and must be a 64-character SHA-256")
    chronology = (
        tuple(config.get("selectorMonths", ())),
        tuple(config.get("lockboxMonths", ())),
        str(config.get("auditStart") or ""),
    )
    expected_chronology = (
        IMMUTABLE_SELECTOR_MONTHS,
        IMMUTABLE_LOCKBOX_MONTHS,
        IMMUTABLE_AUDIT_START,
    )
    if chronology != expected_chronology:
        raise DataContractError(
            f"immutable chronology mismatch: expected {expected_chronology}, observed {chronology}"
        )
    if dict(config.get("evaluationStatus") or {}) != IMMUTABLE_EVALUATION_STATUS:
        raise DataContractError("immutable retrospective evaluationStatus mismatch")
    for key in (
        "expectedAuditRawRows",
        "expectedAuditActionRows",
        "expectedAuditNonActionRows",
        "expectedAuditCustomers",
        "expectedAuditLegacyCustomers",
        "expectedAuditRawCustomers",
        "expectedHistoryRawRows",
    ):
        if key not in config:
            raise DataContractError(f"{key} is required")
    if int(config["expectedAuditActionRows"]) + int(config["expectedAuditNonActionRows"]) != int(
        config["expectedAuditRawRows"]
    ):
        raise DataContractError("expected audit action and non-action rows must sum to raw rows")


def _validate_prepare_provenance(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("prepareManifest", "rawWorkbook"):
        if not str(config.get(key) or ""):
            raise DataContractError(f"{key} is required for prepare manifest provenance")
    prepare_path = Path(str(config["prepareManifest"])).resolve()
    raw_workbook = Path(str(config["rawWorkbook"])).resolve()
    if _sha256_file(prepare_path) != str(config["prepareManifestSha256"]).lower():
        raise DataContractError("prepare manifest SHA-256 mismatch")
    if _sha256_file(raw_workbook) != str(config["rawWorkbookSha256"]).lower():
        raise DataContractError("prepare manifest raw workbook SHA-256 mismatch")
    try:
        manifest = json.loads(prepare_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError("prepare manifest is not readable JSON") from error
    if not isinstance(manifest, dict):
        raise DataContractError("prepare manifest must be a JSON object")
    operation = manifest.get("operation") or {}
    expected_operation = {
        "routingOnly": True,
        "engineLoaded": False,
        "normalizationApplied": False,
        "candidateSelectionApplied": False,
        "fitApplied": False,
        "scoringApplied": False,
    }
    if any(operation.get(key) != expected for key, expected in expected_operation.items()):
        raise DataContractError("prepare manifest operation is not routing-only")
    source = manifest.get("source") or {}
    if source.get("scriptSha256") != str(config["prepareScriptSha256"]).lower():
        raise DataContractError("prepare manifest script SHA-256 mismatch")
    workbook = source.get("rawWorkbook") or {}
    if workbook.get("sha256") != str(config["rawWorkbookSha256"]).lower():
        raise DataContractError("prepare manifest recorded workbook SHA-256 mismatch")
    if int(workbook.get("sizeBytes") or -1) != raw_workbook.stat().st_size:
        raise DataContractError("prepare manifest recorded workbook size mismatch")

    split_contracts = {
        "history": (
            "historyRowsSha256",
            "expectedHistoryRawRows",
        ),
        "audit2026": (
            "auditRowsSha256",
            "expectedAuditRawRows",
        ),
    }
    for split_name, (hash_key, rows_key) in split_contracts.items():
        split = (manifest.get("splits") or {}).get(split_name) or {}
        if split.get("sha256") != str(config[hash_key]).lower():
            raise DataContractError(f"prepare manifest {split_name} split SHA-256 mismatch")
        if int(split.get("rows") or -1) != int(config[rows_key]):
            raise DataContractError(f"prepare manifest {split_name} row-count mismatch")
    return MappingProxyType(manifest)


def _clean_token(value: Any) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).upper()


def _parse_date_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 20_000 < float(value) < 80_000:
            return (date(1899, 12, 30) + timedelta(days=int(float(value)))).isoformat()
        return ""
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return ""
    separated = re.search(r"\b(\d{2,4})[./-](\d{1,2})[./-](\d{1,2})\b", text)
    if separated:
        year = int(separated.group(1))
        year = 1900 + year if 70 <= year < 100 else 2000 + year if year < 70 else year
        return _validated_date(year, int(separated.group(2)), int(separated.group(3)))
    compact = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", text)
    if compact:
        return _validated_date(*(int(compact.group(index)) for index in (1, 2, 3)))
    return ""


def _validated_date(year: int, month: int, day: int) -> str:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return ""


def _date_range(rows: Sequence[FeatureRow]) -> tuple[str, str]:
    if not rows:
        return "", ""
    dates = [row.date for row in rows]
    return min(dates), max(dates)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataContractError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise DataContractError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

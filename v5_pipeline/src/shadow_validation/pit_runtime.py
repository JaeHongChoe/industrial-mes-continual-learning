from __future__ import annotations

from datetime import date
from types import MappingProxyType
from typing import Any, Iterable, Mapping
import warnings

from scipy import sparse

from src import baseline_risk_veto as danger_runtime
from src.baseline_risk_veto import DangerScoreRow


SHIP_LABEL = "RETURN"
REPAIR_LABEL = "REPAIR"
STATE_KIND = "pit_danger_model_state"


class PitDangerModel:
    """Inference-only view of a fitted point-in-time danger model.

    The training wrapper deliberately exports only fitted encoders, scalers,
    heads and calibrators. Labels and the 2026 audit outcome are therefore not
    available on this runtime path.
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state = _validated_state(state)
        self._raw_encoder = self._state["rawEncoder"]
        self._domain_scaler = self._state["domainScaler"]
        self._temporal_scaler = self._state["temporalScaler"]
        self._domain_schema_sha256 = str(self._state["domainSchemaSha256"])
        self._temporal_schema_sha256 = str(self._state["temporalSchemaSha256"])
        self._calibration_cutoff = date.fromisoformat(
            str(self._state["calibrationCutoff"])
        )
        self._heads = self._state["heads"]
        self._calibrators = self._state["calibrators"]
        self._fit_receipt = MappingProxyType(dict(self._state["fitReceipt"]))
        self._feature_names = tuple(self._state["featureNames"])

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PitDangerModel":
        return cls(state)

    @property
    def fit_receipt(self) -> Mapping[str, Any]:
        return self._fit_receipt

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def score_rows(
        self,
        rows: Iterable[Any],
        *,
        temporal: Any | None,
        domain: sparse.spmatrix,
        domain_schema_sha256: str,
    ) -> tuple[DangerScoreRow, ...]:
        materialized = tuple(rows)
        if any(date.fromisoformat(row.date) <= self._calibration_cutoff for row in materialized):
            raise ValueError("PIT danger score rows must be after probability calibration")
        if str(domain_schema_sha256) != self._domain_schema_sha256:
            raise ValueError("domain feature schema does not match the PIT danger model")
        domain_matrix = sparse.csr_matrix(domain, dtype="float32")
        if domain_matrix.shape[0] != len(materialized):
            raise ValueError("PIT domain feature row count mismatch")
        temporal_matrix = danger_runtime._transform_temporal_block(
            temporal,
            tuple(row.row_id for row in materialized),
            self._temporal_scaler,
            self._temporal_schema_sha256,
        )
        matrix = sparse.hstack(
            (
                self._raw_encoder.transform(materialized),
                self._domain_scaler.transform(domain_matrix),
                temporal_matrix,
            ),
            format="csr",
        )
        # The frozen LightGBM heads were fitted through a named dataframe, while
        # this inference-only path deliberately keeps the large matrix sparse.
        # LightGBM emits one harmless warning per fitted submodel for that
        # representation difference. Suppress that exact message only; model,
        # schema and calibration warnings remain visible and fail normally.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMClassifier "
                    "was fitted with feature names"
                ),
                category=UserWarning,
            )
            shipment = danger_runtime._calibrated_probability(
                self._heads[SHIP_LABEL],
                self._calibrators[SHIP_LABEL],
                matrix,
            )
            repair = danger_runtime._calibrated_probability(
                self._heads[REPAIR_LABEL],
                self._calibrators[REPAIR_LABEL],
                matrix,
            )
        return tuple(
            DangerScoreRow(
                row_id=row.row_id,
                month=row.month,
                shipment_probability=float(shipment[index]),
                repair_probability=float(repair[index]),
            )
            for index, row in enumerate(materialized)
        )


def _validated_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schemaVersion",
        "kind",
        "rawEncoder",
        "domainScaler",
        "temporalScaler",
        "domainSchemaSha256",
        "temporalSchemaSha256",
        "calibrationCutoff",
        "heads",
        "calibrators",
        "fitReceipt",
        "featureNames",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"PIT danger state is missing fields: {missing}")
    if state["schemaVersion"] != 1 or state["kind"] != STATE_KIND:
        raise ValueError("unsupported PIT danger state schema")
    date.fromisoformat(str(state["calibrationCutoff"]))
    if set(state["heads"]) != {SHIP_LABEL, REPAIR_LABEL}:
        raise ValueError("PIT danger head label universe mismatch")
    if set(state["calibrators"]) != {SHIP_LABEL, REPAIR_LABEL}:
        raise ValueError("PIT danger calibrator label universe mismatch")
    if not isinstance(state["fitReceipt"], Mapping):
        raise ValueError("PIT danger fit receipt must be a mapping")
    return MappingProxyType(dict(state))

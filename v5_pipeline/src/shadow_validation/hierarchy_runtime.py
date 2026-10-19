from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from scripts.materialize_hierarchical_evidence import CandidateProfile
from scripts.materialize_policy_algebra import algebra_policies_from_registry
from scripts.run_hierarchical_lockbox_audit import (
    _apply_process_veto,
    _load_routes,
    _score_phase,
)
from src.data_contract import FeatureRow

from .integrity import sha256_file


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = PACKAGE_ROOT / "frozen"


@dataclass(frozen=True, slots=True)
class HierarchyResult:
    predictions: Mapping[str, str]
    vetoed_row_ids: tuple[str, ...]


class HierarchyPredictor:
    def __init__(self, frozen_root: str | Path = FROZEN_ROOT) -> None:
        self.frozen_root = Path(frozen_root).resolve()
        profile_payload = _load_json(self.frozen_root / "hierarchy/profiles.json")
        generation = _load_json(self.frozen_root / "generation.json")
        self.profiles = MappingProxyType(
            {
                item["candidateId"]: _restore_profile(
                    item,
                    frozen_root=self.frozen_root,
                    source_dataset_id=str(generation["sourceDatasetId"]),
                )
                for item in profile_payload["profiles"]
            }
        )
        evidence = _load_json(self.frozen_root / "hierarchy/evidence.json")
        self.algebra = algebra_policies_from_registry(evidence["candidates"])
        budget = _load_json(self.frozen_root / "hierarchy/budgeted_policy.json")
        self.company_routes, self.line_routes = _load_routes(budget)
        report = _load_json(self.frozen_root / "hierarchy/lockbox_audit_report.json")
        self.veto_mode = str(report["lockbox"]["vetoMode"])
        self.rejected_scopes = frozenset(
            tuple(str(value) for value in scope)
            for scope in report["lockbox"]["rejectedProcessScopes"]
        )

    def predict(
        self,
        rows: Sequence[FeatureRow],
        baseline_predictions: Mapping[str, str],
    ) -> HierarchyResult:
        expected = {row.row_id for row in rows}
        if set(baseline_predictions) != expected:
            raise ValueError("hierarchy baseline row universe mismatch")
        with tempfile.TemporaryDirectory(prefix="mes-shadow-hierarchy-") as temporary:
            baseline_path = Path(temporary) / "baseline.jsonl"
            baseline_path.write_text(
                "".join(
                    json.dumps(
                        {"row_id": row_id, "predicted": prediction},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for row_id, prediction in sorted(baseline_predictions.items())
                ),
                encoding="utf-8",
            )
            initial = _score_phase(
                rows,
                baseline_path=baseline_path,
                profiles=self.profiles,
                company_routes=self.company_routes,
                line_routes=self.line_routes,
                algebra=self.algebra,
            )
        final = _apply_process_veto(
            rows,
            initial,
            self.rejected_scopes,
            mode=self.veto_mode,
        )
        vetoed = tuple(
            sorted(
                row_id
                for row_id in expected
                if initial[row_id] == "PROCESS" and final[row_id] != "PROCESS"
            )
        )
        return HierarchyResult(
            predictions=MappingProxyType(dict(sorted(final.items()))),
            vetoed_row_ids=vetoed,
        )


def _restore_profile(
    item: Mapping[str, object],
    *,
    frozen_root: Path,
    source_dataset_id: str,
) -> CandidateProfile:
    manifest = (frozen_root / str(item["runtimeManifest"])).resolve()
    expected_sha = str(item["runtimeManifestSha256"])
    if sha256_file(manifest) != expected_sha:
        raise ValueError(f"hierarchy runtime manifest SHA mismatch: {manifest}")
    return CandidateProfile(
        candidate_id=str(item["candidateId"]),
        family=str(item["family"]),
        base_model_id=str(item["baseModelId"]),
        fitted_manifest_path=manifest,
        fitted_manifest_sha256=expected_sha,
        thresholds=MappingProxyType(
            {str(key): float(value) for key, value in item["thresholds"].items()}
        ),
        threshold_enabled=MappingProxyType(
            {str(key): bool(value) for key, value in item["thresholdEnabled"].items()}
        ),
        threshold_profile=str(item["thresholdProfile"]),
        train_end=str(item["trainEnd"]),
        selector_months=tuple(str(value) for value in item["selectorMonths"]),
        source_dataset_id=source_dataset_id,
        source_run=frozen_root / "hierarchy/profiles.json",
        scope_company=(
            str(item["scopeCompany"]) if item.get("scopeCompany") is not None else None
        ),
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

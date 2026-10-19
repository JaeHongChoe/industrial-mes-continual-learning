from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.materialize_hierarchical_evidence import (
    LOOP93_CANDIDATE_ID,
    _sha256_file,
    _sha256_json,
    _write_gzip_jsonl_atomic,
    _write_json_atomic,
)
from src.hierarchical_evidence import PredictionDecision, build_scope_evidence
from src.policy_algebra import AlgebraPolicy, combine_prediction


CA990 = "binary_process_customer_aware_v1__precision_0_990"
CA995 = "binary_process_customer_aware_v1__precision_0_995"
CA000 = "binary_process_customer_aware_v1__zero_unsafe"
PI990 = "binary_process_product_independent_v1__precision_0_990"
PI995 = "binary_process_product_independent_v1__precision_0_995"
PI000 = "binary_process_product_independent_v1__zero_unsafe"
LGBM995 = "customer_aware_lightgbm_v1__precision_0_995"
LGBM000 = "customer_aware_lightgbm_v1__zero_unsafe"
TEMPORAL = "temporal_signature_v1"


ALGEBRA_POLICIES = (
    AlgebraPolicy("alg_recover_ca995", (), 0, (CA995,), 1),
    AlgebraPolicy("alg_recover_pi995", (), 0, (PI995,), 1),
    AlgebraPolicy("alg_recover_any995", (), 0, (CA995, PI995), 1),
    AlgebraPolicy("alg_recover_both995", (), 0, (CA995, PI995), 2),
    AlgebraPolicy("alg_recover_any990", (), 0, (CA990, PI990), 1),
    AlgebraPolicy("alg_recover_both990", (), 0, (CA990, PI990), 2),
    AlgebraPolicy("alg_recover_any_zero", (), 0, (CA000, PI000), 1),
    AlgebraPolicy("alg_recover_both_zero", (), 0, (CA000, PI000), 2),
    AlgebraPolicy("alg_guard_ca995", (CA995,), 1, (), 0),
    AlgebraPolicy("alg_guard_pi995", (PI995,), 1, (), 0),
    AlgebraPolicy("alg_guard_any995", (CA995, PI995), 1, (), 0),
    AlgebraPolicy("alg_guard_both995", (CA995, PI995), 2, (), 0),
    AlgebraPolicy(
        "alg_guard_any995_recover_both995",
        (CA995, PI995),
        1,
        (CA995, PI995),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_both995_recover_both995",
        (CA995, PI995),
        2,
        (CA995, PI995),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any990_recover_both995",
        (CA990, PI990),
        1,
        (CA995, PI995),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_both990_recover_both995",
        (CA990, PI990),
        2,
        (CA995, PI995),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_both990",
        (CA995, PI995),
        1,
        (CA990, PI990),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_both995_recover_both990",
        (CA995, PI995),
        2,
        (CA990, PI990),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_ca_lgbm",
        (CA995, PI995),
        1,
        (CA995, LGBM000),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_pi_lgbm",
        (CA995, PI995),
        1,
        (PI995, LGBM000),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_2of3",
        (CA995, PI995),
        1,
        (CA995, PI995, LGBM000),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_both995_recover_2of3",
        (CA995, PI995),
        2,
        (CA995, PI995, LGBM000),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_ca_temporal",
        (CA995, PI995),
        1,
        (CA995, TEMPORAL),
        2,
    ),
    AlgebraPolicy(
        "alg_guard_any995_recover_2of4",
        (CA995, PI995),
        1,
        (CA995, PI995, LGBM995, TEMPORAL),
        2,
    ),
)


def build_scoped_algebra_policies(
    candidate_registry: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[AlgebraPolicy, str], ...]:
    """Create deterministic baseline overlays for company-local model opinions."""

    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    output: list[tuple[AlgebraPolicy, str]] = []
    for candidate_id, candidate in sorted(candidate_registry.items()):
        owner = candidate.get("scopeCompany")
        family = str(candidate.get("family") or "")
        profile = str(candidate.get("thresholdProfile") or "")
        if not owner or family not in {"binary_process", "multiclass"}:
            continue
        owner = str(owner)
        safe_id = _safe_policy_token(candidate_id)
        output.extend(
            (
                (
                    AlgebraPolicy(
                        f"scoped_{safe_id}__recover",
                        (),
                        0,
                        (candidate_id,),
                        1,
                    ),
                    owner,
                ),
                (
                    AlgebraPolicy(
                        f"scoped_{safe_id}__guard_recover",
                        (candidate_id,),
                        1,
                        (candidate_id,),
                        1,
                    ),
                    owner,
                ),
            )
        )
        grouped.setdefault((owner, profile), {}).setdefault(family, []).append(
            candidate_id
        )

    for (owner, profile), families in sorted(grouped.items()):
        binaries = sorted(families.get("binary_process", ()))
        multiclass = sorted(families.get("multiclass", ()))
        for binary in binaries:
            for multi in multiclass:
                pair = (binary, multi)
                token = _safe_policy_token(f"{owner}_{profile}_{binary}_{multi}")
                output.extend(
                    (
                        (
                            AlgebraPolicy(
                                f"scoped_{token}__recover_any",
                                (),
                                0,
                                pair,
                                1,
                            ),
                            owner,
                        ),
                        (
                            AlgebraPolicy(
                                f"scoped_{token}__recover_both",
                                (),
                                0,
                                pair,
                                2,
                            ),
                            owner,
                        ),
                        (
                            AlgebraPolicy(
                                f"scoped_{token}__guard_any_recover_both",
                                pair,
                                1,
                                pair,
                                2,
                            ),
                            owner,
                        ),
                        (
                            AlgebraPolicy(
                                f"scoped_{token}__guard_both_recover_both",
                                pair,
                                2,
                                pair,
                                2,
                            ),
                            owner,
                        ),
                    )
                )
    ids = [policy.policy_id for policy, _ in output]
    if len(ids) != len(set(ids)):
        raise ValueError("scoped algebra policy ids are not unique")
    return tuple(output)


def algebra_policies_from_registry(
    candidate_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, AlgebraPolicy]:
    output: dict[str, AlgebraPolicy] = {}
    for candidate_id, candidate in candidate_registry.items():
        if candidate.get("family") != "policy_algebra":
            continue
        output[candidate_id] = AlgebraPolicy(
            candidate_id,
            tuple(str(item) for item in candidate.get("guardCandidates", ())),
            int(candidate.get("guardVotesRequired", 0)),
            tuple(str(item) for item in candidate.get("recoveryCandidates", ())),
            int(candidate.get("recoveryVotesRequired", 0)),
        )
    return output


def materialize_policy_algebra(
    evidence_path: str | Path,
    output_dir: str | Path,
) -> Path:
    source_path = Path(evidence_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    output_root = Path(output_dir).resolve()
    decision_dir = output_root / "selector_decisions"
    decision_dir.mkdir(parents=True, exist_ok=True)

    policy_specs = (
        *((policy, None) for policy in ALGEBRA_POLICIES),
        *build_scoped_algebra_policies(source["candidates"]),
    )
    required_candidates = sorted(
        {
            candidate_id
            for policy, _owner in policy_specs
            for candidate_id in (
                *policy.guard_candidates,
                *policy.recovery_candidates,
            )
        }
    )
    baseline = tuple(
        _read_jsonl_gzip(
            Path(source["candidates"][LOOP93_CANDIDATE_ID]["decisionPath"])
        )
    )
    baseline_by_id = {str(row["row_id"]): row for row in baseline}
    candidate_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for candidate_id in required_candidates:
        rows = {
            str(row["row_id"]): row
            for row in _read_jsonl_gzip(
                Path(source["candidates"][candidate_id]["decisionPath"])
            )
        }
        if set(rows) != set(baseline_by_id):
            raise ValueError(f"algebra candidate row identity mismatch: {candidate_id}")
        candidate_rows[candidate_id] = rows

    scope_evidence = list(source["scopeEvidence"])
    registry = dict(source["candidates"])
    for index, (policy, scope_company) in enumerate(policy_specs, start=1):
        decisions: list[PredictionDecision] = []
        needed = set(policy.guard_candidates) | set(policy.recovery_candidates)
        for baseline_row in baseline:
            row_id = str(baseline_row["row_id"])
            rows = {candidate_id: candidate_rows[candidate_id][row_id] for candidate_id in needed}
            predictions = {
                candidate_id: str(row["predicted"])
                for candidate_id, row in rows.items()
            }
            predicted = combine_prediction(
                str(baseline_row["predicted"]), predictions, policy
            )
            train_dates = [
                str(row.get("train_end", ""))
                for row in (baseline_row, *rows.values())
                if str(row.get("train_end", ""))
            ]
            decisions.append(
                PredictionDecision(
                    candidate_id=policy.policy_id,
                    row_id=row_id,
                    episode_id=str(baseline_row["episode_id"]),
                    company=str(baseline_row["company"]),
                    raw_customer=str(baseline_row["raw_customer"]),
                    line=str(baseline_row.get("line", "")),
                    month=str(baseline_row["month"]),
                    actual=str(baseline_row["actual"]),
                    predicted=predicted,
                    train_end=max(train_dates) if train_dates else "",
                )
            )
        evidence = build_scope_evidence(decisions)
        scope_evidence.extend(asdict(item) for item in evidence)
        decision_path = decision_dir / f"{policy.policy_id}.jsonl.gz"
        _write_gzip_jsonl_atomic(
            decision_path, (item.as_dict() for item in decisions)
        )
        global_rows = [item for item in evidence if item.scope_id == "global"]
        registry[policy.policy_id] = {
            "candidateId": policy.policy_id,
            "baseModelId": LOOP93_CANDIDATE_ID,
            "family": "policy_algebra",
            "scopeCompany": scope_company,
            "thresholdProfile": "frozen_vote_algebra",
            "decisionPath": str(decision_path),
            "decisionSha256": _sha256_file(decision_path),
            "rows": len(decisions),
            "autoRows": sum(item.auto_rows for item in global_rows),
            "correctAutoRows": sum(item.correct_auto_rows for item in global_rows),
            "guardCandidates": list(policy.guard_candidates),
            "guardVotesRequired": policy.guard_votes_required,
            "recoveryCandidates": list(policy.recovery_candidates),
            "recoveryVotesRequired": policy.recovery_votes_required,
        }
        print(f"[{index}/{len(policy_specs)}] {policy.policy_id}", flush=True)

    payload = {
        **{key: value for key, value in source.items() if key != "payloadSha256"},
        "kind": "hierarchical_selector_evidence_with_policy_algebra",
        "parentEvidencePath": str(source_path),
        "parentEvidenceSha256": _sha256_file(source_path),
        "candidates": registry,
        "scopeEvidence": sorted(
            scope_evidence,
            key=lambda item: (
                item["candidate_id"], item["scope_id"], item["month"]
            ),
        ),
        "rowCounts": {
            **source["rowCounts"],
            "candidates": len(registry),
            "decisionRows": source["rowCounts"]["selector"] * len(registry),
            "algebraCandidates": len(policy_specs),
        },
    }
    payload["payloadSha256"] = _sha256_json(payload)
    report_path = output_root / "scope_evidence_algebra_2025.json"
    _write_json_atomic(report_path, payload)
    return report_path


def _safe_policy_token(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )


def _read_jsonl_gzip(path: Path) -> Iterable[Mapping[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence", default="outputs/evidence/scope_evidence_2025.json"
    )
    parser.add_argument("--output-dir", default="outputs/algebra")
    args = parser.parse_args()
    print(materialize_policy_algebra(args.evidence, args.output_dir))


if __name__ == "__main__":
    main()

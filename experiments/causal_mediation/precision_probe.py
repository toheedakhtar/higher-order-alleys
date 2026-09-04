"""Synthetic-only numerical qualification. Does not load model or judgments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .precision import POLICY, realize_coordinate, realize_random_pair
from .upstream import DEFAULT_DIRECTION, DEFAULT_UPSTREAM, load_upstream, sha


def run_synthetic(vector):
    rows = []
    gen = torch.Generator().manual_seed(20260905)
    base = torch.randn(vector.numel(), generator=gen).to(torch.bfloat16)
    # Fixed scale grid includes easy and difficult lattice geometries. These
    # are NOT estimated natural scales or a surrogate for real smoke items.
    cases = [(f"scale_{scale}_delta_{delta}", (base.float() * scale).bfloat16(), delta)
             for scale in (0.01, 1.0, 100.0)
             for delta in (0.001, 0.01, 0.1, 1.0)]
    cases += [("zero_residual", torch.zeros_like(base), 0.01), ("sham", base, 0.0)]
    for name, h, delta in cases:
        _, report = realize_coordinate(h, vector, delta)
        report.update({"case": name, "kind": "candidate", "jlens_score_before": None,
                       "jlens_score_after": None, "jlens_status": "unavailable_without_pinned_model_and_lens"})
        rows.append(report)
        for _, control in realize_random_pair(h, vector, report["total_patch_l2"],
                seed=42, item_id=name, branch="synthetic", donor="synthetic_donor"):
            control.update({"case": name, "kind": "random", "jlens_score_before": None,
                            "jlens_score_after": None, "jlens_status": "unavailable_without_pinned_model_and_lens"})
            rows.append(control)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--direction", type=Path, default=DEFAULT_DIRECTION)
    args = parser.parse_args(argv)
    # Never overwrite an existing diagnostic.
    if args.output.exists():
        raise FileExistsError(args.output)
    vector, _, _, identity = load_upstream(args.upstream, args.direction)
    rows = run_synthetic(vector)
    candidates = [row for row in rows if row["kind"] == "candidate"]
    report = {
        "stage": "synthetic_precision_only", "policy": asdict(POLICY),
        "policy_sha256": POLICY.digest(), "identity": identity,
        "implementation_sha256": {p.name: sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "behavioral_outcomes_used": False, "real_smoke_executed": False,
        "mediation_execution_authorized": False,
        "candidate_cases": len(candidates),
        "candidate_cases_passed": sum(row["precision_gate_passed"] for row in candidates),
        "candidate_cases_certified_infeasible": sum(row["lattice_bound"]["certified_infeasible"] for row in candidates),
        "measurements": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("stage", "candidate_cases", "candidate_cases_passed", "candidate_cases_certified_infeasible", "real_smoke_executed")}))
    return 0  # Diagnostic completion is never experimental authorization.


if __name__ == "__main__":
    raise SystemExit(main())

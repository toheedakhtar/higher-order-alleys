"""Synthetic geometry with the exact frozen vector; no model outcomes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .mixed_precision import POLICY, realize_patch
from .upstream import DEFAULT_DIRECTION, DEFAULT_UPSTREAM, load_upstream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    vector, _, _, identity = load_upstream(DEFAULT_UPSTREAM, DEFAULT_DIRECTION)
    generator = torch.Generator().manual_seed(1729)
    rows = []
    for scale in (.01, 1., 100.):
        h = (scale*torch.randn(5120, generator=generator)).bfloat16()
        for delta in (0., .001, .003, .01, .03, -.01):
            _, report = realize_patch(h, delta*vector.double(), vector, intended_coordinate=delta)
            report["coordinate_accuracy_passed"] = report["absolute_coordinate_error"] <= .01*abs(delta)
            rows.append({"synthetic_residual_scale": scale, "delta": delta, **report})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"policy": asdict(POLICY), "candidate": identity["frozen_candidate"],
            "behavioral_outcomes_used": False, "rows": rows,
            "all_passed": all(r["coordinate_accuracy_passed"] and r["noise_bound_passed"] for r in rows)}, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"cases": len(rows), "max_relative_coordinate_error": max(r["relative_coordinate_error"] or 0 for r in rows),
        "max_orthogonal_leakage_l2": max(r["orthogonal_leakage_l2"] for r in rows),
        "all_passed": all(r["coordinate_accuracy_passed"] and r["noise_bound_passed"] for r in rows)}))


if __name__ == "__main__":
    main()

"""Fail-closed analysis for the provenance experiment.

Reads a completed holdout run directory (records.jsonl + status.json) and
scores the three predeclared rungs on support-matched items only:

  R1 content exclusion : v separates planted_a from clean with identical text.
  R2 damage exclusion  : v separates planted_a from the matched random
                         operator, the exact contrast v was discovered under.
  R3 causal mediation  : transplanting the clean coordinate into the planted
                         state moves the judgment margin toward the clean
                         state, beyond a norm-matched random transplant.

Also reported (non-blocking): P1-vs-P2 convergence diagnostic, support-drop
matching, sham parity, patch geometry, and the delta-vs-support-drop
relationship.

  uv run python -m experiments.provenance.analysis assets/provenance-v1/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.provenance.protocol import bootstrap_ci_mean

# Predeclared decision thresholds (do not relax after seeing data)
MIN_POSITIVE_FRACTION = 0.65   # share of items agreeing in sign, bootstrap draws
MIN_ITEMS = 12                 # minimum support-matched holdout items for any verdict
R2_EFFECT_FLOOR = 0.0          # R2 must exceed the random operator, CI lo > 0


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def pass_or_fail(ci, need_lo_above_zero=True):
    ok = ci["lo"] > 0 and ci["positive_fraction"] >= MIN_POSITIVE_FRACTION
    return ok if need_lo_above_zero else (ci["positive_fraction"] >= MIN_POSITIVE_FRACTION)


def analyze(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text())
    docs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines() if l.strip()]
    if run_dir.name.endswith("calib") or all("score_planted_a" not in d for d in docs):
        el = [d for d in docs if d.get("eligible")]
        return dict(kind="calib_summary", n_eligible=len(el))
    matched = [d for d in docs if d.get("support_match_ok") and "score_planted_a" in d]
    cfg = json.loads((run_dir / "run_manifest.json").read_text())["config"]

    r1 = bootstrap_ci_mean([d["score_planted_a"] - d["score_clean"] for d in matched], cfg["bootstrap"], cfg["seed"])
    r2 = bootstrap_ci_mean([d["score_planted_a"] - d["score_random"] for d in matched], cfg["bootstrap"], cfg["seed"])
    step_restore = [d["margin_restored"] - d["margin_planted_a"] for d in matched]
    r3_restore = bootstrap_ci_mean(step_restore, cfg["bootstrap"], cfg["seed"])
    r3_contrast = bootstrap_ci_mean(
        [d["margin_restored"] - d["margin_random_transplant"] for d in matched],
        cfg["bootstrap"], cfg["seed"])
    conv = bootstrap_ci_mean([d["score_planted_a"] - d["score_planted_b"] for d in matched],
                             cfg["bootstrap"], cfg["seed"])

    drop_care = spearman(
        [d["delta"] for d in matched],
        [(d["op_a"]["drop"] + d["op_b"]["drop"] + d["op_m"]["drop"]) / 3 for d in matched])

    gates = status.get("gates", {})
    gates_ok = bool(status.get("complete")) and bool(gates.get("patch_geometry")) \
        and bool(gates.get("sham_parity"))

    verdicts = dict(
        R1_content_exclusion=dict(ci=r1, passed=pass_or_fail(r1) and len(matched) >= MIN_ITEMS),
        R2_damage_exclusion=dict(ci=r2, passed=pass_or_fail(r2) and len(matched) >= MIN_ITEMS),
        R3_causal_mediation=dict(restore_ci=r3_restore, contrast_ci=r3_contrast,
                                 passed=(r3_restore["lo"] > 0 and r3_contrast["lo"] > 0
                                         and r3_contrast["positive_fraction"] >= MIN_POSITIVE_FRACTION
                                         and len(matched) >= MIN_ITEMS)),
    )
    summary = dict(
        run_dir=str(run_dir), status_complete=bool(status.get("complete")),
        run_gates_ok=bool(gates_ok), n_matched=len(matched),
        rungs=verdicts,
        convergence_diagnostic_p1_vs_p2=dict(ci=conv,
                                             interpretation=("compatible with convergence" if conv["lo"] <= 0 <= conv["hi"] else "not converged"),
                                             blocking=False),
        delta_vs_support_spearman=round(drop_care, 4),
        support_match=dict(
            mean_delta_magnitude=round(sum(abs(d["delta"]) for d in matched) / max(len(matched), 1), 6),
            max_patch_abs_err=max((d.get("patch_abs_err", 0.0) for d in matched), default=None),
            max_sham_margin_error=max((abs(d["margin_sham"] - d["margin_planted_a"]) for d in matched), default=None)),
        earned_claim=("higher-order/provenance candidate EARNS the frozen claim chain only if "
                      "R1, R2 and R3 all pass and the run gates were true; otherwise report "
                      "boundary-limited evidence exactly as in the repo's sprint reports")
        if all(v["passed"] for v in verdicts.values()) and gates_ok else
        "R1/R2/R3 chain not earned on this run",
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "report.md").write_text(render_report(summary))
    return summary


def render_report(s: dict) -> str:
    lines = [
        "# Provenance experiment report",
        "",
        f"Run: `{s['run_dir']}`",
        f"Matched holdout items: {s['n_matched']}",
        f"Run gates passed: {s['run_gates_ok']}",
        "",
        "## Rungs",
        "",
    ]
    for name, v in s["rungs"].items():
        cis = [f"`{k}={json.dumps(c)}`" for k, c in v.items() if isinstance(c, dict) and "lo" in c]
        lines += [f"- **{name}**: {'PASS' if v['passed'] else 'FAIL'}; " + "; ".join(cis)]
    c = s["convergence_diagnostic_p1_vs_p2"]
    lines += ["", f"Convergence diagnostic (P1 vs P2, non-blocking): {c['interpretation']}; `ci={json.dumps(c['ci'])}`"]
    lines += ["",
              "## Claim",
              "",
              s["earned_claim"],
              "",
              "R3 depends on patch-fidelity and sham-parity gates recorded in `status.json`."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.provenance.analysis <run_dir>")
    s = analyze(Path(sys.argv[1]))
    print(json.dumps(s, indent=2)[:3000])

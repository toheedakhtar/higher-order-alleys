"""Mode-aware, sample-level analysis for global J-Lens experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return center - half, center + half


def bootstrap_item_interval(
    frame: pd.DataFrame,
    value: str,
    rng: np.random.Generator,
    iterations: int = 4000,
) -> tuple[float, float]:
    """Bootstrap item means so attempts never masquerade as independent samples."""
    values = frame.groupby("item_id")[value].mean().dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def primary_rows(attempts: pd.DataFrame) -> pd.DataFrame:
    result = attempts[
        (attempts["analysis_family"] == "frozen_primary")
        & as_bool(attempts["is_primary_estimand"])
        & (attempts["intervention_mode"] == "neuronpedia_global")
    ].copy()
    identity = ["item_id", "requested_strength"]
    if result.duplicated(identity).any():
        duplicates = result.loc[result.duplicated(identity, keep=False), identity]
        raise ValueError(f"primary rows are not sample-unique: {duplicates.to_dict('records')}")
    return result


def plot_global_flip_effects(primary: pd.DataFrame, plots: Path) -> bool:
    if primary.empty:
        return False
    rows = []
    for strength, group in primary.groupby("requested_strength"):
        sample = group.drop_duplicates("item_id")
        for effect in ("improved", "worsened"):
            successes = int((sample["flip_effect"] == effect).sum())
            low, high = wilson_interval(successes, len(sample))
            rows.append({
                "strength": strength, "effect": effect, "count": successes,
                "total": len(sample), "rate": successes / len(sample), "low": low, "high": high,
            })
    data = pd.DataFrame(rows)
    strengths = sorted(data["strength"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(strengths))
    for index, effect in enumerate(("improved", "worsened")):
        subset = data[data["effect"] == effect].set_index("strength").loc[strengths]
        locations = x + (-0.18 if index == 0 else 0.18)
        rates = subset["rate"].to_numpy()
        ax.bar(locations, rates, width=0.34, label=effect)
        ax.errorbar(
            locations, rates,
            yerr=np.maximum(0, np.vstack([rates - subset["low"], subset["high"] - rates])),
            fmt="none", color="black", capsize=3,
        )
    ax.set_xticks(x, [str(value) for value in strengths])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Requested global strength")
    ax.set_ylabel("Per-sample flip rate")
    ax.set_title("Frozen global primary: judgment rescues and harms")
    ax.legend()
    save(fig, plots / "01_primary_global_rescue_harm.png")
    return True


def plot_primary_margin(primary: pd.DataFrame, plots: Path, rng: np.random.Generator) -> bool:
    if primary.empty:
        return False
    strengths = sorted(primary["requested_strength"].unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    for index, strength in enumerate(strengths):
        group = primary[primary["requested_strength"] == strength]
        items = group.groupby("item_id")["delta_margin"].mean().dropna()
        low, high = bootstrap_item_interval(group, "delta_margin", rng)
        jitter = rng.normal(0, 0.04, len(items))
        ax.scatter(np.full(len(items), index) + jitter, items, alpha=0.4, s=20)
        mean = float(items.mean())
        ax.errorbar(
            index, mean,
            yerr=np.maximum(0.0, [[mean - low], [high - mean]]),
            fmt="o", color="black", capsize=4,
        )
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xticks(range(len(strengths)), [str(value) for value in strengths])
    ax.set_xlabel("Requested global strength")
    ax.set_ylabel("Change in first-label margin")
    ax.set_title("Frozen global PASS/FAIL-style margin movement")
    save(fig, plots / "02_primary_global_margin_change.png")
    return True


def plot_global_vs_localized(attempts: pd.DataFrame, plots: Path) -> bool:
    global_rows = primary_rows(attempts)
    local = attempts[
        (attempts["analysis_family"] == "localized_control")
        & (attempts["localized_target_selector"] == "question_mark")
    ].copy()
    keys = ["item_id", "requested_strength"]
    paired = global_rows.merge(local, on=keys, suffixes=("_global", "_local"))
    if paired.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(paired["delta_oriented_margin_local"], paired["delta_oriented_margin_global"], alpha=0.55)
    values = np.concatenate([
        paired["delta_oriented_margin_local"].to_numpy(dtype=float),
        paired["delta_oriented_margin_global"].to_numpy(dtype=float),
    ])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    ax.plot([low, high], [low, high], "--", color="grey")
    ax.set_xlabel("Localized question-mark oriented-margin change")
    ax.set_ylabel("Global oriented-margin change")
    ax.set_title("Paired global versus localized effects")
    save(fig, plots / "03_global_vs_localized.png")
    return True


def plot_susceptibility(primary: pd.DataFrame, plots: Path) -> bool:
    if primary.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 5))
    for strength, group in primary.groupby("requested_strength"):
        ax.scatter(
            group["baseline_oriented_margin"].abs(), group["delta_oriented_margin"],
            alpha=0.5, label=str(strength),
        )
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xlabel("Absolute baseline oriented margin")
    ax.set_ylabel("Change in oriented margin")
    ax.set_title("Baseline saturation versus global susceptibility")
    ax.legend(title="strength")
    save(fig, plots / "04_baseline_margin_vs_susceptibility.png")
    return True


def plot_adaptive(attempts: pd.DataFrame, plots: Path) -> bool:
    data = attempts[attempts["analysis_family"] == "adaptive_rescue"].copy()
    if data.empty:
        return False
    data["flipped_bool"] = as_bool(data["flipped"])
    counts = data.pivot_table(index="token_id", columns="layer", values="flipped_bool", aggfunc="sum", fill_value=0)
    totals = data.pivot_table(index="token_id", columns="layer", values="flipped_bool", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(max(6, len(counts.columns) * 0.8), 3.5))
    image = ax.imshow(counts.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            ax.text(column, row, f"{int(counts.iloc[row, column])}/{int(totals.iloc[row, column])}", ha="center", va="center")
    ax.set_xticks(range(len(counts.columns)), [str(value) for value in counts.columns])
    ax.set_yticks(range(len(counts.index)), [str(value) for value in counts.index])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Candidate token ID")
    ax.set_title("Descriptive adaptive rescue flips/attempts")
    fig.colorbar(image, ax=ax, label="flip count")
    save(fig, plots / "05_adaptive_rescue_token_layer.png")
    return True


def plot_local_positions(attempts: pd.DataFrame, plots: Path) -> bool:
    local = attempts[attempts["analysis_family"] == "localized_control"].copy()
    pivot = local.pivot_table(
        index=["item_id", "requested_strength"], columns="localized_target_selector",
        values="delta_oriented_margin", aggfunc="first",
    )
    required = {"question_mark", "meaningful_token_before_question_mark"}
    if pivot.empty or not required.issubset(pivot.columns):
        return False
    pivot = pivot.dropna(subset=list(required))
    if pivot.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    x = pivot["meaningful_token_before_question_mark"].to_numpy(dtype=float)
    y = pivot["question_mark"].to_numpy(dtype=float)
    ax.scatter(x, y, alpha=0.5)
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([low, high], [low, high], "--", color="grey")
    ax.set_xlabel("Localized preceding-token effect")
    ax.set_ylabel("Localized question-mark effect")
    ax.set_title("Localized control positions only")
    save(fig, plots / "06_localized_position_comparison.png")
    return True


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No observed rows."
    display = frame.copy().fillna("")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def stratified_primary_summary(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, title in (
        ("condition", "condition"),
        ("baseline_output_normalized", "baseline judgment"),
        ("difficulty", "difficulty"),
        ("factual_correct", "factual correctness"),
    ):
        for (strength, value), group in primary.groupby(
            ["requested_strength", column], dropna=False
        ):
            sample = group.drop_duplicates("item_id")
            flips = int(as_bool(sample["flipped"]).sum())
            low, high = wilson_interval(flips, len(sample))
            rows.append({
                "stratum_type": title, "stratum": value, "strength": strength,
                "samples": len(sample), "flips": flips,
                "rescues": int((sample["flip_effect"] == "improved").sum()),
                "harms": int((sample["flip_effect"] == "worsened").sum()),
                "flip_rate": round(flips / len(sample), 4),
                "wilson_95%": f"[{low:.4f}, {high:.4f}]",
            })
    return pd.DataFrame(rows)


def write_results(
    run_dir: Path,
    manifest: dict,
    summaries: pd.DataFrame,
    attempts: pd.DataFrame,
    primary: pd.DataFrame,
    plotted: dict[str, bool],
    rng: np.random.Generator,
) -> None:
    rows = []
    for strength, group in primary.groupby("requested_strength"):
        sample = group.drop_duplicates("item_id")
        low, high = bootstrap_item_interval(group, "delta_oriented_margin", rng)
        rows.append({
            "strength": strength, "samples": len(sample),
            "flips": int(as_bool(sample["flipped"]).sum()),
            "rescues": int((sample["flip_effect"] == "improved").sum()),
            "harms": int((sample["flip_effect"] == "worsened").sum()),
            "mean_oriented_delta": round(float(sample["delta_oriented_margin"].mean()), 4),
            "item_bootstrap_95%": f"[{low:.4f}, {high:.4f}]",
        })
    primary_summary = pd.DataFrame(rows)
    strata = stratified_primary_summary(primary) if len(primary) else pd.DataFrame()
    strata.to_csv(run_dir / "primary_stratified_summary.csv", index=False)
    families = attempts.groupby("analysis_family").size().rename("attempt_rows").reset_index()
    text = f"""# Results: `{manifest['run_id']}`

## Scope

The primary unit is the sample. Only `frozen_primary` global rows enter the
primary estimates. Localized controls and adaptive rescue attempts are shown
separately and are never treated as independent confirmatory samples.

## Accounting

- Completed sample summaries: {len(summaries)}
- Frozen primary samples represented: {primary['item_id'].nunique() if len(primary) else 0}
- Invalid primary outputs: {int((~as_bool(primary['intervened_valid'])).sum()) if len(primary) else 0}

{markdown_table(families)}

## Frozen global primary

{markdown_table(primary_summary)}

## Frozen global primary by baseline, difficulty, and factual correctness

{markdown_table(strata)}

## Plot status

{markdown_table(pd.DataFrame({'plot': plotted.keys(), 'created': plotted.values()}))}

A global flip demonstrates a causal distributed intervention effect. It does
not localize the mechanism to the direction-source question mark and does not
establish a higher-order representation M(P).
"""
    (run_dir / "results.md").write_text(text, encoding="utf-8")


def analyze(run_dir: Path) -> dict[str, bool]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summaries = pd.read_csv(run_dir / "trial_summary.csv")
    attempts = pd.read_csv(run_dir / "intervention_results.csv")
    summaries = summaries.drop_duplicates(["item_id", "condition"], keep="last")
    attempts = attempts.drop_duplicates(
        ["item_id", "analysis_family", "attempt_order"], keep="last"
    )
    numeric(attempts, [
        "requested_strength", "effective_strength_after_cap", "layer", "raw_rank",
        "word_filtered_rank", "baseline_margin", "intervened_margin", "delta_margin",
        "baseline_oriented_margin", "intervened_oriented_margin", "delta_oriented_margin",
    ])
    primary = primary_rows(attempts)
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    rng = np.random.default_rng(int(manifest["seed"]))
    plotted = {
        "primary global rescue and harm": plot_global_flip_effects(primary, plots),
        "primary global margin movement": plot_primary_margin(primary, plots, rng),
        "global versus localized": plot_global_vs_localized(attempts, plots),
        "baseline susceptibility": plot_susceptibility(primary, plots),
        "adaptive rescue token/layer": plot_adaptive(attempts, plots),
        "localized position comparison": plot_local_positions(attempts, plots),
    }
    write_results(run_dir, manifest, summaries, attempts, primary, plotted, rng)
    return plotted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

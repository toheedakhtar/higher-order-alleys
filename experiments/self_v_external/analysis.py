"""Paired analysis and plots for the SELF-versus-OTHER experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "item_id", "requested_strength", "same_question_and_answer",
    "candidate_token_id", "candidate_layer",
    "candidate_score_self", "candidate_rank_self", "candidate_score_other",
    "candidate_rank_other", "self_minus_other_candidate_score",
    "steering_delta_self", "steering_delta_other",
    "self_minus_other_steering_effect", "flipped_self", "flipped_other",
    "flip_effect_self", "flip_effect_other",
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def paired_bootstrap_ci(
    values: Iterable[float], *, samples: int = 2000, seed: int = 42
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def validate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"paired results missing columns: {sorted(missing)}")
    if frame.duplicated(["item_id", "requested_strength"]).any():
        raise ValueError("paired results contain duplicate item/strength rows")
    if not as_bool(frame["same_question_and_answer"]).all():
        raise ValueError("paired results contain a non-identical question/answer prefix")
    numeric = [
        "requested_strength", "candidate_score_self", "candidate_rank_self",
        "candidate_score_other", "candidate_rank_other",
        "self_minus_other_candidate_score", "steering_delta_self",
        "steering_delta_other", "self_minus_other_steering_effect",
        "candidate_token_id", "candidate_layer",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["flipped_self"] = as_bool(frame["flipped_self"])
    frame["flipped_other"] = as_bool(frame["flipped_other"])
    return frame


def _paired_axis(
    axis: plt.Axes, frame: pd.DataFrame, self_column: str, other_column: str,
    *, ylabel: str, log_scale: bool = False,
) -> None:
    for _, row in frame.iterrows():
        axis.plot(
            [0, 1], [row[self_column], row[other_column]],
            color="#6b7280", alpha=0.22, linewidth=0.8,
        )
    axis.scatter(
        np.zeros(len(frame)), frame[self_column], color="#2563eb", alpha=0.65,
        s=18, label="SELF",
    )
    axis.scatter(
        np.ones(len(frame)), frame[other_column], color="#dc2626", alpha=0.65,
        s=18, label="OTHER",
    )
    means = [frame[self_column].mean(), frame[other_column].mean()]
    axis.plot([0, 1], means, color="black", marker="D", linewidth=2.2, label="mean")
    axis.set_xticks([0, 1], ["SELF", "OTHER"])
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
    axis.grid(alpha=0.2)


def plot_candidate_score(frame: pd.DataFrame, path: Path) -> None:
    items = frame.drop_duplicates("item_id")
    fig, axis = plt.subplots(figsize=(7, 6))
    _paired_axis(
        axis, items, "candidate_score_self", "candidate_score_other",
        ylabel="Layer-40 candidate score",
    )
    axis.set_title("Paired candidate score: SELF vs OTHER")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_rank(frame: pd.DataFrame, path: Path) -> None:
    items = frame.drop_duplicates("item_id")
    fig, axis = plt.subplots(figsize=(7, 6))
    _paired_axis(
        axis, items, "candidate_rank_self", "candidate_rank_other",
        ylabel="Raw vocabulary rank (lower is stronger)", log_scale=True,
    )
    axis.invert_yaxis()
    axis.set_title("Paired candidate rank: SELF vs OTHER")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_steering_effect(frame: pd.DataFrame, path: Path) -> None:
    strengths = sorted(frame["requested_strength"].unique())
    fig, axes = plt.subplots(1, len(strengths), figsize=(7 * len(strengths), 6), squeeze=False)
    for axis, strength in zip(axes[0], strengths):
        subset = frame[frame["requested_strength"] == strength]
        _paired_axis(
            axis, subset, "steering_delta_self", "steering_delta_other",
            ylabel="Correct-oriented sequence-margin change",
        )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
        axis.set_title(f"Steering effect, strength {strength:+g}")
    axes[0][0].legend(frameon=False)
    fig.suptitle("Paired steering effect size: SELF vs OTHER")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_difference_scatter(frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    palette = {strength: color for strength, color in zip(
        sorted(frame["requested_strength"].unique()), ("#2563eb", "#dc2626", "#16a34a")
    )}
    for strength, subset in frame.groupby("requested_strength", sort=True):
        axis.scatter(
            subset["self_minus_other_candidate_score"],
            subset["self_minus_other_steering_effect"],
            s=28, alpha=0.72, color=palette[strength], label=f"{strength:+g}",
        )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.55)
    axis.axvline(0, color="black", linewidth=0.8, alpha=0.55)
    axis.set_xlabel("Candidate score: SELF − OTHER")
    axis.set_ylabel("Steering effect: SELF − OTHER")
    axis.set_title("Candidate difference vs causal steering difference")
    axis.grid(alpha=0.2)
    axis.legend(title="strength", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summary_table(frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rows = []
    for strength, subset in frame.groupby("requested_strength", sort=True):
        candidate_diff = subset["self_minus_other_candidate_score"]
        steering_diff = subset["self_minus_other_steering_effect"]
        candidate_ci = paired_bootstrap_ci(
            candidate_diff, samples=bootstrap_samples, seed=42
        )
        steering_ci = paired_bootstrap_ci(
            steering_diff, samples=bootstrap_samples, seed=43
        )
        rows.append({
            "requested_strength": strength, "paired_items": len(subset),
            "candidate_score_self_mean": subset["candidate_score_self"].mean(),
            "candidate_score_other_mean": subset["candidate_score_other"].mean(),
            "candidate_score_self_minus_other_mean": candidate_diff.mean(),
            "candidate_difference_ci_low": candidate_ci[0],
            "candidate_difference_ci_high": candidate_ci[1],
            "candidate_rank_self_median": subset["candidate_rank_self"].median(),
            "candidate_rank_other_median": subset["candidate_rank_other"].median(),
            "steering_delta_self_mean": subset["steering_delta_self"].mean(),
            "steering_delta_other_mean": subset["steering_delta_other"].mean(),
            "self_minus_other_steering_effect_mean": steering_diff.mean(),
            "steering_difference_ci_low": steering_ci[0],
            "steering_difference_ci_high": steering_ci[1],
            "self_flips": int(subset["flipped_self"].sum()),
            "other_flips": int(subset["flipped_other"].sum()),
            "self_improvements": int((subset["flip_effect_self"] == "improved").sum()),
            "other_improvements": int((subset["flip_effect_other"] == "improved").sum()),
            "self_harms": int((subset["flip_effect_self"] == "worsened").sum()),
            "other_harms": int((subset["flip_effect_other"] == "worsened").sum()),
        })
    return pd.DataFrame(rows)


def write_results(run_dir: Path, frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    unique = frame.drop_duplicates("item_id")
    candidate_ids = unique["candidate_token_id"].unique()
    candidate_layers = unique["candidate_layer"].unique()
    if len(candidate_ids) != 1 or len(candidate_layers) != 1:
        raise ValueError("one frozen candidate token/layer is required per run")
    candidate_id = int(candidate_ids[0])
    candidate_layer = int(candidate_layers[0])
    lines = [
        f"# Paired SELF-versus-OTHER results: `{run_dir.name}`",
        "",
        "## Pairing audit",
        "",
        f"- Paired factual items: {unique['item_id'].nunique()}",
        f"- Paired item-strength rows: {len(frame)}",
        "- Exact question/answer prefix shared in every pair: yes",
        f"- Candidate: vocabulary token {candidate_id} at layer {candidate_layer}, measured without a rank gate",
        "- Steering delta: change in the correct-oriented label-sequence log-probability margin",
        "",
        "## Candidate presence",
        "",
        "| SELF mean score | OTHER mean score | SELF − OTHER | SELF median raw rank | OTHER median raw rank |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {unique['candidate_score_self'].mean():.4f} | "
            f"{unique['candidate_score_other'].mean():.4f} | "
            f"{unique['self_minus_other_candidate_score'].mean():.4f} | "
            f"{unique['candidate_rank_self'].median():.1f} | "
            f"{unique['candidate_rank_other'].median():.1f} |"
        ),
        "",
        "## Paired causal effects",
        "",
        "| Strength | n | SELF delta | OTHER delta | SELF − OTHER effect | Paired bootstrap 95% CI | SELF flips | OTHER flips |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['requested_strength']:+g} | {int(row['paired_items'])} | "
            f"{row['steering_delta_self_mean']:.4f} | "
            f"{row['steering_delta_other_mean']:.4f} | "
            f"{row['self_minus_other_steering_effect_mean']:.4f} | "
            f"[{row['steering_difference_ci_low']:.4f}, "
            f"{row['steering_difference_ci_high']:.4f}] | "
            f"{int(row['self_flips'])} | {int(row['other_flips'])} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "Similar candidate presence and causal effects support a generic evaluator/readout. "
        "A reliably larger and selective SELF effect supports a self-evaluation-selective "
        "evaluator candidate. Neither result establishes a higher-order representation M(P); "
        "that requires a later same-output/different-internal-P experiment.",
        "",
    ])
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_dir: Path) -> dict[str, bool]:
    paired_path = run_dir / "paired_results.csv"
    if not paired_path.is_file():
        raise FileNotFoundError(f"missing {paired_path}")
    frame = validate_pairs(pd.read_csv(paired_path))
    if frame.empty:
        raise ValueError("paired results are empty")
    config_path = run_dir / "config.json"
    bootstrap_samples = 2000
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bootstrap_samples = int(config.get("analysis", {}).get("bootstrap_samples", 2000))
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plot_candidate_score(frame, plots / "01_paired_candidate_score.png")
    plot_candidate_rank(frame, plots / "02_paired_candidate_rank.png")
    plot_steering_effect(frame, plots / "03_paired_steering_effect.png")
    plot_difference_scatter(frame, plots / "04_candidate_vs_steering_difference.png")
    summary = summary_table(frame, bootstrap_samples)
    summary.to_csv(run_dir / "paired_summary.csv", index=False)
    write_results(run_dir, frame, summary)
    return {
        "paired candidate score": True,
        "paired candidate rank": True,
        "paired steering effect": True,
        "candidate versus steering difference": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.run_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

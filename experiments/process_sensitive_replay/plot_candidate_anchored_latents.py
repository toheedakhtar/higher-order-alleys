"""Plot actual layer-42 residual geometry around the frozen candidate axis.

This consumes residual tensors saved by either causal-mediation precision smoke:

* BF16 run: ``residuals_<item>_<branch>.pt``
* mixed run: ``item_<item>_<branch>_<condition>.pt``

Each residual is centered on its own item's clean state. The horizontal axis is
the exact frozen candidate-coordinate change. The vertical axis is the leading
uncentered principal axis of the residual change after removing that candidate
component. Thus item/content baselines cannot dominate the visual, and PCA
cannot rotate the frozen candidate direction away.

Example:
    python -m experiments.process_sensitive_replay.plot_candidate_anchored_latents \
      --run-dir assets/psr-mediation-precision-blackwell-v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "assets/psr-mediation-precision-blackwell-v1"
DEFAULT_DIRECTION = ROOT / "assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt"
DEFAULT_OUTPUT = ROOT / "docs/figures/candidate_anchored_layer42_latents.png"
EXPECTED_CONDITIONS = ("clean", "primary", "alternative")
EXPECTED_BRANCHES = ("confidence", "correctness")

PAPER = "#F7F6F1"
INK = "#17252A"
MUTED = "#6B7B80"
GRID = "#D8E0DE"
TEAL = "#087E8B"
ORANGE = "#E57A44"
PURPLE = "#6657A8"


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def load_direction(path: Path) -> tuple[torch.Tensor, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "direction" not in saved:
        raise RuntimeError("direction file does not contain the frozen direction")
    vector = saved["direction"].detach().cpu().double().reshape(-1)
    if (vector.numel() != 5120 or not torch.isfinite(vector).all()
            or int(saved.get("token_id", -1)) != 75075
            or int(saved.get("layer", -1)) != 42):
        raise RuntimeError("unexpected frozen candidate identity")
    norm = float(vector.norm())
    if abs(norm - 1.0) > 1e-5:
        raise RuntimeError(f"candidate vector is not unit length: {norm}")
    return vector, {
        "token_id": 75075,
        "layer": 42,
        "orientation": -1,
        "tensor_sha256": saved.get("sha256", tensor_sha(saved["direction"])),
        "file": str(path.resolve()),
    }


def _validate_residual(value, *, source: Path) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise RuntimeError(f"non-tensor residual in {source}")
    value = value.detach().cpu().reshape(-1)
    if value.numel() != 5120 or not torch.isfinite(value).all():
        raise RuntimeError(f"invalid layer-42 residual in {source}")
    if value.dtype != torch.bfloat16:
        raise RuntimeError(f"expected original BF16 residual in {source}, got {value.dtype}")
    return value


def load_residuals(run_dir: Path) -> tuple[dict[tuple[str, str, str], torch.Tensor], list[str]]:
    """Load smoke tensors without using logits or behavioral patch outcomes."""
    records: dict[tuple[str, str, str], torch.Tensor] = {}
    sources: list[str] = []
    native_files = sorted(run_dir.glob("residuals_*_*.pt"))
    native_pattern = re.compile(r"residuals_(.+)_(confidence|correctness)\.pt$")
    if native_files:
        for path in native_files:
            match = native_pattern.fullmatch(path.name)
            if match is None:
                continue
            item, branch = match.groups()
            saved = torch.load(path, map_location="cpu", weights_only=True)
            residuals = saved.get("residuals", {})
            for condition in EXPECTED_CONDITIONS:
                if condition not in residuals:
                    raise RuntimeError(f"{path} is missing {condition!r}")
                records[(item, branch, condition)] = _validate_residual(residuals[condition], source=path)
            sources.append(str(path.resolve()))
    else:
        mixed_pattern = re.compile(r"item_(.+)_(confidence|correctness)_(clean|primary|alternative)\.pt$")
        for path in sorted(run_dir.glob("item_*.pt")):
            match = mixed_pattern.fullmatch(path.name)
            if match is None:
                continue
            item, branch, condition = match.groups()
            saved = torch.load(path, map_location="cpu", weights_only=True)
            records[(item, branch, condition)] = _validate_residual(saved.get("bf16_pre_tail"), source=path)
            sources.append(str(path.resolve()))
    if not records:
        raise RuntimeError(
            f"no saved layer-42 smoke residuals found under {run_dir}; copy the Blackwell "
            "precision-smoke output directory or run this script on that host"
        )
    groups = sorted({(item, branch) for item, branch, _ in records})
    for item, branch in groups:
        missing = [condition for condition in EXPECTED_CONDITIONS
                   if (item, branch, condition) not in records]
        if missing:
            raise RuntimeError(f"incomplete residual group item={item} branch={branch}: {missing}")
    if set(branch for _, branch in groups) != set(EXPECTED_BRANCHES):
        raise RuntimeError("both confidence and correctness residual branches are required")
    if len({item for item, _ in groups}) < 2:
        raise RuntimeError("at least two smoke items are required for latent geometry")
    return records, sources


def canonical_axis(vector: torch.Tensor) -> torch.Tensor:
    index = int(torch.argmax(vector.abs()))
    return vector if float(vector[index]) >= 0 else -vector


def project_residuals(records, candidate: torch.Tensor) -> tuple[list[dict], dict]:
    """Exact candidate coordinate plus uncentered SVD of orthogonal changes."""
    candidate_norm_sq = candidate @ candidate
    changes = []
    orthogonal = []
    for item, branch in sorted({(i, b) for i, b, _ in records}):
        clean = records[(item, branch, "clean")].double()
        for condition in ("primary", "alternative"):
            delta = records[(item, branch, condition)].double() - clean
            raw_coordinate = float(candidate @ delta)
            component = candidate * ((candidate @ delta) / candidate_norm_sq)
            remainder = delta - component
            changes.append({
                "item_id": item,
                "branch": branch,
                "condition": condition,
                # Orientation -1 makes the discovery-frozen effect point right.
                "candidate_coordinate": -raw_coordinate,
                "raw_candidate_coordinate": raw_coordinate,
                "total_l2": float(delta.norm()),
                "candidate_component_l2": float(component.norm()),
                "orthogonal_l2": float(remainder.norm()),
                "candidate_energy_fraction": float(component.square().sum() / delta.square().sum()),
                "delta": delta,
            })
            orthogonal.append(remainder)
    matrix = torch.stack(orthogonal)
    # Uncentered SVD preserves the clean origin: axes describe displacement
    # energy rather than deviations around an artificial displacement mean.
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    pc1 = canonical_axis(vh[0])
    pc2 = canonical_axis(vh[1]) if len(vh) > 1 else torch.zeros_like(pc1)
    total_energy = float(singular.square().sum())
    for row, remainder in zip(changes, orthogonal, strict=True):
        row["orthogonal_pc1"] = float(remainder @ pc1)
        row["orthogonal_pc2"] = float(remainder @ pc2)
    deltas = torch.stack([row["delta"] for row in changes])
    normalized = deltas / deltas.norm(dim=1, keepdim=True)
    cosine = (normalized @ normalized.T).numpy()
    return changes, {
        "orthogonal_pc1_energy_fraction": float(singular[0].square() / total_energy),
        "orthogonal_pc2_energy_fraction": float(singular[1].square() / total_energy),
        "orthogonal_singular_values": singular.tolist(),
        "cosine": cosine,
    }


def style_axis(ax):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.75, alpha=0.7)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK, linewidth=0.8, alpha=0.55)
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.55)


def draw_latent_panel(ax, rows, branch, pc_fraction):
    style_axis(ax)
    subset = [row for row in rows if row["branch"] == branch]
    offsets = {"primary": (6, 6), "alternative": (6, -10)}
    for row in subset:
        color = TEAL if row["condition"] == "primary" else ORANGE
        marker = "o" if row["condition"] == "primary" else "s"
        x, y = row["candidate_coordinate"], row["orthogonal_pc1"]
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.38, linewidth=1.25))
        ax.scatter(x, y, s=64, marker=marker, color=color, edgecolor=PAPER,
                   linewidth=1.2, zorder=3)
        ax.annotate(f"item {row['item_id']}", (x, y), xytext=offsets[row["condition"]],
                    textcoords="offset points", color=MUTED, fontsize=8)
    ax.scatter(0, 0, marker="+", s=95, color=PURPLE, linewidth=1.8, zorder=4)
    ax.text(0, 0, "  clean", va="bottom", color=PURPLE, fontsize=8.2)
    ax.set_xlabel("Frozen candidate coordinate  −vᵀΔh", color=INK, labelpad=8)
    ax.set_ylabel("Orthogonal residual axis 1", color=INK, labelpad=8)
    title = "Confidence prompt" if branch == "confidence" else "Correctness prompt"
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="normal", pad=12)
    ax.text(0, 1.01, f"Axis 1 contains {100*pc_fraction:.1f}% of orthogonal displacement energy",
            transform=ax.transAxes, fontsize=8.2, color=MUTED, va="bottom")


def draw_cosine_panel(ax, rows, cosine):
    labels = [f"{'C' if row['branch']=='confidence' else 'K'}{row['item_id']}·"
              f"{'P' if row['condition']=='primary' else 'A'}" for row in rows]
    image = ax.imshow(cosine, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5, color=MUTED)
    ax.tick_params(length=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cosine[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.7,
                    color=PAPER if abs(value) > 0.58 else INK)
    ax.set_title("Full residual-change cosine", loc="left", color=INK,
                 fontsize=11, fontweight="normal", pad=10)
    ax.text(0, 1.01, "Mechanism similarity before dimensionality reduction",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.ax.tick_params(labelsize=7, colors=MUTED)
    colorbar.outline.set_edgecolor(GRID)


def draw_energy_panel(ax, rows):
    values = np.asarray([max(row["candidate_energy_fraction"], 1e-16) for row in rows]) * 100
    labels = [f"{'C' if row['branch']=='confidence' else 'K'}{row['item_id']}·"
              f"{'P' if row['condition']=='primary' else 'A'}" for row in rows]
    colors = [TEAL if row["condition"] == "primary" else ORANGE for row in rows]
    positions = np.arange(len(rows))
    ax.barh(positions, values, color=colors, height=0.58, alpha=0.88)
    ax.set_xscale("log")
    ax.set_yticks(positions, labels, fontsize=7.5, color=MUTED)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Displacement energy on candidate axis (%)", fontsize=8.2, color=INK)
    ax.set_title("How much of Δh lies on v?", loc="left", color=INK,
                 fontsize=11, fontweight="normal", pad=8)


def make_figure(rows, geometry):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
        "text.color": INK, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    figure = plt.figure(figsize=(15.8, 8.4))
    grid = figure.add_gridspec(2, 3, width_ratios=(1.12, 1.12, 0.88),
                              height_ratios=(1.2, 0.8), left=0.06, right=0.965,
                              bottom=0.13, top=0.79, wspace=0.34, hspace=0.42)
    draw_latent_panel(figure.add_subplot(grid[:, 0]), rows, "confidence",
                      geometry["orthogonal_pc1_energy_fraction"])
    draw_latent_panel(figure.add_subplot(grid[:, 1]), rows, "correctness",
                      geometry["orthogonal_pc1_energy_fraction"])
    draw_cosine_panel(figure.add_subplot(grid[0, 2]), rows, geometry["cosine"])
    draw_energy_panel(figure.add_subplot(grid[1, 2]), rows)

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TEAL,
               markeredgecolor=PAPER, markersize=8, label="Primary process change · layer 31"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=ORANGE,
               markeredgecolor=PAPER, markersize=8, label="Alternative process change · layer 23"),
        Line2D([0], [0], marker="+", linestyle="none", color=PURPLE,
               markersize=9, label="Within-item clean origin"),
    ]
    figure.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 0.85),
                  ncol=3, frameon=False, fontsize=9, handletextpad=0.45, columnspacing=1.5)
    figure.suptitle("Candidate-anchored geometry of the layer-42 residual stream",
                    x=0.06, y=0.955, ha="left", fontsize=21, fontweight="normal", color=INK)
    item_count = len({row["item_id"] for row in rows})
    figure.text(0.06, 0.91,
                "Exact process-sensitive direction × dominant orthogonal displacement axis; "
                f"real BF16 changes from {item_count} predeclared smoke items",
                ha="left", fontsize=10.2, color=MUTED)
    figure.text(0.06, 0.055,
                "C = confidence, K = correctness, P = primary, A = alternative. "
                "Orthogonal axes use uncentered SVD after removing v; the candidate axis is never fit or rotated.",
                ha="left", fontsize=8.3, color=MUTED)
    figure.text(0.965, 0.055, "Geometry is descriptive; proximity does not establish causal mediation.",
                ha="right", fontsize=8.3, color=MUTED)
    return figure


def write_projection_csv(path: Path, rows: list[dict]):
    fields = (
        "item_id", "branch", "condition", "candidate_coordinate",
        "raw_candidate_coordinate", "orthogonal_pc1", "orthogonal_pc2",
        "total_l2", "candidate_component_l2", "orthogonal_l2",
        "candidate_energy_fraction",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--direction", type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png")
    candidate, identity = load_direction(args.direction.resolve())
    records, sources = load_residuals(args.run_dir.resolve())
    rows, geometry = project_residuals(records, candidate)
    figure = make_figure(rows, geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", metadata={
        "Title": "Candidate-anchored geometry of the layer-42 residual stream",
        "Creator": "higher-order-alleys",
    })
    if not args.no_pdf:
        figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", metadata={
            "Title": "Candidate-anchored geometry of the layer-42 residual stream",
            "Creator": "higher-order-alleys",
        })
    plt.close(figure)
    write_projection_csv(args.output.with_suffix(".csv"), rows)
    metadata = {
        "projection": "exact frozen candidate coordinate plus uncentered SVD of candidate-orthogonal process changes",
        "candidate": identity,
        "source_run_dir": str(args.run_dir.resolve()),
        "source_files": sources,
        "samples": len(rows),
        "item_ids": sorted({row["item_id"] for row in rows}),
        "orthogonal_pc1_energy_fraction": geometry["orthogonal_pc1_energy_fraction"],
        "orthogonal_pc2_energy_fraction": geometry["orthogonal_pc2_energy_fraction"],
        "behavioral_mediation_claim": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if not args.no_pdf:
        print(f"wrote {args.output.with_suffix('.pdf')}")
    print(f"wrote {args.output.with_suffix('.csv')}")
    print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

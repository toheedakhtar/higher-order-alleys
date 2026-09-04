"""Model-free held-out statistics and plots for process-sensitive replay."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_LABELS = {
    "clean_preserved": "CLEAN",
    "targeted_weak_preserved": "TARGETED_WEAK",
    "targeted_strong_preserved": "TARGETED_STRONG",
    "support_matched_alternative_preserved": "SUPPORT_MATCHED_ALTERNATIVE",
    "random_strong_preserved": "RANDOM_NORM_MATCHED",
    "targeted_strong_reset": "RESET",
}


def json_safe(value: Any) -> Any:
    """Replace undefined/non-finite statistics with JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def paired_bootstrap_interval(
    values: Sequence[float], *, seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choices(finite, k=len(finite))) for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[min(samples - 1, int(0.975 * samples))]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else math.nan


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    offset = 0
    while offset < len(order):
        end = offset + 1
        while end < len(order) and values[order[end]] == values[order[offset]]:
            end += 1
        tied_rank = (offset + 1 + end) / 2
        for index in order[offset:end]:
            result[index] = tied_rank
        offset = end
    return result


def _effect_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    low, high = paired_bootstrap_interval(finite, seed=seed)
    return {
        "n": len(finite),
        "mean": mean(finite) if finite else math.nan,
        "median": median(finite) if finite else math.nan,
        "item_bootstrap_95_ci": [low, high],
    }


def _fixed_effect_mechanism(
    targeted_effect: Sequence[float],
    alternative_effect: Sequence[float],
    targeted_drop: Sequence[float],
    alternative_drop: Sequence[float],
) -> dict[str, Any]:
    # Differencing the two rows within each item removes the item fixed effect:
    # y_alt - y_target = beta_support * (x_alt - x_target) + beta_mechanism.
    def fit(indices: Sequence[int]) -> tuple[float, float, float]:
        x = [alternative_drop[index] - targeted_drop[index] for index in indices]
        y = [alternative_effect[index] - targeted_effect[index] for index in indices]
        x_mean, y_mean = mean(x), mean(y)
        denominator = sum((value - x_mean) ** 2 for value in x)
        beta_support = (
            sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y, strict=True))
            / denominator
            if denominator
            else 0.0
        )
        beta_mechanism = y_mean - beta_support * x_mean
        shared_scale = mean([
            0.5 * (abs(targeted_effect[index]) + abs(alternative_effect[index]))
            for index in indices
        ])
        ratio = abs(beta_mechanism) / shared_scale if shared_scale else math.inf
        return beta_support, beta_mechanism, ratio

    indices = list(range(len(targeted_effect)))
    beta_support, beta_mechanism, ratio = fit(indices)
    rng = random.Random(107)
    bootstrap = [fit(rng.choices(indices, k=len(indices))) for _ in range(2000)]

    def interval(offset: int) -> list[float]:
        values = sorted(row[offset] for row in bootstrap if math.isfinite(row[offset]))
        if not values:
            return [math.nan, math.nan]
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {
        "beta_support": beta_support,
        "beta_mechanism": beta_mechanism,
        "abs_mechanism_to_shared_effect_ratio": ratio,
        "item_bootstrap_95_ci": {
            "beta_support": interval(0),
            "beta_mechanism": interval(1),
            "abs_mechanism_to_shared_effect_ratio": interval(2),
        },
    }


def _item_centered_relationship(
    by_item: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    value_key: str,
) -> dict[str, Any]:
    items = list(by_item.values())

    def fit(selected_items: Sequence[Mapping[str, Mapping[str, Any]]]) -> tuple[float, float, float]:
        xs: list[float] = []
        ys: list[float] = []
        for conditions in selected_items:
            selected = [
                conditions[name]
                for name in (
                    "clean_preserved",
                    "targeted_weak_preserved",
                    "targeted_strong_preserved",
                )
            ]
            item_x = [float(row["support_drop"]) for row in selected]
            item_y = [float(row[value_key]) for row in selected]
            center_x, center_y = mean(item_x), mean(item_y)
            xs.extend(value - center_x for value in item_x)
            ys.extend(value - center_y for value in item_y)
        denominator = sum(value * value for value in xs)
        slope = (
            sum(x * y for x, y in zip(xs, ys, strict=True)) / denominator
            if denominator else math.nan
        )
        return slope, _pearson(xs, ys), _pearson(_ranks(xs), _ranks(ys))

    slope, pearson, spearman = fit(items)
    rng = random.Random(108 if value_key == "oriented_candidate_score" else 109)
    bootstrap = [fit(rng.choices(items, k=len(items))) for _ in range(2000)]

    def interval(offset: int) -> list[float]:
        values = sorted(row[offset] for row in bootstrap if math.isfinite(row[offset]))
        if not values:
            return [math.nan, math.nan]
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {
        "slope": slope,
        "pearson": pearson,
        "spearman": spearman,
        "item_bootstrap_95_ci": {
            "slope": interval(0),
            "pearson": interval(1),
            "spearman": interval(2),
        },
        "observations": len(items) * 3,
    }


def analyze_candidate_effects(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            int(row["candidate_rank"]),
            int(row["candidate_token_id"]),
            str(row["branch"]),
        )].append(row)
    summaries = []
    for (candidate_rank, token_id, branch), group in sorted(groups.items()):
        by_item: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in group:
            by_item[str(row["item_id"])][str(row["condition"])] = row
        required = set(CONDITION_LABELS)
        if any(set(conditions) != required for conditions in by_item.values()):
            raise ValueError("held-out effect rows do not contain every frozen condition")
        targeted_effect: list[float] = []
        alternative_effect: list[float] = []
        targeted_random: list[float] = []
        alternative_random: list[float] = []
        targeted_reset: list[float] = []
        targeted_alternative: list[float] = []
        target_drop: list[float] = []
        alternative_drop: list[float] = []
        normalized_target: list[float] = []
        normalized_alternative: list[float] = []
        for conditions in by_item.values():
            score = {
                name: float(row["oriented_candidate_score"])
                for name, row in conditions.items()
            }
            target = score["targeted_strong_preserved"] - score["clean_preserved"]
            alternative = (
                score["support_matched_alternative_preserved"] - score["clean_preserved"]
            )
            targeted_effect.append(target)
            alternative_effect.append(alternative)
            targeted_random.append(
                score["targeted_strong_preserved"] - score["random_strong_preserved"]
            )
            alternative_random.append(
                score["support_matched_alternative_preserved"]
                - score["random_strong_preserved"]
            )
            targeted_reset.append(
                score["targeted_strong_preserved"] - score["targeted_strong_reset"]
            )
            targeted_alternative.append(target - alternative)
            target_support = float(conditions["targeted_strong_preserved"]["support_drop"])
            alternative_support = float(
                conditions["support_matched_alternative_preserved"]["support_drop"]
            )
            target_drop.append(target_support)
            alternative_drop.append(alternative_support)
            if target_support != 0 and alternative_support != 0:
                normalized_target.append(target / target_support)
                normalized_alternative.append(alternative / alternative_support)
        summaries.append({
            "candidate_rank": candidate_rank,
            "candidate_token_id": token_id,
            "candidate_layer": int(group[0]["candidate_layer"]),
            "candidate_label": str(group[0]["candidate_label"]),
            "branch": branch,
            "h1_targeted_minus_clean": _effect_summary(targeted_effect, seed=101),
            "h2_alternative_minus_clean": _effect_summary(alternative_effect, seed=102),
            "h3_targeted_minus_alternative": _effect_summary(targeted_alternative, seed=103),
            "h3_support_normalized_response_difference": _effect_summary(
                [t - a for t, a in zip(normalized_target, normalized_alternative, strict=True)],
                seed=104,
            ),
            "h3_item_fixed_effect_model": _fixed_effect_mechanism(
                targeted_effect, alternative_effect, target_drop, alternative_drop
            ),
            "h4_targeted_minus_random": _effect_summary(targeted_random, seed=105),
            "h4_alternative_minus_random": _effect_summary(
                alternative_random, seed=110
            ),
            "h5_targeted_preserved_minus_reset": _effect_summary(targeted_reset, seed=106),
            "h6_candidate_score_vs_support": _item_centered_relationship(
                by_item, value_key="oriented_candidate_score"
            ),
            "h7_confidence_margin_vs_support": (
                _item_centered_relationship(by_item, value_key="choice_margin")
                if branch == "confidence"
                else None
            ),
            "interpretation_ceiling": (
                "evidence for a process-property / process-sensitive representation; "
                "an M(P)-like candidate"
            ),
        })
    return summaries


def generate_support_match_plot(
    support_rows: list[Mapping[str, Any]], path: Path
) -> Path:
    if not support_rows:
        raise ValueError("support-match plot received no held-out rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_support = sorted(support_rows, key=lambda row: str(row["item_id"]))
    x = list(range(len(ordered_support)))
    target_values = [float(row["support_drop_targeted"]) for row in ordered_support]
    alternative_values = [float(row["support_drop_alternative"]) for row in ordered_support]
    tolerances = [float(row["item_tolerance"]) for row in ordered_support]
    matched_flags = [
        value if isinstance(value := row["support_matched"], bool)
        else str(value).lower() == "true"
        for row in ordered_support
    ]
    colors = ["tab:green" if matched else "tab:red" for matched in matched_flags]
    plt.figure(figsize=(12, 5))
    plt.fill_between(
        x,
        [target - tolerance for target, tolerance in zip(target_values, tolerances, strict=True)],
        [target + tolerance for target, tolerance in zip(target_values, tolerances, strict=True)],
        color="tab:blue",
        alpha=0.12,
        label="Frozen per-item tolerance around targeted drop",
    )
    for index, (target_value, alternative_value, color) in enumerate(
        zip(target_values, alternative_values, colors, strict=True)
    ):
        plt.plot([index, index], [target_value, alternative_value], color=color, linewidth=0.8)
    plt.scatter(x, target_values, s=13, label="TARGETED_STRONG")
    plt.scatter(x, alternative_values, s=13, label="SUPPORT_MATCHED_ALTERNATIVE")
    matched = sum(matched_flags)
    absolute_mismatches = [float(row["absolute_mismatch"]) for row in ordered_support]
    plt.title(
        f"Item-level support matching: {matched}/{len(ordered_support)} = "
        f"{matched / len(ordered_support):.1%}\n"
        f"mean |mismatch|={mean(absolute_mismatches):.3g} nat; "
        f"median |mismatch|={median(absolute_mismatches):.3g} nat"
    )
    plt.xlabel("Held-out item (green matched, red unmatched)")
    plt.ylabel("Support drop (nat)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def generate_required_plots(
    effect_rows: list[Mapping[str, Any]],
    support_rows: list[Mapping[str, Any]],
    candidate_score_rows: list[Mapping[str, Any]],
    output_dir: Path,
    *,
    primary_branch: str,
    generic_token_ids: Sequence[int],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)

    primary = [row for row in effect_rows if str(row["branch"]) == primary_branch]
    first_rank = min(int(row["candidate_rank"]) for row in primary)
    primary = [row for row in primary if int(row["candidate_rank"]) == first_rank]
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in primary:
        by_condition[str(row["condition"])].append(row)

    manipulation_conditions = [
        "clean_preserved", "targeted_weak_preserved", "targeted_strong_preserved",
        "support_matched_alternative_preserved", "random_strong_preserved",
    ]
    plt.figure(figsize=(10, 5))
    plt.boxplot([
        [float(row["support_drop"]) for row in by_condition[name]]
        for name in manipulation_conditions
    ], tick_labels=[CONDITION_LABELS[name] for name in manipulation_conditions])
    plt.ylabel("Answer-sequence support drop (nat)")
    plt.xticks(rotation=20, ha="right")
    save("01_manipulation_check.png")

    plt.figure(figsize=(7, 5))
    for condition in manipulation_conditions:
        rows = by_condition[condition]
        plt.scatter(
            [float(row["support_drop"]) for row in rows],
            [float(row["choice_margin"]) for row in rows],
            s=16,
            alpha=0.65,
            label=CONDITION_LABELS[condition],
        )
    plt.xlabel("Support drop (nat)")
    plt.ylabel("HIGH_CONFIDENCE - LOW_CONFIDENCE log-probability margin")
    plt.legend(fontsize=7)
    save("02_confidence_vs_support.png")

    candidate_ranks = sorted({int(row["candidate_rank"]) for row in effect_rows})
    for candidate_rank in candidate_ranks:
        candidate = [
            row for row in effect_rows
            if int(row["candidate_rank"]) == candidate_rank
            and str(row["branch"]) == primary_branch
        ]
        condition_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidate:
            condition_rows[str(row["condition"])].append(row)
        suffix = f"candidate_{candidate_rank}"

        plt.figure(figsize=(7, 5))
        for condition in manipulation_conditions:
            rows = condition_rows[condition]
            plt.scatter(
                [float(row["support_drop"]) for row in rows],
                [float(row["oriented_candidate_score"]) for row in rows],
                s=16, alpha=0.65, label=CONDITION_LABELS[condition],
            )
        plt.xlabel("Support drop (nat)")
        plt.ylabel("Oriented candidate score")
        plt.legend(fontsize=7)
        save(f"03_candidate_vs_support_{suffix}.png")

        def paired_plot(left: str, right: str, number: str, title: str) -> None:
            left_rows = sorted(condition_rows[left], key=lambda row: str(row["item_id"]))
            right_rows = sorted(condition_rows[right], key=lambda row: str(row["item_id"]))
            plt.figure(figsize=(6, 5))
            for left_row, right_row in zip(left_rows, right_rows, strict=True):
                values = [
                    float(left_row["oriented_candidate_score"]),
                    float(right_row["oriented_candidate_score"]),
                ]
                plt.plot([0, 1], values, color="0.75", linewidth=0.6)
            plt.scatter([0] * len(left_rows), [float(row["oriented_candidate_score"]) for row in left_rows], s=14)
            plt.scatter([1] * len(right_rows), [float(row["oriented_candidate_score"]) for row in right_rows], s=14)
            plt.xticks([0, 1], [CONDITION_LABELS[left], CONDITION_LABELS[right]], rotation=15)
            plt.ylabel("Oriented candidate score")
            plt.title(title)
            save(f"{number}_{suffix}.png")

        paired_plot("targeted_strong_preserved", "random_strong_preserved", "04_targeted_vs_random", "Targeted versus random")
        paired_plot("targeted_strong_preserved", "targeted_strong_reset", "05_preserved_vs_reset", "Preserved versus reset")
        paired_plot("clean_preserved", "targeted_strong_preserved", "06_clean_vs_targeted", "Identical text: clean versus targeted")

        plt.figure(figsize=(8, 5))
        token_id = int(candidate[0]["candidate_token_id"])
        profile = [
            row for row in candidate_score_rows
            if int(row["token_id"]) == token_id and str(row["branch"]) == primary_branch
        ]
        for condition in (
            "clean_preserved", "targeted_strong_preserved",
            "support_matched_alternative_preserved", "random_strong_preserved",
            "targeted_strong_reset",
        ):
            layer_values: dict[int, list[float]] = defaultdict(list)
            for row in profile:
                if str(row["condition"]) == condition:
                    layer_values[int(row["layer"])].append(float(row["score"]))
            layers = sorted(layer_values)
            plt.plot(layers, [mean(layer_values[layer]) for layer in layers], marker="o", label=CONDITION_LABELS[condition])
        plt.xlabel("Layer")
        plt.ylabel("Mean candidate score")
        plt.legend(fontsize=7)
        save(f"08_layer_profile_{suffix}.png")

        clean = {str(row["item_id"]): row for row in condition_rows["clean_preserved"]}
        comparisons = [
            ("targeted-clean", "targeted_strong_preserved", "clean_preserved"),
            ("alternative-clean", "support_matched_alternative_preserved", "clean_preserved"),
            ("targeted-random", "targeted_strong_preserved", "random_strong_preserved"),
            ("alternative-random", "support_matched_alternative_preserved", "random_strong_preserved"),
            ("targeted-reset", "targeted_strong_preserved", "targeted_strong_reset"),
            ("targeted-alternative", "targeted_strong_preserved", "support_matched_alternative_preserved"),
        ]
        means, lows, highs, labels = [], [], [], []
        maps = {
            condition: {str(row["item_id"]): row for row in rows}
            for condition, rows in condition_rows.items()
        }
        for offset, (label, left, right) in enumerate(comparisons):
            values = [
                float(maps[left][item]["oriented_candidate_score"])
                - float(maps[right][item]["oriented_candidate_score"])
                for item in clean
            ]
            low, high = paired_bootstrap_interval(values, seed=200 + offset)
            labels.append(label)
            means.append(mean(values))
            lows.append(mean(values) - low)
            highs.append(high - mean(values))
        plt.figure(figsize=(8, 5))
        plt.errorbar(range(len(labels)), means, yerr=[lows, highs], fmt="o", capsize=4)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
        plt.ylabel("Mean oriented effect with item-bootstrap 95% CI")
        save(f"09_effect_summary_{suffix}.png")

        five = [
            "clean_preserved", "targeted_strong_preserved",
            "support_matched_alternative_preserved", "random_strong_preserved",
            "targeted_strong_reset",
        ]
        plt.figure(figsize=(10, 5))
        five_maps = {
            condition: {
                str(row["item_id"]): float(row["oriented_candidate_score"])
                for row in condition_rows[condition]
            }
            for condition in five
        }
        for item_id in sorted(five_maps[five[0]]):
            plt.plot(
                range(len(five)),
                [five_maps[condition][item_id] for condition in five],
                color="0.82",
                linewidth=0.5,
                zorder=0,
            )
        for index, condition in enumerate(five):
            values = [float(row["oriented_candidate_score"]) for row in condition_rows[condition]]
            plt.scatter([index] * len(values), values, s=12, alpha=0.4)
            plt.scatter(index, mean(values), marker="D", s=55, color="black")
        plt.xticks(range(len(five)), [CONDITION_LABELS[name] for name in five], rotation=20, ha="right")
        plt.ylabel("Oriented candidate score")
        save(f"10_five_condition_comparison_{suffix}.png")

        target = maps["targeted_strong_preserved"]
        alternative = maps["support_matched_alternative_preserved"]
        xs, ys = [], []
        for item, target_row in target.items():
            target_drop = float(target_row["support_drop"])
            alternative_drop = float(alternative[item]["support_drop"])
            if target_drop == 0 or alternative_drop == 0:
                continue
            clean_score = float(clean[item]["oriented_candidate_score"])
            xs.append((float(target_row["oriented_candidate_score"]) - clean_score) / target_drop)
            ys.append((float(alternative[item]["oriented_candidate_score"]) - clean_score) / alternative_drop)
        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, s=18)
        for targeted_response, alternative_response in zip(xs, ys, strict=True):
            plt.plot(
                [targeted_response, targeted_response],
                [targeted_response, alternative_response],
                color="0.75",
                linewidth=0.6,
            )
        if xs and ys:
            low, high = min((*xs, *ys)), max((*xs, *ys))
            plt.plot([low, high], [low, high], linestyle="--", color="black")
        plt.xlabel("Targeted response / targeted support drop")
        plt.ylabel("Alternative response / alternative support drop")
        save(f"11_support_normalized_convergence_{suffix}.png")

    generic = [
        row for row in candidate_score_rows
        if int(row["token_id"]) in {int(value) for value in generic_token_ids}
        and str(row["branch"]) == primary_branch
    ]
    plt.figure(figsize=(10, 5))
    for token_offset, token_id in enumerate(generic_token_ids):
        means_by_condition = []
        for condition in CONDITION_LABELS:
            values = [
                float(row["score"]) for row in generic
                if int(row["token_id"]) == int(token_id)
                and str(row["condition"]) == condition
            ]
            means_by_condition.append(mean(values) if values else math.nan)
        plt.plot(range(len(CONDITION_LABELS)), means_by_condition, marker="o", label=str(token_id))
    plt.xticks(range(len(CONDITION_LABELS)), [CONDITION_LABELS[name] for name in CONDITION_LABELS], rotation=20, ha="right")
    plt.ylabel("Mean generic evaluator score across layers/items")
    plt.legend(title="Token ID")
    save("07_generic_evaluator_controls.png")

    paths.append(generate_support_match_plot(
        support_rows, output_dir / "12_item_level_support_matching.png"
    ))
    return paths

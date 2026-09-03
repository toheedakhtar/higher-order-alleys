"""Deterministic discovery-only strength selection machinery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AlphaTrial:
    alpha: float
    support_drops: tuple[float, ...]
    finite: bool = True


@dataclass(frozen=True)
class BetaTrial:
    beta: float
    targeted_support_drops: tuple[float, ...]
    alternative_support_drops: tuple[float, ...]
    median_norm_ratio: float
    max_abs_cosine: float
    finite: bool = True


def _finite(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def select_alpha_strengths(
    trials: Sequence[AlphaTrial], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen global weak/strong discovery rule rule."""
    strengths = config["strengths"]
    ordered = sorted(trials, key=lambda trial: trial.alpha)
    declared = [float(value) for value in strengths["alpha_grid"]]
    if [float(trial.alpha) for trial in ordered] != declared:
        raise ValueError("alpha trials do not exactly cover the frozen grid")
    valid = [
        trial for trial in ordered
        if trial.finite and _finite(trial.support_drops)
    ]
    weak = next(
        (
            trial for trial in valid
            if median(trial.support_drops) >= float(strengths["weak_min_median_drop_nat"])
            and sum(value > 0 for value in trial.support_drops)
            >= int(strengths["weak_min_positive_items"])
        ),
        None,
    )
    low, high = (float(value) for value in strengths["strong_median_drop_range_nat"])
    strong = next(
        (trial for trial in valid if low <= median(trial.support_drops) <= high),
        None,
    )
    fallback = False
    if strong is None:
        below = [trial for trial in valid if median(trial.support_drops) < high]
        if below:
            strong = max(below, key=lambda trial: trial.alpha)
            fallback = True
    if weak is None or strong is None:
        raise RuntimeError("alpha_strength_gate_failed")
    if weak.alpha >= strong.alpha:
        raise RuntimeError("alpha_strength_gate_failed: WEAK must be strictly smaller than STRONG")
    return {
        "weak_alpha": weak.alpha,
        "strong_alpha": strong.alpha,
        "strong_fallback_below_four_nats": fallback,
        "weak_median_support_drop": median(weak.support_drops),
        "strong_median_support_drop": median(strong.support_drops),
    }


def evaluate_beta_trial(trial: BetaTrial, config: Mapping[str, Any]) -> dict[str, Any]:
    support = config["support_matching"]
    alternative = config["alternative"]
    if len(trial.targeted_support_drops) != len(trial.alternative_support_drops):
        raise ValueError("targeted/alternative beta rows are unpaired")
    finite = (
        trial.finite
        and _finite(trial.targeted_support_drops)
        and _finite(trial.alternative_support_drops)
        and math.isfinite(trial.median_norm_ratio)
        and math.isfinite(trial.max_abs_cosine)
    )
    if not finite:
        return {"passed": False, "reason": "non_finite"}
    target_median = median(trial.targeted_support_drops)
    alternative_median = median(trial.alternative_support_drops)
    mismatches = [
        abs(a - t)
        for t, a in zip(
            trial.targeted_support_drops,
            trial.alternative_support_drops,
            strict=True,
        )
    ]
    mismatch_median = median(mismatches)
    tolerance = max(
        float(support["absolute_tolerance_nat"]),
        float(support["relative_tolerance"]) * abs(target_median),
    )
    target_positive = target_median > 0
    alternative_positive = alternative_median > 0
    median_drop_match = (
        target_positive
        and abs(alternative_median - target_median)
        <= float(support["discovery_median_drop_relative_tolerance"]) * abs(target_median)
    )
    mismatch_match = mismatch_median <= tolerance
    norm_ok = trial.median_norm_ratio <= float(alternative["max_median_norm_ratio_to_targeted"])
    cosine_ok = trial.max_abs_cosine <= float(alternative["max_abs_cosine_with_answer_gradient"])
    passed = alternative_positive and median_drop_match and mismatch_match and norm_ok and cosine_ok
    return {
        "passed": passed,
        "targeted_median_support_drop": target_median,
        "alternative_median_support_drop": alternative_median,
        "median_absolute_support_mismatch": mismatch_median,
        "support_mismatch_tolerance": tolerance,
        "median_norm_ratio": trial.median_norm_ratio,
        "max_abs_cosine": trial.max_abs_cosine,
        "checks": {
            "targeted_positive": target_positive,
            "alternative_positive": alternative_positive,
            "median_drop_match": median_drop_match,
            "median_mismatch_match": mismatch_match,
            "norm_ratio": norm_ok,
            "cosine": cosine_ok,
        },
    }


def select_beta_strength(
    trials: Sequence[BetaTrial], config: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda trial: trial.beta)
    declared = [float(value) for value in config["strengths"]["beta_grid"]]
    if [float(trial.beta) for trial in ordered] != declared:
        raise ValueError("beta trials do not exactly cover the frozen grid")
    evaluated = [(trial, evaluate_beta_trial(trial, config)) for trial in ordered]
    eligible = [(trial, result) for trial, result in evaluated if result["passed"]]
    if not eligible:
        raise RuntimeError("support_match_gate_failed")
    chosen, diagnostics = min(
        eligible,
        key=lambda pair: (
            pair[1]["median_absolute_support_mismatch"],
            pair[0].beta,
        ),
    )
    return {
        "beta": chosen.beta,
        "diagnostics": diagnostics,
        "grid_diagnostics": {
            str(trial.beta): result for trial, result in evaluated
        },
    }

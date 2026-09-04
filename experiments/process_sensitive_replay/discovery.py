"""Deterministic discovery-only strength selection machinery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence

import torch

from .cache_state import assert_hybrid_cache_integrity, assert_process_propagated
from .gradient_intervention import (
    InterventionSchedule,
    build_interventions,
    compute_clean_gradients,
)
from .replay import replay_teacher_forced


DISCOVERY_CANDIDATE_CONDITIONS = (
    "clean_preserved",
    "targeted_weak_preserved",
    "targeted_strong_preserved",
    "random_strong_preserved",
    "support_matched_alternative_preserved",
    "alternative_random_preserved",
    "targeted_strong_reset",
)


@dataclass(frozen=True)
class AlphaTrial:
    alpha: float
    support_drops: tuple[float, ...]
    finite: bool = True


@dataclass(frozen=True)
class BetaTrial:
    alternative_layer: int
    beta: float
    targeted_support_drops: tuple[float, ...]
    alternative_support_drops: tuple[float, ...]
    median_norm_ratio: float
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
        raise RuntimeError(
            "alpha_strength_gate_failed: WEAK must be strictly smaller than STRONG; "
            f"selected weak={weak.alpha:g} strong={strong.alpha:g}"
        )
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
    passed = alternative_positive and median_drop_match and mismatch_match and norm_ok
    return {
        "passed": passed,
        "alternative_layer": int(trial.alternative_layer),
        "targeted_median_support_drop": target_median,
        "alternative_median_support_drop": alternative_median,
        "median_absolute_support_mismatch": mismatch_median,
        "support_mismatch_tolerance": tolerance,
        "median_norm_ratio": trial.median_norm_ratio,
        "checks": {
            "targeted_positive": target_positive,
            "alternative_positive": alternative_positive,
            "median_drop_match": median_drop_match,
            "median_mismatch_match": mismatch_match,
            "norm_ratio": norm_ok,
        },
    }


def select_beta_strength(
    trials: Sequence[BetaTrial], config: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda trial: (trial.alternative_layer, trial.beta))
    declared = [
        (int(layer), float(beta))
        for layer in config["layers"]["alternative_candidates"]
        for beta in config["strengths"]["beta_grid"]
    ]
    if [(trial.alternative_layer, float(trial.beta)) for trial in ordered] != declared:
        raise ValueError("alternative-layer/beta trials do not exactly cover the frozen grid")
    evaluated = [(trial, evaluate_beta_trial(trial, config)) for trial in ordered]
    eligible = [(trial, result) for trial, result in evaluated if result["passed"]]
    if not eligible:
        raise RuntimeError("support_match_gate_failed")
    chosen, diagnostics = min(
        eligible,
        key=lambda pair: (
            pair[1]["median_absolute_support_mismatch"],
            pair[0].alternative_layer,
            pair[0].beta,
        ),
    )
    return {
        "beta": chosen.beta,
        "alternative_layer": int(chosen.alternative_layer),
        "diagnostics": diagnostics,
        "grid_diagnostics": {
            f"layer_{trial.alternative_layer}/beta_{trial.beta:g}": result
            for trial, result in evaluated
        },
    }


def _replay(adapter: Any, answer: Mapping[str, Any], schedule: InterventionSchedule | None) -> Any:
    return replay_teacher_forced(
        adapter,
        post_answer_token_ids=answer["post_answer_token_ids"],
        question_prefix_token_ids=answer["question_prefix_token_ids"],
        answer_token_ids=answer["answer_token_ids"],
        intervention=schedule,
    )


def _schedule(
    bundle: Any,
    config: Mapping[str, Any],
    *,
    item_id: str,
    family: str,
    strength: float,
    process_layer: int,
) -> InterventionSchedule:
    if int(bundle.process_layer) != int(process_layer):
        raise AssertionError("gradient bundle layer does not match intervention layer")
    return InterventionSchedule(
        process_layer=int(process_layer),
        positions=build_interventions(
            bundle,
            family=family,
            strength=float(strength),
            campaign_seed=int(config["split"]["seed"]),
            item_id=str(item_id),
            max_abs_cosine=float(
                config["alternative"]["random_max_abs_cosine_with_answer_gradient"]
            ),
        ),
    )


def _position_measurements(schedule: InterventionSchedule) -> list[dict[str, Any]]:
    return [
        {
            "process_layer": int(schedule.process_layer),
            "position": int(spec.position),
            "token_id": int(spec.token_id),
            "family": str(spec.family),
            "strength": float(spec.strength),
            "residual_norm": float(spec.residual_norm),
            "answer_gradient_norm": float(spec.answer_gradient_norm),
            "perturbation_norm": float(torch.linalg.vector_norm(spec.delta.float()).item()),
            "direction_cosine": float(spec.direction_cosine),
            "rng_seed": spec.rng_seed,
            "used_fallback": bool(spec.used_fallback),
        }
        for spec in schedule.positions.values()
    ]


def measure_strength_grid_item(
    adapter: Any,
    answer: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    families: Sequence[str] = ("alpha", "beta"),
) -> dict[str, Any]:
    """Measure the complete frozen alpha/beta grid for one discovery item."""
    if bool(answer.get("invalid")):
        raise ValueError(f"discovery item {answer['item_id']} has an invalid answer")
    item_id = str(answer["item_id"])
    family_set = {str(value) for value in families}
    if not family_set or not family_set <= {"alpha", "beta"}:
        raise ValueError("discovery grid families must be alpha and/or beta")
    atol = float(config["reset_parity"]["absolute_tolerance"])
    rtol = float(config["reset_parity"]["relative_tolerance"])
    primary_layer = int(config["layers"]["process"])
    bundle = None
    if "alpha" in family_set:
        bundle = compute_clean_gradients(
            adapter,
            answer["post_answer_token_ids"],
            prefix_length=len(answer["question_prefix_token_ids"]),
            answer_token_ids=answer["answer_token_ids"],
            process_layer=primary_layer,
            atol=atol,
            rtol=rtol,
        )
    clean = _replay(adapter, answer, None)
    if bundle is not None:
        if not math.isclose(
            clean.answer_sequence_logp,
            bundle.answer_sequence_logp,
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise AssertionError("discovery clean recurrent/cached support parity failed")
    if clean.process_hook_positions:
        raise AssertionError("discovery clean replay unexpectedly activated the process hook")

    expected_length = len(answer["post_answer_token_ids"])
    layer_types = list(adapter.text_config.layer_types)

    def validate_grid_outcome(outcome: Any) -> None:
        assert_hybrid_cache_integrity(
            outcome.cache,
            layer_types=layer_types,
            expected_sequence_length=expected_length,
        )
        if (
            outcome.transcript_hash != clean.transcript_hash
            or outcome.question_token_hash != clean.question_token_hash
            or outcome.answer_token_hash != clean.answer_token_hash
        ):
            raise AssertionError("discovery strength grid visible-token parity failed")
        if not math.isfinite(outcome.answer_sequence_logp):
            raise AssertionError("discovery strength grid produced non-finite support")

    validate_grid_outcome(clean)
    clean.release_cache()

    alpha_grid: dict[str, Any] = {}
    beta_grid: dict[str, dict[str, Any]] = {}
    alternative_gradient_parity: dict[str, Any] = {}
    if "alpha" in family_set:
        if bundle is None:
            raise AssertionError("alpha discovery is missing its primary gradient bundle")
        for alpha_value in config["strengths"]["alpha_grid"]:
            alpha = float(alpha_value)
            schedule = _schedule(
                bundle, config, item_id=item_id, family="targeted", strength=alpha,
                process_layer=primary_layer,
            )
            outcome = _replay(adapter, answer, schedule)
            assert_process_propagated(
                clean.cache_audit,
                outcome.cache_audit,
                process_layer=primary_layer,
            )
            validate_grid_outcome(outcome)
            alpha_grid[str(alpha)] = {
                "support_after": float(outcome.answer_sequence_logp),
                "support_drop": float(clean.answer_sequence_logp - outcome.answer_sequence_logp),
                "cache_digest": outcome.cache_audit.digest,
                "positions": _position_measurements(schedule),
            }
            outcome.release_cache()
    if "beta" in family_set:
        for alternative_layer_value in config["layers"]["alternative_candidates"]:
            alternative_layer = int(alternative_layer_value)
            alternative_bundle = compute_clean_gradients(
                adapter,
                answer["post_answer_token_ids"],
                prefix_length=len(answer["question_prefix_token_ids"]),
                answer_token_ids=answer["answer_token_ids"],
                process_layer=alternative_layer,
                atol=atol,
                rtol=rtol,
            )
            if not math.isclose(
                clean.answer_sequence_logp,
                alternative_bundle.answer_sequence_logp,
                abs_tol=atol,
                rel_tol=rtol,
            ):
                raise AssertionError(
                    "alternative-layer recurrent/cached support parity failed"
                )
            beta_grid[str(alternative_layer)] = {}
            alternative_gradient_parity[str(alternative_layer)] = alternative_bundle.parity
            for beta_value in config["strengths"]["beta_grid"]:
                beta = float(beta_value)
                schedule = _schedule(
                    alternative_bundle,
                    config,
                    item_id=item_id,
                    family="alternative_targeted",
                    strength=beta,
                    process_layer=alternative_layer,
                )
                outcome = _replay(adapter, answer, schedule)
                assert_process_propagated(
                    clean.cache_audit,
                    outcome.cache_audit,
                    process_layer=alternative_layer,
                )
                validate_grid_outcome(outcome)
                beta_grid[str(alternative_layer)][str(beta)] = {
                    "alternative_layer": alternative_layer,
                    "support_after": float(outcome.answer_sequence_logp),
                    "support_drop": float(
                        clean.answer_sequence_logp - outcome.answer_sequence_logp
                    ),
                    "cache_digest": outcome.cache_audit.digest,
                    "positions": _position_measurements(schedule),
                }
                outcome.release_cache()

    return {
        "item_id": item_id,
        "clean_support": float(clean.answer_sequence_logp),
        "clean_cache_digest": clean.cache_audit.digest,
        "transcript_hash": clean.transcript_hash,
        "question_token_hash": clean.question_token_hash,
        "answer_token_hash": clean.answer_token_hash,
        "gradient_parity": None if bundle is None else bundle.parity,
        "alternative_gradient_parity": alternative_gradient_parity,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
    }


def _alpha_trials(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[AlphaTrial]:
    if not records:
        raise ValueError("discovery alpha selection received no records")
    return [
        AlphaTrial(
            alpha=float(alpha_value),
            support_drops=tuple(
                float(record["alpha_grid"][str(float(alpha_value))]["support_drop"])
                for record in records
            ),
            finite=_finite(tuple(
                float(record["alpha_grid"][str(float(alpha_value))]["support_drop"])
                for record in records
            )),
        )
        for alpha_value in config["strengths"]["alpha_grid"]
    ]


def alpha_grid_diagnostics(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    strengths = config["strengths"]
    low, high = (float(value) for value in strengths["strong_median_drop_range_nat"])
    rows = []
    for trial in _alpha_trials(records, config):
        trial_median = median(trial.support_drops)
        positive_items = sum(value > 0 for value in trial.support_drops)
        rows.append({
            "alpha": trial.alpha,
            "support_drops": list(trial.support_drops),
            "median_support_drop": trial_median,
            "positive_items": positive_items,
            "finite": trial.finite,
            "weak_eligible": (
                trial.finite
                and trial_median >= float(strengths["weak_min_median_drop_nat"])
                and positive_items >= int(strengths["weak_min_positive_items"])
            ),
            "strong_in_target_range": trial.finite and low <= trial_median <= high,
            "strong_fallback_eligible": trial.finite and trial_median < high,
        })
    return {
        "item_count": len(records),
        "weak_rule": {
            "minimum_median_drop_nat": float(strengths["weak_min_median_drop_nat"]),
            "minimum_positive_items": int(strengths["weak_min_positive_items"]),
        },
        "strong_target_range_nat": [low, high],
        "grid": rows,
    }


def _beta_trials(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    strong_alpha: float,
) -> list[BetaTrial]:
    targeted_drops = tuple(
        float(record["alpha_grid"][str(strong_alpha)]["support_drop"])
        for record in records
    )
    beta_trials: list[BetaTrial] = []
    for alternative_layer_value in config["layers"]["alternative_candidates"]:
        alternative_layer = int(alternative_layer_value)
        for beta_value in config["strengths"]["beta_grid"]:
            beta = float(beta_value)
            alternative_drops = tuple(
                float(
                    record["beta_grid"][str(alternative_layer)][str(beta)][
                        "support_drop"
                    ]
                )
                for record in records
            )
            norm_ratios: list[float] = []
            for record in records:
                targeted_positions = {
                    int(row["position"]): row
                    for row in record["alpha_grid"][str(strong_alpha)]["positions"]
                }
                alternative_positions = {
                    int(row["position"]): row
                    for row in record["beta_grid"][str(alternative_layer)][str(beta)][
                        "positions"
                    ]
                }
                if set(targeted_positions) != set(alternative_positions):
                    raise AssertionError(
                        "discovery targeted/alternative positions are unpaired"
                    )
                for position in sorted(targeted_positions):
                    target_norm = float(
                        targeted_positions[position]["perturbation_norm"]
                    )
                    alternative_norm = float(
                        alternative_positions[position]["perturbation_norm"]
                    )
                    if target_norm <= 0:
                        raise AssertionError(
                            "discovery targeted perturbation norm is non-positive"
                        )
                    norm_ratios.append(alternative_norm / target_norm)
            beta_trials.append(BetaTrial(
                alternative_layer=alternative_layer,
                beta=beta,
                targeted_support_drops=targeted_drops,
                alternative_support_drops=alternative_drops,
                median_norm_ratio=median(norm_ratios),
                finite=_finite((*targeted_drops, *alternative_drops, *norm_ratios)),
            ))
    return beta_trials


def beta_grid_diagnostics(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    strong_alpha: float,
) -> dict[str, Any]:
    trials = _beta_trials(records, config, strong_alpha=float(strong_alpha))
    return {
        "item_count": len(records),
        "frozen_strong_alpha": float(strong_alpha),
        "alternative_layer_candidates": [
            int(value) for value in config["layers"]["alternative_candidates"]
        ],
        "grid": {
            f"layer_{trial.alternative_layer}/beta_{trial.beta:g}":
                evaluate_beta_trial(trial, config)
            for trial in trials
        },
    }


def select_discovery_strengths(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select global alpha/beta from complete paired discovery-grid records."""
    if not records:
        raise ValueError("discovery strength selection received no records")
    alpha_selection = select_discovery_alpha(records, config)
    beta_trials = _beta_trials(
        records,
        config,
        strong_alpha=float(alpha_selection["strong_alpha"]),
    )
    beta_selection = select_beta_strength(beta_trials, config)
    return {"alpha": alpha_selection, "beta": beta_selection}


def select_discovery_alpha(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("discovery alpha selection received no records")
    return select_alpha_strengths(_alpha_trials(records, config), config)


def _descending_ordinal_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.float64)
    return ranks


def rank_candidate_grid(
    scores: torch.Tensor,
    support_drops: torch.Tensor,
    *,
    layers: Sequence[int],
    condition_names: Sequence[str],
    eligible_token_ids: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute the frozen discovery metrics and equal-weight rank aggregation."""
    condition_names = tuple(str(value) for value in condition_names)
    if condition_names != DISCOVERY_CANDIDATE_CONDITIONS:
        raise ValueError("candidate score conditions do not match the frozen order")
    if scores.ndim != 4:
        raise ValueError("candidate scores must have shape [item, condition, layer, vocab]")
    item_count, condition_count, layer_count, vocab_size = scores.shape
    if condition_count != len(condition_names) or layer_count != len(layers):
        raise ValueError("candidate score axes do not match their metadata")
    if support_drops.shape != (item_count, condition_count):
        raise ValueError("support-drop matrix is not aligned with candidate scores")
    if item_count < 2:
        raise ValueError("candidate discovery requires multiple items")
    values = scores.float()
    drops = support_drops.float()
    index = {name: condition_names.index(name) for name in condition_names}
    clean_index = index["clean_preserved"]
    weak_index = index["targeted_weak_preserved"]
    strong_index = index["targeted_strong_preserved"]
    random_index = index["random_strong_preserved"]
    alternative_index = index["support_matched_alternative_preserved"]
    alternative_random_index = index["alternative_random_preserved"]
    reset_index = index["targeted_strong_reset"]

    metric_names = tuple(config["candidate_selection"]["rank_metrics"])
    metric_layers: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}
    eligibility_layers: list[torch.Tensor] = []
    orientation_layers: list[torch.Tensor] = []
    divergence_layers: list[torch.Tensor] = []
    structured_consistency_layers: list[torch.Tensor] = []
    random_consistency_layers: list[torch.Tensor] = []
    token_mask = torch.zeros(vocab_size, dtype=torch.bool)
    eligible_ids = sorted(set(int(value) for value in eligible_token_ids))
    if not eligible_ids or eligible_ids[0] < 0 or eligible_ids[-1] >= vocab_size:
        raise ValueError("eligible candidate token IDs are empty or out of vocabulary")
    token_mask[eligible_ids] = True

    relation_indices = [clean_index, weak_index, strong_index]
    relation_x = drops[:, relation_indices]
    centered_x = relation_x - relation_x.mean(dim=1, keepdim=True)
    slope_denominator = (centered_x * centered_x).sum()
    if not torch.isfinite(slope_denominator) or float(slope_denominator.item()) <= 0:
        raise RuntimeError("candidate support-slope denominator is degenerate")
    pooled_drops = drops[:, [strong_index, alternative_index]]
    pooled_denominator = (pooled_drops * pooled_drops).sum().clamp_min(1e-12)

    for layer_offset in range(layer_count):
        layer_scores = values[:, :, layer_offset, :]
        clean = layer_scores[:, clean_index, :]
        targeted = layer_scores[:, strong_index, :]
        alternative = layer_scores[:, alternative_index, :]
        random = layer_scores[:, random_index, :]
        alternative_random = layer_scores[:, alternative_random_index, :]
        reset = layer_scores[:, reset_index, :]
        relation_y = layer_scores[:, relation_indices, :]
        centered_y = relation_y - relation_y.mean(dim=1, keepdim=True)
        slope = (centered_x[:, :, None] * centered_y).sum(dim=(0, 1)) / slope_denominator
        orientation = torch.sign(slope)

        target_effects = targeted - clean
        alternative_effects = alternative - clean
        random_effects = random - clean
        oriented_target = orientation * target_effects.mean(dim=0)
        oriented_alternative = orientation * alternative_effects.mean(dim=0)
        targeted_minus_random = orientation * (targeted - random).mean(dim=0)
        alternative_minus_random = orientation * (
            alternative - alternative_random
        ).mean(dim=0)
        targeted_minus_reset = orientation * (targeted - reset).mean(dim=0)
        structured_consistency = 0.5 * (
            (orientation[None, :] * target_effects > 0).float().mean(dim=0)
            + (orientation[None, :] * alternative_effects > 0).float().mean(dim=0)
        )
        alternative_random_effects = alternative_random - clean
        random_consistency = 0.5 * (
            (orientation[None, :] * random_effects > 0).float().mean(dim=0)
            + (
                orientation[None, :] * alternative_random_effects > 0
            ).float().mean(dim=0)
        )

        pooled_effects = torch.stack((target_effects, alternative_effects), dim=1)
        pooled_slope = (
            pooled_drops[:, :, None] * pooled_effects
        ).sum(dim=(0, 1)) / pooled_denominator
        residuals = pooled_effects - pooled_drops[:, :, None] * pooled_slope[None, None, :]
        agreement_mismatch = (residuals[:, 0, :] - residuals[:, 1, :]).abs().mean(dim=0)
        effect_scale = (
            target_effects.abs().mean(dim=0)
            + alternative_effects.abs().mean(dim=0)
        ).clamp_min(1e-12)
        divergence_ratio = agreement_mismatch / effect_scale
        score_variance = layer_scores.var(dim=(0, 1), unbiased=False)

        metrics = {
            "item_centered_support_slope": slope.abs(),
            "targeted_strong_effect": oriented_target,
            "support_matched_alternative_effect": oriented_alternative,
            "targeted_minus_random": targeted_minus_random,
            "alternative_minus_random": alternative_minus_random,
            "targeted_preserved_minus_reset": targeted_minus_reset,
            "support_adjusted_agreement": -agreement_mismatch,
            "item_sign_consistency": structured_consistency,
        }
        unknown = set(metric_names) - set(metrics)
        if unknown:
            raise ValueError(f"unknown frozen candidate rank metrics: {sorted(unknown)}")
        finite = torch.isfinite(score_variance) & torch.isfinite(divergence_ratio)
        for metric in metrics.values():
            finite &= torch.isfinite(metric)
        eligible = (
            token_mask
            & finite
            & (score_variance > float(config["candidate_selection"]["nontrivial_variance_epsilon"]))
            & (orientation != 0)
            & (oriented_target > 0)
            & (oriented_alternative > 0)
            & (targeted_minus_reset > 0)
            & (
                (targeted_minus_random > 0)
                | (structured_consistency > random_consistency)
            )
            & (
                (alternative_minus_random > 0)
                | (structured_consistency > random_consistency)
            )
            & (
                divergence_ratio
                <= float(config["candidate_selection"]["max_support_adjusted_divergence_ratio"])
            )
        )
        for name in metric_names:
            metric_layers[name].append(metrics[name].cpu())
        eligibility_layers.append(eligible.cpu())
        orientation_layers.append(orientation.cpu())
        divergence_layers.append(divergence_ratio.cpu())
        structured_consistency_layers.append(structured_consistency.cpu())
        random_consistency_layers.append(random_consistency.cpu())

    metric_tensors = {name: torch.stack(rows) for name, rows in metric_layers.items()}
    eligibility = torch.stack(eligibility_layers)
    eligible_flat = torch.nonzero(eligibility.reshape(-1), as_tuple=False).flatten()
    if eligible_flat.numel() == 0:
        raise RuntimeError("candidate_selection_gate_failed: no eligible directions")
    aggregate_rank = torch.zeros(eligible_flat.numel(), dtype=torch.float64)
    for name in metric_names:
        selected = metric_tensors[name].reshape(-1)[eligible_flat].double()
        ranks = _descending_ordinal_ranks(selected)
        aggregate_rank += ranks
    aggregate_rank /= len(metric_names)
    aggregate_order = torch.argsort(aggregate_rank, stable=True)
    ranked_flat_indices = eligible_flat[aggregate_order]
    aggregate_rank_sorted = aggregate_rank[aggregate_order]

    return {
        "layers": tuple(int(value) for value in layers),
        "vocab_size": int(vocab_size),
        "metric_names": metric_names,
        "metrics": metric_tensors,
        "eligibility": eligibility,
        "orientation": torch.stack(orientation_layers),
        "support_adjusted_divergence_ratio": torch.stack(divergence_layers),
        "structured_sign_consistency": torch.stack(structured_consistency_layers),
        "random_sign_consistency": torch.stack(random_consistency_layers),
        "ranked_flat_indices": ranked_flat_indices,
        "aggregate_rank": aggregate_rank_sorted,
        "eligible_count": int(eligible_flat.numel()),
    }


def candidate_ranking_row(
    ranking: Mapping[str, Any],
    rank_offset: int,
) -> dict[str, Any]:
    flat_index = int(ranking["ranked_flat_indices"][rank_offset].item())
    vocab_size = int(ranking["vocab_size"])
    layer_offset, token_id = divmod(flat_index, vocab_size)
    row = {
        "aggregate_rank": int(rank_offset + 1),
        "aggregate_mean_ordinal_rank": float(ranking["aggregate_rank"][rank_offset].item()),
        "layer": int(ranking["layers"][layer_offset]),
        "token_id": int(token_id),
        "orientation": int(ranking["orientation"][layer_offset, token_id].item()),
        "support_adjusted_divergence_ratio": float(
            ranking["support_adjusted_divergence_ratio"][layer_offset, token_id].item()
        ),
        "structured_sign_consistency": float(
            ranking["structured_sign_consistency"][layer_offset, token_id].item()
        ),
        "random_sign_consistency": float(
            ranking["random_sign_consistency"][layer_offset, token_id].item()
        ),
    }
    row["metrics"] = {
        name: float(ranking["metrics"][name][layer_offset, token_id].item())
        for name in ranking["metric_names"]
    }
    return row

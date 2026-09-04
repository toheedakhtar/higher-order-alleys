"""Pre-discovery engineering and post-freeze critical smoke execution."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch

from .cache_state import (
    assert_hybrid_cache_integrity,
    assert_numeric_parity,
    assert_process_propagated,
    assert_storage_disjoint,
    clone_hybrid_cache,
    release_cache_storage,
)
from .gradient_intervention import (
    InterventionSchedule,
    build_interventions,
    compute_clean_gradients,
)
from .jlens_readout import readout_layers
from .protocol import item_support_matched, support_match_summary
from .replay import (
    QwenReplayAdapter,
    append_meta_prompt,
    build_turn3_suffix,
    generate_from_cache,
    replay_teacher_forced,
    score_label_pair_from_cache,
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
            item_id=item_id,
            max_abs_cosine=float(
                config["alternative"]["random_max_abs_cosine_with_answer_gradient"]
            ),
        ),
    )


def _replay(
    adapter: QwenReplayAdapter,
    answer: Mapping[str, Any],
    schedule: InterventionSchedule | None,
) -> Any:
    return replay_teacher_forced(
        adapter,
        post_answer_token_ids=answer["post_answer_token_ids"],
        question_prefix_token_ids=answer["question_prefix_token_ids"],
        answer_token_ids=answer["answer_token_ids"],
        intervention=schedule,
    )


def _meta_branch(
    adapter: QwenReplayAdapter,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    source: Any,
    answer: Mapping[str, Any],
    branch_name: str,
    branch_config: Mapping[str, Any],
    config: Mapping[str, Any],
    explicit_token_ids: Sequence[int],
    *,
    include_full_vocab: bool = False,
) -> dict[str, Any]:
    turn3 = build_turn3_suffix(
        tokenizer,
        frozen_question_rendered=str(answer["question_rendered"]),
        frozen_answer_text=str(answer["answer"]),
        post_answer_token_ids=answer["post_answer_token_ids"],
        meta_prompt=str(branch_config["prompt"]),
    )
    suffix = list(turn3.token_ids)
    question_position = turn3.question_position
    boundary, logits, residuals, cache_audit, process_hook_registrations = append_meta_prompt(
        adapter,
        source.cache,
        suffix_token_ids=suffix,
        question_position=question_position,
        capture_layers=config["layers"]["readout"],
    )
    readout = readout_layers(
        lens_model,
        lens,
        residuals,
        layers=config["layers"]["readout"],
        top_k=int(config["readout"]["top_k"]),
        explicit_token_ids=explicit_token_ids,
        include_full_scores=include_full_vocab,
    )
    labels = [str(value) for value in branch_config["labels"]]
    scores = score_label_pair_from_cache(adapter, boundary, logits, tokenizer, labels)
    generated = generate_from_cache(
        adapter,
        boundary,
        logits,
        tokenizer,
        max_new_tokens=int(config["generation"]["max_choice_tokens"]),
    )
    result = {
        "branch": branch_name,
        "prompt": branch_config["prompt"],
        "labels": labels,
        "rendered": turn3.rendered_transcript,
        "rendered_suffix": turn3.rendered_suffix,
        "suffix_token_ids": suffix,
        "prefix_token_hash": turn3.prefix_token_hash,
        "suffix_token_hash": turn3.suffix_token_hash,
        "boundary_token_hash": turn3.boundary_token_hash,
        "final_transcript_token_hash": turn3.final_transcript_token_hash,
        "question_position": question_position,
        "question_token_id": int(tokenizer("?", add_special_tokens=False)["input_ids"][0]),
        "process_hook_calls": process_hook_registrations,
        "boundary_cache_digest": cache_audit.digest,
        "_boundary_logits": logits.detach().float().cpu(),
        "_question_residuals": {
            str(layer): residual.detach().float().cpu()
            for layer, residual in residuals.items()
        },
        "scores": scores,
        "generation": generated,
        "jlens": readout,
    }
    release_cache_storage(boundary)
    return result


def _condition_meta(
    adapter: QwenReplayAdapter,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    replay: Any,
    answer: Mapping[str, Any],
    config: Mapping[str, Any],
    explicit_token_ids: Sequence[int],
    *,
    include_full_vocab: bool = False,
) -> dict[str, Any]:
    isolation_clone = clone_hybrid_cache(replay.cache)
    assert_storage_disjoint(replay.cache, isolation_clone)
    release_cache_storage(isolation_clone)
    return {
        name: _meta_branch(
            adapter, lens_model, lens, tokenizer, replay, answer, name,
            branch_config, config, explicit_token_ids,
            include_full_vocab=include_full_vocab,
        )
        for name, branch_config in config["meta_branches"].items()
    }


def _assert_hash_parity(outcomes: Mapping[str, Any]) -> None:
    for field in ("transcript_hash", "question_token_hash", "answer_token_hash"):
        values = {getattr(outcome, field) for outcome in outcomes.values()}
        if len(values) != 1:
            raise AssertionError(f"visible-token parity failed for {field}")
    if not all(outcome.teacher_forced for outcome in outcomes.values()):
        raise AssertionError("one or more matched conditions did not use teacher forcing")


def run_smoke_item(
    adapter: QwenReplayAdapter,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    answer: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    phase: str,
    frozen_protocol: Mapping[str, Any] | None = None,
    include_full_vocab: bool = False,
) -> dict[str, Any]:
    if phase not in {
        "pre_discovery_smoke", "post_freeze_smoke", "discovery", "heldout"
    }:
        raise ValueError(f"unsupported replay phase: {phase}")
    if bool(answer.get("invalid")):
        raise ValueError(f"smoke item {answer['item_id']} has an invalid canonical answer")
    if phase in {"post_freeze_smoke", "heldout"} and frozen_protocol is None:
        raise RuntimeError(f"{phase} requires frozen_protocol.json")
    if phase == "discovery" and frozen_protocol is None:
        raise RuntimeError("discovery candidate replay requires selected strengths")
    if include_full_vocab and phase != "discovery":
        raise ValueError("full-vocabulary readout is restricted to discovery")
    item_id = str(answer["item_id"])
    primary_layer = int(config["layers"]["process"])
    bundle = compute_clean_gradients(
        adapter,
        answer["post_answer_token_ids"],
        prefix_length=len(answer["question_prefix_token_ids"]),
        answer_token_ids=answer["answer_token_ids"],
        process_layer=primary_layer,
        atol=float(config["reset_parity"]["absolute_tolerance"]),
        rtol=float(config["reset_parity"]["relative_tolerance"]),
    )
    clean = _replay(adapter, answer, None)
    atol = float(config["reset_parity"]["absolute_tolerance"])
    rtol = float(config["reset_parity"]["relative_tolerance"])
    if not math.isclose(clean.answer_sequence_logp, bundle.answer_sequence_logp, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(
            "clean cached support does not match recurrent gradient-pass support: "
            f"cached={clean.answer_sequence_logp:.9g} "
            f"gradient={bundle.answer_sequence_logp:.9g} "
            f"abs_diff={abs(clean.answer_sequence_logp - bundle.answer_sequence_logp):.9g}"
        )
    assert_numeric_parity(
        torch.tensor(clean.token_logprobs),
        torch.tensor(bundle.token_logprobs),
        atol=atol,
        rtol=rtol,
        context="clean recurrent gradient per-token answer log probabilities",
    )
    layer_types = list(adapter.text_config.layer_types)
    expected_length = len(answer["post_answer_token_ids"])

    def validate_grid_replay(outcome: Any, *, intervention_layer: int) -> None:
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
            raise AssertionError("engineering grid visible-token parity failed")
        if outcome.process_hook_positions != outcome.intervention_positions:
            raise AssertionError("engineering grid intervention hook scope failed")
        if not math.isfinite(float(outcome.answer_sequence_logp)):
            raise AssertionError("engineering grid produced non-finite support")
        assert_process_propagated(
            clean.cache_audit,
            outcome.cache_audit,
            process_layer=int(intervention_layer),
        )

    alpha_values = (
        [float(frozen_protocol["weak_alpha"]), float(frozen_protocol["strong_alpha"])]
        if frozen_protocol is not None
        else [float(value) for value in config["strengths"]["alpha_grid"]]
    )
    beta_values = (
        [float(frozen_protocol["beta"])]
        if frozen_protocol is not None
        else [float(value) for value in config["strengths"]["beta_grid"]]
    )
    alternative_layers = (
        [int(frozen_protocol["alternative_layer"])]
        if frozen_protocol is not None
        else [int(value) for value in config["layers"]["alternative_candidates"]]
    )
    strong_alpha = (
        float(frozen_protocol["strong_alpha"])
        if frozen_protocol is not None else max(alpha_values)
    )
    selected_beta = (
        float(frozen_protocol["beta"])
        if frozen_protocol is not None else max(beta_values)
    )
    selected_alternative_layer = (
        int(frozen_protocol["alternative_layer"])
        if frozen_protocol is not None else max(alternative_layers)
    )
    selected_alternative_key = (selected_alternative_layer, selected_beta)
    alternative_bundles: dict[int, Any] = {}
    for alternative_layer in alternative_layers:
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
                f"alternative layer {alternative_layer} recurrent/cached support parity failed"
            )
        assert_numeric_parity(
            torch.tensor(clean.token_logprobs),
            torch.tensor(alternative_bundle.token_logprobs),
            atol=atol,
            rtol=rtol,
            context=(
                f"alternative layer {alternative_layer} recurrent gradient per-token "
                "answer log probabilities"
            ),
        )
        alternative_bundles[alternative_layer] = alternative_bundle
    target_grid: dict[float, Any] = {}
    target_support_grid: dict[float, float] = {}
    target_schedules: dict[float, InterventionSchedule] = {}
    for alpha in alpha_values:
        schedule = _schedule(
            bundle, config, item_id=item_id, family="targeted", strength=alpha,
            process_layer=primary_layer,
        )
        outcome = _replay(adapter, answer, schedule)
        validate_grid_replay(outcome, intervention_layer=primary_layer)
        target_support_grid[alpha] = float(outcome.answer_sequence_logp)
        retained_alphas = (
            {float(frozen_protocol["weak_alpha"]), strong_alpha}
            if frozen_protocol is not None else {strong_alpha}
        )
        if alpha in retained_alphas:
            target_grid[alpha] = outcome
        else:
            outcome.release_cache()
        target_schedules[alpha] = schedule
    alternative_grid: dict[tuple[int, float], Any] = {}
    alternative_support_grid: dict[tuple[int, float], float] = {}
    alternative_schedules: dict[tuple[int, float], InterventionSchedule] = {}
    for alternative_layer, alternative_bundle in alternative_bundles.items():
        for beta in beta_values:
            key = (alternative_layer, beta)
            schedule = _schedule(
                alternative_bundle,
                config,
                item_id=item_id,
                family="alternative_targeted",
                strength=beta,
                process_layer=alternative_layer,
            )
            outcome = _replay(adapter, answer, schedule)
            validate_grid_replay(outcome, intervention_layer=alternative_layer)
            alternative_support_grid[key] = float(outcome.answer_sequence_logp)
            if key == selected_alternative_key:
                alternative_grid[key] = outcome
            else:
                outcome.release_cache()
            alternative_schedules[key] = schedule
    targeted = target_grid[strong_alpha]
    alternative = alternative_grid[selected_alternative_key]
    random_schedule = _schedule(
        bundle, config, item_id=item_id, family="random", strength=strong_alpha,
        process_layer=primary_layer,
    )
    random_control = _replay(adapter, answer, random_schedule)
    alternative_random_schedule = _schedule(
        alternative_bundles[selected_alternative_layer],
        config,
        item_id=item_id,
        family="alternative_random",
        strength=selected_beta,
        process_layer=selected_alternative_layer,
    )
    alternative_random_control = _replay(
        adapter, answer, alternative_random_schedule
    )
    targeted_reset = _replay(adapter, answer, None)
    clean_reset = _replay(adapter, answer, None)
    outcomes = {
        "clean_preserved": clean,
        "targeted_strong_preserved": targeted,
        "random_strong_preserved": random_control,
        "support_matched_alternative_preserved": alternative,
        "alternative_random_preserved": alternative_random_control,
        "targeted_strong_reset": targeted_reset,
        "clean_reset": clean_reset,
    }
    if frozen_protocol is not None:
        outcomes["targeted_weak_preserved"] = target_grid[float(frozen_protocol["weak_alpha"])]
    _assert_hash_parity(outcomes)
    for condition, outcome in outcomes.items():
        if outcome.process_hook_positions != outcome.intervention_positions:
            raise AssertionError(
                f"process hook escaped declared positions for {condition}: "
                f"declared={outcome.intervention_positions} "
                f"observed={outcome.process_hook_positions}"
            )
    if len({id(outcome.cache) for outcome in outcomes.values()}) != len(outcomes):
        raise AssertionError("experimental conditions share a cache object")
    for outcome in outcomes.values():
        assert_hybrid_cache_integrity(
            outcome.cache,
            layer_types=layer_types,
            expected_sequence_length=expected_length,
        )
        if (
            not math.isfinite(float(outcome.answer_sequence_logp))
            or not all(math.isfinite(float(value)) for value in outcome.token_logprobs)
        ):
            raise AssertionError("replay produced non-finite answer support or logits")
    assert_process_propagated(
        clean.cache_audit, targeted.cache_audit,
        process_layer=primary_layer,
    )
    assert_process_propagated(
        clean.cache_audit, random_control.cache_audit,
        process_layer=primary_layer,
    )
    assert_process_propagated(
        clean.cache_audit, alternative.cache_audit,
        process_layer=selected_alternative_layer,
    )
    assert_process_propagated(
        clean.cache_audit, alternative_random_control.cache_audit,
        process_layer=selected_alternative_layer,
    )
    manipulated_digests = {
        targeted.cache_audit.digest,
        random_control.cache_audit.digest,
        alternative.cache_audit.digest,
        alternative_random_control.cache_audit.digest,
    }
    if len(manipulated_digests) != 4:
        raise AssertionError(
            "targeted mechanisms/random controls produced aliased persistent states"
        )
    if targeted.cache_audit.digest == clean.cache_audit.digest:
        raise AssertionError("targeted cache was accidentally replaced by clean cache")
    if targeted_reset.cache_audit.digest != clean.cache_audit.digest:
        raise AssertionError("targeted reset did not reconstruct the clean token-only state")
    if clean_reset.cache_audit.digest != clean.cache_audit.digest:
        raise AssertionError("clean reset cache parity failed")
    assert_numeric_parity(
        torch.tensor(clean.token_logprobs), torch.tensor(clean_reset.token_logprobs),
        atol=atol, rtol=rtol, context="clean answer support",
    )
    target_drop = clean.answer_sequence_logp - targeted.answer_sequence_logp
    alternative_drop = clean.answer_sequence_logp - alternative.answer_sequence_logp
    alternative_random_drop = (
        clean.answer_sequence_logp
        - alternative_random_control.answer_sequence_logp
    )
    if target_drop <= 0 and phase in {"pre_discovery_smoke", "post_freeze_smoke"}:
        raise AssertionError("targeted intervention did not reduce answer support")
    if phase == "pre_discovery_smoke":
        smallest_alpha = min(alpha_values)
        if (
            clean.answer_sequence_logp
            - target_support_grid[smallest_alpha]
        ) <= 0:
            raise AssertionError(
                "primary negative-gradient finite-difference sign check failed"
            )
        smallest_beta = min(beta_values)
        for alternative_layer in alternative_layers:
            if (
                clean.answer_sequence_logp
                - alternative_support_grid[(alternative_layer, smallest_beta)]
            ) <= 0:
                raise AssertionError(
                    "alternative negative-gradient finite-difference sign check failed "
                    f"at layer {alternative_layer}"
                )

    explicit = [int(value) for value in config["readout"]["generic_evaluator_token_ids"]]
    if frozen_protocol is not None:
        explicit.extend(int(value) for value in frozen_protocol.get("candidate_token_ids", ()))
    meta = {
        condition: _condition_meta(
            adapter, lens_model, lens, tokenizer, outcome, answer, config, explicit,
            include_full_vocab=include_full_vocab,
        )
        for condition, outcome in outcomes.items()
    }
    turn3_process_hook_calls = sum(
        int(branch["process_hook_calls"])
        for branches in meta.values()
        for branch in branches.values()
    )
    if turn3_process_hook_calls != 0:
        raise AssertionError(
            f"process hook fired {turn3_process_hook_calls} times during Turn 3"
        )
    for condition, branches in meta.items():
        for branch_name, branch in branches.items():
            if not bool(torch.isfinite(branch["_boundary_logits"]).all().item()):
                raise AssertionError(
                    f"non-finite Turn-3 boundary logits for {condition}/{branch_name}"
                )
            if not all(
                bool(torch.isfinite(value).all().item())
                for value in branch["_question_residuals"].values()
            ):
                raise AssertionError(
                    f"non-finite Turn-3 residual for {condition}/{branch_name}"
                )
            score_values = [float(branch["scores"]["margin"])]
            score_values.extend(
                float(branch["scores"][label]["sequence_logprob"])
                for label in branch["labels"]
            )
            if not all(math.isfinite(value) for value in score_values):
                raise AssertionError(
                    f"non-finite Turn-3 label score for {condition}/{branch_name}"
                )
            for layer_data in branch["jlens"].values():
                readout_scores = [
                    float(entry["score"]) for entry in layer_data["top_k"]
                ]
                readout_scores.extend(
                    float(entry["score"])
                    for entry in layer_data["explicit"].values()
                )
                if not all(math.isfinite(value) for value in readout_scores):
                    raise AssertionError(
                        f"non-finite J-Lens score for {condition}/{branch_name}"
                    )
    for branch_name in config["meta_branches"]:
        clean_branch = meta["clean_preserved"][branch_name]
        reset_branch = meta["clean_reset"][branch_name]
        if clean_branch["suffix_token_hash"] != reset_branch["suffix_token_hash"]:
            raise AssertionError("clean reset Turn-3 token parity failed")
        if clean_branch["boundary_cache_digest"] != reset_branch["boundary_cache_digest"]:
            raise AssertionError("clean reset Turn-3 boundary cache parity failed")
        assert_numeric_parity(
            clean_branch["_boundary_logits"], reset_branch["_boundary_logits"],
            atol=atol, rtol=rtol, context=f"clean reset {branch_name} full logits",
        )
        assert_numeric_parity(
            torch.tensor([clean_branch["scores"]["margin"]]),
            torch.tensor([reset_branch["scores"]["margin"]]),
            atol=atol, rtol=rtol, context=f"clean reset {branch_name} margin",
        )
        for layer in config["layers"]["readout"]:
            assert_numeric_parity(
                clean_branch["_question_residuals"][str(layer)],
                reset_branch["_question_residuals"][str(layer)],
                atol=atol, rtol=rtol,
                context=f"clean reset {branch_name} layer {layer} residual",
            )
            for token_id in explicit:
                left = clean_branch["jlens"][str(layer)]["explicit"][str(token_id)]["score"]
                right = reset_branch["jlens"][str(layer)]["explicit"][str(token_id)]["score"]
                assert_numeric_parity(
                    torch.tensor([left]), torch.tensor([right]),
                    atol=atol, rtol=rtol,
                    context=f"clean reset {branch_name} layer {layer} token {token_id}",
                )
        targeted_reset_branch = meta["targeted_strong_reset"][branch_name]
        if (
            clean_branch["suffix_token_hash"]
            != targeted_reset_branch["suffix_token_hash"]
        ):
            raise AssertionError("targeted reset Turn-3 token parity failed")
        if (
            clean_branch["boundary_cache_digest"]
            != targeted_reset_branch["boundary_cache_digest"]
        ):
            raise AssertionError("targeted reset Turn-3 boundary cache parity failed")
        assert_numeric_parity(
            clean_branch["_boundary_logits"], targeted_reset_branch["_boundary_logits"],
            atol=atol, rtol=rtol,
            context=f"targeted reset {branch_name} full logits",
        )
        assert_numeric_parity(
            torch.tensor([clean_branch["scores"]["margin"]]),
            torch.tensor([targeted_reset_branch["scores"]["margin"]]),
            atol=atol, rtol=rtol, context=f"targeted reset {branch_name} margin",
        )
        for layer in config["layers"]["readout"]:
            assert_numeric_parity(
                clean_branch["_question_residuals"][str(layer)],
                targeted_reset_branch["_question_residuals"][str(layer)],
                atol=atol, rtol=rtol,
                context=f"targeted reset {branch_name} layer {layer} residual",
            )
            for token_id in explicit:
                left = clean_branch["jlens"][str(layer)]["explicit"][str(token_id)]["score"]
                right = targeted_reset_branch["jlens"][str(layer)]["explicit"][str(token_id)]["score"]
                assert_numeric_parity(
                    torch.tensor([left]), torch.tensor([right]),
                    atol=atol, rtol=rtol,
                    context=f"targeted reset {branch_name} layer {layer} token {token_id}",
                )
    for branch_name in config["meta_branches"]:
        hash_fields = (
            "prefix_token_hash",
            "suffix_token_hash",
            "boundary_token_hash",
            "final_transcript_token_hash",
        )
        for field in hash_fields:
            values = {branches[branch_name][field] for branches in meta.values()}
            if len(values) != 1:
                raise AssertionError(
                    f"Turn-3 {field} differs across conditions for {branch_name}"
                )
        question_positions = {
            branches[branch_name]["question_position"] for branches in meta.values()
        }
        question_token_ids = {
            branches[branch_name]["question_token_id"] for branches in meta.values()
        }
        if len(question_positions) != 1 or len(question_token_ids) != 1:
            raise AssertionError(f"Turn-3 token/? alignment differs across conditions for {branch_name}")
    discovery_vocab_scores: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for condition, branches in meta.items():
        if include_full_vocab:
            discovery_vocab_scores[condition] = {}
        for branch in branches.values():
            if include_full_vocab:
                discovery_vocab_scores[condition][str(branch["branch"])] = {
                    str(layer): layer_data.pop("_full_scores")
                    for layer, layer_data in branch["jlens"].items()
                }
            branch.pop("_boundary_logits")
            branch.pop("_question_residuals")
    random_specs = random_schedule.positions
    targeted_specs = target_schedules[strong_alpha].positions
    for position in targeted_specs:
        target_norm = torch.linalg.vector_norm(targeted_specs[position].delta.float())
        random_norm = torch.linalg.vector_norm(random_specs[position].delta.float())
        assert_numeric_parity(
            target_norm.reshape(1), random_norm.reshape(1),
            atol=1e-5, rtol=1e-5, context=f"random norm position {position}",
        )
    alternative_norms = [
        float(torch.linalg.vector_norm(value.delta.float()).item())
        for value in alternative_schedules[selected_alternative_key].positions.values()
    ]
    for position in alternative_schedules[selected_alternative_key].positions:
        alternative_target_norm = torch.linalg.vector_norm(
            alternative_schedules[selected_alternative_key]
            .positions[position]
            .delta.float()
        )
        alternative_random_norm = torch.linalg.vector_norm(
            alternative_random_schedule.positions[position].delta.float()
        )
        assert_numeric_parity(
            alternative_target_norm.reshape(1),
            alternative_random_norm.reshape(1),
            atol=1e-5,
            rtol=1e-5,
            context=f"alternative random norm position {position}",
        )
    targeted_norms = [
        float(torch.linalg.vector_norm(value.delta.float()).item())
        for value in targeted_specs.values()
    ]
    norm_ratios = [
        alternative_norm / targeted_norm
        for alternative_norm, targeted_norm in zip(alternative_norms, targeted_norms, strict=True)
    ]
    median_norm_ratio = float(torch.tensor(norm_ratios).median().item())
    if median_norm_ratio > float(config["alternative"]["max_median_norm_ratio_to_targeted"]):
        raise AssertionError("support-matched alternative exceeds perturbation-norm ceiling")

    def intervention_records(
        specs: Mapping[int, Any], *, condition: str, support_after: float,
        process_layer: int,
    ) -> list[dict[str, Any]]:
        return [
            {
                "condition": condition,
                "process_layer": int(process_layer),
                "token_index": value.position,
                "token_id": value.token_id,
                "token": tokenizer.decode(
                    [value.token_id], clean_up_tokenization_spaces=False
                ),
                "family": value.family,
                "strength": value.strength,
                "alpha": value.strength if value.family in {"targeted", "random"} else None,
                "beta": value.strength if value.family in {
                    "alternative_targeted", "alternative_random"
                } else None,
                "residual_norm": value.residual_norm,
                "answer_gradient_norm": value.answer_gradient_norm,
                "perturbation_norm": float(torch.linalg.vector_norm(value.delta.float()).item()),
                "direction_cosine": value.direction_cosine,
                "rng_seed": value.rng_seed,
                "used_fallback": value.used_fallback,
                "support_before": clean.answer_sequence_logp,
                "support_after": support_after,
            }
            for value in specs.values()
        ]
    intervention_records_all: list[dict[str, Any]] = []
    for alpha, schedule in target_schedules.items():
        label = (
            "targeted_weak_preserved"
            if frozen_protocol is not None and alpha == float(frozen_protocol["weak_alpha"])
            else "targeted_strong_preserved"
            if frozen_protocol is not None and alpha == float(frozen_protocol["strong_alpha"])
            else f"engineering_targeted_alpha_{alpha:g}"
        )
        intervention_records_all.extend(intervention_records(
            schedule.positions,
            condition=label,
            support_after=target_support_grid[alpha],
            process_layer=primary_layer,
        ))
        if (
            frozen_protocol is not None
            and alpha == float(frozen_protocol["strong_alpha"])
        ):
            reset_rows = intervention_records(
                schedule.positions,
                condition="targeted_strong_reset",
                support_after=target_support_grid[alpha],
                process_layer=primary_layer,
            )
            for row in reset_rows:
                row["state_disposition"] = "discarded_and_reconstructed_clean_before_turn3"
            intervention_records_all.extend(reset_rows)
    for (alternative_layer, beta), schedule in alternative_schedules.items():
        label = (
            "support_matched_alternative_preserved"
            if frozen_protocol is not None
            else f"engineering_alternative_layer_{alternative_layer}_beta_{beta:g}"
        )
        intervention_records_all.extend(intervention_records(
            schedule.positions,
            condition=label,
            support_after=alternative_support_grid[(alternative_layer, beta)],
            process_layer=alternative_layer,
        ))
    intervention_records_all.extend(intervention_records(
        random_specs,
        condition="random_strong_preserved",
        support_after=random_control.answer_sequence_logp,
        process_layer=primary_layer,
    ))
    intervention_records_all.extend(intervention_records(
        alternative_random_schedule.positions,
        condition="alternative_random_preserved",
        support_after=alternative_random_control.answer_sequence_logp,
        process_layer=selected_alternative_layer,
    ))
    result = {
        "item_id": item_id,
        "phase": phase,
        "gradient": {
            "clean_support": bundle.answer_sequence_logp,
            "primary_layer": primary_layer,
            "predictor_positions": list(bundle.predictor_positions),
            "answer_token_ids": list(bundle.answer_token_ids),
            "token_logprobs": list(bundle.token_logprobs),
            "parity": bundle.parity,
            "alternative_layers": {
                str(layer): {
                    "clean_support": alternative_bundle.answer_sequence_logp,
                    "predictor_positions": list(
                        alternative_bundle.predictor_positions
                    ),
                    "answer_token_ids": list(alternative_bundle.answer_token_ids),
                    "token_logprobs": list(alternative_bundle.token_logprobs),
                    "parity": alternative_bundle.parity,
                }
                for layer, alternative_bundle in alternative_bundles.items()
            },
        },
        "support": {
            "clean": clean.answer_sequence_logp,
            "target_grid": {
                str(key): value for key, value in target_support_grid.items()
            },
            "alternative_grid": {
                str(layer): {
                    str(beta): alternative_support_grid[(layer, beta)]
                    for beta in beta_values
                }
                for layer in alternative_layers
            },
            "alternative_layer": selected_alternative_layer,
            "targeted_drop": target_drop,
            "alternative_drop": alternative_drop,
            "random_drop": clean.answer_sequence_logp - random_control.answer_sequence_logp,
            "alternative_random_drop": alternative_random_drop,
            "condition_drops": {
                "clean_preserved": 0.0,
                "targeted_weak_preserved": (
                    clean.answer_sequence_logp
                    - target_support_grid[float(frozen_protocol["weak_alpha"])]
                    if frozen_protocol is not None else None
                ),
                "targeted_strong_preserved": target_drop,
                "random_strong_preserved": (
                    clean.answer_sequence_logp - random_control.answer_sequence_logp
                ),
                "support_matched_alternative_preserved": alternative_drop,
                "alternative_random_preserved": alternative_random_drop,
                # The targeted first-order process occurred; only its stored
                # state is discarded and reconstructed cleanly before Turn 3.
                "targeted_strong_reset": target_drop,
            },
        },
        "hashes": {
            condition: {
                "transcript": outcome.transcript_hash,
                "question": outcome.question_token_hash,
                "answer": outcome.answer_token_hash,
                "cache": outcome.cache_audit.digest,
            }
            for condition, outcome in outcomes.items()
        },
        "state_audits": {
            condition: {
                "cache_object_id": id(outcome.cache),
                "digest": outcome.cache_audit.digest,
                "structure_digest": outcome.cache_audit.structure_digest,
                "layer_digests": list(outcome.cache_audit.layer_digests),
                "tensors": [asdict(tensor) for tensor in outcome.cache_audit.tensors],
            }
            for condition, outcome in outcomes.items()
        },
        "interventions": intervention_records_all,
        "meta": meta,
        "checks": {
            "visible_hash_parity": True,
            "teacher_forcing_all_conditions": True,
            "clean_gradient_cache_support_parity": True,
            "gradient_token_logit_parity": True,
            "gradient_total_support_parity": True,
            "gradient_residual_parity": True,
            "gradient_finite_nonzero": True,
            "gradient_hook_scope": True,
            "gradient_sign_finite_difference": True,
            "alternative_gradient_parity": True,
            "intervention_hook_scope": True,
            "hybrid_cache_integrity": True,
            "downstream_state_changed": True,
            "reset_parity": True,
            "branch_isolation": True,
            "turn3_suffix_integrity": True,
            "turn3_process_hook_calls": turn3_process_hook_calls,
            "random_norm_match": True,
            "alternative_random_norm_match": True,
            "alternative_norm_ceiling": True,
        },
    }
    if include_full_vocab:
        result["_discovery_vocab_scores"] = discovery_vocab_scores
    for outcome in outcomes.values():
        outcome.release_cache()
    return result


def summarize_smoke(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("smoke produced no records")
    required_checks = {
        "visible_hash_parity", "teacher_forcing_all_conditions",
        "clean_gradient_cache_support_parity", "hybrid_cache_integrity",
        "downstream_state_changed", "reset_parity", "branch_isolation",
        "random_norm_match", "alternative_norm_ceiling",
        "gradient_token_logit_parity", "gradient_total_support_parity",
        "gradient_residual_parity", "gradient_finite_nonzero",
        "gradient_hook_scope", "alternative_gradient_parity",
        "gradient_sign_finite_difference",
        "intervention_hook_scope",
        "turn3_suffix_integrity",
        "alternative_random_norm_match",
    }
    for record in records:
        checks = record["checks"]
        if not all(checks.get(name) is True for name in required_checks):
            raise AssertionError(f"critical smoke assertion missing/failed for item {record['item_id']}")
        if checks.get("turn3_process_hook_calls") != 0:
            raise AssertionError(f"process hook fired during Turn 3 for item {record['item_id']}")
    result: dict[str, Any] = {
        "passed": True,
        "phase": phase,
        "items": len(records),
        "critical_checks": sorted(required_checks),
    }
    if phase == "post_freeze_smoke":
        support_config = config["support_matching"]
        item_measurements = []
        for record in records:
            targeted_drop = float(record["support"]["targeted_drop"])
            alternative_drop = float(record["support"]["alternative_drop"])
            tolerance = max(
                float(support_config["absolute_tolerance_nat"]),
                float(support_config["relative_tolerance"]) * abs(targeted_drop),
            )
            item_measurements.append({
                "item_id": str(record["item_id"]),
                "targeted_drop": targeted_drop,
                "alternative_drop": alternative_drop,
                "signed_mismatch": alternative_drop - targeted_drop,
                "absolute_mismatch": abs(alternative_drop - targeted_drop),
                "item_tolerance": tolerance,
                "targeted_drop_positive": targeted_drop > 0,
                "support_matched": item_support_matched(
                    targeted_drop,
                    alternative_drop,
                    absolute_tolerance_nat=float(
                        support_config["absolute_tolerance_nat"]
                    ),
                    relative_tolerance=float(support_config["relative_tolerance"]),
                ),
            })
        matching = support_match_summary(
            [
                (record["support"]["targeted_drop"], record["support"]["alternative_drop"])
                for record in records
            ],
            config,
        )
        result["support_matching"] = matching
        result["item_support_matching"] = item_measurements
        if not matching["passed"]:
            result["passed"] = False
            result["failure_reason"] = "support_match_gate_failed"
    else:
        result["support_matching"] = {
            "required": False,
            "reason": "pre-discovery engineering smoke does not select or require frozen beta",
        }
    return result

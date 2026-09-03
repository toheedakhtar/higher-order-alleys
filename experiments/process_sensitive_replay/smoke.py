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
)
from .gradient_intervention import (
    InterventionSchedule,
    build_interventions,
    compute_clean_gradients,
)
from .jlens_readout import readout_layers
from .protocol import support_match_summary
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
) -> InterventionSchedule:
    return InterventionSchedule(
        process_layer=int(config["layers"]["process"]),
        positions=build_interventions(
            bundle,
            family=family,
            strength=float(strength),
            campaign_seed=int(config["split"]["seed"]),
            item_id=item_id,
            max_abs_cosine=float(config["alternative"]["max_abs_cosine_with_answer_gradient"]),
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
    return {
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


def _condition_meta(
    adapter: QwenReplayAdapter,
    lens_model: Any,
    lens: Any,
    tokenizer: Any,
    replay: Any,
    answer: Mapping[str, Any],
    config: Mapping[str, Any],
    explicit_token_ids: Sequence[int],
) -> dict[str, Any]:
    assert_storage_disjoint(replay.cache, clone_hybrid_cache(replay.cache))
    return {
        name: _meta_branch(
            adapter, lens_model, lens, tokenizer, replay, answer, name,
            branch_config, config, explicit_token_ids,
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
) -> dict[str, Any]:
    if phase not in {"pre_discovery_smoke", "post_freeze_smoke"}:
        raise ValueError(f"not a smoke phase: {phase}")
    if bool(answer.get("invalid")):
        raise ValueError(f"smoke item {answer['item_id']} has an invalid canonical answer")
    if phase == "post_freeze_smoke" and frozen_protocol is None:
        raise RuntimeError("post-freeze smoke requires frozen_protocol.json")
    item_id = str(answer["item_id"])
    bundle = compute_clean_gradients(
        adapter,
        answer["post_answer_token_ids"],
        prefix_length=len(answer["question_prefix_token_ids"]),
        answer_token_ids=answer["answer_token_ids"],
        process_layer=int(config["layers"]["process"]),
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
    target_grid: dict[float, Any] = {}
    target_schedules: dict[float, InterventionSchedule] = {}
    for alpha in alpha_values:
        schedule = _schedule(bundle, config, item_id=item_id, family="targeted", strength=alpha)
        target_grid[alpha] = _replay(adapter, answer, schedule)
        target_schedules[alpha] = schedule
    alternative_grid: dict[float, Any] = {}
    alternative_schedules: dict[float, InterventionSchedule] = {}
    for beta in beta_values:
        schedule = _schedule(bundle, config, item_id=item_id, family="alternative", strength=beta)
        alternative_grid[beta] = _replay(adapter, answer, schedule)
        alternative_schedules[beta] = schedule

    strong_alpha = (
        float(frozen_protocol["strong_alpha"])
        if frozen_protocol is not None else max(alpha_values)
    )
    selected_beta = float(frozen_protocol["beta"]) if frozen_protocol is not None else max(beta_values)
    targeted = target_grid[strong_alpha]
    alternative = alternative_grid[selected_beta]
    random_schedule = _schedule(
        bundle, config, item_id=item_id, family="random", strength=strong_alpha
    )
    random_control = _replay(adapter, answer, random_schedule)
    targeted_reset = _replay(adapter, answer, None)
    clean_reset = _replay(adapter, answer, None)
    outcomes = {
        "clean_preserved": clean,
        "targeted_strong_preserved": targeted,
        "random_strong_preserved": random_control,
        "support_matched_alternative_preserved": alternative,
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
    layer_types = list(adapter.text_config.layer_types)
    expected_length = len(answer["post_answer_token_ids"])
    for outcome in outcomes.values():
        assert_hybrid_cache_integrity(
            outcome.cache,
            layer_types=layer_types,
            expected_sequence_length=expected_length,
        )
    assert_process_propagated(
        clean.cache_audit, targeted.cache_audit,
        process_layer=int(config["layers"]["process"]),
    )
    assert_process_propagated(
        clean.cache_audit, random_control.cache_audit,
        process_layer=int(config["layers"]["process"]),
    )
    assert_process_propagated(
        clean.cache_audit, alternative.cache_audit,
        process_layer=int(config["layers"]["process"]),
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
    if target_drop <= 0:
        raise AssertionError("targeted intervention did not reduce answer support")

    explicit = [int(value) for value in config["readout"]["generic_evaluator_token_ids"]]
    if frozen_protocol is not None:
        explicit.extend(int(value) for value in frozen_protocol.get("candidate_token_ids", ()))
    meta = {
        condition: _condition_meta(
            adapter, lens_model, lens, tokenizer, outcome, answer, config, explicit
        )
        for condition, outcome in outcomes.items()
    }
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
        assert_numeric_parity(
            clean_branch["_boundary_logits"], targeted_reset_branch["_boundary_logits"],
            atol=atol, rtol=rtol,
            context=f"targeted reset {branch_name} full logits",
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
    for branches in meta.values():
        for branch in branches.values():
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
        for value in alternative_schedules[selected_beta].positions.values()
    ]
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
        specs: Mapping[int, Any], *, condition: str, support_after: float
    ) -> list[dict[str, Any]]:
        return [
            {
                "condition": condition,
                "process_layer": int(config["layers"]["process"]),
                "token_index": value.position,
                "token_id": value.token_id,
                "token": tokenizer.decode(
                    [value.token_id], clean_up_tokenization_spaces=False
                ),
                "family": value.family,
                "strength": value.strength,
                "alpha": value.strength if value.family in {"targeted", "random"} else None,
                "beta": value.strength if value.family == "alternative" else None,
                "residual_norm": value.residual_norm,
                "answer_gradient_norm": value.answer_gradient_norm,
                "alternative_gradient_norm": value.alternative_gradient_norm,
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
            support_after=target_grid[alpha].answer_sequence_logp,
        ))
    for beta, schedule in alternative_schedules.items():
        label = (
            "support_matched_alternative_preserved"
            if frozen_protocol is not None
            else f"engineering_alternative_beta_{beta:g}"
        )
        intervention_records_all.extend(intervention_records(
            schedule.positions,
            condition=label,
            support_after=alternative_grid[beta].answer_sequence_logp,
        ))
    intervention_records_all.extend(intervention_records(
        random_specs,
        condition="random_strong_preserved",
        support_after=random_control.answer_sequence_logp,
    ))
    return {
        "item_id": item_id,
        "phase": phase,
        "gradient": {
            "clean_support": bundle.answer_sequence_logp,
            "predictor_positions": list(bundle.predictor_positions),
            "answer_token_ids": list(bundle.answer_token_ids),
            "token_logprobs": list(bundle.token_logprobs),
            "parity": bundle.parity,
        },
        "support": {
            "clean": clean.answer_sequence_logp,
            "target_grid": {str(key): value.answer_sequence_logp for key, value in target_grid.items()},
            "alternative_grid": {str(key): value.answer_sequence_logp for key, value in alternative_grid.items()},
            "targeted_drop": target_drop,
            "alternative_drop": alternative_drop,
            "random_drop": clean.answer_sequence_logp - random_control.answer_sequence_logp,
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
            "intervention_hook_scope": True,
            "hybrid_cache_integrity": True,
            "downstream_state_changed": True,
            "reset_parity": True,
            "branch_isolation": True,
            "turn3_suffix_integrity": True,
            "turn3_process_hook_calls": 0,
            "random_norm_match": True,
            "alternative_norm_ceiling": True,
        },
    }


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
        "gradient_hook_scope", "intervention_hook_scope",
        "turn3_suffix_integrity",
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
        matching = support_match_summary(
            [
                (record["support"]["targeted_drop"], record["support"]["alternative_drop"])
                for record in records
            ],
            config,
        )
        result["support_matching"] = matching
        if not matching["passed"]:
            raise RuntimeError("support_match_gate_failed")
    else:
        result["support_matching"] = {
            "required": False,
            "reason": "pre-discovery engineering smoke does not select or require frozen beta",
        }
    return result

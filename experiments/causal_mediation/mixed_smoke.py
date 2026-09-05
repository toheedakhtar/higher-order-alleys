"""Two-item BF16/FP32 no-patch equivalence study. Never launches mediation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from experiments.process_sensitive_replay.cache_state import (
    audit_cache, clone_hybrid_cache, release_cache_storage, assert_cache_unchanged,
    assert_hybrid_cache_integrity, assert_process_propagated,
)
from experiments.process_sensitive_replay.cuda_memory import CudaMemoryTrendGuard, reclaim_cuda_memory
from experiments.process_sensitive_replay.gradient_intervention import compute_clean_gradients, hidden_tensor, replace_hidden
from experiments.process_sensitive_replay.protocol import sha256_json, hash_token_ids
from experiments.process_sensitive_replay.replay import build_turn3_suffix, label_token_ids, eos_token_ids
from experiments.process_sensitive_replay.runner import append_jsonl, load_runtime, run_cuda_item
from experiments.process_sensitive_replay.smoke import _replay, _schedule

from .mixed_precision import POLICY, FP32Tail, candidate_patch, realize_patch, promote_tail_cache, state_topology, tensor_sha
from .precision import random_direction
from .precision_smoke import _support_reference, candidate_score
from .upstream import DEFAULT_DIRECTION, DEFAULT_UPSTREAM, ROOT, SMOKE_IDS, load_upstream, require_runtime_packages, sha


# A proposal, deliberately not an authorization gate. Fixed before any model
# judgments are inspected; approval still required after the observed report.
EQUIVALENCE_PROPOSAL = {
    "status": "proposed_not_approved",
    "all_12_generated_labels_identical_and_valid": True,
    "all_complete_label_sequence_logp_absolute_errors_max_nats": .05,
    "all_margin_absolute_errors_max_nats": .05,
    "all_process_minus_clean_margin_contrast_absolute_errors_max_nats": .05,
    "process_contrast_sign_preserved": True,
    "process_contrast_relative_error_max": .10,
    "zero_reference_contrast_requires_absolute_error_max_nats": 1e-5,
    "justification": "0.05 nats bounds likelihood/odds ratios by exp(0.05)=1.0513; "
        "10% contrast drift additionally protects naturally small process effects. "
        "These are conservative operational limits, not proof of equivalence or mediation.",
}


def check_state(adapter, cache):
    audit = assert_hybrid_cache_integrity(cache, layer_types=adapter.text_config.layer_types,
        expected_sequence_length=adapter.cache_length(cache))
    topology = state_topology(cache)
    if getattr(adapter, "_mixed_tail_active", False):
        for tensor in topology:
            if any(tensor["path"].startswith(f"cache.layers[{layer}].") for layer in range(43, 64)):
                if tensor["dtype"] in ("torch.bfloat16", "torch.float16", "torch.float64"):
                    raise AssertionError("tail cache left the declared FP32 representation")
    return {"digest": audit.digest, "structure_digest": audit.structure_digest,
            "topology": topology}


@torch.no_grad()
def judgments(adapter, cache, logits, tokenizer, labels, max_new_tokens):
    """Original scoring equations, with explicit per-prefix relevant logits."""
    before = audit_cache(cache)
    results = {}
    for label in labels:
        branch = clone_hybrid_cache(cache)
        try:
            ids = label_token_ids(tokenizer, label)
            scores, raw_logits = [], []
            current = logits
            for index, token in enumerate(ids):
                scores.append(float(current.log_softmax(-1)[token]))
                raw_logits.append(float(current[token]))
                if index + 1 < len(ids):
                    current, _ = adapter.step(token, branch, expected_position=adapter.cache_length(branch))
            results[label] = {"token_ids": ids, "token_logprobs": scores,
                "relevant_token_logits": raw_logits, "sequence_logprob": sum(scores),
                "terminal_state": check_state(adapter, branch)}
        finally:
            release_cache_storage(branch)
            current = None
    results["margin"] = results[labels[0]]["sequence_logprob"] - results[labels[1]]["sequence_logprob"]
    branch = clone_hybrid_cache(cache)
    try:
        generated = []
        current = logits
        eos = eos_token_ids(tokenizer, adapter.hf_model)
        for _ in range(max_new_tokens):
            token = int(current.argmax())
            generated.append(token)
            if token in eos:
                break
            current, _ = adapter.step(token, branch, expected_position=adapter.cache_length(branch))
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        results["generated"] = {"raw": raw, "token_ids": generated,
            "label": raw.strip() if raw.strip() in labels else None,
            "terminal_state": check_state(adapter, branch)}
        assert_cache_unchanged(before, cache, "mixed qualification scoring/generation")
    finally:
        release_cache_storage(branch)
    return results


@torch.no_grad()
def continuation(adapter, source, tokens, *, fp32, sham=False, replacement=None,
                 tokenizer=None, labels=None, max_new_tokens=8):
    """Source is immediately BEFORE '?'; clone, optionally promote, continue.

    FP32Tail must already be active for fp32=True. A nonzero replacement is
    reserved for forward engineering tests; real smoke uses geometry only.
    """
    before = audit_cache(source)
    cache = clone_hybrid_cache(source)
    position = adapter.cache_length(cache)
    handle = None
    pre = post = logits = None
    calls = []
    try:
        promotion = promote_tail_cache(cache) if fp32 else None
        def hook(_module, _inputs, output):
            nonlocal pre, post
            hidden = hidden_tensor(output)
            if hidden.dtype != torch.bfloat16 or hidden.shape[:2] != (1, 1):
                raise AssertionError("pre-tail recipient is not the original BF16 residual")
            pre = hidden.detach().cpu().reshape(-1).clone()
            if fp32:
                value = hidden.float()
                if sham:
                    value = value.clone()
                if replacement is not None:
                    if replacement.dtype != torch.float32 or replacement.shape != pre.shape:
                        raise AssertionError("FP32 replacement shape/dtype mismatch")
                    value = replacement.to(hidden.device).reshape_as(hidden)
                post = value.detach().cpu().reshape(-1).clone()
                calls.append(position)
                return replace_hidden(output, value)
            post = pre.clone()
            calls.append(position)
            return output
        for offset, token in enumerate(tokens):
            if offset == 0:
                handle = adapter.layers[42].register_forward_hook(hook)
            try:
                logits, _ = adapter.step(int(token), cache, expected_position=position + offset)
            finally:
                if handle is not None:
                    handle.remove()
                    handle = None
        if calls != [position] or pre is None:
            raise AssertionError("layer-42 '?' hook did not execute exactly once")
        state = check_state(adapter, cache)
        result = {"patch_positions": calls, "state": state,
            "promotion": promotion, "pre_tail_dtype": str(pre.dtype), "post_tail_dtype": str(post.dtype)}
        if labels is not None:
            result["judgments"] = judgments(adapter, cache, logits, tokenizer, labels, max_new_tokens)
        assert_cache_unchanged(before, source, "mixed continuation source")
        return pre, post, logits.detach().cpu(), result
    finally:
        if handle is not None:
            handle.remove()
        release_cache_storage(cache)


def discrepancy(bf, mixed, labels):
    a, b = bf["judgments"], mixed["judgments"]
    logp = {s: b[s]["sequence_logprob"] - a[s]["sequence_logprob"] for s in labels}
    logits = {s: [y-x for x, y in zip(a[s]["relevant_token_logits"], b[s]["relevant_token_logits"], strict=True)] for s in labels}
    def topology(state):
        return [(t["path"], t["shape"]) for t in state["topology"]]
    if topology(bf["state"]) != topology(mixed["state"]):
        raise AssertionError("boundary cache topology changed beyond dtype")
    for label in labels:
        if topology(a[label]["terminal_state"]) != topology(b[label]["terminal_state"]):
            raise AssertionError("label-scoring cache topology changed beyond dtype")
    return {"generated_label_identical_and_valid": a["generated"]["label"] is not None and a["generated"]["label"] == b["generated"]["label"],
        "generated_token_ids_identical": a["generated"]["token_ids"] == b["generated"]["token_ids"],
        "sequence_logp_differences": logp, "margin_difference": b["margin"]-a["margin"],
        "relevant_token_logit_differences": logits,
        "proposed_absolute_bounds_passed": max(abs(x) for x in logp.values()) <= .05 and abs(b["margin"]-a["margin"]) <= .05}


def patch_geometry(residuals, vector, item_id, branch, lens_model, lens, out):
    rows = []
    for name, h in residuals.items():
        q, report = candidate_patch(h, h, vector)
        row = {"item_id": item_id, "branch": branch, "recipient": name, "donor": name,
            "kind": "sham", **report, "jlens_score_before": candidate_score(lens_model, lens, h),
            "jlens_score_after": candidate_score(lens_model, lens, q),
            "jlens_execution": "original BF16-calibrated unembed outside FP32 context", "nonzero_judgment_computed": False}
        append_jsonl(out / "patch_geometry.jsonl", row)
        rows.append(row)
    for recipient, donor in (("primary", "clean"), ("alternative", "clean"), ("clean", "primary"), ("clean", "alternative")):
        h, d = residuals[recipient], residuals[donor]
        q, candidate = candidate_patch(h, d, vector)
        proposals = [("candidate", q, candidate)]
        direction, seed = random_direction(vector, seed=POLICY.random_seed, item_id=item_id, branch=branch, donor=f"{recipient}_from_{donor}")
        for sign in (-1, 1):
            rq, report = realize_patch(h, direction.double() * sign * candidate["total_patch_l2"], vector, intended_coordinate=0.0)
            report.update(random_derived_seed=seed, random_direction_sha256=tensor_sha(direction), sign=sign)
            report["realized_norm_matching_error"] = abs(report["total_patch_l2"] - candidate["total_patch_l2"])
            report["realized_projection_leakage_onto_candidate"] = report["realized_candidate_coordinate_change"]
            report["realized_projection_leakage_over_candidate_patch_norm"] = abs(report["realized_candidate_coordinate_change"])/candidate["total_patch_l2"] if candidate["total_patch_l2"] else 0.0
            report["precision_gate_passed"] = (report["noise_bound_passed"]
                and report["realized_norm_matching_error"] <= report["floating_point_noise_l2_bound"]
                and abs(report["realized_candidate_coordinate_change"]) <= report["floating_point_noise_l2_bound"] * float(vector.double().norm()))
            proposals.append((f"random_{sign:+d}", rq, report))
        fq, full = realize_patch(h, d.double()-h.double(), vector)
        full["exact_donor_restored"] = torch.equal(fq, d.float())
        full["precision_gate_passed"] = full["noise_bound_passed"] and full["exact_donor_restored"]
        proposals.append(("full_residual", fq, full))
        for kind, value, report in proposals:
            row = {"item_id": item_id, "branch": branch, "recipient": recipient, "donor": donor,
                "kind": kind, **report, "jlens_score_before": candidate_score(lens_model, lens, h),
                "jlens_score_after": candidate_score(lens_model, lens, value),
                "jlens_execution": "original BF16-calibrated unembed outside FP32 context",
                "nonzero_judgment_computed": False}
            append_jsonl(out / "patch_geometry.jsonl", row)
            rows.append(row)
    return rows


def numerical_item(adapter, tokenizer, lens_model, lens, answer, config, vector, reference, out):
    item = str(answer["item_id"])
    if item not in SMOKE_IDS or answer["invalid"]:
        raise AssertionError("only valid predeclared smoke items allowed")
    outcomes, comparisons, residuals_by_branch = {}, [], {}
    bundle = schedule = None
    try:
        for field, ids in (("question_token_hash", "question_prefix_token_ids"), ("answer_token_hash", "answer_token_ids"), ("transcript_hash", "post_answer_token_ids")):
            if hash_token_ids(answer[ids]) != answer[field]:
                raise AssertionError("frozen answer token identity changed")
        outcomes["clean"] = _replay(adapter, answer, None)
        for name, layer, family, strength in (("primary", 31, "targeted", .11), ("alternative", 23, "alternative_targeted", .20)):
            bundle = compute_clean_gradients(adapter, answer["post_answer_token_ids"], prefix_length=len(answer["question_prefix_token_ids"]),
                answer_token_ids=answer["answer_token_ids"], process_layer=layer, gradient_answer_token_limit=32, atol=1e-5, rtol=1e-5)
            schedule = _schedule(bundle, config, item_id=item, family=family, strength=strength, process_layer=layer)
            outcomes[name] = _replay(adapter, answer, schedule)
            append_jsonl(out / "gradient_audits.jsonl", {"item_id": item, "mechanism": name, "parity": bundle.parity,
                "predictor_positions": bundle.predictor_positions, "process_hook_positions": outcomes[name].process_hook_positions})
            bundle = schedule = None
            assert_process_propagated(outcomes["clean"].cache_audit, outcomes[name].cache_audit, process_layer=layer)
            observed = outcomes["clean"].answer_sequence_logp - outcomes[name].answer_sequence_logp
            expected = float(reference["support_drop_targeted" if name == "primary" else "support_drop_alternative"])
            passed = abs(observed-expected) <= 1e-5 + 1e-5*abs(expected)
            append_jsonl(out / "support_reproduction.jsonl", {"item_id": item, "mechanism": name,
                "original_drop": expected, "recomputed_drop": observed, "absolute_error": abs(observed-expected), "passed": passed})
            if not passed:
                raise AssertionError("original smoke support drop did not reproduce")
        for name, outcome in outcomes.items():
            for key in ("transcript_hash", "question_token_hash", "answer_token_hash"):
                if getattr(outcome, key) != getattr(outcomes["clean"], key):
                    raise AssertionError("visible transcript changed")
        for branch in ("confidence", "correctness"):
            meta = config["meta_branches"][branch]
            turn3 = build_turn3_suffix(tokenizer, frozen_question_rendered=answer["question_rendered"], frozen_answer_text=answer["answer"],
                post_answer_token_ids=answer["post_answer_token_ids"], meta_prompt=meta["prompt"])
            append_jsonl(out / "tokenizations.jsonl", {"item_id": item, "branch": branch, **asdict(turn3)})
            residuals_by_branch[branch] = {}
            for condition, outcome in outcomes.items():
                print(f"no-patch item={item} branch={branch} process={condition}", flush=True)
                source_audit = audit_cache(outcome.cache)
                prefix = clone_hybrid_cache(outcome.cache)
                try:
                    split = turn3.question_position - adapter.cache_length(prefix)
                    if split < 0 or split >= len(turn3.token_ids):
                        raise AssertionError("question token outside suffix")
                    for token in turn3.token_ids[:split]:
                        prefix_logits, _ = adapter.step(int(token), prefix, expected_position=adapter.cache_length(prefix))
                        del prefix_logits
                    tokens = turn3.token_ids[split:]
                    options = dict(tokenizer=tokenizer, labels=meta["labels"], max_new_tokens=int(config["generation"]["max_choice_tokens"]))
                    h, _, bf_logits, bf = continuation(adapter, prefix, tokens, fp32=False, **options)
                    jlens_score = candidate_score(lens_model, lens, h)
                    with FP32Tail(adapter) as tail:
                        hf, _, fp_logits, fp = continuation(adapter, prefix, tokens, fp32=True, **options)
                        hs, _, sham_logits, sham = continuation(adapter, prefix, tokens, fp32=True, sham=True, **options)
                        weights = tail.weight_audit
                    if not torch.equal(h, hf) or not torch.equal(h, hs):
                        raise AssertionError("BF16 pre-tail state changed")
                    restored_jlens_score = candidate_score(lens_model, lens, hf)
                    if abs(restored_jlens_score-jlens_score) > 1e-5 + 1e-5*abs(jlens_score):
                        raise AssertionError("original J-Lens scoring path was not restored")
                    torch.testing.assert_close(fp_logits, sham_logits, atol=1e-5, rtol=1e-5)
                    sham_diff = discrepancy(fp, sham, meta["labels"])
                    if not sham_diff["generated_token_ids_identical"] or fp["state"]["digest"] != sham["state"]["digest"]:
                        raise AssertionError("FP32 sham changed generated sequence or cache")
                    for label in meta["labels"]:
                        torch.testing.assert_close(torch.tensor(fp["judgments"][label]["token_logprobs"]),
                            torch.tensor(sham["judgments"][label]["token_logprobs"]), atol=1e-5, rtol=1e-5)
                        torch.testing.assert_close(torch.tensor(fp["judgments"][label]["relevant_token_logits"]),
                            torch.tensor(sham["judgments"][label]["relevant_token_logits"]), atol=1e-5, rtol=1e-5)
                    delta = discrepancy(bf, fp, meta["labels"])
                    row = {"item_id": item, "branch": branch, "process": condition, "bf16": bf, "fp32": fp,
                        "discrepancy": delta, "sham": {"passed": True, "logit_max_abs_error": float((fp_logits-sham_logits).abs().max()), "discrepancy": sham_diff},
                        "pre_tail_bitwise_identical": True, "candidate_coordinate": float(vector.double() @ h.double()),
                        "jlens_score_before_conversion": jlens_score,
                        "jlens_score_same_residual_after_context": restored_jlens_score,
                        "full_boundary_logit_max_abs_difference": float((fp_logits-bf_logits).abs().max())}
                    stem = f"item_{item}_{branch}_{condition}"
                    torch.save({"bf16_pre_tail": h, "fp32_pre_tail": hf, "sham_pre_tail": hs,
                        "bf16_boundary_logits": bf_logits, "fp32_boundary_logits": fp_logits}, out / f"{stem}.pt")
                    (out / f"{stem}_weights.json").write_text(json.dumps(weights, indent=2)+"\n", encoding="utf-8")
                    append_jsonl(out / "equivalence.jsonl", row)
                    comparisons.append(row)
                    residuals_by_branch[branch][condition] = h
                    del hf, hs, bf_logits, fp_logits, sham_logits, bf, fp, sham, tail
                    assert_cache_unchanged(source_audit, outcome.cache, "original post-answer state")
                finally:
                    release_cache_storage(prefix)
            selected = {r["process"]: r for r in comparisons if r["branch"] == branch}
            for process in ("primary", "alternative"):
                old = selected[process]["bf16"]["judgments"]["margin"] - selected["clean"]["bf16"]["judgments"]["margin"]
                new = selected[process]["fp32"]["judgments"]["margin"] - selected["clean"]["fp32"]["judgments"]["margin"]
                error = abs(new-old)
                append_jsonl(out / "process_contrasts.jsonl", {"item_id": item, "branch": branch, "process": process,
                    "bf16_process_minus_clean_margin": old, "fp32_process_minus_clean_margin": new,
                    "absolute_error": error, "relative_error": error/abs(old) if old else None,
                    "proposed_bounds_passed": error <= .05 and (error <= .1*abs(old) and new*old>0 if old else error <= 1e-5)})
        # No patched judgments are evaluated. Geometry uses the saved native
        # residuals only; BF16-calibrated J-Lens is restored before this call.
        geometry = []
        for branch, residuals in residuals_by_branch.items():
            geometry.extend(patch_geometry(residuals, vector, item, branch, lens_model, lens, out))
        return {"comparisons": comparisons, "geometry": geometry}
    finally:
        bundle = schedule = None
        for outcome in outcomes.values():
            outcome.release_cache()
        outcomes.clear()
        reclaim_cuda_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--direction", type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument("--hf-cache-dir", type=Path)
    args = parser.parse_args(argv)
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    try:
        if any(int(os.environ.get(k, "1")) > 1 for k in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
            raise RuntimeError("one CUDA process/worker required")
        vector, config, answers, identity = load_upstream(args.upstream, args.direction)
        source = ROOT / "experiments/process_sensitive_replay"
        code = sha256_json({p.name: sha(p.read_bytes().replace(b"\r\n", b"\n")) for p in sorted(source.glob("*.py"))})
        if code != identity["upstream_hashes"]["code"]:
            raise RuntimeError("upstream replay source changed")
        packages = require_runtime_packages(identity["runtime_packages"])
        reference, reference_audit = _support_reference(args.upstream, identity)
        hashes.update(policy=sha256_json(asdict(POLICY)), upstream_code=code, identity=sha256_json(identity), packages=sha256_json(packages),
            code=sha256_json({p.name: sha(p.read_bytes().replace(b"\r\n", b"\n")) for p in sorted(Path(__file__).parent.glob("*.py"))}))
        manifest = {"stage": "mixed_precision_equivalence_only", "policy": asdict(POLICY), "hashes": hashes.copy(),
            "identity": identity, "support_reference": reference_audit, "item_ids": list(SMOKE_IDS), "equivalence_proposal": EQUIVALENCE_PROPOSAL,
            "mediation_execution_authorized": False, "behavioral_outcomes_used_for_patch_policy": False,
            "claim_boundary": "Causal follow-up on the same eight-item exploratory quick-run set, not independent replication or full-profile confirmation. "
                "Any future mediation uses exact BF16-generated process state followed by a numerically validated FP32 continuation after layer 42."}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        model, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
        runtime["attention_implementation"] = adapter.text_config._attn_implementation
        runtime["cuda_matmul_fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
        runtime["cudnn_conv_fp32_precision"] = torch.backends.cudnn.conv.fp32_precision
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2)+"\n", encoding="utf-8")
        reclaim_cuda_memory()
        guard = CudaMemoryTrendGuard.from_config(config)
        guard.baseline_allocated_bytes = int(torch.cuda.memory_allocated())
        guard.previous_allocated_bytes = guard.baseline_allocated_bytes
        results = []
        for item in SMOKE_IDS:
            results.append(run_cuda_item(guard, out / "cuda_memory.jsonl", hashes, hash_name="cuda_memory", phase="mixed_smoke", stage="no_patch_equivalence", item_id=item,
                operation=lambda: numerical_item(adapter, tokenizer, lens_model, lens, answers[item], config, vector, reference[item], out)))
        rows = [r for result in results for r in result["comparisons"]]
        geometry = [r for result in results for r in result["geometry"]]
        if len(rows) != 12 or len(geometry) != 76:
            raise AssertionError("incomplete qualification matrix")
        labels_ok = all(r["discrepancy"]["generated_label_identical_and_valid"] for r in rows)
        precision_ok = all(r["precision_gate_passed"] for r in geometry)
        contrasts = [json.loads(line) for line in (out / "process_contrasts.jsonl").read_text().splitlines()]
        proposed_ok = all(r["discrepancy"]["proposed_absolute_bounds_passed"] for r in rows) and all(r["proposed_bounds_passed"] for r in contrasts)
        status = ("stopped_material_label_change" if not labels_ok else "failed_precision" if not precision_ok
                  else "stopped_proposed_equivalence_limits_exceeded" if not proposed_ok else "observed_pending_equivalence_review")
        gate = {"status": status,
            "comparisons": len(rows), "patch_geometry_proposals": len(geometry), "generated_labels_preserved": labels_ok,
            "patch_precision_passed": precision_ok, "mediation_execution_authorized": False,
            "proposed_equivalence_limits_met": proposed_ok,
            "nonzero_patch_behavior_evaluated": False, "equivalence_proposal_approved": False, "hashes": hashes}
        summary = {"maximum_absolute_sequence_logp_difference": max(abs(x) for r in rows for x in r["discrepancy"]["sequence_logp_differences"].values()),
            "maximum_absolute_margin_difference": max(abs(r["discrepancy"]["margin_difference"]) for r in rows),
            "maximum_absolute_process_contrast_difference": max(r["absolute_error"] for r in contrasts),
            "maximum_candidate_relative_coordinate_error": max(r["relative_coordinate_error"] or 0 for r in geometry if r["kind"] == "candidate"),
            "maximum_candidate_orthogonal_leakage_l2": max(r["orthogonal_leakage_l2"] for r in geometry if r["kind"] == "candidate"),
            "equivalence_proposal": EQUIVALENCE_PROPOSAL}
        (out / "numerical_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
        hashes["artifacts"] = sha256_json({p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
        (out / "gate_status.json").write_text(json.dumps(gate, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(gate))
        return 0 if labels_ok and precision_ok and proposed_ok else 2
    except Exception as exc:
        gate = {"status": "failed_engineering", "error": str(exc), "hashes": hashes, "mediation_execution_authorized": False}
        append_jsonl(out / "errors.jsonl", gate)
        (out / "gate_status.json").write_text(json.dumps(gate, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(gate))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

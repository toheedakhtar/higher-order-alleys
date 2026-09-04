"""Two-item, geometry-only CUDA qualification; no mediation experiment.

Uses only frozen items 0 and 2. No generated labels, sequence label scores, or
nonzero-patch judgment effects are computed. Model step logits are discarded;
only identical-run/sham logits are compared for engineering parity.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.process_sensitive_replay.cache_state import (
    assert_cache_unchanged, assert_hybrid_cache_integrity, assert_process_propagated,
    audit_cache, clone_hybrid_cache, release_cache_storage,
)
from experiments.process_sensitive_replay.cuda_memory import CudaMemoryTrendGuard, reclaim_cuda_memory
from experiments.process_sensitive_replay.gradient_intervention import compute_clean_gradients
from experiments.process_sensitive_replay.jlens_readout import readout_residual
from experiments.process_sensitive_replay.protocol import hash_token_ids, sha256_json
from experiments.process_sensitive_replay.replay import build_turn3_suffix
from experiments.process_sensitive_replay.runner import append_jsonl, load_runtime, run_cuda_item
from experiments.process_sensitive_replay.smoke import _replay, _schedule

from .precision import POLICY, realize_coordinate, realize_random_pair, vector_hash
from .upstream import (
    DEFAULT_DIRECTION, DEFAULT_UPSTREAM, ROOT, SMOKE_IDS,
    checked_bytes, load_upstream, require_runtime_packages, sha,
)


@torch.no_grad()
def capture_branch(adapter, source, turn3, *, replacement=None):
    """Clone pristine post-answer cache; optionally patch '?' during its forward.

    Downstream layers and the rest of the suffix run after the patch. The hook
    exists during one adapter.step only, so no later cache can bypass it.
    """
    before = audit_cache(source)
    branch = clone_hybrid_cache(source)
    registrations = adapter.intervention_hook_registrations
    positions = []
    residual = None
    logits = None
    handle = None
    start = adapter.cache_length(branch)
    try:
        for offset, token_id in enumerate(turn3.token_ids):
            position = start + offset
            if position == turn3.question_position and replacement is not None:
                def hook(_module, _inputs, output):
                    from experiments.process_sensitive_replay.gradient_intervention import hidden_tensor, replace_hidden
                    hidden = hidden_tensor(output)
                    if hidden.dtype != torch.bfloat16 or replacement.dtype != torch.bfloat16:
                        raise AssertionError("native BF16 patch path changed")
                    if hidden.shape[:2] != (1, 1) or replacement.numel() != hidden.shape[-1]:
                        raise AssertionError("unexpected residual shape at patch")
                    changed = hidden.clone()
                    changed[0, 0] = replacement.reshape(-1).to(hidden.device)
                    positions.append(position)
                    return replace_hidden(output, changed)

                handle = adapter.layers[42].register_forward_hook(hook)
            try:
                logits, captured = adapter.step(int(token_id), branch,
                    expected_position=position, capture_layers=(42,) if position == turn3.question_position else ())
            finally:
                if handle is not None:
                    handle.remove()
                    handle = None
            if captured:
                residual = captured[42].detach().cpu().reshape(-1)
        if residual is None or residual.dtype != torch.bfloat16:
            raise AssertionError("missing native layer-42 '?' residual")
        expected_calls = [] if replacement is None else [turn3.question_position]
        if positions != expected_calls:
            raise AssertionError("patch hook count or position mismatch")
        if adapter.intervention_hook_registrations != registrations:
            raise AssertionError("factual process hook active during Turn 3")
        if replacement is not None and not torch.equal(residual.view(torch.int16), replacement.cpu().reshape(-1).view(torch.int16)):
            raise AssertionError("forward did not contain the exact BF16 proposal")
        assert_cache_unchanged(before, source, "numerical Turn-3 branch")
        after = assert_hybrid_cache_integrity(branch,
            layer_types=adapter.text_config.layer_types,
            expected_sequence_length=start + len(turn3.token_ids))
        return residual, logits.detach().cpu(), {
            "patch_positions": positions, "source_cache_digest": before.digest,
            "boundary_cache_digest": after.digest,
            "cache_structure_digest": after.structure_digest,
            "process_hook_calls": 0,
            "cache_tensor_dtypes": sorted({t.dtype for t in after.tensors}),
        }
    finally:
        if handle is not None:
            handle.remove()
        release_cache_storage(branch)
        del branch, logits


def candidate_score(lens_model, lens, residual):
    result = readout_residual(lens_model, lens, residual, layer=42, top_k=1, explicit_token_ids=(75075,))
    return result["explicit"]["75075"]["score"]


def _support_reference(upstream, identity):
    data, audit = checked_bytes(upstream / "heldout_support_match.csv",
        identity["upstream_manifest_hashes"]["heldout_support_match"], text=True)
    rows = {row["item_id"]: row for row in csv.DictReader(io.StringIO(data.decode()))}
    return rows, audit


def numerical_item(adapter, tokenizer, lens_model, lens, answer, config, vector, expected, out):
    item_id = str(answer["item_id"])
    if item_id not in SMOKE_IDS:
        raise RuntimeError("precision development is restricted to the two smoke items")
    outcomes = {}
    bundle = schedule = None
    measurements = []
    try:
        if answer["invalid"]:
            raise AssertionError("frozen smoke answer is invalid")
        for field, ids in (("question_token_hash", "question_prefix_token_ids"),
                           ("answer_token_hash", "answer_token_ids"),
                           ("transcript_hash", "post_answer_token_ids")):
            if hash_token_ids(answer[ids]) != answer[field]:
                raise AssertionError(f"frozen answer {field} mismatch")
        outcomes["clean"] = _replay(adapter, answer, None)
        for name, layer, strength, family in (("primary", 31, 0.11, "targeted"),
                                             ("alternative", 23, 0.20, "alternative_targeted")):
            bundle = compute_clean_gradients(adapter, answer["post_answer_token_ids"],
                prefix_length=len(answer["question_prefix_token_ids"]), answer_token_ids=answer["answer_token_ids"],
                process_layer=layer, gradient_answer_token_limit=32, atol=1e-5, rtol=1e-5)
            schedule = _schedule(bundle, config, item_id=item_id, family=family,
                                 strength=strength, process_layer=layer)
            outcomes[name] = _replay(adapter, answer, schedule)
            append_jsonl(out / "gradient_audits.jsonl", {
                "item_id": item_id, "mechanism": name, "layer": layer,
                "strength": strength, "parity": bundle.parity,
                "predictor_positions": bundle.predictor_positions,
                "process_hook_positions": outcomes[name].process_hook_positions,
            })
            bundle = schedule = None
            assert_process_propagated(outcomes["clean"].cache_audit, outcomes[name].cache_audit, process_layer=layer)
            drop = outcomes["clean"].answer_sequence_logp - outcomes[name].answer_sequence_logp
            previous = float(expected["support_drop_targeted" if name == "primary" else "support_drop_alternative"])
            reproduced = abs(drop - previous) <= 1e-5 + 1e-5 * abs(previous)
            append_jsonl(out / "support_reproduction.jsonl", {
                "item_id": item_id, "mechanism": name, "original_drop": previous,
                "recomputed_drop": drop, "absolute_error": abs(drop-previous), "passed": reproduced,
            })
            if not reproduced or drop <= 0:
                raise AssertionError("original smoke support drop did not reproduce")
        for name, outcome in outcomes.items():
            for field in ("transcript_hash", "question_token_hash", "answer_token_hash"):
                if getattr(outcome, field) != getattr(outcomes["clean"], field):
                    raise AssertionError("visible transcript changed across conditions")
            state = assert_hybrid_cache_integrity(outcome.cache, layer_types=adapter.text_config.layer_types,
                expected_sequence_length=len(answer["post_answer_token_ids"]))
            append_jsonl(out / "state_audits.jsonl", {"item_id": item_id, "condition": name, **asdict(state)})

        for branch in ("confidence", "correctness"):
            turn3 = build_turn3_suffix(tokenizer, frozen_question_rendered=answer["question_rendered"],
                frozen_answer_text=answer["answer"], post_answer_token_ids=answer["post_answer_token_ids"],
                meta_prompt=config["meta_branches"][branch]["prompt"])
            append_jsonl(out / "tokenizations.jsonl", {"item_id": item_id, "branch": branch, **asdict(turn3)})
            residuals = {}
            for name, source in outcomes.items():
                h, baseline_logits, audit = capture_branch(adapter, source.cache, turn3)
                residuals[name] = h
                sham, sham_report = realize_coordinate(h, vector, 0.0)
                hs, sham_logits, sham_audit = capture_branch(adapter, source.cache, turn3, replacement=sham)
                torch.testing.assert_close(sham_logits, baseline_logits, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(hs, h, atol=1e-5, rtol=1e-5)
                if (audit["boundary_cache_digest"] != sham_audit["boundary_cache_digest"]
                        or candidate_score(lens_model, lens, hs) != candidate_score(lens_model, lens, h)):
                    raise AssertionError("sham cache or readout changed")
                append_jsonl(out / "sham_checks.jsonl", {"item_id": item_id, "branch": branch,
                    "condition": name, "passed": True, "audit": sham_audit, "geometry": sham_report})
                del baseline_logits, sham_logits, hs, sham

            # Both restoration and reverse transplant, without reading labels.
            for recipient, donor in (("primary", "clean"), ("alternative", "clean"),
                                     ("clean", "primary"), ("clean", "alternative")):
                h, d = residuals[recipient], residuals[donor]
                delta = float((d.double() - h.double()).dot(vector.double()))
                proposal, report = realize_coordinate(h, vector, delta)
                candidates = [("candidate", proposal, report)]
                candidates += [("random", q, r) for q, r in realize_random_pair(h, vector,
                    report["total_patch_l2"], seed=42, item_id=item_id, branch=branch, donor=donor)]
                for kind, q, geometry in candidates:
                    passed = geometry.get("control_precision_gate_passed", geometry["precision_gate_passed"])
                    row = {"item_id": item_id, "branch": branch, "recipient": recipient, "donor": donor,
                        "kind": kind, "raw_recipient_coordinate": float(h.double().dot(vector.double())),
                        "raw_donor_coordinate": float(d.double().dot(vector.double())),
                        "raw_post_patch_coordinate": float(q.double().dot(vector.double())),
                        "jlens_score_before": candidate_score(lens_model, lens, h),
                        "jlens_score_after": candidate_score(lens_model, lens, q), **geometry}
                    # An invalid geometric proposal is NEVER sent through the model.
                    # Accepted proposals are checked causally, but their boundary
                    # logits are discarded without scoring or inspecting effects.
                    if passed:
                        actual, unused_logits, causal_audit = capture_branch(adapter, outcomes[recipient].cache, turn3, replacement=q)
                        del unused_logits, actual
                        row["causal_forward_audit"] = causal_audit
                    else:
                        row["causal_forward_audit"] = "not_applied_precision_gate_failed"
                    append_jsonl(out / "patch_precision.jsonl", row)
                    measurements.append(row)
                del proposal, candidates
            # CPU-only, numerical residual archive for reproducible precision audits.
            torch.save({"item_id": item_id, "branch": branch, "residuals": residuals,
                        "candidate_vector_sha256": vector_hash(vector)}, out / f"residuals_{item_id}_{branch}.pt")
            del residuals
        return measurements
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
        source_dir = ROOT / "experiments/process_sensitive_replay"
        source_hash = sha256_json({p.name: sha(p.read_bytes().replace(b"\r\n", b"\n")) for p in sorted(source_dir.glob("*.py"))})
        if source_hash != identity["upstream_hashes"]["code"]:
            raise RuntimeError("upstream replay implementation changed")
        packages = require_runtime_packages(identity["runtime_packages"])
        reference, reference_audit = _support_reference(args.upstream, identity)
        code_hash = sha256_json({p.name: sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})
        hashes.update({"policy": POLICY.digest(), "code": code_hash, "upstream_code": source_hash,
                       "upstream_identity": sha256_json(identity), "packages": sha256_json(packages)})
        manifest = {"stage": "precision_smoke_only", "identity": identity, "policy": asdict(POLICY),
            "hashes": hashes, "support_reference": reference_audit, "item_ids": list(SMOKE_IDS),
            "behavioral_outcomes_used": False, "mediation_execution_authorized": False,
            "claim_boundary": "Same exploratory eight-item set; not independent replication or full-profile confirmation."}
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        model, tokenizer, lens_model, lens, adapter, runtime = load_runtime(args, config)
        (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
        # Establish memory baseline before item one; preserve upstream thresholds.
        reclaim_cuda_memory()
        guard = CudaMemoryTrendGuard.from_config(config)
        guard.baseline_allocated_bytes = int(torch.cuda.memory_allocated())
        guard.previous_allocated_bytes = guard.baseline_allocated_bytes
        rows = []
        for item in SMOKE_IDS:
            print(f"numerical precision smoke item={item}", flush=True)
            rows += run_cuda_item(guard, out / "cuda_memory.jsonl", hashes, hash_name="cuda_memory",
                phase="precision_smoke", stage="geometry_only", item_id=item,
                operation=lambda: numerical_item(adapter, tokenizer, lens_model, lens, answers[item],
                    config, vector, reference[item], out))
        if len(rows) != 48 or {r["item_id"] for r in rows} != set(SMOKE_IDS):
            raise AssertionError("incomplete two-item numerical precision matrix")
        # Bind every produced diagnostic, including residual binary files. A
        # passing status alone must never authorize different measurements.
        hashes["artifacts"] = sha256_json({p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
        passed = all(r.get("control_precision_gate_passed", r["precision_gate_passed"]) for r in rows)
        gate = {"status": "numerically_qualified_pending_review" if passed else "failed_precision",
                "hashes": hashes, "proposals": len(rows), "item_ids": list(SMOKE_IDS),
                "behavioral_outcomes_used": False, "mediation_execution_authorized": False}
        (out / "gate_status.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        # Deliberately no mediation success marker or eight-item launcher.
        print(json.dumps(gate))
        return 0 if passed else 2
    except Exception as exc:
        gate = {"status": "failed_engineering", "error": str(exc), "hashes": hashes,
                "behavioral_outcomes_used": False, "mediation_execution_authorized": False}
        append_jsonl(out / "errors.jsonl", gate)
        (out / "gate_status.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

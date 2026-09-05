# Mixed-precision mediation qualification

Status: **implemented for two-item qualification; full-model CUDA validation and
equivalence approval are pending. No eight-item experiment is authorized or
exposed by this runner.**

This extends [the experiment implementation](CAUSAL_MEDIATION_IMPLEMENTATION.md).
The BF16-native approach is closed under its frozen criteria: the supplied
Blackwell results reproduced all four upstream support drops exactly, but all
candidate patches failed orthogonal leakage despite passing coordinate accuracy.
The reported BF16 lattice bounds certify infeasibility under those criteria.
That is an engineering limitation, not scientific evidence about mediation.

## Exact execution policy

The implementation is in
[`mixed_precision.py`](../experiments/causal_mediation/mixed_precision.py), with
the qualification runner in
[`mixed_smoke.py`](../experiments/causal_mediation/mixed_smoke.py).

1. Load the frozen model, tokenizer, J-Lens, candidate vector, configuration and
   answer bank through the existing identity checks. Require the original runtime
   package versions and replay source hash. Reproduce the primary layer-31/0.11
   and alternative layer-23/0.20 support drops on items **0 and 2**, using the
   unchanged `atol=rtol=1e-5` rule. Retain the original cached replay and complete
   factual answer, including the 32-token gradient objective.
2. Build Turn 3 with the existing suffix constructor. Run everything before the
   declared `?` in BF16. Clone this immediately pre-question hybrid state for
   each continuation. The original post-answer cache is never promoted or edited.
3. Finish zero-based block **42** at `?` in BF16. Record its output before any
   conversion. Only then promote this residual to FP32. Blocks **43–63**, final
   RMSNorm and the language-model head run in FP32. For subsequent suffix and
   judgment tokens, blocks 0–42 remain BF16 and their outputs are promoted before
   block 43. Thus this is a per-token split, not a whole-model FP32 continuation.
4. Tail parameter values come exclusively from the already-loaded BF16 tensors.
   The same module objects and forward methods are used. BF16 representations
   are temporarily saved on CPU, their lossless FP32 representations are used
   for the tail, and originals are restored on exit, including exceptional exit.
   Tied/aliased storage reaching upstream modules is rejected.
5. Promote floating cache tensors belonging to blocks 43–63 on the isolated
   branch only. Preserve shapes, tensor paths, integer state, initialization
   flags, sequence positions and recurrence semantics. Existing FP32 recurrent
   state stays FP32. Verify lossless values and unchanged blocks 0–42.
6. Positional embeddings and masks are constructed by the original BF16 forward
   path. Their existing values are promoted at tail block inputs; they are not
   recomputed under a different positional or masking implementation.
7. Preserve the configured attention implementation and its existing dispatch.
   Do not force another SDPA backend, replace recurrent kernels, compile the
   model, or add fallback equations. Unsupported FP32 execution is an engineering
   failure and stops the run. This policy does not promise identical low-level
   kernels for different input dtypes.
8. Set CUDA matmul and cuDNN convolution `fp32_precision="ieee"` only from block
   43 through the output head, restoring original settings before the next
   token's upstream layers. This prevents TF32 from silently defining the FP32
   tail while preserving settings for existing FP32 arithmetic inside earlier
   BF16 blocks. See [PyTorch's precision documentation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

J-Lens scores are always evaluated **outside** the FP32-tail context, after the
original BF16 final norm/head have been restored. This retains the calibrated
readout path. A J-Lens score is not treated as the raw candidate coordinate.

## Patch construction and numerical criteria

For the exact frozen vector `v`, compute natural donor-minus-recipient coordinate
`delta = vᵀ(h_donor - h_recipient)` in FP64 for numerical measurement. Do not
renormalize or rediscover `v`. Form the mathematical displacement `d = delta*v`,
round that displacement to FP32, then perform the residual addition in FP32:

```python
h_prime = h_bf16.float() + displacement.float()
```

The frozen vector's tiny deviation from mathematical unit length is reported
through coordinate error rather than hidden by changing the vector. For zero
displacement, return the losslessly promoted original residual directly.

Measure realized displacement in FP64. Candidate coordinate error must be at
most **1%** of the intended coordinate change. Orthogonal leakage and total
rounding error must both be no larger than the forward rounding-error bound:

```text
B = 4 * eps_float32 * ||abs(h) + abs(d)||_2
```

This conservative arithmetic bound covers displacement conversion and FP32
addition. It is not the old BF16 10% leakage allowance. Report actual leakage,
its ratio to the intended intervention, and the bound separately; a bound does
not imply that every representable residual scale offers negligible relative
contamination. Relative leakage at naturally tiny deltas remains visible for
review.

Both signed orthogonal random controls use the same FP32 realization, targeting
the **realized candidate-patch norm**. Report realized norm mismatch and candidate
projection leakage. The deterministic Gaussian control seed is 1729, derived
with SHA-256 from item, branch, and directed donor/recipient identifiers. Reverse
candidate transplants use the same construction. Full-residual proposals use
the same FP32 addition and must exactly restore the donor's promoted BF16 values.

Reports include intended and realized candidate-coordinate changes, absolute and
relative errors, patch norm, orthogonal leakage and ratios, rounding bound,
J-Lens before/after, and residual dtype. For random/full-residual patches,
orthogonal *rounding leakage* is measured relative to their own intended
displacement; the realized candidate projection is reported separately.

No confidence/correctness outcomes enter the patch algorithm, numerical bounds,
control directions, or synthetic validation. Real-smoke nonzero proposals are
**geometry-only**; no patched judgments or mediation effect sizes are computed.
Synthetic forward tests verify that a supplied FP32 residual is written once
and recomputed through the same continuation path used by sham and baseline.

## Two different validation comparisons

The smoke matrix contains **12 comparisons**: items 0 and 2, clean/primary/
alternative process states, and confidence/correctness branches.

Each compares original BF16 continuation against FP32-tail delta-zero
continuation, recording:

- exact generated text, token IDs, valid label and generation terminal state;
- every complete-label sequence log probability and its token contributions;
- the confidence/correctness margin and each relevant token's logit at its
  actual label-prefix position;
- full boundary logits in tensor artifacts;
- bitwise-identical BF16 layer-42 recipient tensors, candidate coordinates and
  original-path J-Lens scores;
- boundary and label-scoring cache topology, dtypes, finiteness and digests;
- lossless cache promotion audits and source weight hashes.

An independent FP32-tail **SHAM versus FP32-tail baseline** check requires
identical generated token IDs and cache digests, plus logits and complete-label
token log probabilities within the unchanged `atol=rtol=1e-5` tolerance. This is
an engineering check, not a substitute for BF16/FP32 equivalence.

## Proposed equivalence criterion: not approved

The following proposal was written before viewing any real FP32 judgment or
mediation outcome. It is emitted with the observed discrepancies for review:

- All 12 generated labels must be valid and unchanged.
- Each complete-label log probability and each judgment margin may differ by
  at most **0.05 nats**.
- For each branch and process mechanism, the process-minus-clean margin contrast
  may differ by at most **0.05 nats and 10%** of its original BF16 magnitude,
  with its sign preserved. An exactly zero reference contrast instead requires
  absolute discrepancy at most `1e-5`.
- Pre-tail residual identity, cache integrity, patch precision and sham checks
  must all pass.

The absolute limit bounds the corresponding probability/odds ratio change by
`exp(0.05) ≈ 1.0513`. The relative contrast limit protects small natural process
effects that an absolute-only criterion could obscure. These are operational
proposals, not a statistical equivalence theorem. No number is selected by
looking at mediation effects. Raw-logit differences are diagnostic, because
common logit shifts need not change label probabilities.

Exceeding proposed limits stops progress; meeting them produces only
`observed_pending_equivalence_review`. **Neither outcome authorizes mediation.**
Do not widen limits automatically after seeing results. Inspect and approve the
criterion explicitly before an eight-item experiment could be implemented/run.

## Local validation and the remaining CUDA run

Local validation uses CPU-only synthetic weights and the actual Qwen hybrid
forward implementation. It cannot establish full-model BF16/FP32 equivalence.
The existing 11 BF16 tests and six new mixed-policy tests pass. Coverage includes
complete multi-token scoring, exact pre-tail state identity, active downstream
patching, sham equality, source-cache isolation, restoration of original weight
bytes and math settings, exceptional cleanup, and the complete geometry matrix.
The CPU-only smoke invocation fails closed before model loading, as intended.

The [synthetic geometry artifact](../experiments/causal_mediation/diagnostics/mixed_synthetic_v1.json)
uses the exact frozen candidate with 18 synthetic cases: BF16 residual scales
0.01, 1 and 100; intended deltas 0, 0.001, 0.003, 0.01, 0.03 and -0.01.
All 18 pass. Maximum relative coordinate error is **0.3762%**; maximum absolute
orthogonal leakage is **0.00018232**, within its arithmetic noise bound. These
are synthetic results, not measurements on the two real items.

On the original Blackwell environment, use a fresh directory:

```bash
python -m experiments.causal_mediation.mixed_smoke \
  --run-dir assets/psr-mediation-mixed-blackwell-v1 \
  --upstream assets/psr-quick-v3 \
  --direction assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt

cat assets/psr-mediation-mixed-blackwell-v1/gate_status.json
cat assets/psr-mediation-mixed-blackwell-v1/numerical_summary.json
cat assets/psr-mediation-mixed-blackwell-v1/support_reproduction.jsonl
cat assets/psr-mediation-mixed-blackwell-v1/process_contrasts.jsonl
```

Detailed records are in `equivalence.jsonl` and `patch_geometry.jsonl`. A failed
run is not resumable or promotable; diagnostics remain in its fresh directory.
No eight-item CLI or success marker is created.
The geometry matrix has 76 records: 16 candidate transplants (including reverse),
32 signed random controls, 16 full-residual proposals and 12 shams.

Use one CUDA worker. Temporary tail representations need additional GPU memory
and CPU storage for BF16 originals; no model is duplicated in full. Cache clones,
scoring branches, gradient bundles and temporary representations are released.
The existing per-item CUDA allocated/reserved/peak logging and growth guard are
retained. OOM, unsupported kernels, failed support reproduction or cache/hook
incompatibility stop the run without changing policy.

## Scientific scope

Any later positive result must describe a causal follow-up on the **same eight
exploratory quick-run held-out items**, using **the exact BF16-generated process
state, followed by a numerically validated FP32 continuation from immediately
after layer 42 onward**. It is neither an independent replication nor full-profile
confirmation. It may support an M(P)M(P)-like causal process-monitoring
interpretation, but cannot establish a dataset-wide or philosophically definitive
higher-order claim. No such scientific result exists from this qualification.

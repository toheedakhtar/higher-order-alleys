# Actual API mapping

This note records the implementation mapping inspected before code was added.
It is not a modification of the frozen scientific protocol.

## Qwen3.6-27B

- Model and tokenizer revision:
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Runtime architecture: `Qwen3_5ForConditionalGeneration`.
- Text decoder used by J-Lens: `model.language_model`.
- Decoder shape: 64 blocks, residual width 5120.
- Layer 31 is zero-based and is a `full_attention` block.
- Predeclared alternative layers 15, 19, and 23 are also `full_attention`
  blocks. They are 16, 12, and 8 layers before layer 31 and below readout layer
  40. The pinned architecture metadata is checked at runtime.
- The checkpoint alternates three `linear_attention` blocks with one
  `full_attention` block.
- Transformers 5.15.1 constructs `DynamicCache(config=model.config)` with
  `LinearAttentionLayer` instances for linear blocks and `DynamicLayer`
  instances for full-attention blocks.
- Full-attention layers store keys and values. Linear-attention layers store
  convolution states, recurrent states, initialization flags,
  `has_previous_state`, kernel sizes, and record-state metadata.
- Cache updates are in-place. Every condition and every Turn-3 scoring or
  generation branch therefore receives a deep, storage-disjoint clone whose
  source digest is checked after the branch runs.

Each intervention hook is attached to the output of its declared block. That
block's own state has already been computed, so cache layers through the
intervention layer must retain clean digests while at least one later layer
must change. This is enforced separately for layer 31 and the frozen earlier
alternative layer.

## J-Lens

- Installed package: `jlens==0.1.0`, repository commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Checkpoint repository revision: `0731326edff4ae730ffc5356fe1a4728c748b3a6`;
  file SHA-256: `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.
- The fitted checkpoint used in prior runs contains source layers 0–62, so
  required readout layers 36–44 are available.
- `HFLensModel.forward()` forces `use_cache=False`; it is not used for cached
  Turn-3 continuation.
- Cached replay calls the underlying Qwen text module directly. Residuals
  captured at the Turn-3 `?` are passed through the public
  `JacobianLens.transport()` method and the adapter's `unembed()` operation.
- `jlens.from_hf(..., force_bos=False)` is mandatory because tokenizer IDs are
  frozen by the protocol and the adapter otherwise may mutate BOS behavior.

## Phase implementation plan

1. Hash and validate the dataset, configuration, code, and frozen protocol.
2. Generate one clean answer bank and freeze exact question/answer/transcript
   token IDs and deterministic discovery/held-out membership.
3. Run pre-discovery engineering smoke over discovery items only.
4. Permit discovery only after that smoke gate passes.
5. Re-run all critical checks after discovery freezes alpha, the alternative
   layer/global beta pair, and candidates.
6. Permit held-out execution only after the post-freeze smoke gate passes.
7. Execute held-out replay with frozen strengths/candidates, enforce aggregate
   and item-level support matching fail-closed, and retain item-level matching
   diagnostics even when the campaign is invalid.
8. Run model-free H1-H7 statistics and required plotting only after a valid
   held-out gate, with the process-sensitive/M(P)-like interpretation ceiling.

`psr-v7` selected the improved alpha grid but failed the paired support-match
gate for its same-layer entropy/orthogonal alternative. `psr-v8` retains every
threshold and fail-closed rule while replacing that alternative with the same
answer-support-reducing objective at one discovery-selected layer from the
predeclared full-attention set `{15, 19, 23}`. Discovery selects a single
layer/global-beta pair without J-space, confidence, correctness, or held-out
inputs. Both the primary and alternative mechanisms have distinct exact
same-layer norm-matched random controls.

The first CUDA execution of that design, `psr-v8`, exhausted a 94.97-GiB GPU
after completing 12/16 beta-grid items. The allocated-memory trajectory showed
live state accumulation rather than an intrinsically oversized single trial.
The engineering-only `psr-v9` correction preserves the exact alpha/beta grids,
selection rules, conditions, prompts, splits, and gates. It removes the
unneeded primary-layer gradient computation from beta-only measurement,
releases disposable replay/cache objects deterministically, and replaces the
temporary recurrent-cache bound-method override with a weak-reference function
so cache layers cannot retain their own autograd graph through a reference
cycle.

Every smoke, discovery, candidate, and held-out item now writes allocated,
reserved, peak, baseline, step, and total CUDA bytes to phase-local
`cuda_memory.jsonl`. After garbage
collection and `empty_cache`, the run fails closed if post-cleanup allocation
exceeds its baseline by more than 1024 MiB while rising by at least 128 MiB for
two consecutive items. These engineering thresholds and each completed
memory-log artifact participate in configuration/phase hashes.

Discovery and freeze are now executable. Discovery performs the complete
16-item alpha grid and freezes weak/strong alpha before beginning the separate
16-item beta calibration. It then reruns those same discovery IDs with the
selected global strengths for full-vocabulary candidate scoring. It records an
empty held-out-access list, saves the score and metric tensors, reuses the
existing meaningful-token filter and effective J-Lens direction construction,
and freezes up to three cosine-deduplicated directions. Freeze is model-free:
it validates the split, all discovery input hashes, direction-file hashes,
strength-grid membership, prompts, conditions, and selection rules before
atomically gating `frozen_protocol.json`.

Alpha item rows and aggregate eligibility diagnostics are persisted and hashed
before alpha selection; beta rows and complete support-match diagnostics are
persisted and hashed before beta selection. Selection failure therefore leaves
auditable measurements but no success marker, frozen protocol, or path to the
next phase.

## Answer-bank compatibility correction

- Greedy answer generation has a frozen 256-token hard cap and still uses
  `enable_thinking=False`.
- Before generation, one example from each of the three item families must
  render with Qwen's closed empty thinking block. Token hashes and suffix IDs
  are saved in `answer_bank/thinking_mode_verification.json`.
- Generated answer-content IDs are never rewritten. A valid generated terminal
  ID is logged, while only the invisible replay delimiter is canonicalized to
  the chat template's one-token `<|im_end|>` delimiter.
- An item that reaches the cap without a valid terminator is diagnostic-only
  and is excluded from both discovery and held-out splits.
- The stored factual rendering must tokenize exactly to `post_answer_token_ids`.
  The completed Qwen template must equal that rendering plus exactly one
  newline separator, which remains part of the later Turn-3 suffix.

## Integrity hardening

- Model/tokenizer commits, J-Lens revision/checksum, and runtime package
  versions participate in every gate's base identity hash.
- Hybrid-cache validation requires true convolution and recurrent
  initialization flags for every recurrent state slot.
- Clean-reset and targeted-reset branches receive symmetric Turn-3 cache,
  logit, margin, residual, and J-Lens parity checks.
- A post-freeze support-match failure writes phase-local `trials.jsonl`, the
  aggregate smoke report, and per-item support diagnostics before returning a
  nonzero status and withholding the success marker.
- Held-out reporting includes both structured-minus-own-layer-random contrasts and is
  descriptive only; no candidate is automatically classified without an
  approved frozen convergence decision rule.

## Recurrent gradient compatibility correction

- The former full-sequence, no-cache gradient route is not used: Qwen3.6 runs
  its chunked delta-rule kernel there, whereas experimental replay runs the
  one-token recurrent kernel.
- Gradients are computed by the same token-by-token recurrent path as ordinary
  replay. The gradient-only cache clones convolution storage before its
  in-place update and replaces recurrent-state `copy_` with graph-preserving
  assignment. No model equation or kernel is replaced.
- An ordinary `DynamicCache` replay runs alongside the differentiable pass.
  Every intended position must pass full-logit and layer-31 residual parity;
  total answer support must also pass at the frozen `1e-5` tolerances.
- Answer-support gradients at the primary and alternative layers must be
  finite and nonzero. Gradient and intervention hooks must fire exactly at
  answer-predicting positions.
  Any mismatch is fail-closed and logged with its numerical differences.

## Suffix-only Turn-3 compatibility correction

- Qwen's chat template can alter an earlier assistant turn when a later user
  turn is included, so the frozen factual history is never rerendered.
- Turn 3 is rendered as a standalone user/generation suffix and appended to
  the immutable `answer_bank.post_answer_token_ids` and its preserved hybrid
  cache/state.
- The constructor first re-encodes the stored answer-bank rendering without
  calling the chat template, then requires exact factual-prefix parity.
- Exact token/hash assertions cover the factual prefix, suffix, joint
  `<|im_end|>` boundary, and complete concatenated transcript. The Turn-3 `?`
  position and all four hashes must also agree across conditions.
- A changed prefix, invalid boundary, or hash/token mismatch is fail-closed and
  classified as invalid cache/state rather than interpreted.

The present CPU workspace can unit-test this infrastructure but has
`torch==2.13.0+cpu`; real model phases are intentionally blocked here.

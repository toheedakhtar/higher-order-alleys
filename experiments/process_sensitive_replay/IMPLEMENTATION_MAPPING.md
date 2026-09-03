# Actual API mapping

This note records the implementation mapping inspected before code was added.
It is not a modification of the frozen scientific protocol.

## Qwen3.6-27B

- Runtime architecture: `Qwen3_5ForConditionalGeneration`.
- Text decoder used by J-Lens: `model.language_model`.
- Decoder shape: 64 blocks, residual width 5120.
- Layer 31 is zero-based and is a `full_attention` block.
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

The intervention hook is attached to the output of block 31. Its own K/V state
has already been computed at that point, so layers 0–31 must retain clean cache
digests while at least one layer 32–63 must change. That is enforced directly.

## J-Lens

- Installed package: `jlens==0.1.0`, repository commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
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
5. Re-run all critical checks after discovery freezes alpha, beta, and
   candidates.
6. Permit held-out execution only after the post-freeze smoke gate passes.

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
- Answer-support and entropy gradients must be finite and nonzero. Gradient
  and intervention hooks must fire exactly at answer-predicting positions.
  Any mismatch is fail-closed and logged with its numerical differences.

The present CPU workspace can unit-test this infrastructure but has
`torch==2.13.0+cpu`; real model phases are intentionally blocked here.

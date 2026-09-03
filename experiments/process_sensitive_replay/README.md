# Process-sensitive replay

This package implements the smoke-test infrastructure for the frozen protocol in
[`docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md`](../../docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md).

The scientific comparison keeps the factual question, teacher-forced answer,
Turn-3 prompts, and tokenizer IDs identical while varying only the hidden
first-order answer process and whether its hybrid cache is preserved.

## Implemented phase boundary

The current runner exposes:

1. `validate`
2. `answer_bank`
3. `pre_discovery_smoke`
4. `discovery`
5. `freeze`
6. `post_freeze_smoke`

Held-out and confirmatory analysis remain deliberately non-callable. Discovery
and freeze now enforce their protocol dependencies and produce the immutable,
hash-gated `frozen_protocol.json` required by post-freeze smoke.

## CPU verification

```powershell
python -m unittest discover -s experiments -p "test*.py" -v
```

Static validation can run on CPU. Model-bearing phases explicitly refuse to
start when CUDA is unavailable, before loading or downloading Qwen or J-Lens.

## CUDA-host sequence

The original `assets/psr`, `assets/psr-v2`, and `assets/psr-v3` campaigns are
retained as failed diagnostics, and `assets/psr-v4` is retained as the passing
pre-discovery smoke diagnostic. Adding executable discovery/freeze changes the
code and protocol hashes, so use a fresh `assets/psr-v5` campaign on the same
CUDA host:

```bash
python -m experiments.process_sensitive_replay.runner --phase validate --run-dir assets/psr-v5
python -m experiments.process_sensitive_replay.runner --phase answer_bank --run-dir assets/psr-v5
python -m experiments.process_sensitive_replay.runner --phase pre_discovery_smoke --run-dir assets/psr-v5
```

The pre-discovery smoke validates replay/token parity, hybrid state cloning,
state isolation, intervention indexing and sign, reset mechanics, J-Lens
alignment, Turn-3 hook lifetime, and the complete alpha/beta grid machinery.
It does not select or require a frozen beta.

Turn 3 is constructed by rendering only its new suffix; the already-computed
factual history is never rerendered. Prefix, suffix, boundary, and final
transcript token hashes are exact critical gates.

After pre-discovery smoke passes, run discovery and freeze in order:

```bash
python -m experiments.process_sensitive_replay.runner --phase discovery --run-dir assets/psr-v5
python -m experiments.process_sensitive_replay.runner --phase freeze --run-dir assets/psr-v5
python -m experiments.process_sensitive_replay.runner --phase post_freeze_smoke --run-dir assets/psr-v5
```

Discovery uses only the 16 IDs in `split_manifest.json`. It completes the alpha
grid and selects global weak/strong alpha before beginning the separate beta
support-matching pass. It then saves full-vocabulary discovery scores and
deterministic candidate metrics and selects up to three cosine-deduplicated
token/layer directions. Freeze is model-free and validates every source and
direction-file hash before writing `frozen_protocol.json`.

The post-freeze smoke reruns every critical check and additionally enforces
support matching with the frozen beta. Any failed gate returns a nonzero exit
status, removes/withholds the phase-success marker, and prevents the next
phase from starting.

There is intentionally no override for failed, skipped, missing, or stale
gates.

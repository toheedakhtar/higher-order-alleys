# Process-sensitive replay

This package implements the smoke-test infrastructure for the frozen protocol in
[`docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md`](../../docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md).

The scientific comparison keeps the factual question, teacher-forced answer,
Turn-3 prompts, and tokenizer IDs identical while varying only the hidden
first-order answer process and whether its hybrid cache is preserved.

## Implemented phase boundary

The current runner exposes only:

1. `validate`
2. `answer_bank`
3. `pre_discovery_smoke`
4. `post_freeze_smoke`

Discovery, freeze, held-out, and confirmatory analysis are deliberately not
callable in this smoke-infrastructure milestone. Their protocol dependency
graph is implemented and unit-tested, so later runners must satisfy the
pre-discovery and post-freeze gate records rather than bypass them.

## CPU verification

```powershell
python -m unittest discover -s experiments -p "test*.py" -v
```

Static validation can run on CPU. Model-bearing phases explicitly refuse to
start when CUDA is unavailable, before loading or downloading Qwen or J-Lens.

## CUDA-host sequence

The original `assets/psr`, `assets/psr-v2`, and `assets/psr-v3` campaigns are
retained as failed diagnostics. After syncing the suffix-only Turn-3
correction, use the fresh `assets/psr-v4` campaign directory and the same CUDA
host used for the prior runs:

```bash
python -m experiments.process_sensitive_replay.runner --phase validate --run-dir assets/psr-v4
python -m experiments.process_sensitive_replay.runner --phase answer_bank --run-dir assets/psr-v4
python -m experiments.process_sensitive_replay.runner --phase pre_discovery_smoke --run-dir assets/psr-v4
```

The pre-discovery smoke validates replay/token parity, hybrid state cloning,
state isolation, intervention indexing and sign, reset mechanics, J-Lens
alignment, Turn-3 hook lifetime, and the complete alpha/beta grid machinery.
It does not select or require a frozen beta.

Turn 3 is constructed by rendering only its new suffix; the already-computed
factual history is never rerendered. Prefix, suffix, boundary, and final
transcript token hashes are exact critical gates.

After the separately implemented discovery/freeze phase writes a hash-gated
`frozen_protocol.json` and passing `freeze` gate, run:

```bash
python -m experiments.process_sensitive_replay.runner --phase post_freeze_smoke --run-dir assets/psr-v4
```

The post-freeze smoke reruns every critical check and additionally enforces
support matching with the frozen beta. Any failed gate returns a nonzero exit
status, removes/withholds the phase-success marker, and prevents the next
phase from starting.

There is intentionally no override for failed, skipped, missing, or stale
gates.

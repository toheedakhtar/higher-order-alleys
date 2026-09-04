# Process-sensitive replay

This package implements the fail-closed runner for the frozen protocol in
[`docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md`](../../docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md).

The scientific comparison keeps the factual question, teacher-forced answer,
Turn-3 prompts, and tokenizer IDs identical while varying only the hidden
first-order answer process and whether its hybrid cache is preserved.

## Implemented phases

The current runner exposes:

1. `validate`
2. `answer_bank`
3. `pre_discovery_smoke`
4. `discovery`
5. `freeze`
6. `post_freeze_smoke`
7. `heldout`
8. `analyze`

Every phase enforces its direct predecessor and the complete current input-hash
set. Discovery and freeze produce the immutable `frozen_protocol.json`;
held-out remains inaccessible until post-freeze smoke passes fail-closed.
The base gate identity also binds the pinned model/tokenizer commits, J-Lens
file checksum, and runtime package versions, so a changed environment cannot
reuse earlier success markers.

## CPU verification

```powershell
python -m unittest discover -s experiments -p "test*.py" -v
```

Static validation can run on CPU. Model-bearing phases explicitly refuse to
start when CUDA is unavailable, before loading or downloading Qwen or J-Lens.
Because runtime package versions are gate-bound, a CPU validation campaign is
diagnostic-only: the real campaign's `validate` phase and every later phase
must all run in the same unchanged CUDA environment.

## CUDA-host sequence

The original `assets/psr` through `assets/psr-v6` campaigns are retained as
diagnostics. In particular, `psr-v6` failed because the declared alpha grid
jumped from the weak point at `0.10` to an overshooting point at `0.20`. The
declared `psr-v7` grid adds `0.11`, `0.125`, and `0.15` without changing any
selection threshold or fallback rule. Start a fresh `assets/psr-v7` campaign
on the same CUDA host:

```bash
python -m experiments.process_sensitive_replay.runner --phase validate --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase answer_bank --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase pre_discovery_smoke --run-dir assets/psr-v7
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
python -m experiments.process_sensitive_replay.runner --phase discovery --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase freeze --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase post_freeze_smoke --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase heldout --run-dir assets/psr-v7
python -m experiments.process_sensitive_replay.runner --phase analyze --run-dir assets/psr-v7
```

Discovery uses only the 16 IDs in `split_manifest.json`. It completes the alpha
grid and selects global weak/strong alpha before beginning the separate beta
support-matching pass. It then saves full-vocabulary discovery scores and
deterministic candidate metrics and selects up to three cosine-deduplicated
token/layer directions. Freeze is model-free and validates every source and
direction-file hash before writing `frozen_protocol.json`.

Alpha and beta item rows plus aggregate eligibility diagnostics are written and
hashed before their respective selectors run. A failed strength gate therefore
retains `alpha_grid.jsonl` / `alpha_grid_diagnostics.json` or
`beta_grid.jsonl` / `beta_grid_diagnostics.json` for inspection without
creating a success marker or permitting freeze.

The post-freeze smoke reruns every critical check and additionally enforces
support matching with the frozen beta. Any failed gate returns a nonzero exit
status, removes/withholds the phase-success marker, and prevents the next
phase from starting.
Support-match failures retain a phase-local full trial log and diagnostic
report before halting.

Held-out uses the frozen strengths and candidates without adaptation. It runs
the token/cache/state/hook checks for every held-out item, writes item-level
support-match diagnostics, and withholds `heldout_effects.csv` and the success
marker unless the aggregate criteria and 65% coverage gate pass. `analyze` is
model-free and can run only after that success marker; it emits H1-H7
statistics, all required plots, hashes, and the conservative interpretation
ceiling. Smoke summaries are phase-local, so post-freeze execution cannot
overwrite pre-discovery summaries.

`RESULTS.md` reports H1-H7, both structured-minus-random contrasts, and the
reset/convergence diagnostics descriptively. It does not automatically call a
candidate process-sensitive or M(P)-like because no numerical held-out
convergence decision threshold has been frozen.

There is intentionally no override for failed, skipped, missing, or stale
gates.

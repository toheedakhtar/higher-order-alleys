# Final causal mediation experiment: implementation and handoff

Last updated: 2026-09-05

## Current status

The new package is [`experiments/causal_mediation/`](../experiments/causal_mediation/).
**Only the BF16 patch-precision qualification stage is implemented.** Its
synthetic tests and small-model integration tests have passed. The real
two-item CUDA smoke has not run, and the complete eight-item mediation runner,
behavioral analysis, and final scientific plots remain unimplemented.

This pause follows an explicit protocol requirement: validate and review the
patch-precision policy before proceeding to the eight-item experiment. The
current workspace has `torch=2.13.0+cpu`, no available CUDA device, and cannot
perform the real smoke. No model precision change or mixed-precision fallback
has been made.

## Scientific purpose

The successful exploratory process-sensitive replay campaign supplied evidence
for a later candidate changing when the hidden answer process changes while the
visible question and complete answer remain fixed:

```text
P -> M*(P)
```

The final experiment targets the missing causal link:

```text
P -> M*(P) -> confidence / correctness judgment
```

Its intended intervention restores or transplants the naturally occurring
donor-recipient residual coordinate along the exact frozen candidate direction.
It does not rediscover a candidate or introduce an arbitrary steering strength.

This is a causal follow-up on the **same eight exploratory quick-run held-out
items**, not an independent replication or full-profile confirmation. A positive
result could support an M(P)-like causal process-monitoring interpretation,
subject to its controls. It cannot establish a dataset-wide or philosophically
definitive higher-order claim.

Research context:

- [Sprint summary](E2SUM_first_order_vs_higher_order_research_sprint.md)
- [Quick-v3 results](PROCESS_SENSITIVE_REPLAY_QUICK_V3_RESULTS.md)
- [Upstream replay implementation plan](PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md)

## Frozen upstream identity

| Component | Required identity |
|---|---|
| Upstream campaign | `assets/psr-quick-v3` |
| Model | `Qwen/Qwen3.6-27B` |
| Model/tokenizer revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| J-Lens revision | `0731326edff4ae730ffc5356fe1a4728c748b3a6` |
| J-Lens file SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Candidate | Token `75075`, layer `42`, orientation `-1` |
| Residual width | `5120` |
| Candidate tensor SHA-256 | `8515912d78e8afe1a932bf06dc43cb41a54e446e91a8abbbe008a28fac54ac84` |
| Candidate file SHA-256 | `7661c6529508c2f571ffd68a15e0837669303ba304aacd1c377c9caa31f695aa` |
| PyTorch / Transformers | `2.13.0` / `5.15.1` |
| J-Lens / Hugging Face Hub | `0.1.0` / `1.27.0` |
| Primary process intervention | Layer `31`, strong alpha `0.11` |
| Alternative process intervention | Layer `23`, beta `0.20` |
| Gradient/intervention window | First `min(answer_length, 32)` answer tokens |
| Visible answer | Entire original frozen answer, teacher-forced |
| Held-out item order | `0, 2, 3, 4, 57, 67, 68, 82` |
| New numerical smoke items | `0, 2` |

The exact candidate artifact is available at
[`assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt`](../assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt).
Both its file and tensor hashes were verified. Its decoded vocabulary label,
` UIImagePickerController`, is an identifier, not an established semantic
interpretation of the direction.

Orientation is for analysis only. Coordinate patches use the raw saved vector.
The new smoke items `0, 2` are selected by frozen held-out order; they are
different from the upstream discovery smoke items `1, 14`.

## Why precision qualification came first

For recipient residual `h`, unit candidate vector `v`, and donor coordinate
`c_donor`, the requested ideal patch is:

```text
delta = c_donor - dot(h, v)
h_patched = h + delta * v
```

The original model runs in BF16. Simply casting this ideal patch to BF16 can
erase much of a small coordinate change and introduce an orthogonal component.
Keeping the patch in float32 is also not a drop-in solution: the installed Qwen
normalization preserves the input dtype, so the following BF16 linear layer
encounters a dtype mismatch.

The user approved development of a **BF16-native compensated restoration**
policy. Model weights, downstream compute dtype, attention backend, cache dtype,
and existing global parity tolerances must remain unchanged. Any mixed-precision
tail would require a separately approved policy and equivalence validation.

## Implementation map

| File | Implemented responsibility |
|---|---|
| [`upstream.py`](../experiments/causal_mediation/upstream.py) | Verify and load the exact frozen candidate, protocol, config, answer bank, split, revisions, and runtime package identity |
| [`precision.py`](../experiments/causal_mediation/precision.py) | Deterministic compensated BF16 proposals, geometry metrics, infeasibility certificates, and matched random proposals |
| [`precision_probe.py`](../experiments/causal_mediation/precision_probe.py) | Outcome-blind synthetic numerical matrix and machine-readable diagnostic report |
| [`precision_smoke.py`](../experiments/causal_mediation/precision_smoke.py) | Two-item CUDA support reproduction, residual capture, sham parity, numerical patch checks, causal forward checks, and memory safeguards |
| [`test_precision.py`](../experiments/causal_mediation/test_precision.py) | Frozen identity, deterministic geometry, sham, random controls, certificate, and invalid-input tests |
| [`test_precision_forward.py`](../experiments/causal_mediation/test_precision_forward.py) | Synthetic-weight Qwen/J-Lens integration tests for the native BF16 forward path |
| [`README.md`](../experiments/causal_mediation/README.md) | Detailed algorithm specification and operational commands |
| [`NUMERICAL_RESULTS.md`](../experiments/causal_mediation/NUMERICAL_RESULTS.md) | Synthetic measurements, validation evidence, and current qualification limits |

The package reuses the existing `process_sensitive_replay` model loader,
recurrent-gradient computation, teacher-forced replay, hybrid-cache utilities,
suffix-only Turn-3 construction, J-Lens scoring, and CUDA memory guard. The
upstream experiment code was not modified.

## Compensated BF16 algorithm

The policy is named `bf16_compensated_v1_proposed` and remains proposed until
real numerical smoke and review are complete.

1. Verify the BF16 recipient and exact frozen unit vector. Compute patch
   geometry on CPU float64 without changing the model's execution precision.
2. Return a bitwise copy for zero delta. Canonically reflect negative-target
   problems so tie-breaking does not depend on patch polarity.
3. Form the ideal point using `h + delta*v/dot(v,v)`. The denominator accounts
   for the saved vector's tiny unit-norm rounding error without replacing it.
4. Find nearest BF16 neighbors and bracket the desired projection using
   `BF16(h + lambda*v)`: at most 64 doubling steps and 80 bisection steps.
5. Starting from nearest rounding and both bracket endpoints, make up to 512
   deterministic single-coordinate BF16-neighbor corrections that improve
   coordinate accuracy, favoring efficient norm changes.
6. Make up to 512 additional neighbor changes that reduce orthogonal leakage
   while staying inside the coordinate-error budget.
7. Select the coordinate-feasible proposal with least orthogonal leakage.
   Retain failed proposals for diagnostics, but do not apply them to the model.

This is a bounded local search, not a globally optimal integer solver. The
complete tie-breaking and update rules are specified in the package README.
No behavioral outcome is accepted as an input to the solver.

### Proposed separate patch criteria

| Check | Proposed limit |
|---|---|
| Nonzero coordinate error | At most `0.01 * abs(delta)` |
| Orthogonal leakage L2 | At most `0.10 * abs(delta)` |
| Random realized-norm mismatch | At most 1% of the realized candidate-patch norm |
| Random projection onto candidate | At most 1% of the realized candidate-patch norm |
| SHAM residual | Bitwise unchanged |
| Existing engineering/sham parity | Original `atol=rtol=1e-5` |

These are proposed geometric fidelity limits, not thresholds adjusted to make
the observed probes pass. The 10% leakage-amplitude budget limits off-axis
energy to 1% of the squared intended coordinate change. There is no absolute
floor that silently permits erasure of very small nonzero patches.

### Independent infeasibility check

For any BF16-representable `q`, let `d=q-h`, coordinate error
`e=dot(v,d)-delta`, and orthogonal movement `d_perp`. Then:

```text
||q - (h + delta*v/||v||^2)||^2 = e^2/||v||^2 + ||d_perp||^2
```

Nearest coordinatewise BF16 rounding gives the global minimum of the left side.
If that lower bound exceeds the combined coordinate/leakage budgets, no BF16
procedure can meet both limits for that input. Such a result is labeled
`certified_infeasible`. A solver failure without this certificate is reported
separately and does not imply mathematical impossibility.

### Random controls

Each seeded Gaussian control direction is orthogonalized against the exact
candidate and normalized. Seeds derive from campaign seed `42`, item ID, branch,
and donor condition. Both polarities undergo the same compensated BF16
procedure, targeting the **realized candidate-patch L2 norm**. The implementation
logs the actual norm mismatch and actual projection leakage onto the candidate;
it does not silently rescale a rounded control afterward.

The final behavioral analysis must preserve both polarities and use their mean
as its primary random-control estimate.

## Numerical smoke execution and safeguards

The implemented smoke runs only items `0` and `2` and does not execute the final
mediation analysis. For each item it:

1. Reuses the exact frozen question and answer tokens and computes clean,
   primary-targeted, and alternative-targeted post-answer states once.
2. Requires primary and alternative support drops to reproduce their original
   machine-readable held-out values at the existing `1e-5` tolerances.
3. Checks token identity, recurrent-gradient parity, process-hook scope,
   persistent downstream state changes, and complete hybrid-cache integrity.
4. Creates independent, storage-disjoint confidence and correctness branches
   with the original suffix-only Turn-3 construction.
5. Captures the layer-42 question-mark residuals and checks candidate restoration,
   reverse transplant, both random polarities, and true sham numerically.
6. Applies only geometrically qualified proposals during the actual question-mark
   forward, before downstream layers and suffix tokens are computed. The hook
   must fire exactly once; float32 writes and stale-cache reuse are rejected.
7. Discards nonzero-patch output logits without scoring or inspecting behavioral
   effects. Boundary logits are compared only for sham engineering parity.

It produces 48 candidate/random numerical proposals across two items, two
branches, and four donor-recipient directions, plus sham checks for all three
base states. Every proposal reports coordinate accuracy, total norm, orthogonal
leakage, dtype, and J-Lens score before/after. Random proposals add their seed,
vector hash, norm mismatch, and projection onto the candidate.

J-Lens scores are not treated as raw projections. The installed path transports
the residual with `J`, casts to the LM-head dtype, and applies final normalization
before unembedding. Both quantities are therefore reported separately.

### Identity and artifact handling

The runner binds candidate and source artifact hashes, resolved config, runtime
versions, upstream replay code, and the new implementation/policy into its
records. Candidate binaries must match byte-for-byte. Historical text artifacts
may be verified after explicitly logged CRLF-to-LF transfer recovery; source
files are not rewritten.

### CUDA lifetime handling

One CUDA process/worker is used. Caches, gradient bundles, schedules, and branch
states are released explicitly. Per-item allocated, reserved, and peak memory
are logged with the existing growth guard. The original thresholds remain:

- More than 1024 MiB total post-cleanup allocation growth above baseline;
- Growth steps of at least 128 MiB;
- At least two consecutive growing items.

A systematic-growth failure stops execution. A baseline is recorded before
the first item. No concurrent CUDA workers or altered attention backend are
introduced.

### Gate behavior

Engineering failures exit nonzero with diagnostics. Failed precision proposals
remain visible and cause the numerical qualification to fail. Successful smoke
is only `numerically_qualified_pending_review`; it does not create a mediation
success marker or an eight-item execution path.

## Completed validation

Eleven focused tests passed. They cover candidate identity, deterministic
compensation, sign symmetry, bitwise sham, matched random realization, lattice
bounds, invalid inputs, and a tiny 64-block Qwen with synthetic BF16 weights.
Integration checks verify native BF16 writes, one hook invocation, hook cleanup,
downstream cache recomputation, source-state isolation, unchanged cache dtypes,
and sham parity. A J-Lens test demonstrates normalization's effect on the
relationship between displayed score and raw projection.

The synthetic matrix used the exact frozen vector with 12 predeclared
scale/target combinations, a zero-residual case, and sham:

| Result | Count |
|---|---:|
| Candidate proposals within coordinate-error budget | 14/14 |
| Candidate proposals within both precision budgets | 4/14, including sham |
| Candidate cases certified infeasible for both budgets | 10/14 |
| Random proposals passing their full precision criteria | 8/28 |

For a synthetic residual scale of 1 and target change 0.001, naive rounding
lost about 99.19% of the intended coordinate. Compensation reduced the coordinate
error below 1%, but introduced orthogonal leakage about 3.64 times the intended
change. The independent bound shows that even an optimal BF16 procedure would
require at least about 99.62% orthogonal leakage at the 1% coordinate-error
budget for that input.

These are synthetic geometry results, not measurements of the natural
eight-item residuals or evidence for/against behavioral mediation. Real-model
J-Lens scores are explicitly unavailable in the synthetic report.

Evidence:

- [Detailed numerical results](../experiments/causal_mediation/NUMERICAL_RESULTS.md)
- [All synthetic measurements](../experiments/causal_mediation/diagnostics/synthetic_precision_final.json)
- [CPU-only smoke guard result](../experiments/causal_mediation/diagnostics/cpu_guard_check/gate_status.json)

## Commands

Run from the repository root. Diagnostic outputs require fresh paths.

CPU tests:

```powershell
.venv/Scripts/python.exe -m unittest experiments.causal_mediation.test_precision experiments.causal_mediation.test_precision_forward -v
```

Synthetic geometry report:

```powershell
.venv/Scripts/python.exe -m experiments.causal_mediation.precision_probe --output assets/precision-synthetic-new.json
```

Two-item numerical smoke on the original CUDA host and pinned environment:

```bash
python -m experiments.causal_mediation.precision_smoke \
  --run-dir assets/psr-mediation-precision-v1 \
  --upstream assets/psr-quick-v3 \
  --direction assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt
```

An optional `--hf-cache-dir PATH` selects an existing Hugging Face cache. The
runner offers no item-selection, precision, backend, or tolerance override.

## Remaining work after numerical qualification

First obtain CUDA access, run the two numerical smoke items, and return their
achieved coordinate errors, orthogonal leakage, random-control matching, and
J-Lens changes for review. If naturally small coordinate differences cannot
meet the agreed budgets, stop rather than silently changing precision or
discarding items.

If qualification is accepted, the remaining implementation must add:

- All eight fixed items and both independent meta branches, with confidence
  primary and correctness secondary; retain item 57 and all inconvenient signs.
- Baseline candidate/confidence/correctness comparisons against upstream values
  for all eight items. **Scientific effect reproduction is diagnostic, not a
  gate:** weakened or reversed effects must remain reportable negative evidence.
  This does not waive the identity, engineering, or two-item support gates.
- Candidate-coordinate restoration and reverse transplant, matched positive and
  negative random controls, sham, and full-residual restoration positive controls.
- Full-sequence judgment scoring, generated-output validity logging, and
  donor-distance reduction:

  ```text
  DDR = abs(J_recipient - J_donor) - abs(J_patched - J_donor)
  ```

- Primary/alternative mechanism summaries and their within-item mean, using
  whole-item bootstraps that preserve paired mechanisms and control polarities.
- The requested scientific plots, patch tables, complete campaign logging, and
  `RESULTS.md`. Poor full-residual restoration must be prominent rather than
  hidden by item exclusion.
- Explicit discussion of J-Lens direct-output accessibility: changing an output
  alone is insufficient. Interpretation requires natural-sized, bidirectional,
  control-specific effects across both mechanisms and both judgment branches.

The existing precision stage does not supply these behavioral results. The
next implementation step is conditional on numerical qualification, not on a
desired mediation outcome.

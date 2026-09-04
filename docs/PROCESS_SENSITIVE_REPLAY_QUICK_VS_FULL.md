# Process-Sensitive Replay: Quick vs Full Profile

Last updated: 2026-09-04

## Purpose

The `process_sensitive_replay` experiment provides two execution profiles:

- `full`: the complete confirmatory protocol.
- `quick`: a reduced exploratory protocol designed to finish faster and avoid
  the recurrent-autograd CUDA peak encountered around beta item 12/16.

Both profiles test the same central mechanistic question:

```text
same factual question + same complete answer text
                    + different hidden answer process
                              |
                              v
          later candidate readout and judgment
```

The quick profile preserves the causal structure and critical validity checks.
It reduces dataset coverage, search resolution, layer coverage, candidate count,
and the length of the differentiable answer-support objective. It therefore
provides comparable *types* of mechanistic insight, but not equal statistical
power or confirmatory strength.

## Side-by-side comparison

| Component | Full profile | Quick profile |
|---|---:|---:|
| Answers generated | 82 | 16 predeclared items |
| Discovery items | 16 | 8 |
| Held-out items | Up to 66; 57 in the last valid answer bank | 8 |
| Alpha grid | `0.01, 0.02, 0.05, 0.10, 0.11, 0.125, 0.15, 0.20` | `0.10, 0.11` |
| Beta grid | `0.05, 0.08, 0.10, 0.11, 0.125, 0.15, 0.20, 0.30, 0.40` | `0.10, 0.20, 0.30` |
| Alternative intervention layers | `15, 19, 23` | `15, 19, 23` |
| J-Lens readout layers | `36` through `44` | `38, 40, 42` |
| Maximum frozen candidates | 3 | 1 |
| Top vocabulary entries per readout | 100 | 25 |
| Pre/post-freeze smoke items | 4 | 2 |
| Differentiable answer tokens | Entire answer, up to 256 | First 32 answer tokens |
| Complete answer teacher-forced | Yes | Yes |
| Complete perturbed state preserved into Turn 3 | Yes | Yes |
| Correctness and confidence branches | Both | Both |
| Seven causal conditions | All | All |
| Discovery and held-out support-match gates | Full criteria | Same criteria, reduced samples |
| Post-freeze support replay, reset, cache, token, and hook gates | All | All |
| Intended interpretation | Confirmatory if every gate passes | Exploratory only |

## What remains identical

Both profiles retain:

1. The exact factual question and complete teacher-forced answer within every
   matched condition.
2. Suffix-only Turn-3 construction over the preserved factual state.
3. The primary process intervention at zero-based layer 31.
4. Alternative-layer search across layers 15, 19, and 23.
5. Both correctness and confidence evaluation branches.
6. All seven experimental conditions:
   - `clean_preserved`
   - `targeted_weak_preserved`
   - `targeted_strong_preserved`
   - `random_strong_preserved`
   - `support_matched_alternative_preserved`
   - `alternative_random_preserved`
   - `targeted_strong_reset`
7. Norm-matched, same-layer random controls for both structured mechanisms.
8. Exact transcript/token parity, hybrid-cache integrity, storage isolation,
   hook-lifetime, reset-parity, and frozen-support reproduction gates.
9. Discovery/held-out isolation and immutable, hash-bound phase transitions.
10. Candidate discovery followed by held-out H1-H7 analysis.

Consequently, quick mode can still show whether:

- targeted answer-process perturbation changes later candidate activity;
- an earlier-layer support-reducing mechanism produces a convergent response;
- structured interventions exceed their own norm-matched random controls;
- the later effect depends on preserving the perturbed state;
- confidence and correctness margins track the hidden-process manipulation;
- the support-matching and candidate-discovery pipeline is scientifically
  promising enough to justify a larger run.

## Principal memory difference

Full mode differentiates the complete answer-sequence support objective. For a
long answer, recurrent state must retain an autograd graph across every
answer-predicting position. The preserved answer bank includes prospective and
knowledge-boundary answers of roughly 158-182 tokens, and the full beta grid
repeats this operation at three alternative layers. This produces the observed
single-item CUDA peak even when caches are cleaned between items.

Quick mode bounds the differentiable objective and intervention schedule to the
first 32 answer tokens:

```text
gradient objective:
    sum log P(X_t | Q, X_<t), for t = 1...min(answer length, 32)

subsequent replay:
    teacher-force every remaining token of the same answer
    with the factual-process hook disabled

Turn 3:
    evaluate from the complete resulting preserved state
```

The answer is not truncated in the visible transcript or in the preserved
state. Only the differentiable support objective and the positions receiving
the process intervention are bounded. This sharply reduces peak autograd memory
while retaining a controlled hidden-process difference that propagates through
the rest of the answer.

## What the quick profile gives up

### Lower population coverage

Eight held-out items cannot estimate dataset-wide effects as reliably as the
full held-out set. Domain-, difficulty-, correctness-, and answer-length
subgroups are especially underpowered.

### Coarser strength selection

The quick beta grid may miss a useful strength between its three tested values.
A quick support-match failure can therefore mean either that the alternative
mechanism genuinely fails or that the reduced grid was too coarse.

### Reduced layer information

Layers 38, 40, and 42 retain a small depth profile around the primary layer-40
readout, but quick mode cannot characterize the complete layer 36-44 trajectory.

### One frozen candidate

Quick mode studies the highest-ranked eligible candidate only. It cannot assess
whether several cosine-deduplicated candidates replicate the same pattern.

### Prefix-defined intervention

For answers longer than 32 tokens, quick mode tests a perturbation to the early
answer process whose consequences persist through the complete answer. It does
not test a direction derived from every token in the full answer objective.

### Wider uncertainty

Bootstrap intervals and correlations based on eight held-out items will be
unstable. Effect direction, control separation, support matching, and item-level
consistency should receive more weight than precise interval endpoints.

## Single-GPU performance policy

Use one experiment process and one CUDA worker. The phase dependency chain is
serial, and concurrent item workers would each need independent recurrent cache
and autograd state around the same 27B model. That raises peak memory and usually
reduces throughput on one already-busy GPU.

The safe single-GPU optimizations are therefore bounded-work optimizations:

- quick mode loads only its three requested J-Lens Jacobians onto CUDA rather
  than all nine fitted readout layers;
- the answer bank includes only the 16 predeclared quick items;
- differentiable recurrent replay is bounded to 32 answer tokens while the full
  answer remains teacher-forced;
- unused caches and graphs are released and checked after every item;
- phases remain separate processes so a completed phase cannot retain CUDA
  allocations in the next one.

After the pinned model and lens are present in the selected Hugging Face cache,
offline cache lookup can remove network metadata checks from each phase startup:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python -m experiments.process_sensitive_replay.run_all_phases `
  --profile quick `
  --hf-cache-dir PATH_TO_EXISTING_CACHE `
  --run-dir assets/psr-quick-v2
```

`HF_HUB_OFFLINE` should be set only after both pinned artifacts are already in
that cache. `expandable_segments` can reduce allocator fragmentation; it does
not change the measurements. Neither setting replaces the per-item CUDA memory
gate.

Do not enable concurrent phases, multiple item processes, ad-hoc batching,
`torch.compile`, a different attention backend, or a lower precision inside a
protocol-bound campaign. Those changes need separate numerical-parity and memory
validation before they can be treated as equivalent execution optimizations.

## Mechanistic-fidelity contract

Quick mode preserves the original experiment's causal design and mechanistic
question, while reducing the estimand's answer window, coverage, and resolution.
It does not claim that the 32-token estimand is identical to the full-answer
estimand. Validation and regression tests lock the following across both
profiles:

- the pinned model, tokenizer, J-Lens artifact, dataset, process layer, meta
  layer, and all three alternative intervention layers;
- the same seven causal conditions, two meta branches, gradient-based targeted
  direction construction, and same-layer norm-matched orthogonal random
  controls;
- suffix-only Turn-3 replay from the complete teacher-forced answer state;
- discovery/held-out isolation, support-match thresholds, reset parity, cache
  integrity, hook scope, candidate-ranking metrics, and interpretation ceiling;
- the same H1-H7 analysis and the same prohibition on a higher-order causal
  claim without candidate-to-judgment mediation.

Accordingly, quick mode answers the same *kinds* of mechanistic questions: does
the intervention persist, affect the later candidate/readout, exceed its random
control, depend on preserved state, and converge with a support-matched earlier
layer mechanism? It does not provide the same precision or claim strength. The
eight-item held-out set, early-32-token gradient objective, three readout layers,
coarser strength grids, and one candidate make it exploratory. A full run remains
necessary for dataset-wide confirmation and full-answer/full-layer coverage.
The resolved quick config explicitly names its estimand
`early_answer_process_first_32_tokens_with_complete_answer_state`; it never
labels the bounded objective as full-answer support.

## Commands

Run the exploratory profile in a new campaign directory:

```bash
python -m experiments.process_sensitive_replay.run_all_phases \
  --profile quick \
  --run-dir assets/psr-quick-v1
```

Run the full profile explicitly:

```bash
python -m experiments.process_sensitive_replay.run_all_phases \
  --profile full \
  --run-dir assets/psr-full-v1
```

`full` is the default when `--profile` is omitted.

If phases are executed individually, the same profile argument must be supplied
to every command:

```bash
python -m experiments.process_sensitive_replay.runner \
  --phase validate \
  --profile quick \
  --run-dir assets/psr-quick-v1
```

Never reuse a failed or partially completed directory, and never mix quick and
full phases in one directory. The resolved execution profile is included in the
campaign hash, manifest, gates, frozen protocol, and results report.

## Interpreting a quick run

### Quick run passes and effects converge

This is evidence that the process-sensitive mechanism and its controls are
worth pursuing. Inspect item-level plots and effect directions before deciding
whether to optimize or rerun the full protocol.

### Quick run passes but effects do not converge

This is useful exploratory negative evidence. It suggests that targeted and
alternative perturbations leave mechanism-specific traces rather than a shared
representation of answer-process reliability.

### Quick support matching fails

If discovery fails, treat the result as diagnostic. The alternative may
genuinely fail, but the coarse beta grid and small discovery set are additional
explanations. If held-out fails, the frozen alternative did not generalize under
the reduced protocol. Do not retune within either failed campaign.

The post-freeze smoke is different: it reports the two-item match rate only as a
diagnostic and gates exact reproduction of those items' hash-bound discovery
support drops. This avoids turning the 65% held-out rule into an accidental
100% requirement when quick smoke contains two items. A reproduction failure is
an engineering/state-consistency failure and still halts fail-closed.

### Quick run passes but full run later disagrees

The full run controls the confirmatory conclusion because it has greater item,
strength, layer, and candidate coverage.

## Claim boundary

A successful quick run may be described as exploratory evidence for a
process-sensitive effect under the reduced protocol. It must not be presented
as full-dataset confirmation.

Even a successful full run would establish only:

```text
process manipulation -> process-sensitive candidate/readout
```

It would not establish:

```text
candidate -> judgment
```

A later candidate restoration or causal-mediation experiment remains necessary
for a stronger higher-order representation claim.

# Higher-order alleys

Research code, frozen protocols, and experiment artifacts for distinguishing
ordinary output readout from process-sensitive and potentially higher-order
representations in Qwen3.6-27B using Jacobian Lens.

## Final sprint status

The sprint finds **exploratory evidence for a later process-sensitive latent**,
but does **not** establish that this latent causally mediates metacognitive
judgment:

```text
P -> R(P): exploratory evidence
P -> M(P) -> judgment: unresolved
```

`P` is the hidden computation that produced a factual answer. `R(P)` is a later
representation or readout that changes when that hidden process is degraded,
even while the visible question and complete answer remain identical. Calling
it a proven higher-order representation would require a valid causal test of the
candidate-to-judgment path.

The strongest defensible conclusion is:

> Qwen3.6-27B shows exploratory evidence for a later internal signal that tracks
> controlled degradation of its preceding answer process despite identical
> visible output. The candidate responds in the same frozen direction under two
> structured support-reducing mechanisms, exceeds their respective matched
> random controls, and returns to the clean state when the perturbed process
> history is reset. Process damage also covaries with both the candidate score
> and subsequent confidence. Whether that candidate causally mediates judgment
> remains unresolved because neither qualified intervention route could test the
> path faithfully.

This conclusion concerns the same eight-item exploratory quick-run held-out set.
It is not an independent replication, full-profile confirmation, or dataset-wide
claim.

## Evidence progression

| Stage | Question | Result |
|---|---|---|
| Global J-Lens steering | Do evaluation-related directions affect metacognitive-looking outputs? | Yes, but this is compatible with generic output evaluation. |
| Matched SELF vs. OTHER | Are the earlier candidates specific to evaluating the model's own answer? | No convincing self-specific effect for tokens `97817` or `99973`. |
| Process-sensitive replay | Does a later frozen signal track hidden process degradation when visible output is fixed? | Exploratory evidence for `P -> R(P)` on eight held-out items. |
| Causal mediation follow-up | Does changing the frozen candidate itself change confidence/correctness? | Unresolved: both intervention routes failed qualification before mediation outcomes were measured. |

The frozen process-sensitive candidate is token `75075` at layer `42`, with
orientation `-1`. Its decoded vocabulary label, ` UIImagePickerController`, is
not interpreted semantically; the research object is the frozen direction and
its measured behavior.

## Process-sensitive replay result

The successful `psr-quick-v3` run held the visible question and complete factual
answer constant while comparing:

- a clean process state;
- the primary answer-support perturbation at layer 31, strength 0.11;
- a support-matched alternative perturbation at layer 23, strength 0.20;
- separate same-layer, norm-matched random controls;
- reset conditions that discarded the perturbed history before judgment.

Across the eight exploratory held-out items:

- both structured mechanisms moved the layer-42 candidate in the frozen
  direction;
- each structured effect exceeded its corresponding random control;
- the intervention-induced candidate difference returned to clean under reset;
- candidate movement covaried with answer-support damage;
- greater hidden damage covaried with a lower later confidence margin.

The two mechanisms were compatible with convergence, but mechanism equivalence
was not established. The experiment supports the term **process-sensitive
representation/readout**, with an `M(P)`-like candidate as the interpretation
ceiling.

## Why causal mediation remains unresolved

The final experiment required a selective natural coordinate transplant at the
layer-42 `?` residual, followed by the unchanged judgment computation. Two
outcome-blind numerical policies were qualified on predeclared smoke items 0 and
2.

### BF16-native restoration

The Blackwell run reproduced all four upstream primary/alternative support drops
exactly. All 16 candidate proposals met the 1% coordinate-accuracy criterion,
but every candidate proposal failed the frozen orthogonal-leakage ceiling.
Certified BF16 lattice bounds showed the failure was unavoidable. Realized
off-axis movement was approximately 89.77% to 248.61% of the intended coordinate
change, versus the 10% maximum.

### FP32-tail restoration

The FP32 tail represented the patches cleanly: maximum candidate-coordinate
relative error was 0.0005853%, and maximum orthogonal leakage L2 was `2.14e-6`.
However, switching blocks 43-63, final normalization, and the output head to FP32
materially changed the unpatched judgments:

| Equivalence diagnostic | Maximum discrepancy | Proposed limit |
|---|---:|---:|
| Complete-label sequence log probability | 0.203638 nats | 0.05 nats |
| Judgment margin | 0.203789 nats | 0.05 nats |
| Process-minus-clean margin contrast | 0.195183 nats | 0.05 nats and 10% |

Generated labels were unchanged, but only two of eight process contrasts met
the proposed bounds. The eight-item mediation runner was therefore never
authorized. No nonzero candidate patch was evaluated behaviorally, so there is
no positive, negative, or null mediation estimate.

The two failures are engineering and identifiability results. They are not
evidence against higher-order representation. Thresholds were not relaxed and a
third patching policy was not added after observing the failures.

Read the complete result in
[Causal mediation follow-up: final results](docs/CAUSAL_MEDIATION_FINAL_RESULTS.md).

## Repository layout

| Path | Purpose |
|---|---|
| [`experiments/higher_v_readout_global/`](experiments/higher_v_readout_global/) | Global J-Lens steering and evaluator-direction experiments |
| [`experiments/self_v_external/`](experiments/self_v_external/) | Matched SELF-versus-OTHER experiment |
| [`experiments/process_sensitive_replay/`](experiments/process_sensitive_replay/) | Frozen replay, hidden process perturbations, controls, analysis, and latent visualization |
| [`experiments/causal_mediation/`](experiments/causal_mediation/) | BF16 and FP32-tail numerical qualification; no eight-item mediation runner |
| [`dataset/metacognition.csv`](dataset/metacognition.csv) | Ninety-row factual/metacognitive experiment dataset |
| [`assets/psr-quick-v3/`](assets/psr-quick-v3/) | Frozen quick-run protocol, held-out records, analysis, and plots |
| [`assets/psr_quick_big_files/`](assets/psr_quick_big_files/) | Frozen candidate direction and large tensors |
| [`docs/`](docs/) | Research narrative, implementation records, protocols, and final results |

## Key documents

- [Final causal mediation results](docs/CAUSAL_MEDIATION_FINAL_RESULTS.md)
- [Full first-order vs. higher-order sprint summary](docs/E2SUM_first_order_vs_higher_order_research_sprint.md)
- [Process-sensitive replay implementation plan](docs/PROCESS_SENSITIVE_REPLAY_IMPLEMENTATION_PLAN.md)
- [Causal mediation implementation](docs/CAUSAL_MEDIATION_IMPLEMENTATION.md)
- [Mixed-precision qualification policy](docs/MIXED_PRECISION_MEDIATION_POLICY.md)
- [SELF vs. OTHER results](docs/SELF_V_EXTERNAL_TOKEN_COMPARISON_RESULTS.md)

## Setup

Python 3.11 or newer and `uv` are expected. From the repository root:

```powershell
uv sync
```

The model experiments require a single CUDA worker and sufficient GPU and host
memory for Qwen3.6-27B, J-Lens, hybrid caches, and temporary states. Exact model,
tokenizer, lens, package, and artifact revisions are frozen in the run manifests.
Cross-hardware numerical reproduction is not assumed: an A100 failed the
predeclared upstream support gate, while the Blackwell environment reproduced
the four smoke values exactly.

Static validation and CPU tests do not download or load the full model.

## Verify the project

Run all experiment tests:

```powershell
uv run python -m unittest discover -s experiments -p "test*.py"
```

Validate the earlier runnable experiment protocols:

```powershell
uv run python -m experiments.higher_v_readout_global.runner --phase validate
uv run python -m experiments.self_v_external.runner --phase validate
```

Test the candidate-anchored latent visualizer:

```powershell
uv run python -m unittest experiments.process_sensitive_replay.test_latent_visualization -v
```

## Visualize the actual latent geometry

The precision smoke saved real BF16 layer-42 residuals for items 0 and 2. On the
Blackwell host containing those artifacts, create a candidate-anchored residual
map with:

```bash
python -m experiments.process_sensitive_replay.plot_candidate_anchored_latents \
  --run-dir assets/psr-mediation-precision-blackwell-v1
```

The plot centers every primary and alternative residual on its matched clean
state. Its horizontal axis is the exact frozen candidate coordinate
`-v^T delta_h`; the remaining axes are derived only from the
candidate-orthogonal residual changes. It also reports full residual-change
cosine similarity and the fraction of displacement energy carried by the
candidate axis.

Outputs default to:

```text
docs/figures/candidate_anchored_layer42_latents.png
docs/figures/candidate_anchored_layer42_latents.pdf
docs/figures/candidate_anchored_layer42_latents.csv
docs/figures/candidate_anchored_layer42_latents.json
```

The CSV and JSON make the projection auditable. Because raw residual tensors
were retained only for smoke items 0 and 2, this is a two-item geometric
visualization, not an eight-item latent-space result or causal analysis.

## Reproducibility and interpretation

The repository uses fail-closed gates for frozen identity, tokenization, replay
parity, hook scope, cache isolation, CUDA memory growth, patch geometry, and
precision equivalence. Failed campaigns retain diagnostics but do not create a
success marker or authorize a later phase.

When reporting this project:

- describe `psr-quick-v3` as exploratory evidence from eight held-out items;
- describe token `75075` as a frozen process-sensitive direction, not by the
  semantics of its decoded vocabulary label;
- say the primary and alternative mechanisms are compatible with convergence,
  not proven equivalent;
- distinguish disappearance of the intervention effect under reset from
  disappearance of the representation;
- state explicitly that candidate-to-judgment mediation was never tested.

Any future causal attempt should be a new, separately preregistered experiment,
not a post-hoc extension of this sprint.

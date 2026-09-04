# Process-Sensitive Replay Quick v3: Results and Interpretation

Run: `assets/psr-quick-v3`  
Recovered discovery records: `assets/psr_quick-v3-discovery`  
Model: `Qwen/Qwen3.6-27B`  
Execution profile: `quick`  
Answer-support objective: first 32 answer tokens, with the complete answer
teacher-forced into the preserved Turn-3 state  
Run date: 2026-09-04  
Status: **all eight phases passed**

## Executive summary

The quick campaign provides positive exploratory evidence for a persistent,
direction-specific process-sensitive readout associated with later confidence
and correctness evaluation. The primary layer-31 intervention and a separately
calibrated layer-23 intervention both changed the frozen layer-42 candidate in
the discovery-selected direction, exceeded their respective same-layer random
controls, and passed held-out support matching on all eight items. Resetting the
perturbed state removed the targeted effect, supporting the claim that the later
response depended on preservation of the manipulated answer state.

The result does **not** establish that the two mechanisms encode an identical
representation, and it does not prove a higher-order representation. Mechanism
convergence is compatible with the held-out estimates but remains inconclusive:
the normalized targeted-versus-alternative differences are close to zero, while
the mechanism-specific regression terms have wide intervals and no predeclared
equivalence threshold exists. The eight-item sample and 32-token objective make
this a successful exploratory screen rather than a confirmatory result.

## Campaign integrity

The following phase gates passed:

1. `validate`
2. `answer_bank`
3. `pre_discovery_smoke`
4. `discovery`
5. `freeze`
6. `post_freeze_smoke`
7. `heldout`
8. `analyze`

The run logged no experiment errors. Post-freeze smoke reproduced the discovery
targeted and alternative support drops exactly on both smoke items and passed
all 19 critical gradient, intervention, token, cache, reset, branch, and hook
checks. Held-out support matching passed before candidate-effect analysis was
authorized.

The discovery gate records zero held-out access. Candidate and strength
selection were therefore completed without using the eight held-out outcomes.

## Discovery

### Strength selection

| Quantity | Selected value |
|---|---:|
| Weak primary strength | `alpha = 0.10` |
| Weak median support drop | `1.30371` nat |
| Strong primary strength | `alpha = 0.11` |
| Strong median support drop | `1.44593` nat |
| Alternative intervention layer | `23` |
| Alternative strength | `beta = 0.20` |
| Alternative median support drop | `1.19860` nat |
| Median paired support mismatch | `0.39087` nat |
| Frozen mismatch tolerance | `0.5` nat |
| Median alternative/target norm ratio | `1.51823` |

Layer 23 at beta `0.20` was the only one of the nine tested layer/strength
combinations that passed every alternative-selection check. Layer 19 at beta
`0.20` matched the aggregate median but failed paired mismatch (`0.64929 > 0.5`),
while the remaining settings missed the support criteria.

The strong alpha used the protocol's allowed below-target fallback. Its
discovery median support drop (`1.44593` nat) did not enter the preferred
`2–4`-nat interval. Weak and strong alpha were consequently close in both
strength (`0.10` versus `0.11`) and median effect. This does not invalidate the
run, but it limits conclusions about a broad weak-to-strong dose response.

### Candidate selection

Discovery evaluated 248,131 eligible token/layer directions derived from
217,494 eligible vocabulary tokens. The quick profile froze one candidate:

| Field | Value |
|---|---|
| Token ID | `75075` |
| Token label | ` UIImagePickerController` |
| J-Lens layer | `42` |
| Orientation | `-1` |
| Discovery structured sign consistency | `0.875` |
| Targeted-strong effect | `0.04492` |
| Alternative effect | `0.06055` |
| Targeted minus random | `0.04102` |
| Alternative minus random | `0.07813` |
| Targeted preserved minus reset | `0.04492` |

The token label is an index label, not a semantic interpretation. Nothing in
this experiment indicates that the relevant representation concerns an image
picker or software APIs. The defensible object of analysis is the oriented
layer-42 direction associated with token ID 75075.

## Held-out support matching

The frozen layer-23/beta-0.20 intervention passed the complete held-out gate:

| Quantity | Result |
|---|---:|
| Valid held-out items | 8 |
| Matched items | 8/8 (`100%`) |
| Required match fraction | `65%` |
| Targeted median support drop | `2.94407` nat |
| Alternative median support drop | `3.12691` nat |
| Median signed mismatch | `0.05449` nat |
| Median absolute mismatch | `0.26395` nat |
| Mean absolute mismatch | `3.20962` nat |

The median behavior is tightly matched, but the mean absolute mismatch is much
larger because the long-answer items have large support scales. Items 67, 68,
and 82 had absolute mismatches of `5.762`, `2.900`, and `16.263` nats,
respectively, while still satisfying their frozen 25%-relative tolerance bands.
Thus, 8/8 means that all items passed the predeclared rule; it does not mean the
two interventions had nearly identical absolute drops on every item.

## Held-out candidate results

All effects below are oriented so that positive values follow the direction
selected during discovery. Intervals are item-bootstrap 95% intervals with only
eight held-out resampling units.

### Confidence branch

| Hypothesis or control | Estimate | Median | 95% interval |
|---|---:|---:|---:|
| H1: targeted minus clean | `0.02930` | `0.03906` | `[0.00781, 0.05078]` |
| H2: alternative minus clean | `0.04688` | `0.03906` | `[0.01758, 0.07617]` |
| H3: targeted minus alternative | `-0.01758` | `-0.01563` | `[-0.04883, 0.01563]` |
| H3: support-normalized difference | `-0.02105` | `-0.00068` | `[-0.06290, 0.00516]` |
| H4: targeted minus random | `0.02930` | `0.03906` | `[0.01563, 0.04297]` |
| H4: alternative minus its random | `0.05859` | `0.03906` | `[0.02539, 0.09375]` |
| H5: targeted preserved minus reset | `0.02930` | `0.03906` | `[0.00781, 0.04883]` |
| H6: candidate-score/support slope | `0.000482` | — | `[0.000033, 0.001497]` |
| H7: confidence-margin/support slope | `-0.01830` | — | `[-0.03551, -0.01416]` |

The targeted and alternative effects were positive on six of eight confidence
items. Both structured-versus-random contrasts were also positive on six of
eight items. The negative H7 slope means that greater intervention-induced
answer-support damage was associated with a lower `HIGH_CONFIDENCE -
LOW_CONFIDENCE` log-probability margin. H7 correlations were Pearson `-0.830`
and Spearman `-0.890`.

### Correctness branch

| Hypothesis or control | Estimate | Median | 95% interval |
|---|---:|---:|---:|
| H1: targeted minus clean | `0.03271` | `0.02734` | `[0.01660, 0.05078]` |
| H2: alternative minus clean | `0.09521` | `0.03906` | `[0.00488, 0.20703]` |
| H3: targeted minus alternative | `-0.06250` | `-0.03125` | `[-0.16309, 0.01855]` |
| H3: support-normalized difference | `0.01022` | `-0.00127` | `[-0.00818, 0.03799]` |
| H4: targeted minus random | `0.03369` | `0.03516` | `[0.00879, 0.05615]` |
| H4: alternative minus its random | `0.10498` | `0.05078` | `[0.01709, 0.21875]` |
| H5: targeted preserved minus reset | `0.03271` | `0.02734` | `[0.01660, 0.05029]` |
| H6: candidate-score/support slope | `0.000978` | — | `[0.000746, 0.001276]` |

The targeted effect was positive on seven of eight correctness items; the
alternative effect was positive on five of eight. The alternative mean and its
uncertainty were influenced by the long-answer items, particularly items 67 and
68. Candidate-score/support correlations were Pearson `0.815` and Spearman
`0.675`.

## Interpretation by hypothesis

| Test | Outcome | Reading |
|---|---|---|
| H1 | Supported exploratorily | Primary answer-process perturbation changes the later candidate. |
| H2 | Supported exploratorily | A support-matched earlier-layer perturbation changes the same candidate direction. |
| H3 | Compatible but unresolved | Near-zero normalized medians favor convergence, but absence of a difference is not an equivalence test. |
| H4 | Supported exploratorily | Both structured directions exceed their own norm-matched random controls. |
| H5 | Supported exploratorily | The effect depends on preserving the perturbed answer state. |
| H6 | Supported exploratorily | Candidate activity increases with support damage in both branches. |
| H7 | Supported exploratorily | Greater support damage is associated with lower model confidence. |

The generic evaluator-token controls were nearly flat across conditions, which
argues against a uniform shift in every evaluation-related vocabulary
direction. The layer profile also localizes the selected candidate's largest
mean score to layer 42 among the three quick-profile readout layers. These are
useful specificity checks, not proof of a uniquely metacognitive feature.

## Plot guide and visual results

The analysis produced 13 numbered figures (14 PNG files because figure 4 has
three panels). Figures 1–11 and 13 use the protocol's frozen primary branch,
`confidence`. Figure 12 visualizes answer-support matching, which is shared by
the confidence and correctness branches. The correctness estimates reported
above are therefore machine-readable results, but this quick run did not emit a
second set of correctness-specific figures.

Candidate plots use the oriented layer-42 score unless stated otherwise. The
frozen orientation is `-1`, so a higher oriented score means a lower raw J-Lens
score for token 75075. The absolute score is negative and is not a probability;
the paired differences between conditions are the relevant quantities.

### 1. Manipulation check

[Open plot 01](../assets/psr-quick-v3/analyze/plots/01_manipulation_check.png)

The box plot shows that clean support drop is exactly zero and both random
controls remain centered near zero. In contrast, the structured primary and
alternative interventions have positive median drops of `2.944` and `3.127`
nats. Their ranges extend to `71.325` and `87.588` nats because three held-out
answers are much longer than the five short calibration answers. The figure
therefore confirms that both structured directions manipulate answer support,
while also showing why medians and per-item tolerances are more informative
than raw means for this mixed-length sample.

### 2. Confidence margin versus support

[Open plot 02](../assets/psr-quick-v3/analyze/plots/02_confidence_vs_support.png)

The high-support-drop points lie at lower `HIGH_CONFIDENCE - LOW_CONFIDENCE`
margins than most near-zero points. This visual pattern agrees with H7:
item-centered slope `-0.01830`, Pearson `-0.830`, and Spearman `-0.890`, with a
bootstrap slope interval wholly below zero. The plotted points include primary,
alternative, and random conditions, whereas the frozen H7 estimate is computed
from the 24 clean/weak-primary/strong-primary observations after centering
within item. The figure is supportive context for H7, not a separate pooled
regression or evidence that support loss alone causes lower confidence.

### 3. Candidate score versus support

[Open plot 03](../assets/psr-quick-v3/analyze/plots/03_candidate_vs_support_candidate_1.png)

The oriented candidate tends to become less negative as answer-support damage
increases. The frozen item-centered H6 estimate is positive: slope `0.000482`,
Pearson `0.468`, and Spearman `0.636`; its slope interval is
`[0.000033, 0.001497]`. As in figure 2, the scatter displays more conditions
than the formal 24-observation H6 model, which uses only clean, weak-primary,
and strong-primary observations. A few long-answer points provide much of the
horizontal range, so the rank correlation and item bootstrap are important
companions to the visual trend.

### 4. Paired mechanism and random-control comparisons

- [Open plot 04](../assets/psr-quick-v3/analyze/plots/04_targeted_vs_random_candidate_1.png):
  primary targeted versus its layer-31 norm-matched random control. Six items
  favor the targeted direction and two are tied; mean difference `0.02930`,
  95% interval `[0.01563, 0.04297]`.
- [Open plot 04a](../assets/psr-quick-v3/analyze/plots/04a_primary_vs_alternative_candidate_1.png):
  primary versus support-matched layer-23 intervention. Item trajectories are
  mixed, and the alternative is stronger on average: targeted minus alternative
  `-0.01758`, interval `[-0.04883, 0.01563]`. Crossing zero means this panel does
  not establish a reliable mechanism difference—or equivalence.
- [Open plot 04b](../assets/psr-quick-v3/analyze/plots/04b_alternative_vs_random_candidate_1.png):
  alternative versus its same-layer norm-matched random control. Six items favor
  the structured alternative and two are tied; mean difference `0.05859`,
  interval `[0.02539, 0.09375]`.

Each grey segment connects the same held-out item. These paired panels are more
informative than comparing two unpaired group averages because they expose item
heterogeneity directly.

### 5. Preserved versus reset state

[Open plot 05](../assets/psr-quick-v3/analyze/plots/05_preserved_vs_reset_candidate_1.png)

Six items favor the preserved targeted state, one is tied, and item 57 moves in
the opposite direction. The mean preserved-minus-reset effect is `0.02930`,
with interval `[0.00781, 0.04883]`. Reset scores return exactly to the clean
scores in this run, so the H5 mean equals the H1 mean. This is the visual reset
control: the downstream candidate change depends on retaining the manipulated
answer state rather than merely replaying identical visible text.

### 6. Clean versus targeted under identical text

[Open plot 06](../assets/psr-quick-v3/analyze/plots/06_clean_vs_targeted_candidate_1.png)

With visible text held constant, six items increase in the oriented direction,
one is unchanged, and item 57 decreases. The mean targeted-minus-clean effect is
`0.02930`, with interval `[0.00781, 0.05078]`. This is the clearest item-level
view of H1, although it remains a selected-candidate result rather than a claim
that every item responds uniformly.

### 7. Generic evaluator-token controls

[Open plot 07](../assets/psr-quick-v3/analyze/plots/07_generic_evaluator_controls.png)

The two generic evaluator tokens are nearly flat across all conditions. Token
97817 varies by only about `0.0234` in mean score and token 99973 by about
`0.0286`, across layers and items. Clean and reset means are identical for both.
This argues against a broad intervention-induced shift in arbitrary evaluator
tokens. It does not prove that token 75075 is semantically metacognitive or that
all possible generic controls would remain unchanged.

### 8. Layer profile

[Open plot 08](../assets/psr-quick-v3/analyze/plots/08_layer_profile_candidate_1.png)

This figure plots **raw**, not oriented, mean candidate scores at the three
quick-profile readout layers (`38`, `40`, and `42`). Layer 42 has the largest
raw baseline score (`2.723`) and is the frozen candidate layer. Because the
orientation is `-1`, the relevant targeted movement at layer 42 is downward:
clean `2.723` to primary `2.693` and alternative `2.676`, corresponding to
positive oriented effects of `0.0293` and `0.0469`. The primary effect has the
opposite sign at layers 38 and 40, while the alternative effect is already
positive there. Thus the panel supports late-layer specificity for the primary
effect within the three sampled layers, but it is not a complete depth scan.

### 9. Effect summary with uncertainty

[Open plot 09](../assets/psr-quick-v3/analyze/plots/09_effect_summary_candidate_1.png)

This is the compact confidence-branch summary. The targeted-clean,
alternative-clean, targeted-random, alternative-random, and targeted-reset
intervals are all above zero. Only targeted-minus-alternative crosses zero.
Visually, this is the strongest summary of H1, H2, H4, and H5 support and H3
uncertainty. The intervals are item-bootstrap intervals over just eight items;
they should not be read as confirmatory population estimates.

### 10. All mechanism and control conditions

[Open plot 10](../assets/psr-quick-v3/analyze/plots/10_mechanism_control_comparison_candidate_1.png)

Grey lines trace individual items and black diamonds show condition means. The
mean oriented scores are clean `-2.7227`, primary `-2.6934`, primary random
`-2.7227`, alternative `-2.6758`, alternative random `-2.7344`, and reset
`-2.7227`. Both structured interventions move upward in the selected direction;
the primary random and reset means coincide with clean, and the alternative
random moves slightly downward. The crossing item trajectories show why the
positive average effects should not be described as universal.

### 11. Support-normalized convergence

[Open plot 11](../assets/psr-quick-v3/analyze/plots/11_support_normalized_convergence_candidate_1.png)

The dashed diagonal denotes equal candidate response per nat of support damage.
Most points lie close to it, especially the three long-answer items whose large
denominators put both normalized responses near zero. Item 57 is a conspicuous
outlier: primary normalized response `-0.0431` versus alternative `0.1109`.
Consequently, the median targeted-minus-alternative normalized difference is
near zero (`-0.00068`), but the mean is `-0.02105`. This supports compatibility
with convergence for typical items while demonstrating that convergence is not
uniform and has not passed a predeclared equivalence test.

### 12. Item-level support matching

[Open plot 12](../assets/psr-quick-v3/analyze/plots/12_item_level_support_matching.png)

Plot positions 0–7 correspond to held-out item IDs `0`, `2`, `3`, `4`, `57`,
`67`, `68`, and `82`. The shaded band is the frozen per-item tolerance around
the primary support drop; every alternative point falls inside its band, so the
gate passes 8/8. The short-answer items cluster below 4.1 nats, whereas items
67, 68, and 82 occupy the 43–88-nat range. Item 82 has the largest absolute
mismatch (`16.263` nats) but still passes its length-scaled tolerance. This plot
shows successful protocol matching, not exact equality of intervention damage.

### 13. Shared-support component versus mechanism residual

[Open plot 13](../assets/psr-quick-v3/analyze/plots/13_support_vs_mechanism_residual_candidate_1.png)

The confidence item-fixed-effect decomposition estimates a shared-support
component of about `-0.085` and an alternative-layer residual of `+0.0257`.
The latter has interval `[-0.0156, 0.0585]`, and the absolute residual/shared
ratio is `0.584` with a wide interval `[0.040, 1.388]`. This decomposition uses
only the paired primary and alternative observations after item effects, so its
negative support component is not contradictory to the positive H6 trend,
which uses clean/weak/strong primary observations. With eight items, the plot is
best read as evidence that mechanism-specific residual variation remains; it
cannot establish either complete convergence or reliable separation.

## Mechanism-convergence assessment

The raw and support-normalized targeted-versus-alternative intervals include
zero in both branches, and the normalized medians are close to zero. This is
consistent with the two mechanisms producing a related downstream response.

It is not sufficient to claim equivalence. The item-fixed-effect
mechanism/shared-effect ratios were `0.584` for confidence and `0.900` for
correctness, with bootstrap intervals extending above one. The protocol did not
freeze an equivalence margin or an acceptable ratio threshold. Consequently,
the result supports “compatible with convergence,” not “demonstrated shared
representation.”

## Memory behavior

The quick profile resolved the observed CUDA-lifetime problem:

| Stage | Maximum allocated peak |
|---|---:|
| Discovery alpha grid | `56.63 GiB` |
| Discovery alternative-layer beta grid | `59.13 GiB` |
| Discovery candidate replay | `58.04 GiB` |
| Post-freeze smoke | `52.81 GiB` |
| Held-out | `58.06 GiB` |

Post-cleanup allocation returned to approximately `51.31 GiB` after each item.
All recorded CUDA trend gates passed with zero cumulative post-cleanup growth.
This indicates bounded per-item memory rather than the progressive allocation
growth seen in the failed full campaign.

## Limitations and claim boundary

1. The held-out sample contains only eight items, so bootstrap intervals and
   item-level consistency estimates are unstable.
2. The direction is derived from only the first 32 answer tokens. The complete
   answer state is preserved, but this is not the full-answer gradient estimand.
3. Four-token calibration answers and long prospective/knowledge-boundary
   answers have very different support scales. Long items influence regression
   and mean-mismatch estimates disproportionately.
4. Only one candidate and three readout layers were retained by the quick
   profile. Candidate and depth generality are unknown.
5. Discovery searched many eligible directions. Independent held-out
   replication reduces selection-overfit concerns but does not eliminate them
   at this sample size.
6. There is no frozen convergence-equivalence threshold, so the experiment does
   not automatically classify the candidate as process-sensitive or `M(P)`-like.
7. Candidate-to-judgment causal mediation was not tested. The run therefore
   cannot prove a higher-order representation.

## Artifact provenance

The recovered JSON, JSONL, and other text discovery records match the hashes in
the completed analysis gate after normalizing Windows CRLF line endings back to
their original LF representation. `candidate_scores.csv` and
`trial_summary.csv` match their recorded hashes byte-for-byte. The numeric
discovery content is therefore consistent with the completed campaign.

The recovered directory does not contain these binary artifacts:

- `discovery_vocab_scores.pt`
- `candidate_metrics.pt`
- `directions/layer42_token75075_8515912d78e8.pt`

The completed analysis gate proves that those files existed and passed their
hash checks when the campaign ran, but a complete independent reconstruction of
candidate ranking requires recovering them from the GPU host. Future transfers
should archive the whole run as `.tar.gz` or `.zip` so binary files and exact
line endings are preserved.

## Conclusion

The quick run is a successful exploratory result: two independently calibrated,
support-matched answer-process interventions produced persistent,
direction-specific downstream changes associated with both correctness and
confidence, and those changes survived random-control and reset tests. The
evidence is compatible with a shared process-sensitive readout, but mechanism
equivalence and a higher-order representation remain unproven and require a
larger confirmatory or causal-mediation experiment.

## Source artifacts

- `assets/psr-quick-v3/analyze/RESULTS.md`
- `assets/psr-quick-v3/analyze/analysis_report.json`
- `assets/psr-quick-v3/analyze/plot_manifest.json`
- `assets/psr-quick-v3/heldout/support_match_summary.json`
- `assets/psr-quick-v3/heldout_support_match.csv`
- `assets/psr-quick-v3/heldout_effects.csv`
- `assets/psr-quick-v3/frozen_protocol.json`
- `assets/psr_quick-v3-discovery/alpha_grid_diagnostics.json`
- `assets/psr_quick-v3-discovery/beta_grid_diagnostics.json`
- `assets/psr_quick-v3-discovery/candidate_discovery.json`
- `assets/psr_quick-v3-discovery/gate_status.json`
- `assets/psr_quick-v3-discovery/cuda_memory.jsonl`

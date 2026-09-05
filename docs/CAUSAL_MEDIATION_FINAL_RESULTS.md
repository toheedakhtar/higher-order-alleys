# Causal mediation follow-up: final results and sprint endpoint

Last updated: 2026-09-05

## Executive conclusion

The causal mediation follow-up did **not** perform a candidate-to-judgment
intervention. Both predeclared ways of making that intervention failed their
engineering qualification gates before behavioral mediation outcomes were
computed:

1. A BF16-native patch could approximately restore the intended candidate
   coordinate, but necessarily introduced too much movement in orthogonal
   residual directions.
2. An FP32 continuation could realize the intended patch with negligible
   numerical contamination, but the precision switch materially changed the
   unpatched downstream judgment computation.

These results leave the causal mediation question unresolved. They are not
evidence that the candidate lacks a causal role, and they are not evidence
against higher-order representation. They identify a precision and
identifiability limit for this frozen model, intervention site, and protocol.

The strongest supported scientific statement at the end of the sprint is:

> Qwen3.6-27B shows exploratory evidence for a later internal signal that tracks
> controlled degradation of its preceding answer process despite identical
> visible output. The candidate responds in the same frozen direction under two
> structured support-reducing mechanisms, exceeds their respective matched
> random controls, and returns to the clean state when the perturbed process
> history is reset. Process damage also covaries with both the candidate score
> and subsequent confidence. These findings go beyond a simple evaluator based
> only on visible output. However, whether the candidate causally mediates
> metacognitive judgment remains unresolved: BF16 coordinate patching at the
> frozen layer-42 site cannot satisfy the predeclared intervention-precision
> criteria, while the tested FP32 continuation materially changes the downstream
> judgment computation.

Accordingly:

```text
P -> R(P): exploratory evidence
P -> M(P) -> judgment: unresolved
```

Here `R(P)` means a later process-sensitive representation or readout. It should
not be promoted to a proven higher-order representation without a valid causal
candidate-to-judgment test.

## Scope and claim boundary

This work is a causal follow-up to the same eight-item exploratory quick-run
held-out set used in `psr-quick-v3`. It is not an independent replication, a
full-profile confirmation, or dataset-wide evidence. The held-out item IDs were:

```text
0, 2, 3, 4, 57, 67, 68, 82
```

Only items `0` and `2`, the predeclared smoke items, were used to qualify the
precision policies. No mediation analysis was run over all eight items. No
nonzero candidate patch was interpreted behaviorally.

The candidate was frozen before this follow-up:

| Field | Frozen value |
|---|---:|
| Model | Qwen3.6-27B / `Qwen3_5ForConditionalGeneration` |
| Model and tokenizer revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| J-Lens revision | `0731326edff4ae730ffc5356fe1a4728c748b3a6` |
| J-Lens SHA-256 | `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1` |
| Candidate token ID | `75075` |
| Candidate vocabulary label | ` UIImagePickerController` |
| Candidate layer | `42` |
| Candidate orientation | `-1` |
| Direction tensor SHA-256 | `8515912d78e8afe1a932bf06dc43cb41a54e446e91a8abbbe008a28fac54ac84` |
| Direction file SHA-256 | `7661c6529508c2f571ffd68a15e0837669303ba304aacd1c377c9caa31f695aa` |
| Primary process intervention | Layer `31`, strength `0.11` |
| Alternative process intervention | Layer `23`, strength `0.20` |
| Model execution dtype | BF16 |

The decoded candidate label is not treated as semantically meaningful. The
object under study is the frozen direction and its measured behavior.

## Evidence entering the causal follow-up

The upstream `psr-quick-v3` experiment held the visible question and complete
answer fixed while changing the hidden answer-process state. It compared:

- `clean`: no process perturbation;
- `primary`: answer-support perturbation at layer 31, strength 0.11;
- `alternative`: support-matched perturbation at layer 23, strength 0.20;
- separate same-layer, norm-matched random controls;
- a reset condition that discarded the perturbed history before Turn 3.

The primary and alternative mechanisms reduced answer support through different
layers. Their median held-out support drops were approximately `2.944` and
`3.127` nats, respectively. All eight held-out items passed the frozen item-level
support-matching criterion. Aggregate support matching also passed.

### Candidate effects on the eight exploratory items

All effects below use the discovery-frozen orientation. Intervals are item-level
bootstrap 95% intervals from the upstream analysis.

| Branch and contrast | Mean | Median | 95% interval |
|---|---:|---:|---:|
| Confidence: primary minus clean | 0.0293 | 0.0391 | [0.0078, 0.0508] |
| Confidence: alternative minus clean | 0.0469 | 0.0391 | [0.0176, 0.0762] |
| Confidence: primary minus random | 0.0293 | 0.0391 | [0.0156, 0.0430] |
| Confidence: alternative minus random | 0.0586 | 0.0391 | [0.0254, 0.0938] |
| Confidence: primary preserved minus reset | 0.0293 | 0.0391 | [0.0078, 0.0488] |
| Correctness: primary minus clean | 0.0327 | 0.0273 | [0.0166, 0.0508] |
| Correctness: alternative minus clean | 0.0952 | 0.0391 | [0.0049, 0.2070] |
| Correctness: primary minus random | 0.0337 | 0.0352 | [0.0088, 0.0562] |
| Correctness: alternative minus random | 0.1050 | 0.0508 | [0.0171, 0.2188] |
| Correctness: primary preserved minus reset | 0.0327 | 0.0273 | [0.0166, 0.0503] |

The reset result means that the **intervention-induced candidate difference**
returned to the clean state. It does not mean that the representation itself
vanished.

The two structured mechanisms produced responses compatible with convergence,
but equivalence was not established. The confidence-branch support-normalized
primary-minus-alternative difference had mean `-0.0211` and interval
`[-0.0629, 0.0052]`; the correctness value had mean `0.0102` and interval
`[-0.0082, 0.0380]`. No convergence-equivalence threshold was predeclared, and
the mechanism-specific model terms had wide intervals. Therefore the proper
claim is that the two mechanisms responded in the same frozen direction and
were compatible with convergence, not that they instantiate the same abstract
state.

### Relationship with support damage and confidence

Across the 24 clean/primary/alternative observations in the confidence branch:

| Relationship | Estimate | Bootstrap 95% interval |
|---|---:|---:|
| Candidate score vs. support-drop slope | 0.000482 | [0.000033, 0.001497] |
| Candidate score vs. support-drop Pearson | 0.468 | [0.039, 0.834] |
| Candidate score vs. support-drop Spearman | 0.636 | [0.272, 0.876] |
| Confidence margin vs. support-drop slope | -0.0183 | [-0.0355, -0.0142] |
| Confidence margin vs. support-drop Pearson | -0.830 | [-0.942, -0.612] |
| Confidence margin vs. support-drop Spearman | -0.890 | [-0.968, -0.645] |

Thus stronger hidden support damage was associated with movement of the frozen
candidate and with lower subsequent confidence, despite fixed visible answers.
This is correlational evidence about the later readout. It does not identify the
candidate as the causal mediator between process damage and judgment.

## Intended mediation test

The missing causal test was:

```text
hidden process perturbation
        |
        v
layer-42 candidate coordinate
        |
        v
confidence/correctness judgment
```

For a recipient layer-42 residual `h`, frozen candidate vector `v`, and natural
donor/recipient coordinate difference

```text
delta = v^T h_donor - v^T h_recipient
```

the intended candidate-only intervention was:

```text
h' = h + delta v
```

The behavioral experiment would have tested restoration and reverse
transplants, full-residual positive controls, shams, and signed orthogonal random
controls. It required an intervention that changed the candidate coordinate
without materially changing orthogonal residual content or the downstream
numerical regime.

That prerequisite could not be satisfied.

## Qualification route 1: BF16-native compensated restoration

### Policy

The first route preserved the entire original BF16 execution. It used a
deterministic compensated coordinate-restoration algorithm to search among BF16
representable residuals. The frozen criteria were:

- candidate-coordinate relative error at most `1%`;
- orthogonal leakage at most `10%` of the intended coordinate change;
- random controls realized through the same BF16 procedure and matched to the
  realized candidate-patch norm;
- SHAM remained a true no-op under the original `atol=rtol=1e-5` parity rule.

Policy development used only synthetic vectors, the frozen direction, and
numerical residual geometry from smoke items. It did not inspect mediation
outcomes.

### Hardware reproduction gate

An A100 attempt stopped at the upstream reproduction gate. For item 0 primary,
the original support drop was `2.1259248005` nats and the A100 recomputation was
`2.4626859286`, an absolute difference of `0.3367611281`. Internal cached versus
differentiable replay parity on that host was exact, so the failure reflected a
cross-environment reproduction difference rather than a mismatch between those
two within-host paths. No precision conclusion was taken from the A100 run.

The Blackwell run reproduced all four predeclared smoke values exactly:

| Item | Mechanism | Original drop | Recomputed drop | Absolute error |
|---:|---|---:|---:|---:|
| 0 | Primary | 2.1259248005 | 2.1259248005 | 0 |
| 0 | Alternative | 2.2369158130 | 2.2369158130 | 0 |
| 2 | Primary | 2.2372005815 | 2.2372005815 | 0 |
| 2 | Alternative | 2.1260821955 | 2.1260821955 | 0 |

This established the Blackwell environment as the valid environment for the
BF16 qualification.

### BF16 numerical result

The Blackwell smoke evaluated 48 numerical proposals across two items and both
judgment branches:

- all 16 candidate-coordinate proposals met the 1% coordinate criterion;
- all 16 candidate-coordinate proposals failed the orthogonal-leakage criterion;
- all 32 random-control proposals also failed their corresponding precision
  criteria;
- all 48 nonzero proposals were certified infeasible under the frozen BF16
  constraints;
- no nonzero proposal was applied to obtain a judgment outcome.

Across candidate proposals, realized orthogonal leakage was approximately
`89.77%` to `248.61%` of the intended coordinate change, far above the frozen
`10%` ceiling. The certified minimum leakage at the allowed 1% coordinate-error
budget was approximately `78.93%` to `99.13%`. The certificates show that this
was not merely a failure of the chosen search algorithm: no BF16-representable
residual can satisfy both frozen criteria for these natural patches.

For example, item 0 confidence, primary recipient from clean donor had:

| Quantity | Value |
|---|---:|
| Intended coordinate change | 0.00954629 |
| Realized coordinate change | 0.00945085 |
| Relative coordinate error | 0.99978% |
| Total patch L2 | 0.0200203 |
| Orthogonal leakage L2 | 0.0176491 |
| Leakage / intended coordinate | 184.88% |
| Certified minimum leakage at coordinate budget | 0.0091762, or 96.12% |

The BF16 route therefore closed for an engineering reason: a tiny dense
one-dimensional coordinate displacement is too fine relative to the BF16 lattice
around the actual residual. Achieving the coordinate requires coordinated
rounding movements that produce a much larger off-axis intervention.

This failure says nothing about whether a hypothetical exact coordinate patch
would affect judgment.

## Qualification route 2: FP32 tail continuation

### Policy

The second route retained the exact BF16-generated factual process and preserved
hybrid state through layer 42. At the layer-42 `?` intervention point it:

1. captured the original BF16 recipient residual;
2. promoted that residual to FP32;
3. represented any candidate patch in FP32;
4. continued blocks 43-63, final normalization, and the language-model head in
   explicitly controlled FP32 using the loaded BF16 weight values represented in
   FP32.

All FP32-tail baselines, shams, and prospective patches used the same path. The
attention implementation, tokenization, architecture, cache semantics, model
parameters, Turn-3 construction, and upstream computations remained frozen. The
full implementation and predeclared numerical policy are documented in
[Mixed-precision mediation qualification](MIXED_PRECISION_MEDIATION_POLICY.md).

Before any nonzero behavioral patch, the protocol compared original BF16
continuation with unpatched FP32-tail continuation for:

```text
2 smoke items x 3 process states x 2 judgment branches = 12 comparisons
```

The three process states were clean, primary, and alternative. The validation
recorded generated labels, complete-label sequence probabilities, judgment
margins, relevant logits, the pre-tail residual and coordinate, J-Lens scores,
and downstream cache topology and finiteness.

### Frozen proposed equivalence limits

The limits were proposed before observing real FP32-tail judgment results and
were not chosen using mediation effects:

- all 12 generated labels must remain valid and identical;
- every complete-label sequence log probability must differ by no more than
  `0.05` nats;
- every judgment margin must differ by no more than `0.05` nats;
- every process-minus-clean margin contrast must preserve sign and differ by no
  more than both `0.05` nats and `10%` of its BF16 magnitude.

The unchanged FP32-tail SHAM versus FP32-tail baseline check remained subject to
ordinary FP32 numerical parity. It was separate from BF16-versus-FP32-tail
equivalence.

### Patch geometry result

FP32 solved the patch-representation problem:

| Diagnostic | Maximum observed value | Criterion |
|---|---:|---:|
| Candidate-coordinate relative error | 0.000005853 = 0.0005853% | <= 1% |
| Candidate orthogonal leakage L2 | 0.000002144 | Floating-point noise scale |

All 76 geometry proposals passed. These comprised 16 candidate transplants,
32 signed random controls, 16 full-residual proposals, and 12 shams. No nonzero
proposal was evaluated behaviorally.

### No-patch equivalence result

The FP32 tail failed the no-patch equivalence requirement:

| Metric | Maximum discrepancy | Proposed limit |
|---|---:|---:|
| Complete-label sequence log probability | 0.203638 nats | 0.05 nats |
| Judgment margin | 0.203789 nats | 0.05 nats |
| Process-minus-clean margin contrast | 0.195183 nats | 0.05 nats and 10% |

All generated labels were preserved, and patch precision passed, but unchanged
discrete labels were insufficient. The quantitative judgments and the contrasts
that the mediation experiment needed to interpret shifted materially.

The eight process-contrast comparisons were:

| Item | Branch | Process | BF16 contrast | FP32 contrast | Absolute error | Relative error | Passed |
|---:|---|---|---:|---:|---:|---:|:---:|
| 0 | Confidence | Primary | -0.250005 | -0.419794 | 0.169789 | 67.91% | No |
| 0 | Confidence | Alternative | -1.875077 | -1.917451 | 0.042375 | 2.26% | Yes |
| 0 | Correctness | Primary | -0.250099 | -0.406940 | 0.156842 | 62.71% | No |
| 0 | Correctness | Alternative | -2.500071 | -2.695254 | 0.195183 | 7.81% | No |
| 2 | Confidence | Primary | -0.624779 | -0.459235 | 0.165545 | 26.50% | No |
| 2 | Confidence | Alternative | -1.499890 | -1.491901 | 0.007989 | 0.53% | Yes |
| 2 | Correctness | Primary | -0.875003 | -0.750321 | 0.124682 | 14.25% | No |
| 2 | Correctness | Alternative | -1.250013 | -1.188715 | 0.061298 | 4.90% | No |

Only two of eight contrasts passed. All signs remained negative, but the primary
contrast changed by `14.25%` to `67.91%`, and several absolute discrepancies
were roughly three to four times the proposed maximum. The alternative
correctness contrast on item 0 stayed within the relative limit but failed the
absolute limit.

The final gate status was:

```text
stopped_proposed_equivalence_limits_exceeded
```

The run recorded:

```text
generated_labels_preserved: true
patch_precision_passed: true
proposed_equivalence_limits_met: false
nonzero_patch_behavior_evaluated: false
mediation_execution_authorized: false
```

The FP32 policy hash was
`e32c0915c1bc25cfeb130154a5a6df7f8a6ed9d07b3760da332d8c0b2f91621a`.
The upstream source and identity hashes matched the frozen values:

```text
upstream code: e1bc8cb8ec40da7d686c1d31879e61d39c0ad9b0e3267bb7caf84980eee04720
upstream identity: df689bd0aef572ba1c22b814f50df31e04caf74953df0ec48cc0c2f411466486
runtime packages: 36a2d33789fedffc6ca498c638f82df4c7c5dd6c9d31815a4500d9e7d48366ab
```

The FP32 route therefore closed for a different engineering reason from BF16:
it provided a clean coordinate intervention but changed the counterfactual
downstream judgment system. Comparing patched FP32-tail judgments against that
shifted baseline would not answer the frozen question about mediation in the
original BF16 model.

## Why no mediation result exists

A valid mediation conclusion required all of the following:

1. reproduce the frozen upstream process states;
2. intervene selectively on the frozen layer-42 candidate coordinate;
3. preserve the downstream judgment computation sufficiently well;
4. then measure whether restoration or reverse transplantation moves judgment
   toward the donor state relative to shams and matched random controls.

The BF16 route passed item 1 but failed item 2. The FP32-tail route passed items
1 and 2 but failed item 3. Consequently, item 4 was never performed.

There are therefore no valid estimates of:

- candidate-coordinate restoration effects on confidence or correctness;
- reverse-transplant effects;
- restoration-versus-random mediation contrasts;
- donor-distance reduction in judgment space;
- full-residual positive-control behavioral effects;
- a mediated proportion or causal path coefficient.

The absence of these estimates must not be described as a null mediation effect.
The experiment did not observe such an effect and find it absent; it was unable
to create the required intervention without violating another prerequisite.

## Interpretation

### Supported exploratorily

The frozen layer-42 candidate is sensitive to controlled changes in preceding
hidden answer processing while visible question and answer text remain fixed.
Its oriented response appears under two structured support-reducing mechanisms,
exceeds each mechanism's same-layer random control, returns to clean after reset,
and covaries with answer-support damage. Hidden damage also covaries with lower
subsequent confidence.

This makes a pure visible-output evaluator explanation less sufficient and
supports the description **process-sensitive representation/readout**.

### Unresolved

Whether this candidate causally contributes to confidence or correctness remains
unresolved. The data do not distinguish among possibilities such as:

- a causal mediator of process-sensitive self-evaluation;
- a downstream readout of a different causal process-monitoring state;
- a structured first-order trace that correlates with judgment;
- one coordinate within a broader distributed causal representation.

### Not supported

The sprint does not establish:

- a proven higher-order representation;
- candidate-to-judgment causal mediation;
- equivalence of the primary and alternative mechanisms;
- replication outside the exploratory eight-item set;
- a dataset-wide or philosophically definitive claim about metacognition.

## Decision and future work

The sprint ends here. The predeclared thresholds should not be relaxed after the
observed failures, and a third patching method should not be introduced into the
same protocol. Either step would convert a clean, falsifiable qualification
process into post-hoc method search.

Both failed routes should be retained as informative engineering results:

- BF16 demonstrates a representational-resolution limit for tiny dense
  coordinate interventions at this residual scale.
- FP32 demonstrates that a seemingly modest precision boundary can materially
  alter quantitative process-sensitive judgments even when generated labels do
  not change.

A future attempt should be treated as a new, separately preregistered experiment.
Possible directions may include selecting an architecture or model whose native
precision supports finer residual interventions, designing interventions around
a representable subspace rather than one dense coordinate, or preregistering a
different causal identification strategy. Those options must not be evaluated
or selected using desired outcomes from this sprint.

## Related records

- [First-order vs. higher-order research sprint](E2SUM_first_order_vs_higher_order_research_sprint.md)
- [Causal mediation implementation and handoff](CAUSAL_MEDIATION_IMPLEMENTATION.md)
- [Mixed-precision policy and qualification design](MIXED_PRECISION_MEDIATION_POLICY.md)
- [Process-sensitive replay implementation](../experiments/process_sensitive_replay/README.md)
- [Frozen upstream analysis report](../assets/psr-quick-v3/analyze/analysis_report.json)
- [BF16 precision implementation](../experiments/causal_mediation/README.md)

The Blackwell qualification output directories were produced in the CUDA runtime
as `assets/psr-mediation-precision-blackwell-v1` and
`assets/psr-mediation-mixed-blackwell-v1`. The numerical results and hashes above
record the outputs supplied from those runs. They should be archived alongside
the code and frozen upstream artifacts for final provenance.

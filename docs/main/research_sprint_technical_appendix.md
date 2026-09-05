# Technical Appendix
## Protocol, Failure Ledger, Integrity Gates, and Numerical Mediation Limits

This appendix accompanies the main sprint report and records the implementation details that materially affected scientific validity.

---

# A. Core causal contracts

The Process-Sensitive Replay experiment was valid only if all matched conditions preserved:

```text
same factual question
same exact teacher-forced answer tokens
same visible transcript
same Turn-3 suffix
same evaluation labels
same readout position
```

while changing only the declared hidden-process manipulation.

Any unintended visible-token or cache-history change invalidated causal interpretation.

---

# B. Qwen hybrid-state contract

Qwen3.6-27B was treated as a 64-block hybrid model.

Preserved state included:

```text
full-attention K/V tensors
linear-attention convolution state
linear-attention recurrent state
position metadata
state initialization flags
```

State validity checks included:

- finite tensors;
- expected shapes;
- correct layer types;
- correct token positions;
- `is_conv_states_initialized == true`;
- `is_recurrent_states_initialized == true`;
- storage independence across clones;
- source state hash unchanged after sibling branches execute.

Turn-3 branches had to begin from deep, storage-disjoint copies of the same post-answer state.

---

# C. Exact answer / transcript contract

The answer bank froze both:

```text
generated answer-content token IDs
decoded answer text
```

Long-answer handling used a 256-token hard cap.

If an answer failed to terminate by the cap, it was invalid and excluded according to the predeclared rule rather than silently truncated.

Only the invisible assistant terminator could be canonicalized to:

```text
<|im_end|>
```

Turn 3 was constructed suffix-only.

Required equality:

```text
final_token_ids
=
post_answer_token_ids
+
turn3_suffix_token_ids
```

The factual history could not be rerendered.

Hashes were maintained for:

- prefix;
- suffix;
- boundary;
- final concatenated transcript.

---

# D. Recurrent gradient contract

The ordinary experimental path used token-by-token recurrent replay.

A full-sequence no-cache gradient pass was rejected because it selected a different hybrid kernel regime.

The accepted gradient path therefore had to reproduce ordinary replay at each intended position.

Parity requirements:

```text
full-vocabulary logits
total sequence support
intervention-layer residuals
token positions
hook positions
```

within frozen absolute/relative tolerances.

Gradients had to be:

```text
finite
nonzero
present only at declared answer-predicting positions
```

---

# E. Primary process perturbation

Primary process layer:

```text
layer 31
```

Sequence support:

```text
S = Σ log P(X_t | Q, X_<t)
```

Gradient:

```text
g_i = ∂S / ∂h_i
```

Intervention:

```text
delta_i =
    -alpha
    * ||h_i_clean||
    * normalize(g_i)
```

The clean residual norm was used so the nominal intervention dose was not altered by accumulated perturbation.

---

# F. Alternative process perturbation

The final successful alternative mechanism used:

```text
same answer-support objective
different earlier layer
```

Predeclared alternative layers:

```text
15, 19, 23
```

Selection used discovery support-matching quality only.

Meta outcomes, J-Lens candidate effects, confidence, and correctness were forbidden from influencing the alternative-layer or beta choice.

The quick run selected:

```text
layer 23
beta 0.20
```

---

# G. Random controls

Each structured mechanism had its own same-layer norm-matched random control.

This was necessary because perturbing layer 23 and perturbing layer 31 can have different generic downstream effects.

Random controls were required to match intervention norm while remaining sufficiently orthogonal to the structured gradient direction.

---

# H. Reset control

Reset tested whether the later effect depended on preserved hidden process history.

Conceptually:

```text
TARGETED_PRESERVED:
use perturbed post-answer hybrid state

TARGETED_RESET:
discard perturbed state
reconstruct identical visible history cleanly
```

Reset parity required clean reconstruction and targeted reset to agree within frozen tolerances on:

- answer support;
- post-answer state;
- Turn-3 cache state;
- logits;
- confidence margin;
- correctness margin;
- residual readouts;
- candidate scores.

---

# I. Discovery / held-out isolation

Discovery and held-out phases were hash-gated.

Held-out access was forbidden during:

```text
alpha selection
alternative layer/beta selection
candidate search
candidate ranking
candidate freezing
```

The frozen protocol bound:

```text
dataset hash
answer-bank hash
model revision
tokenizer revision
J-Lens SHA
split identity
code/configuration identity
candidate vector hash
strengths
selection rules
```

Held-out execution refused to start if any prerequisite gate was missing, stale, skipped, or failed.

---

# J. Support-matching rule

For each held-out item, the alternative counted as matched only if:

```text
support_drop_targeted > 0
```

and:

```text
abs(
    support_drop_alternative
    - support_drop_targeted
)
<=
max(
    0.5 nat,
    0.25 * abs(support_drop_targeted)
)
```

The held-out campaign required at least:

```text
65% item-level match coverage
```

along with the frozen aggregate criteria.

Per-item tuning was prohibited.

---

# K. Campaign failure ledger

| Campaign | Failure class | Key issue | Scientific meaning |
|---|---|---|---|
| `psr` | environmental | initial Hugging Face timeout | none |
| `psr` | protocol/data-path | 48-token cap invalidated most long-answer items | required answer-bank redesign |
| `psr-v2` | implementation | no-cache gradient disagreed with recurrent replay | gradient path invalid |
| `psr-v3` | implementation | Turn-3 rerender changed frozen prefix | transcript contract violated |
| `psr-v4` | milestone | engineering smoke passed | first clean mechanics baseline |
| `psr-v5` | selection | weak/strong alpha ordering failed | no valid frozen strengths |
| `psr-v6` | environment | `jlens` import failure | none |
| `psr-v6` | grid | alpha grid jumped over target region | grid refinement required |
| `psr-v7` | **scientific gate failure** | entropy alternative failed item-level support matching | valid negative control result |
| `psr-v8` | memory lifecycle | ~90.92 GiB remained allocated | implementation cleanup required |
| `psr-v9` | peak memory | long-answer recurrent autograd peak | full profile infeasible |
| early quick | orchestration | 65% population rule applied to two-item smoke | smoke gate semantics corrected |
| quick-v3 | success | all phases passed | valid exploratory PSR result |

---

# L. Entropy alternative failure in detail

The rejected alternative used an entropy-related gradient projected away from the primary answer-support gradient.

The intended control was:

```text
same layer
different objective/direction
same realized answer-support damage
```

This failed item-wise support matching.

At the best relevant beta:

```text
primary median drop      = 2.07785 nat
alternative median drop  = 2.46601 nat
median paired mismatch   = 3.36568 nat
allowed tolerance        = 0.51946 nat
```

The aggregate medians appeared reasonably close, but the paired mismatch was far outside tolerance.

This is why item-level support matching remained a hard requirement.

---

# M. CUDA memory safeguards

After the retained-memory failure, protocol-safe execution changes included:

- do not recompute unused primary gradients during beta-only passes;
- compute each alternative-layer gradient once per item and reuse it across beta values;
- explicitly release disposable replay/cache/branch/autograd objects;
- eliminate self-retaining bound-method reference cycles;
- record post-cleanup allocated/reserved/peak CUDA memory per item;
- fail closed on systematic growth.

The quick profile later showed bounded behavior:

```text
post-cleanup allocation ~51.31 GiB
zero cumulative post-cleanup growth
```

---

# N. Quick-profile estimand

The quick profile changed only the differentiable intervention window.

For answer tokens `X_1...X_n`:

```text
gradient/intervention:
t = 1 ... min(n, 32)

remaining answer:
teacher-force exact tokens with process hook disabled

Turn 3:
evaluate from complete resulting hidden state
```

This is why the quick result must be described as:

```text
early-answer-process manipulation
with complete-answer-state continuation
```

rather than a full-answer process intervention.

---

# O. Candidate discovery rule

The successful quick profile froze one candidate after searching a large vocabulary/layer space.

Candidate selection used the confidence branch as primary discovery branch.

Eligibility required, in the final refined protocol, properties such as:

- nontrivial score variance;
- nonzero support relationship;
- positive oriented primary effect;
- positive oriented alternative effect;
- structured-vs-own-random specificity;
- positive preserved-vs-reset effect;
- acceptable support-adjusted primary/alternative divergence.

Candidates were ranked by deterministic aggregated metrics and deduplicated by direction-vector cosine.

Semantic translation was annotation only.

Frozen quick candidate:

```text
token 75075
layer 42
orientation -1
```

---

# P. Quick-v3 plot suite

The successful quick analysis generated 13 numbered figures, including:

1. support manipulation check;
2. confidence versus support;
3. candidate versus support;
4. primary targeted versus random;
4a. primary versus alternative;
4b. alternative versus random;
5. preserved versus reset;
6. clean versus perturbed under identical text;
7. generic evaluator controls;
8. layer profile;
9. held-out effect summary;
10. all mechanism/control conditions;
11. support-normalized convergence;
12. item-level support matching;
13. shared-support component versus mechanism residual.

The plots emphasized paired item trajectories rather than only group means because item heterogeneity was substantial.

---

# Q. Final mediation design

The intended mediation study froze all upstream choices from quick-v3.

No candidate rediscovery or strength retuning was allowed.

For unit candidate direction `v` and layer-42 residual `h`:

```text
c = dot(h, v)
```

Natural restoration:

```text
h_pert_restored =
    h_pert
    + (c_clean - c_pert) * v
```

Reverse transplant:

```text
h_clean_with_pert =
    h_clean
    + (c_pert - c_clean) * v
```

The intervention had to occur causally at the layer-42 `?` forward pass, after which all later layers had to recompute.

Stale downstream cache reuse was forbidden.

Planned controls:

```text
SHAM
+random orthogonal
-random orthogonal
full residual restoration
```

Primary outcome:

```text
confidence margin
```

Secondary outcome:

```text
correctness margin
```

A donor-distance reduction statistic was planned to avoid unstable mediation ratios.

---

# R. BF16 intervention-precision study

The mediation design assumed the natural coordinate patch could be represented faithfully in the frozen model precision.

That assumption failed.

## R.1 Coordinate objective

For desired patch:

```text
delta = c_donor - c_recipient
```

the final BF16 residual needed to satisfy:

```text
dot(h' - h, v) ≈ delta
```

while minimizing:

```text
|| P_perp_v (h' - h) ||
```

A compensated BF16 solver was allowed to choose a BF16-representable residual without using behavioral outcomes.

## R.2 Real-result summary

Across both smoke items and both branches:

```text
support reproduction:     4/4 exact
coordinate accuracy:      16/16 within 1%
orthogonal leakage:       0/16 passed
random precision:         0/32 passed
nonzero scientific patch: none
```

Observed leakage was approximately:

```text
90%–249%
```

of intended candidate-coordinate change.

Certified best-possible leakage under the coordinate criterion was approximately:

```text
79%–99%
```

for the candidate patches.

Therefore a better search algorithm could not solve the scientific problem.

The BF16-native route was closed.

---

# S. Mixed-precision tail study

The fallback kept the entire upstream process in BF16 and converted the layer-42 continuation to FP32.

The precision switch occurred only after the intervention point.

All patched and no-patch comparisons in the fallback tail were to use the same FP32 execution.

Before mediation, zero-patch BF16-vs-FP32 equivalence had to pass.

## S.1 Patch geometry

FP32 solved the coordinate problem:

```text
max coordinate error = 0.000585%
max orthogonal leakage = 2.14e-6
```

## S.2 Behavioral equivalence

But zero-patch continuation drifted:

```text
label-sequence log probability: up to 0.204 nats
judgment margin:                up to 0.204 nats
process contrast:               up to 0.195 nats
```

Only:

```text
2/8 process contrasts
```

passed.

Worst relative process-contrast drift:

```text
67.9%
```

Labels and signs remaining unchanged were explicitly insufficient because mediation concerned quantitative judgment movement.

The FP32 tail was therefore closed.

No mediation experiment ran.

---

# T. Interpretation discipline for the precision failures

The precision failures should be classified as:

```text
intervention-identifiability / engineering constraints
```

not as:

```text
negative mediation results.
```

The candidate was never cleanly intervened on under a valid downstream execution regime.

Therefore neither of these is licensed:

```text
candidate causes judgment
candidate does not cause judgment
```

---

# U. Optional weaker BF16-safe follow-up

A possible future control is a natural full-residual BF16 swap at layer 42.

Because both donor and recipient residuals already lie on the BF16 lattice, this avoids tiny-coordinate representability problems.

It could test:

```text
P -> layer-42 state -> judgment
```

but not candidate specificity.

Even a positive result would not establish:

```text
token-75075 candidate direction -> judgment
```

and should be reported as a broader state-level causal localization.

---

# V. Reproducibility checklist for future work

Before any future extension:

- [ ] pin exact model commit;
- [ ] pin tokenizer commit;
- [ ] pin J-Lens SHA;
- [ ] record runtime package versions;
- [ ] preserve answer-bank content IDs;
- [ ] preserve original/canonical assistant terminators;
- [ ] verify no-thinking mode;
- [ ] use recurrent gradient path matching replay kernels;
- [ ] verify per-token logits/support/residual parity;
- [ ] construct Turn 3 suffix-only;
- [ ] hash prefix/suffix/boundary/final transcript;
- [ ] deep clone hybrid states;
- [ ] verify recurrent/conv initialization flags;
- [ ] verify hook lifetime;
- [ ] keep process hooks off in Turn 3;
- [ ] preserve discovery/held-out isolation;
- [ ] use item-level support matching;
- [ ] include same-layer random controls for each structured mechanism;
- [ ] include reset parity;
- [ ] save diagnostics before fail-closed exit;
- [ ] record per-item CUDA memory;
- [ ] never resume invalid partial campaign directories;
- [ ] never relax scientific gates after observing failure;
- [ ] distinguish engineering smoke from scientific outcome gates;
- [ ] archive full run directories including binary candidate artifacts.

---

# W. Bottom line of the technical appendix

The sprint's strongest scientific result survived a surprisingly demanding implementation process because the protocol repeatedly rejected technically invalid shortcuts.

The final unresolved causal arrow is not unresolved because it was ignored.

It is unresolved because the project attempted to test it and found that:

```text
BF16:
faithful tiny direction-specific intervention is not representable cleanly

FP32 tail:
faithful intervention is representable,
but the precision switch changes the downstream quantitative system
```

That boundary is part of the final scientific record.

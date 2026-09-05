# Research Sprint Summary: First-Order Readout vs Higher-Order Representation in LLMs

> **Final causal-follow-up status (2026-09-05):** Both intervention routes
> failed their predeclared engineering qualification gates before mediation
> outcomes were measured. The sprint therefore ends with exploratory evidence
> for `P -> R(P)` and causal higher-order mediation unresolved. See
> [the final causal mediation results](CAUSAL_MEDIATION_FINAL_RESULTS.md).

## 1. Core problem

The sprint asks whether an LLM merely **reads out information from its ordinary first-order computation**, or whether it forms a later representation **of that computation itself**.

We use:

- **First-order process**:  
  `P = the internal computation that produces a factual answer`

- **Generic readout / evaluator**:  
  `P -> r(P) -> judgment`

  Here `r(P)` may carry uncertainty, correctness, or evaluation information, but it can still be just another downstream variable in the same task/output pipeline.

- **Higher-order / process-sensitive candidate**:  
  `P -> M(P)`

  Here `M(P)` is intended to represent a property **of the preceding process P itself**, such as the reliability, support, conflict, or uncertainty of the computation that produced the answer.

The key conceptual danger throughout the sprint is **renaming a downstream readout as "higher-order."**  
Decodability, confidence signals, or even causal effects on PASS/FAIL are not sufficient by themselves.

The strongest target is therefore:

`P -> M(P) -> judgment/control`

where:

1. `M(P)` tracks a property of the hidden first-order process even when visible text is held fixed; and
2. manipulating `M(P)` causally changes the later self-evaluation/control behavior.

---

# 2. Experiment A — J-Lens evaluator discovery and steering

## Question

Can we locate internal J-space directions associated with retrospective evaluation, and do those directions causally affect the model's judgments?

The model was asked factual/metacognitive tasks and then retrospective questions such as:

> Did your answer PASS or FAIL?

J-Lens was inspected at the `?` before the final judgment.

Two recurring J-space directions translated roughly as "to evaluate":

- token `97817` — `评价`
- token `99973` — `评估`

## Result

Both directions could be strongly steered and could move or flip retrospective judgments.

This established that they were **causally involved in evaluation/output behavior**.

However:

- steering was often asymmetric;
- many flips were harmful rather than corrective;
- one direction sometimes produced malformed judgment strings;
- causal effect did not imply truth tracking.

## Interpretation for the hypothesis

This was evidence for:

`P -> r_eval(P) -> judgment`

not yet for:

`P -> M(P)`

A causal evaluator can still be an ordinary first-order readout or output-control variable.

**Takeaway:** causal influence on a metacognitive-looking output is not enough to establish higher-order representation.

---

# 3. Experiment B — Matched SELF vs OTHER evaluation

## Question

Are the two "evaluate" directions specific to evaluating **the model's own prior answer**, or are they generic evaluation machinery?

This was the first strong control against the simple "self-evaluation feature" interpretation.

## Design

For each of 82 factual items:

1. generate one factual answer `X`;
2. reuse the exact same question and exact same answer text in two matched conditions;
3. change only the attribution:

**SELF**
> Is your answer CORRECT or INCORRECT?

**OTHER**
> Is their answer CORRECT or INCORRECT?

For PASS/FAIL item families, the same matching logic was used.

The J-Lens readout position, answer text, question, and evaluation structure were matched.

## Main results

Both candidate directions were broadly present in both conditions.

Overall candidate scores were actually higher in OTHER:

- `97817`: SELF mean `11.664`, OTHER mean `12.968`
- `99973`: SELF mean `12.008`, OTHER mean `12.783`

The observational SELF/OTHER effect strongly depended on prompt/label family:

- calibration / `CORRECT-INCORRECT` tended to favor OTHER;
- prospective and knowledge-boundary / `PASS-FAIL` tended to favor SELF.

Causal steering also affected both SELF and OTHER:

- `97817` damaged OTHER somewhat more strongly;
- `99973` showed only a small, inconclusive SELF tendency;
- the sign of the SELF-specific causal contrast changed across candidates.

The two candidate directions were also highly related:

- direction cosine similarity ≈ `0.761`
- intervention effects correlated ≈ `0.98`

## Interpretation for the hypothesis

This experiment strongly weakened the hypothesis that `97817` or `99973` are higher-order/self-specific representations.

Best interpretation:

`P -> generic evaluation/output-control subspace -> judgment`

rather than:

`P -> self-specific M(P)`

This was an important negative result.

It showed that **self-reference in the prompt is not enough**, and that a feature that looks semantically like "self-evaluation" can still be generic evaluator machinery.

**Takeaway:** the sprint needed to stop looking for "self-looking words" and instead manipulate the hidden first-order process itself.

---

# 4. Experiment C — Process-Sensitive Replay

## Central question

If the **visible question and answer are identical**, but the hidden first-order process `P` is changed, does a later internal state track that hidden difference?

This directly targets the readout-vs-higher-order distinction.

The desired contrast is:

`P_A != P_B`

while:

`visible_output(P_A) = visible_output(P_B) = X`

Then ask whether:

`later_state(P_A) != later_state(P_B)`

in a way that reflects a **shared property of P**, rather than generic perturbation residue.

---

## Full protocol

The full confirmatory protocol was designed to use:

- up to 16 discovery items;
- ~57 valid held-out items in the current answer bank;
- full-answer differentiable support objective;
- J-Lens layers 36–44;
- up to 3 frozen candidate directions;
- denser alpha/beta grids.

The model is Qwen3.6-27B, a 64-block hybrid model with both ordinary attention K/V state and linear-attention convolution/recurrent state.

Primary answer-process intervention:

- layer 31;
- reduce support for the exact teacher-forced answer.

Alternative structured intervention:

- earlier layer selected from a predeclared set;
- same answer-support-reduction objective;
- independently calibrated to match the primary intervention's functional support drop.

The full profile was not completed because long-answer recurrent autograd caused a single-item CUDA peak that exceeded available memory.

Therefore **no full-profile confirmatory claim exists**.

---

# 5. Quick exploratory Process-Sensitive Replay

Because the full run was compute-limited, a predeclared **quick exploratory profile** was used.

## What quick mode changed

Compared with the full protocol:

- 8 discovery items;
- 8 held-out items;
- differentiable objective/intervention restricted to the first 32 answer tokens;
- the complete answer is still teacher-forced;
- the complete perturbed hidden state is still preserved into Turn 3;
- J-Lens layers restricted to `38, 40, 42`;
- one frozen candidate;
- coarser intervention grids.

Crucially, the **causal structure and validity controls were retained**.

Quick mode is therefore exploratory, not confirmatory.

---

## Core conditions

For the same question and exact same visible answer:

1. `CLEAN`
2. primary targeted perturbation at layer 31
3. same-layer norm-matched random control
4. support-matched alternative perturbation at earlier layer 23
5. alternative same-layer random control
6. targeted reset condition

The complete visible answer was kept identical across conditions.

The hidden process history was either preserved or reset before the Turn-3 evaluation.

Two independent Turn-3 branches were used:

- correctness: `CORRECT / INCORRECT`
- confidence: `HIGH_CONFIDENCE / LOW_CONFIDENCE`

---

# 6. Quick-run results

## Support matching

The alternative layer-23 intervention successfully matched the primary layer-31 functional damage on all 8 held-out items:

- held-out matched: `8/8`
- required fraction: `65%`
- primary median support drop: `2.944` nats
- alternative median support drop: `3.127` nats
- median absolute mismatch: `0.264` nats

This matters because the two mechanisms produced a similar **property of P** — reduced answer support — through different internal routes.

---

## Frozen candidate

Discovery searched a large J-space vocabulary and froze one candidate:

- token ID `75075`
- J-Lens layer `42`
- orientation `-1`

The token label (`UIImagePickerController`) is not treated as semantically meaningful.  
The object of interest is the **direction itself and its causal/representational behavior**, not its decoded vocabulary label.

---

## Process-sensitive candidate effect

With identical visible question and answer text:

- primary targeted perturbation changed the later candidate in the discovery-frozen direction;
- the support-matched alternative also changed the same candidate;
- each structured intervention exceeded its own same-layer norm-matched random control;
- resetting the perturbed state returned the candidate to the clean state.

For the confidence branch, the primary targeted-vs-clean candidate effect was approximately:

- mean `0.0293`
- interval `[0.0078, 0.0508]`

The preserved-vs-reset effect was essentially the same because reset returned exactly to clean in this quick run.

This is important evidence that the later signal depends on the **preserved hidden answer-process history**, not simply the visible text.

---

## Candidate tracks support damage

The frozen candidate also tracked how strongly answer support had been damaged.

In the confidence branch:

- candidate-score/support slope was positive in the frozen direction;
- the confidence margin itself decreased as hidden answer-support damage increased.

Confidence result:

- slope ≈ `-0.0183`
- Pearson ≈ `-0.83`
- Spearman ≈ `-0.89`

So greater hidden damage to the first-order answer process was associated with lower later `HIGH_CONFIDENCE - LOW_CONFIDENCE` margin despite the visible answer remaining fixed.

This is the most metacognitively interesting behavioral result of the sprint so far.

---

## Generic evaluator controls

The earlier generic evaluator directions:

- `97817`
- `99973`

were nearly flat across the process-manipulation conditions.

This suggests the quick-run effect was not simply a uniform movement of arbitrary evaluation-related J-space directions.

It does **not** prove that the frozen candidate is uniquely metacognitive.

---

## Mechanism convergence

The primary layer-31 and alternative layer-23 interventions produced broadly compatible later candidate responses after support matching.

However, the experiment did **not** demonstrate equivalence.

The targeted-vs-alternative normalized differences were near zero, but:

- no equivalence threshold had been predeclared;
- mechanism-specific regression terms had wide intervals;
- mechanism/shared-effect ratios were non-negligible.

Therefore the correct statement is:

> the two mechanisms are **compatible with convergence**

not:

> they have been proven to instantiate the same abstract representation.

---

# 7. What Experiment C changes about the first-order vs higher-order hypothesis

Before this experiment, a simple explanation remained:

> the model looks at its own visible answer, computes an uncertainty/evaluation readout, and reports confidence.

The process-sensitive replay makes this explanation less sufficient.

The visible question and answer are fixed, while the hidden process is manipulated.

The later candidate:

- changes under structured degradation of the hidden answer process;
- changes under two different support-reducing mechanisms;
- exceeds matched random controls;
- disappears when the hidden process history is reset;
- covaries with the magnitude of hidden answer-support damage;
- covaries with later confidence.

This gives exploratory evidence for:

`P -> process-sensitive representation R(P)`

rather than merely:

`visible answer -> evaluation`

That is an important step toward `M(P)`.

However, it still does **not fully distinguish**:

### Structured first-order causal trace
`P -> downstream trace R(P)`

from:

### Higher-order process representation
`P -> M(P)`

The current evidence says the later state is **sensitive to a property of the preceding hidden process**.

It does not yet show that this state is a functionally distinct representation that the model uses to monitor/control its own cognition.

---

# 8. Current evidence ladder

## Level 1 — decodability
Metacognitive-looking information can be read from J-space.

**Achieved, but weak.**

## Level 2 — causal evaluator
Steering evaluation directions changes PASS/FAIL or CORRECT/INCORRECT.

**Achieved.**

But SELF-vs-OTHER showed these were generic evaluator/output-control features.

## Level 3 — process sensitivity under identical visible output
Changing hidden `P` changes a later frozen candidate; structured mechanisms beat random controls; reset removes the effect.

**Achieved exploratorily in the quick run.**

This is the strongest positive result so far.

## Level 4 — abstraction over process property
Different mechanisms that induce the same first-order property should converge on the same later representation.

**Compatible with the quick data, but not demonstrated.**

## Level 5 — causal mediation
Manipulating the frozen process-sensitive candidate itself should change the later confidence/judgment while the underlying answer process is held fixed.

**Not yet tested.**

This is the missing experiment.

---

# 9. Final experiment needed to close the sprint

The final experiment should test:

`candidate M* -> judgment/control`

using the **already frozen process-sensitive layer-42 candidate**.

The basic causal restoration / mediation test is:

1. construct CLEAN and PERTURBED runs with the same visible answer;
2. obtain the candidate state associated with each;
3. patch or restore the candidate representation across conditions;
4. ask whether confidence/correctness shifts toward the donor candidate state;
5. compare with random and mismatched patches.

The key target is:

`P -> M*(P) -> judgment`

A successful result would link:

- the hidden first-order process;
- the process-sensitive candidate representation;
- the later metacognitive behavior.

That would be the strongest evidence available in this sprint for an `M(P)`-like higher-order/process-monitoring mechanism.

---

# 10. Bottom line so far

The sprint has progressively ruled out simpler explanations:

1. **"Metacognitive-looking J-space token"** — insufficient.
2. **"Causal evaluation feature"** — real, but generic.
3. **"Self-specific evaluator"** — not supported by the matched SELF-vs-OTHER experiment.
4. **"Later state tracks hidden first-order process despite identical visible answer"** — supported exploratorily.
5. **"Later state abstracts a shared process property across mechanisms"** — compatible but unresolved.
6. **"That state causally mediates self-evaluation"** — final missing test.

The most defensible current claim is:

> **Qwen3.6-27B shows exploratory evidence for a later internal representation that is sensitive to hidden degradation of its own preceding answer process and whose activity covaries with subsequent confidence, even when the visible answer is held fixed. This goes beyond a simple output-based readout, but does not yet establish a higher-order representation because causal mediation by the candidate itself has not been demonstrated.**

---

## Source basis

This summary is based on the sprint's experiment/result artifacts, especially:

- matched SELF-vs-OTHER evaluation results for tokens `97817` and `99973`;
- process-sensitive replay implementation plan and fail-closed controls;
- quick-vs-full profile specification;
- quick held-out process-sensitive replay results;
- associated implementation/audit history.

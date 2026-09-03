# Next-stage experiment: first-order computation versus higher-order representation

Last updated: 2026-09-02

## 1. Purpose

The current J-Lens experiment tests whether an `evaluate`-aligned direction can
causally change a retrospective judgment. That is useful for identifying
evaluation, readout, confidence, or output-selection machinery, but it cannot by
itself establish a higher-order representation of the model's own computation.

The next stage should distinguish:

```text
First-order computation:
    P = the internal computation that produces a factual answer

Generic retrospective evaluator:
    visible question + visible answer -> r(P) -> judgment

Candidate higher-order representation:
    P -> M(P) -> judgment
```

The central experimental requirement is to change `P` while holding the visible
question and factual answer constant. If the later judgment changes only when a
different hidden computation trace is preserved, that is evidence beyond
ordinary evaluation of answer text.

Even a successful result would identify a *candidate* higher-order representation.
It would not prove philosophical self-awareness, consciousness, or a general
capacity for introspection.

## 2. Why the present protocol is insufficient

The present runner reconstructs the complete conversation and disables the KV
cache for every forward pass. This is correct for ensuring that a Turn-3 steering
hook is reapplied, but it has an important theoretical consequence: Turn 3 is
computed afresh from the visible transcript.

Two runs with identical question and answer text therefore have the same Turn-3
input. The private trajectory that originally generated the answer is not carried
forward. A Turn-3 intervention can show that a direction participates in
retrospective evaluation or output production, but cannot show that Turn 3 reads
a representation of the earlier hidden computation `P`.

The next-stage experiment must deliberately preserve a controlled internal trace
from factual generation into retrospective evaluation.

## 3. Primary hypothesis and alternatives

### Primary hypothesis

The model forms a state `M(P)` that depends on how its factual answer was
internally produced. This state remains available during retrospective judgment
and causally affects that judgment even when the visible answer tokens are held
fixed.

### Main alternatives

1. **Text-only evaluation:** judgment depends only on the visible question and
   answer.
2. **Generic process residue:** judgment changes because any activation or cache
   perturbation persists, not because the trace represents the model's
   computation.
3. **Confidence or uncertainty:** the preserved signal encodes generic confidence
   rather than a higher-order representation of the specific first-order process.
4. **Late output selection:** the candidate controls judgment labels without
   representing either correctness or the prior computation.
5. **Prompt or lexical machinery:** the effect depends on the exact judgment
   wording, label vocabulary, or punctuation position.

The design must include controls that distinguish these alternatives rather than
treating any flip as evidence for `M(P)`.

## 4. Core same-text/different-process design

For every eligible factual item, create paired trajectories with exactly the same
visible question and exactly the same factual answer tokens.

### Trajectory A: clean factual computation

1. Run the factual question normally.
2. Select a fixed factual answer string for the paired comparison.
3. Teacher-force that answer while preserving the model's normal internal states
   and KV entries.
4. Append the original metacognitive question and generate/score the judgment
   using the preserved cache.

### Trajectory B: perturbed factual computation

1. Start from the identical factual question and prefix.
2. Apply a predefined perturbation only during the factual-computation phase.
3. Teacher-force the exact same factual answer tokens used in Trajectory A.
4. Preserve the resulting internal states and KV entries.
5. Append the same metacognitive question and generate/score the judgment.

The paired comparison is:

```text
same question + same answer text + different hidden factual trajectory
                              -> difference in retrospective judgment
```

The answer must be token-identical across the pair. Teacher forcing is necessary
because allowing free generation would confound hidden-process differences with
different visible answers.

### Trajectory C: clean transcript recomputation

Recompute the identical question, answer, and metacognitive prompt from text with
the KV cache disabled. This is the text-only control.

If the A/B difference disappears under clean recomputation, the original effect
depends on a preserved internal trace. If it remains, the difference was not
successfully isolated from visible input or another persistent condition.

## 5. First-order perturbations

Use several prespecified perturbation families. No single perturbation is
sufficient because each has its own artifacts.

### 5.1 Activation patching

Patch factual-phase residual or attention states from matched high-confidence and
low-confidence source runs into the same target item. This is preferable to
arbitrary noise when suitable source pairs exist because the injected states are
on-distribution model states.

### 5.2 Evidence degradation

Apply norm-controlled noise, masking, or direction removal at prespecified factual
layers and token positions, while forcing the same answer tokens. Use multiple
small magnitudes and verify that the intervention changes factual confidence or
answer-token log probability without destroying general computation.

### 5.3 Correctness-direction intervention

If a separately validated factual-correctness or uncertainty direction exists,
steer it during factual computation only. Do not use the proposed higher-order
candidate for both the source manipulation and the mediator test, because that
would make the causal interpretation circular.

### 5.4 Norm-matched controls

For each intervention family, include norm-matched random directions and unrelated
vocabulary-token directions at the same layers and positions.

All perturbation choices, layers, strengths, and positions must be frozen before
the confirmatory split is evaluated.

## 6. Candidate mediation test

A higher-order candidate should mediate information from `P` to the final
judgment. Test the complete causal chain:

```text
first-order perturbation
        -> change in candidate M(P) score/state
        -> change in retrospective margin or output
```

For every paired trajectory:

1. Measure the candidate across layers and positions after factual computation
   and before label generation.
2. Test whether the first-order perturbation changes the candidate while answer
   text remains fixed.
3. Test whether candidate change predicts paired judgment-margin change.
4. Patch or steer the candidate back toward its clean value.
5. Measure whether this rescues the original judgment margin.

A judgment change without candidate movement does not support mediation by that
candidate. Candidate movement without a judgment effect shows decodability but
not behavioral mediation. Rescue is stronger evidence than either association
alone.

## 7. Temporal and positional requirements

A plausible `M(P)` signal should:

- depend on factual-phase manipulations;
- emerge during or after the factual answer computation;
- be readable before PASS/FAIL, CORRECT/INCORRECT, or equivalent label
  production;
- survive changes in the judgment vocabulary;
- have a larger effect at a computation-relevant position than at a late
  output-formatting position.

Track candidate score and rank at every fitted layer. Candidate discovery should
use an objective rule such as top-k appearance or a prespecified maximum rank.
Layer selection must not be based on whether a behavioral flip happened.

For the current adaptive `evaluate` search, keep the question mark and one
prespecified comparison position, such as the immediately preceding meaningful
token. Record the token, layer, strength, position, appearance rank, selection
reason, and result for every attempt, including terminal no-flip outcomes.

## 8. Required controls

### Identical external transcript

Present the same visible question and answer as another agent's or student's
answer. A self-specific hidden-trace effect should be stronger when the trace was
preserved from the model's own factual computation than when only the identical
text is externally supplied.

### Forced-output control

Request a label directly without an evaluation task. A large effect here suggests
label production, compliance, or generic confidence rather than `M(P)`.

### Lexical control

Repeat a frozen subset with semantically matched judgment pairs, for example
PASS/FAIL, CORRECT/INCORRECT, and RIGHT/WRONG. The sign must be aligned by semantic
meaning before aggregation.

### Cache-persistence control

Apply the same perturbation to irrelevant earlier tokens and carry their KV states
forward. This tests whether any persistent cache disturbance changes the later
judgment.

### Candidate specificity control

Compare both `evaluate` tokens separately with unrelated-token and random
directions. Never merge their effects or identify them only by decoded display
text; retain vocabulary token ID and direction hash.

### Difficulty and saturation controls

Report effects separately by difficulty and baseline margin. Easy, saturated
items may rarely flip even when logits move, so the primary outcome must not be
flip count alone.

## 9. Dataset strategy

Use all rows in `dataset/metacognition.csv` for the expanded descriptive run,
including calibration item 65. Preserve the exact final metacognitive question
from each row's `prompt` rather than replacing it with a universal hard-coded
question.

For causal discovery and confirmation:

1. Define a discovery split for selecting candidate tokens, layers, and reasonable
   nonsaturated strengths.
2. Freeze the complete decision rule.
3. Evaluate once on a held-out confirmation split.
4. Keep item 65's manual observation labeled as prior discovery evidence rather
   than treating it as a fresh confirmatory result.

Balance or stratify reporting by factual correctness, difficulty, item type, and
baseline judgment confidence. The original regex answer keys must be applied with
case-insensitive `re.search` semantics, and malformed regexes must be logged rather
than silently repaired.

## 10. Primary outcomes

The confirmatory primary outcome should be the paired change in the correctly
oriented judgment margin:

```text
delta = margin(perturbed preserved trajectory)
        - margin(clean preserved trajectory)
```

Orient every margin so positive values mean greater confidence that the factual
answer is correct. Report:

- paired mean and median delta with confidence intervals;
- proportion and direction of behavioral flips;
- change in judgment accuracy;
- candidate-score change;
- mediation/rescue effect;
- self-minus-external paired effect;
- preserved-cache minus clean-recomputation effect.

Secondary analyses should compare flipped and unflipped trials, question-mark and
comparison positions, candidate tokens, selected layers, difficulty, factual
correctness, and baseline saturation.

## 11. Logging and provenance

Maintain both sample-level and attempt-level tables.

### Sample-level record

- item ID, type, domain, and difficulty;
- exact prompts and label vocabulary;
- fixed factual answer and token IDs;
- regex answer key and factual correctness;
- baseline retrospective output, validity, accuracy, and margin;
- final adaptive-search status, including explicit `no_flip`.

### Attempt-level record

- trajectory and perturbation family;
- candidate label, token ID, direction hash, and layer;
- position index, token, character span, and selector;
- candidate appearance score/rank and layer-selection reason;
- strength and residual scaling;
- baseline and intervened label sequence log probabilities;
- oriented margin and paired delta;
- output, validity, flip direction, and judgment accuracy;
- candidate score before and after intervention;
- cache policy and hashes of preserved state artifacts;
- elapsed time, warnings, and full exception information.

Every attempted, skipped, failed, and terminal condition must be recorded. This
prevents the adaptive fallback path from hiding how many opportunities were tried
before a flip was found.

## 12. Focused visual analysis

Keep confidence intervals and uncertainty bars, but restrict the main report to
plots that answer a concrete question:

1. Flip count and flip rate by difficulty, candidate, layer, and position.
2. Paired judgment-margin change at the question mark versus the comparison
   position.
3. Baseline margin and intervention delta for flipped versus unflipped samples.
4. Preserved hidden-trajectory effect versus clean transcript recomputation.
5. Candidate-score change versus judgment-margin change, with rescue results.
6. Factual correctness and retrospective accuracy by difficulty and item type.

Use bootstrap confidence intervals for paired continuous effects and an
appropriate binomial interval for flip proportions. Always show individual item
points when the sample size is small.

## 13. Interpretation ladder

- **Level 0 — Decodability:** candidate appears in J-space.
- **Level 1 — Association:** candidate correlates with factual process,
  correctness, confidence, or judgment.
- **Level 2 — Evaluator causality:** candidate intervention changes retrospective
  judgment and survives basic output controls.
- **Level 3 — Hidden-trace sensitivity:** with question and answer text fixed,
  preserved factual trajectories produce different candidate states and
  judgments relative to clean transcript recomputation.
- **Level 4 — Candidate mediation:** changing `P` changes the candidate and
  judgment, and restoring the candidate rescues the judgment; effects exceed
  external, random-direction, cache-persistence, lexical, and output controls.

Level 4 would justify describing the feature as a candidate higher-order
representation of the model's first-order computation. It still would not prove
that the representation is exhaustive, uniquely metacognitive, or equivalent to
human introspection.

## 14. Go/no-go criteria

Proceed from the expanded descriptive experiment to the preserved-trace study
only if at least one candidate has a graded, nonsaturated internal response and a
repeatable effect on an oriented judgment margin.

Prioritize a candidate for confirmatory mediation only if:

- its effect is consistent across multiple items rather than driven by one hard
  example;
- the effect is not larger in forced-output or irrelevant-cache controls;
- it survives at least one lexical change;
- layer and position choices were made without using confirmatory flip outcomes;
- the factual answer is token-identical across paired trajectories;
- candidate restoration produces a measurable rescue effect.

If these conditions fail, the defensible conclusion remains that the feature is
generic evaluation, confidence, cache residue, or output-selection machinery
rather than evidence for `M(P)`.

## 15. Suggested implementation phases

1. **Protocol repair:** run all CSV rows, preserve each row's original final
   question, score factual answers with their regex keys, and simplify positional
   controls.
2. **Adaptive readout audit:** record candidate appearance layers and every
   token/layer/position attempt without hiding no-flip samples.
3. **Dose calibration:** find smaller positive and negative strengths that produce
   graded candidate changes without saturation.
4. **State-preserving prototype:** implement forced identical answer tokens with
   separate clean and perturbed KV trajectories on a small CPU-testable model.
5. **Artifact validation:** prove that answer token IDs are identical while
   preserved state hashes differ, and that clean recomputation removes the hidden
   difference.
6. **Discovery run:** select perturbation and candidate parameters on a declared
   subset.
7. **Held-out confirmation:** freeze the protocol, execute once, and report all
   outcomes and failures.

The core standard for the next stage is simple: **same visible question, same
visible answer, measurably different first-order computation, and a specific
causal effect transmitted through a candidate higher-order state.**

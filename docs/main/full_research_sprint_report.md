# Full Research Sprint Report
## Distinguishing First-Order Readout from Higher-Order / Process-Sensitive Representation in LLMs

**Model studied:** `Qwen/Qwen3.6-27B`  
**Primary interpretability interface:** J-Lens / J-space residual directions  
**Primary behavioral source:** Metacognitive Monitoring Battery (`metacognition.csv`)  
**Sprint period:** 2026-09-03 to 2026-09-05  
**Status:** Completed exploratory sprint; process-sensitive representation supported exploratorily, candidate-to-judgment mediation unresolved because of intervention-precision limits.

---

# 1. Executive summary

This research sprint asked a deceptively simple mechanistic question:

> When an LLM reports confidence, correctness, uncertainty, or self-evaluation, is it merely reading out information from its ordinary first-order computation, or does it form a later representation of that computation itself?

The distinction matters because a model can contain uncertainty signals, error predictors, confidence features, or evaluator directions without possessing anything that deserves to be called higher-order or metacognitive representation.

The sprint progressively tightened the hypothesis.

The initial hypothesis was weak:

```text
x -> y
versus
x -> m(x) -> y
```

This was rejected as conceptually insufficient because `m(x)` can still be an ordinary readout. The stronger operational distinction became:

```text
First-order process:
P = x0 -> x1 -> x2 -> ... -> y

Ordinary readout:
P -> r(P)

Candidate higher-order / meta-representation:
P -> M(P)
```

The important distinction is not depth. A very deep computation can remain first-order. The key question is whether the later state is merely computed from `P`, or whether it behaves like a representation of a property of `P` as the thing being evaluated.

The research proceeded in three main experiments plus a final attempted mediation study.

## Experiment 1: causal evaluator directions

Two J-space directions roughly translated as “to evaluate” were identified around the retrospective evaluation question:

- token `97817`: `评价`
- token `99973`: `评估`

Steering these directions strongly changed retrospective `PASS/FAIL` or `CORRECT/INCORRECT` judgments. The primary global run showed a large asymmetric causal effect, including 27/82 flips at negative steering strength.

However, the vast majority of flips were harmful, several outputs became malformed, and the pattern tracked judgment-label polarity more reliably than truth.

**Conclusion:** the directions participate causally in an evaluation/output-control pathway, but this does not establish higher-order representation.

## Experiment 2: matched SELF vs OTHER

The next question was whether these evaluator directions were specific to evaluating the model's own answer.

For each item, the exact same factual question and exact same answer were evaluated under:

```text
SELF:  Is your answer CORRECT or INCORRECT?
OTHER: Is their answer CORRECT or INCORRECT?
```

The two evaluator directions were present and causally active in both conditions. Neither exhibited a stable general SELF-specific causal effect. Their SELF-vs-OTHER prominence was strongly dependent on prompt/label family.

**Conclusion:** the easy “self-evaluation feature” interpretation was falsified. The directions are better described as generic evaluation/output-control probes.

## Experiment 3: Process-Sensitive Replay (PSR)

The sprint then moved to a much stronger design:

```text
same question
same complete answer text
same evaluation prompt
different hidden first-order process P
```

The exact factual answer was generated once and then teacher-forced identically across conditions. The hidden answer process was manipulated while visible text was held fixed.

The final exploratory quick profile used:

- a primary answer-support-reducing intervention at layer 31;
- a support-matched alternative intervention at layer 23;
- same-layer norm-matched random controls;
- a reset control;
- independent confidence and correctness branches;
- held-out candidate discovery;
- strict token/cache/hook/state integrity gates.

A new frozen J-space candidate was found:

- token ID `75075`
- layer `42`
- orientation `-1`

The vocabulary label (`UIImagePickerController`) was explicitly treated as semantically meaningless. The scientific object was the frozen layer-42 residual-space direction.

On 8 held-out items:

- support matching passed 8/8;
- both structured mechanisms changed the candidate in the frozen direction;
- both exceeded their own same-layer random controls;
- reset returned the candidate to clean;
- candidate activity tracked support damage;
- greater answer-support damage predicted lower later confidence;
- generic evaluator tokens `97817` and `99973` remained nearly flat.

This gave the strongest result of the sprint:

```text
P manipulation -> process-sensitive later representation/readout
```

with visible output held fixed.

However, convergence across the two mechanisms was only **compatible with**, not demonstrably equivalent to, a shared abstract process-property representation.

## Final attempted experiment: candidate-to-judgment mediation

The missing arrow was:

```text
candidate -> judgment
```

A natural-size coordinate restoration experiment was designed to patch only the naturally occurring clean-vs-perturbed component of the frozen layer-42 candidate.

It could not be executed faithfully.

Two numerically principled implementations were tested and rejected:

1. **BF16-native coordinate patching**
   - coordinate target could be matched;
   - but unavoidable BF16 quantization caused large orthogonal movement;
   - certified lower bounds showed that no better solver could satisfy both coordinate accuracy and low-leakage criteria.

2. **FP32 tail continuation**
   - coordinate precision became excellent;
   - but merely switching the downstream tail to FP32 changed quantitative judgments substantially;
   - therefore the precision regime itself became a confound.

No nonzero mediation patch was accepted into the scientific experiment.

**Final status:**

```text
Supported exploratorily:
P -> process-sensitive readout / representation

Not established:
process-sensitive candidate -> judgment
```

The sprint therefore ends with positive exploratory evidence for a process-sensitive internal representation, but causal higher-order mediation remains unresolved due to an intervention-identifiability / numerical precision limitation.

---

# 2. The scientific problem

## 2.1 Why confidence is not automatically metacognition

A language model can expose many signals correlated with correctness:

- logit margin;
- token entropy;
- hidden-state probes;
- semantic entropy;
- error classifiers;
- evaluator features;
- representations of answer content;
- decision-boundary proximity.

All of these can be computed downstream of the ordinary task process without the model forming a representation of that process itself.

For example:

```text
question -> reasoning -> answer state -> confidence readout
```

does not automatically imply:

```text
question -> reasoning process P
                   |
                   v
             representation of P
                   |
                   v
               confidence
```

This distinction was the organizing principle of the sprint.

## 2.2 Operational definitions

### First-order process

Let:

```text
P = x0 -> x1 -> x2 -> ... -> y
```

where `P` is the internal computation that produces a factual answer.

Depth does not determine order. A 64-layer computation can still be entirely first-order.

### First-order readout

A readout is any downstream variable:

```text
r(P)
```

that extracts or computes information from the process.

`r(P)` may encode:

- answer correctness;
- uncertainty;
- probability of error;
- conflict;
- whether the answer will fail;
- whether the model should choose `PASS` or `FAIL`.

None of these properties alone establish higher-order representation.

### Higher-order / meta-representational candidate

A stronger candidate is:

```text
M(P)
```

where the target of the representation is the first-order process `P` or a property of that process.

The conceptual intuition was:

> Readout = compute from `P`.  
> Higher-order candidate = represent `P`, or a property of `P`, as the thing being evaluated.

### Monitoring versus regulation

The sprint also distinguished:

```text
Monitoring:
P -> M(P)

Regulation:
P -> M(P) -> future control / behavior
```

The completed PSR experiment primarily targeted monitoring. The final mediation experiment was intended to test whether the candidate participates causally in downstream judgment/control.

---

# 3. Cognitive-science grounding

The project was deliberately anchored in cognitive-science distinctions rather than inventing “metacognition” from LLM behavior alone.

Conceptual influences discussed during the sprint included:

- **Rosenthal (1986), “Two Concepts of Consciousness”** — higher-order thought accounts emphasize a mental state being represented by a distinct higher-order state; the higher-order thought is about the particular target state.
- **Fleming, Dolan, Frith and related metacognition literature** — behavioral second-order judgments are not sufficient by themselves to identify a meta-level representation.
- **Fleming & Daw (2017)** — second-order inference provides a formal perspective on metacognitive readout.
- **Timmermans et al. (2012)** — higher-order networks and metacognitive structure.
- **Desender et al. (2021)** — continued evidence accumulation can mimic metacognitive signals, illustrating the need to rule out first-order alternatives.
- **Kanai, Takatsuki & Fujisawa (2025), “Meta-representations as representations of processes”** — especially important to the sprint's stronger definition: simply applying another function downstream is not enough; meta-representation must target the process/function itself.

The practical consequence was a high evidence bar.

A convincing higher-order claim should not rest on:

```text
decodability
or
confidence behavior
or
causal output steering
```

alone.

The sprint therefore sought converging evidence from:

1. controlled hidden-process manipulation;
2. identical visible outputs;
3. alternative mechanisms producing a shared process property;
4. random controls;
5. reset controls;
6. held-out candidate selection;
7. ideally causal candidate-to-judgment mediation.

---

# 4. Model, data, and interpretability setup

## 4.1 Model

The core model was:

```text
Qwen/Qwen3.6-27B
```

The relevant implementation treated it as a 64-block hybrid model rather than a plain KV-only transformer.

Preserved state therefore included both:

- full-attention K/V state;
- linear-attention convolution/recurrent state.

This mattered substantially for replay, gradients, cache cloning, and Turn-3 continuation.

## 4.2 Dataset

The broader Metacognitive Monitoring Battery contained 524 prompt-level items across six datasets.

The primary file used here was:

```text
metacognition.csv
```

with 90 items:

| Item family | Count |
|---|---:|
| Calibration | 66 |
| Prospective monitoring | 8 |
| Error detection | 8 |
| Knowledge boundary | 8 |
| **Total** | **90** |

The dataset contains benchmark items, not response logs.

The source prompts were not always already organized into genuinely interactive multi-turn conversations, so the experiments reconstructed multi-turn interactions while preserving the intended task semantics.

## 4.3 J-Lens / J-space

J-Lens was used as an internal readout and candidate-direction interface.

A recurring lesson throughout the sprint was:

> J-Lens decodability is evidence that information is readable from the residual state, not proof that the model itself uses that direction as a higher-order representation.

The most informative retrospective readout position was found to be the `?` in the final evaluation question rather than the final punctuation after the answer label. The latter could be too late in the computation.

---

# 5. Experiment 1 — evaluator direction discovery and global steering

## 5.1 Initial manual observation

The model was placed in a multi-turn metacognitive setting:

1. prospective prediction;
2. factual answer;
3. retrospective evaluation.

A representative retrospective prompt was:

```text
Did your answer PASS or FAIL? Reply with exactly PASS or FAIL.
```

At layer 40 near the evaluation `?`, two notable J-space vocabulary directions appeared:

```text
97817 = 评价
99973 = 评估
```

Both translate roughly to evaluation/assessment language.

Manual steering suggested that negative intervention along one evaluator direction could flip retrospective judgments.

This created an initially tempting hypothesis:

```text
perhaps this is a self-evaluation / metacognitive feature
```

The rest of the sprint was largely an attempt to test and refine that interpretation.

## 5.2 Full global steering run

The first full causal run used 90 metacognition samples.

The primary candidate was visible for 82 self-condition samples but absent in all 8 external error-detection samples.

That meant the intended self-vs-external specificity comparison could not be made cleanly in this run.

### Baseline performance

The model's baseline judgment was correct on:

```text
86 / 90 = 95.6%
```

### Steering results

For the 82 samples where the primary intervention was available:

| Requested strength | Flips | Flip rate | Improved | Worsened | Mean oriented-margin change |
|---:|---:|---:|---:|---:|---:|
| `-1.7` | 27/82 | 32.93% | 2 | 25 | -7.4761 |
| `+1.8` | 4/82 | 4.88% | 0 | 4 | -0.2936 |

The intervention strongly manipulated the intended internal candidate:

```text
baseline candidate score ~ +12.58
negative steering       ~ -28.40
positive steering       ~ +28.75
```

The requested strengths saturated at an effective ±1.0 norm cap.

Negative steering reduced the mean decision margin from approximately:

```text
8.55 -> 0.64
```

which explained its high flip rate.

### Important failure of the “truthful metacognition” interpretation

The intervention did not improve judgment calibration.

Of the 31 primary flips across the two strengths:

```text
29 were harmful
2 were improvements
```

Four negative-steering outputs became malformed:

```text
INCORIENT
INCORENT
INCORNT
INCORrent
```

The pattern was strongly label-dependent:

- negative steering readily pushed `CORRECT -> INCORRECT`;
- positive steering on prospective / knowledge-boundary items pushed `FAIL -> PASS`;
- positive steering caused no calibration flips.

This looked much more like control of an evaluation/output pathway than a clean internal variable representing truthful self-assessment.

### Adaptive fallback

An adaptive search made 406 additional attempts over 51 samples.

Only one additional flip was found, using token `99973`, and it was harmful.

### Interpretation after Experiment 1

The experiment established:

```text
candidate direction -> causal effect on evaluation/output
```

It did **not** establish:

```text
candidate = self-specific metacognitive representation
```

At this stage, the strongest interpretation was:

```text
generic evaluator / label-control machinery
```

The next experiment therefore targeted self-specificity directly.

---

# 6. Experiment 2 — matched SELF vs OTHER evaluation

## 6.1 Question

Do the evaluator directions specifically participate in evaluating the model's **own** answer, or do they behave similarly when evaluating the exact same answer attributed to someone else?

## 6.2 Design

The design removed several confounds from the earlier external-control setup.

For every factual item:

1. generate the factual answer `X` once;
2. reuse the exact same factual question;
3. reuse the exact same answer text;
4. keep labels and evaluation structure matched;
5. alter only the attribution.

Example:

```text
SELF:
Is your answer CORRECT or INCORRECT?

OTHER:
Is their answer CORRECT or INCORRECT?
```

For source families using `PASS/FAIL`, the same matched structure was used.

Importantly, the clean matched version did **not** include:

- a prospective `WILL_PASS/WILL_FAIL` turn;
- a knowledge-boundary `I_KNOW/I_WOULD_GUESS` turn.

The readout remained at the evaluation `?`, layer 40.

Two candidate directions were independently tested:

```text
97817 = 评价
99973 = 评估
```

## 6.3 Sample accounting

The paired experiment used 82 factual items:

| Family | Count |
|---|---:|
| Calibration | 66 |
| Knowledge boundary | 8 |
| Prospective source items | 8 |

Every SELF/OTHER pair shared the same exact factual question and answer hash.

The model's factual answer was correct on 71/82 items.

Baseline evaluation was strong:

| Condition | Correct judgments |
|---|---:|
| SELF | 77/82 = 93.9% |
| OTHER | 79/82 = 96.3% |

Thus OTHER evaluation was not simply harder.

## 6.4 Observational candidate results

Overall:

| Candidate | SELF mean | OTHER mean | SELF − OTHER | 95% CI |
|---|---:|---:|---:|---|
| `97817` | 11.664 | 12.968 | -1.304 | [-1.647, -0.940] |
| `99973` | 12.008 | 12.783 | -0.775 | [-1.121, -0.405] |

Both were actually higher in OTHER overall.

But the average concealed a striking prompt-family interaction.

### Token 97817

| Family | SELF − OTHER |
|---|---:|
| Calibration | -2.033 |
| Knowledge boundary | +1.492 |
| Prospective | +1.914 |

### Token 99973

| Family | SELF − OTHER |
|---|---:|
| Calibration | -1.523 |
| Knowledge boundary | +2.195 |
| Prospective | +2.422 |

The pattern was nearly perfectly systematic:

- all calibration `CORRECT/INCORRECT` items favored OTHER;
- nearly all `PASS/FAIL` items favored SELF.

This means candidate prominence was strongly sensitive to prompt or label family, not uniformly to SELF perspective.

Because prompt family and label family were confounded, the experiment could not say which one was primary. But it could decisively reject a simple uniform SELF-specific interpretation.

## 6.5 Causal steering results

Correct-oriented margin changes:

| Candidate | Strength | SELF delta | OTHER delta | SELF − OTHER causal contrast |
|---|---:|---:|---:|---:|
| `97817` | -1.7 | -6.142 | -6.754 | +0.612 |
| `97817` | -1.8 | -6.146 | -6.764 | +0.619 |
| `99973` | -1.7 | -5.762 | -5.500 | -0.262 |
| `99973` | -1.8 | -5.772 | -5.499 | -0.274 |

Interpretation:

- token `97817` damaged OTHER somewhat more strongly;
- token `99973` showed a weak SELF tendency, but intervals crossed zero;
- the sign changed across candidates.

There was no stable replicated SELF-selective causal effect.

## 6.6 Decodability did not predict differential causality

The item-level relationship between:

```text
SELF-vs-OTHER candidate prominence
```

and:

```text
SELF-vs-OTHER steering effect
```

was weak to essentially zero.

This reinforced a recurring interpretability lesson:

> A representation being more decodable in one condition does not imply that perturbing it has a correspondingly condition-selective causal effect.

## 6.7 Relationship between the two evaluator directions

The two candidate directions were highly related:

| Comparison | Value |
|---|---:|
| Direction cosine similarity | 0.761 |
| Candidate-score correlation | 0.934 |
| SELF−OTHER score-contrast correlation | 0.986 |
| SELF intervention-delta correlation | 0.979 |
| OTHER intervention-delta correlation | 0.982 |

The correct interpretation was therefore not “two independent metacognition discoveries,” but:

```text
two probes into a substantially overlapping evaluation/output-related subspace
```

## 6.8 Key update from Experiment 2

The easy story was falsified.

The best description became:

> Tokens 97817 and 99973 are correlated, causally active evaluation/output-control probes whose expression depends strongly on evaluation prompt family. They are not supported as representations specific to evaluating the model's own answer.

This forced the project into a stronger question.

Instead of:

```text
Does the model look self-evaluative?
```

the next experiment asked:

```text
Does a later state track properties of the model's hidden first-order process
when the visible answer is exactly identical?
```

---

# 7. Experiment 3 — Process-Sensitive Replay (PSR)

## 7.1 Core idea

PSR was designed around the strongest operational test in the sprint:

```text
same question
same complete answer X
same visible transcript
same evaluation prompt
different hidden process P
```

If a later internal state changes in a structured way despite identical visible text, simple output-based explanations become less sufficient.

The core causal flow was:

```text
generate factual answer X once
          |
          v
freeze exact answer token IDs
          |
          +------------------------------+
          |                              |
          v                              v
        CLEAN                    PROCESS PERTURBED
teacher-force X                  teacher-force same X
          |                              |
          v                              v
preserved hidden state            preserved altered hidden state
          |                              |
          +--------------+---------------+
                         |
                         v
               identical Turn-3 prompt
                         |
                         v
           later candidate / judgment
```

## 7.2 Primary process manipulation

The primary intervention targeted zero-based:

```text
PROCESS_LAYER = 31
```

roughly the midpoint of the 64-block model and a full-attention block in the installed configuration.

For answer sequence support:

```text
S = Σ_t log P(X_t | Q, X_<t)
```

the process intervention used the negative gradient of `S` with respect to the layer-31 residual at the answer-predicting positions.

At each position:

```text
targeted_delta_i =
    -alpha * ||h_i_clean|| * normalize(g_i)
```

where:

```text
g_i = ∂S / ∂h_i
```

Using the clean residual norm prevented accumulated perturbation from changing the nominal dose.

## 7.3 Why exact answer teacher forcing mattered

The point was not to make the model emit a different answer.

The exact same answer tokens were teacher-forced in every matched condition.

Thus:

```text
visible answer = fixed
hidden computation = manipulated
```

This separated hidden-process evidence from surface-text evidence.

## 7.4 Preserved hybrid state

Qwen's hybrid architecture meant the post-answer state was not just a standard K/V cache.

The experiment had to preserve:

- full-attention K/V tensors;
- convolution state;
- recurrent linear-attention state;
- initialization flags;
- exact positions and topology.

These hidden states were then cloned into independent Turn-3 meta branches.

## 7.5 Meta branches

Two independent branches were used:

### Correctness

```text
Was the factual answer you just gave CORRECT or INCORRECT?
```

### Process confidence

```text
How confident were you in the factual answer you just gave?
HIGH_CONFIDENCE or LOW_CONFIDENCE
```

The branches were cloned independently so one scoring pass could not mutate the other's cache.

Readout was taken when the Turn-3 `?` token itself was processed.

---

# 8. Why a support-matched alternative mechanism was necessary

A single perturbation changing a later state can always be dismissed as:

```text
local damage
generic contamination
persistent perturbation residue
```

Therefore PSR required a second structured mechanism that generated the **same functional property of P** through a different causal route.

The relevant shared property was:

```text
reduced support for the exact same answer X
```

The desired logic was:

```text
mechanism A -> reduced answer support
mechanism B -> similar reduced answer support

if both ->
    same later candidate response

while random matched perturbations do not,
then the later state may be tracking the process property
rather than perturbation identity.
```

This was the most important conceptual control in the PSR design.

---

# 9. The PSR development and failure sequence

The PSR implementation went through many iterations. These failures are part of the research story because most were caught by fail-closed gates before any scientific interpretation.

## 9.1 Initial answer-bank failure

The initial answer-generation cap was:

```text
48 tokens
```

This worked for calibration answers but failed badly for longer prospective and knowledge-boundary items.

Result:

- 66/66 calibration valid;
- only 1/8 prospective valid;
- 0/8 knowledge-boundary valid.

The cap was increased to:

```text
256 tokens
```

with these rules:

- preserve generated answer-content token IDs exactly;
- canonicalize only the invisible assistant terminator to `<|im_end|>`;
- log original and canonical terminal IDs;
- mark >256-token unterminated answers invalid;
- verify thinking mode explicitly.

## 9.2 Recurrent-gradient incompatibility

A full-sequence no-cache gradient pass disagreed with the ordinary cached replay.

Reason:

- Qwen selected a chunked delta-rule kernel in the no-cache route;
- ordinary experimental replay used recurrent kernels.

This meant the gradient computation and experimental replay were not actually describing the same computation.

The fix was a differentiable token-by-token recurrent gradient pass using the same:

- token positions;
- recurrent kernels;
- causal cache semantics.

Mandatory parity checks included:

- per-token full-vocabulary logits;
- total answer support;
- intervention-layer residuals;
- finite nonzero gradients;
- exact hook-position lists.

## 9.3 Chat-template / Turn-3 prefix incompatibility

Rerendering the full conversation after adding Turn 3 changed the earlier assistant history, including the empty thinking block.

This violated the “same visible factual prefix” requirement.

The fix was **suffix-only Turn-3 construction**:

- never rerender frozen factual history;
- preserve exact `post_answer_token_ids`;
- render only the Turn-3 suffix;
- append it to the frozen prefix;
- assert exact prefix, suffix, boundary, and concatenated transcript hashes.

This became a core integrity rule.

## 9.4 Strength-grid issue

The original alpha grid skipped too far from `0.10` to `0.20`.

At one point:

```text
alpha 0.10 -> median support drop 1.864 nat
alpha 0.20 -> median support drop 5.694 nat
```

so no grid point occupied the desired strong range of 2–4 nats.

Intermediate values were added:

```text
0.11
0.125
0.15
```

The failed campaign was retained as diagnostics rather than silently reused.

## 9.5 Entropy-based support-matched alternative failed scientifically

The first alternative mechanism tried to use a same-layer entropy objective, projected orthogonally to the answer-support gradient.

The idea was elegant:

```text
primary: answer-support gradient
alternative: entropy-related orthogonal direction
```

while calibrating the alternative to match answer-support damage.

But the control failed the actual item-wise support-matching gate.

In `psr-v7`:

```text
targeted median support drop = 2.07785 nat
best alternative median      = 2.46601 nat
median paired mismatch       = 3.36568 nat
allowed mismatch             = 0.51946 nat
```

This was a **scientific gate failure**, not a code bug.

The implication was important:

> Aggregate matching can look good while item-level functional matching is poor.

The entropy method was therefore abandoned rather than weakened post hoc.

## 9.6 Different-layer, same-objective alternative

The replacement control used:

```text
same answer-support objective
different earlier layer
```

The primary stayed at layer 31.

The alternative layer was selected from a small predeclared set of earlier layers using discovery support-match quality only, not meta outcomes.

The quick run ultimately selected:

```text
layer 23
beta = 0.20
```

This made layer identity a controlled mechanism difference rather than an uncontrolled confound.

Each structured intervention also received its own same-layer norm-matched random control.

## 9.7 Resource-lifetime OOM

A full discovery run exhausted a ~95 GiB GPU after 12/16 beta-grid items.

At failure:

```text
~90.92 GiB allocated
~3.36 GiB reserved but unallocated
~5.56 MiB free
```

Investigation found:

- redundant primary-gradient computation during beta-only passes;
- a temporary bound-method cycle capable of retaining autograd state;
- implicit disposable cache lifetimes.

Corrections:

- remove redundant gradient computation;
- break the reference cycle;
- explicitly release replay/cache/branch/autograd objects;
- record allocated/reserved/peak CUDA memory after items;
- fail closed on systematic post-cleanup growth.

## 9.8 Full run still failed from per-item peak memory

Even after fixing retained memory, the full run still failed when reaching a long prospective answer around 177 tokens.

This showed the remaining problem was not progressive leakage but the **single-item recurrent-autograd peak**.

The full confirmatory profile was therefore blocked.

This motivated a separately declared quick profile rather than quietly shrinking the original experiment.

---

# 10. Full versus quick PSR

The quick profile preserved the causal logic but deliberately reduced the estimand and statistical strength.

| Component | Full | Quick |
|---|---:|---:|
| Answers generated | 82 | 16 predeclared |
| Discovery items | 16 | 8 |
| Held-out items | up to 66; 57 valid in recent bank | 8 |
| Alternative layers | 15, 19, 23 | 15, 19, 23 |
| Readout layers | 36–44 | 38, 40, 42 |
| Max frozen candidates | 3 | 1 |
| Top vocab entries | 100 | 25 |
| Smoke items | 4 | 2 |
| Differentiable answer tokens | full answer, up to 256 | first 32 |
| Complete answer teacher-forced | yes | yes |
| Complete perturbed state preserved | yes | yes |
| Correctness branch | yes | yes |
| Confidence branch | yes | yes |
| Random controls | yes | yes |
| Reset control | yes | yes |
| Interpretation | potentially confirmatory | exploratory only |

The quick estimand was explicitly:

```text
early_answer_process_first_32_tokens_with_complete_answer_state
```

For long answers:

1. compute/perturb only the first 32 answer-token positions;
2. disable the factual-process hook;
3. teacher-force the remaining exact answer;
4. preserve the complete resulting state into Turn 3.

Thus the visible answer and preserved state were complete; only the differentiable intervention window was bounded.

---

# 11. PSR quick-v3 — successful exploratory run

## 11.1 Campaign integrity

The successful quick campaign passed all eight phases:

1. validate
2. answer bank
3. pre-discovery smoke
4. discovery
5. freeze
6. post-freeze smoke
7. held-out
8. analyze

The post-freeze smoke reproduced the frozen support effects and passed all critical gradient, intervention, token, cache, reset, branch, and hook checks.

Held-out outcomes were not accessed during candidate or strength selection.

## 11.2 Frozen strengths

Discovery selected:

```text
weak primary alpha   = 0.10
strong primary alpha = 0.11
alternative layer    = 23
alternative beta     = 0.20
```

Discovery medians:

```text
weak primary support drop      = 1.30371 nat
strong primary support drop    = 1.44593 nat
alternative support drop       = 1.19860 nat
median paired support mismatch = 0.39087 nat
frozen mismatch tolerance      = 0.5 nat
```

The strong alpha used an allowed below-target fallback because the quick grid did not produce a point in the preferred 2–4 nat range.

This limited dose-response claims but did not invalidate the run.

## 11.3 Candidate discovery

Discovery searched:

```text
248,131 eligible token/layer directions
```

from:

```text
217,494 eligible vocabulary tokens
```

One candidate was frozen:

| Field | Value |
|---|---|
| Token ID | `75075` |
| Token label | `UIImagePickerController` |
| J-Lens layer | `42` |
| Orientation | `-1` |
| Discovery structured sign consistency | `0.875` |
| Targeted-strong effect | `0.04492` |
| Alternative effect | `0.06055` |
| Targeted − random | `0.04102` |
| Alternative − random | `0.07813` |
| Targeted preserved − reset | `0.04492` |

The token label was explicitly treated as an index artifact rather than semantic evidence.

The scientifically relevant object was:

```text
the frozen layer-42 effective residual-space direction associated with token 75075
```

## 11.4 Held-out support matching

Held-out items:

```text
0, 2, 3, 4, 57, 67, 68, 82
```

Support-match gate:

```text
8 / 8 passed
required fraction = 65%
```

Medians:

```text
primary support drop     = 2.94407 nat
alternative support drop = 3.12691 nat
median signed mismatch   = 0.05449 nat
median absolute mismatch = 0.26395 nat
```

Mean absolute mismatch was larger (`3.20962` nat) because long-answer items lived on much larger support scales.

For example, items 67, 68, and 82 had absolute mismatches of:

```text
5.762
2.900
16.263 nats
```

but still satisfied their predeclared relative tolerance bands.

Thus “8/8 matched” means:

```text
all items passed the declared functional tolerance
```

not:

```text
the two perturbations produced numerically identical support drops.
```

---

# 12. Quick-v3 held-out results

All candidate effects were oriented according to the discovery-frozen direction.

## 12.1 Confidence branch

| Test | Estimate | Median | 95% bootstrap interval |
|---|---:|---:|---:|
| H1 primary targeted − clean | 0.02930 | 0.03906 | [0.00781, 0.05078] |
| H2 alternative − clean | 0.04688 | 0.03906 | [0.01758, 0.07617] |
| H3 primary − alternative | -0.01758 | -0.01563 | [-0.04883, 0.01563] |
| H3 support-normalized difference | -0.02105 | -0.00068 | [-0.06290, 0.00516] |
| H4 primary − own random | 0.02930 | 0.03906 | [0.01563, 0.04297] |
| H4 alternative − own random | 0.05859 | 0.03906 | [0.02539, 0.09375] |
| H5 preserved − reset | 0.02930 | 0.03906 | [0.00781, 0.04883] |
| H6 candidate-score/support slope | 0.000482 | — | [0.000033, 0.001497] |
| H7 confidence-margin/support slope | -0.01830 | — | [-0.03551, -0.01416] |

The structured effects were positive on six of eight confidence items.

The H7 result was particularly important:

```text
greater hidden answer-support damage
        ->
lower HIGH_CONFIDENCE - LOW_CONFIDENCE margin
```

Correlations:

```text
Pearson  = -0.830
Spearman = -0.890
```

## 12.2 Correctness branch

| Test | Estimate | Median | 95% bootstrap interval |
|---|---:|---:|---:|
| H1 primary targeted − clean | 0.03271 | 0.02734 | [0.01660, 0.05078] |
| H2 alternative − clean | 0.09521 | 0.03906 | [0.00488, 0.20703] |
| H3 primary − alternative | -0.06250 | -0.03125 | [-0.16309, 0.01855] |
| H3 support-normalized difference | 0.01022 | -0.00127 | [-0.00818, 0.03799] |
| H4 primary − own random | 0.03369 | 0.03516 | [0.00879, 0.05615] |
| H4 alternative − own random | 0.10498 | 0.05078 | [0.01709, 0.21875] |
| H5 preserved − reset | 0.03271 | 0.02734 | [0.01660, 0.05029] |
| H6 candidate-score/support slope | 0.000978 | — | [0.000746, 0.001276] |

Primary effects were positive on seven of eight correctness items.

Alternative effects were positive on five of eight.

Long-answer items had substantial influence on some estimates.

---

# 13. What the PSR controls ruled out

## 13.1 Not just visible text

The complete visible question and answer were identical across process conditions.

Therefore the candidate effect cannot be explained simply by the model reading a different answer string.

## 13.2 Not just generic norm-matched damage

Each structured mechanism exceeded its own same-layer norm-matched random control.

This argues against a simple explanation of:

```text
any perturbation at this layer causes the later signal.
```

## 13.3 Not just preserved token text

Reset discarded the perturbed hidden process state and reconstructed the same visible history cleanly.

Reset candidate scores returned to clean.

This indicates the later candidate depended on preserving hidden process history.

## 13.4 Not a broad shift of the earlier evaluator subspace

The earlier generic evaluator tokens:

```text
97817
99973
```

were nearly flat across PSR conditions.

Their mean-score variation was only around:

```text
0.0234
0.0286
```

respectively.

This does not prove candidate 75075 is uniquely metacognitive, but it argues against a trivial global shift in all evaluation-related vocabulary directions.

---

# 14. Mechanism convergence: what worked and what remained unresolved

The central abstraction question was:

> Do two different support-reducing mechanisms produce the same later candidate response once realized support damage is accounted for?

The data were compatible with convergence.

For confidence:

```text
median support-normalized primary-vs-alternative difference ≈ -0.00068
```

For correctness:

```text
median ≈ -0.00127
```

But lack of difference is not evidence of equivalence.

The item-fixed-effect mechanism/shared-effect ratios were:

```text
confidence  = 0.584
correctness = 0.900
```

with wide bootstrap intervals extending above 1.

No equivalence margin or acceptable mechanism-residual threshold had been frozen.

Therefore the correct conclusion was:

```text
compatible with convergence
```

not:

```text
demonstrated shared abstract representation
```

Item 57 was a particularly visible outlier, reinforcing the need not to average away heterogeneity.

---

# 15. PSR memory behavior and computational integrity

The quick profile solved the earlier progressive-memory problem.

Peak allocated GPU memory was approximately:

| Stage | Peak |
|---|---:|
| Discovery alpha grid | 56.63 GiB |
| Discovery beta grid | 59.13 GiB |
| Candidate replay | 58.04 GiB |
| Post-freeze smoke | 52.81 GiB |
| Held-out | 58.06 GiB |

Post-cleanup allocation returned to roughly:

```text
51.31 GiB
```

after each item.

The CUDA trend gates recorded zero cumulative post-cleanup growth.

This supported the interpretation that the quick run was bounded and mechanically stable, unlike the failed full profile.

---

# 16. Main PSR limitations

The successful quick result remained exploratory for several reasons.

1. **Only 8 held-out items.**  
   Bootstrap intervals and sign consistency are unstable at this sample size.

2. **Only the first 32 answer tokens were differentiated/intervened.**  
   The complete answer state was preserved, but this is not the full-answer gradient estimand.

3. **Mixed support scales.**  
   Calibration answers were very short, while some prospective/knowledge-boundary answers were long. Long items strongly affected regression and mean-mismatch quantities.

4. **Only one frozen candidate.**  
   Candidate generality was not tested.

5. **Only three readout layers.**  
   The quick run used layers 38, 40, and 42 rather than a full 36–44 profile.

6. **Large candidate search space.**  
   Held-out testing reduces selection-overfit concerns but does not eliminate them with n=8.

7. **No frozen equivalence margin.**  
   Mechanism convergence could not be promoted to proven abstraction.

8. **No candidate-to-judgment mediation.**  
   This became the final missing causal test.

---

# 17. Evidence update after PSR

Before PSR, a simple account remained:

```text
visible answer
    ->
generic evaluator
    ->
confidence/correctness judgment
```

After PSR, that account was less sufficient.

The experiment showed that:

```text
hidden degradation of first-order process
while visible answer is fixed
    ->
later candidate changes
    ->
later confidence covaries with degradation
```

with random and reset controls.

The strongest defensible representation claim became:

```text
P -> R(P)
```

where `R(P)` is a process-sensitive later representation/readout.

The result was suggestive of:

```text
P -> M(P)
```

but did not prove the stronger “aboutness” claim.

The remaining causal question was:

```text
Does the frozen candidate itself affect the later judgment?
```

---

# 18. Final intended experiment — causal mediation/restoration

The final experiment was designed to test:

```text
P -> M*(P) -> judgment
```

using the already frozen layer-42 candidate.

The central idea was **natural-size coordinate restoration**, not arbitrary steering.

For frozen unit direction `v`:

```text
c_clean = dot(h_clean, v)
c_pert  = dot(h_pert, v)
```

A perturbed residual would be restored along only the candidate coordinate:

```text
h_restored =
    h_pert
    + (c_clean - c_pert) * v
```

The reverse transplant would insert the perturbed candidate coordinate into clean.

This was intended to test both:

- restoration / necessity-like behavior;
- reverse transplant / sufficiency-like behavior.

Planned controls included:

- sham patch;
- ± same-norm orthogonal random patches;
- full-residual restoration positive control;
- confidence primary outcome;
- correctness cross-output generalization;
- both upstream process mechanisms.

The mediation experiment never reached a valid nonzero candidate patch.

---

# 19. Precision failure 1 — BF16-native candidate restoration

The frozen model runs in BF16.

The natural clean-vs-perturbed candidate-coordinate differences were small enough that a naive:

```text
BF16(h + delta * v)
```

could destroy the intended one-dimensional intervention.

A compensated BF16 policy was therefore tested before running the scientific experiment.

## 19.1 Synthetic warning

An initial synthetic check using the exact frozen vector showed a case with:

```text
intended coordinate change        = -0.001576
coordinate error after BF16       =  0.001551
unwanted orthogonal movement L2   =  0.000239
```

This demonstrated that a precision policy was necessary.

## 19.2 Real smoke-item precision study

The compensated solver reproduced the upstream process correctly.

Across both smoke items and both branches:

| Check | Result |
|---|---|
| Original support-drop reproduction | 4/4 exact |
| Candidate coordinate accuracy | 16/16 within 1% |
| Candidate orthogonal-leakage limit | 0/16 passed |
| Random-control precision requirements | 0/32 passed |
| Nonzero candidate patches applied to model | none |

The coordinate target could be reached, but the BF16-representable residual required excessive movement orthogonal to the candidate.

Observed orthogonal leakage:

```text
~90% to ~249% of intended coordinate change
```

against the proposed 10% ceiling.

More importantly, lower-bound calculations certified that even an optimal BF16 solver would require roughly:

```text
79% to 99% minimum orthogonal leakage
```

for these candidate patches while keeping coordinate error within 1%.

Example:

```text
item 0
confidence branch
primary restoration

intended coordinate change = 0.0095463
coordinate error            ≈ 1%
actual orthogonal leakage    = 184.9%
certified minimum leakage    = 96.1%
```

This closed the BF16-native path.

It was not a failure of the compensation algorithm.

It was a representational limitation of the BF16 lattice for these tiny direction-specific residual changes.

No mediation result was obtained.

---

# 20. Precision failure 2 — mixed-precision FP32 tail

A separately approved fallback kept the upstream BF16 process exactly fixed but switched the continuation after the layer-42 intervention point to FP32.

This solved patch geometry extremely well.

## 20.1 Patch precision

The mixed-precision policy achieved:

```text
maximum coordinate error       = 0.000585%
maximum orthogonal leakage     = 2.14e-6
```

Generated labels remained unchanged.

Upstream process reproduction passed.

## 20.2 Equivalence failure

However, the zero-patch FP32 continuation materially changed quantitative judgment behavior relative to the original BF16 continuation.

Observed drifts:

```text
label-sequence log-probability drift: up to 0.204 nats
margin drift:                        up to 0.204 nats
process-contrast drift:              up to 0.195 nats
```

Only:

```text
2 / 8
```

process contrasts passed the frozen equivalence test.

Worst relative contrast drift:

```text
67.9%
```

All process-effect signs remained negative, but sign preservation was intentionally not considered sufficient.

The FP32 tail was therefore rejected before mediation.

Again:

```text
no nonzero mediation effect was scientifically tested.
```

---

# 21. Why the final mediation failure is not a negative result on the hypothesis

The two precision studies tested whether a clean intervention **could be implemented**, not whether the candidate causally affects judgment.

BF16 failed because:

```text
the desired coordinate-only perturbation was not representable
without large orthogonal contamination.
```

FP32 failed because:

```text
the precision change itself altered the downstream quantitative computation.
```

Therefore the sprint cannot conclude:

```text
candidate causes judgment
```

or:

```text
candidate does not cause judgment
```

The correct label is:

> **mediation unresolved because the intervention was not identifiable under the frozen numerical execution regime.**

This is an engineering / causal-identifiability boundary, not negative causal evidence.

---

# 22. Final evidence ladder

## Level 1 — decodability

Metacognitive/evaluation-related information can be read from J-space.

**Supported.**

But decodability alone is weak evidence.

## Level 2 — causal evaluator feature

Steering evaluator directions changes retrospective judgments.

**Supported.**

But the effects were largely harmful and label-dependent.

## Level 3 — self-specific evaluator

Evaluator directions should be preferentially present or causally stronger when the model evaluates its own answer.

**Not supported.**

Matched SELF-vs-OTHER falsified the easy self-specific story.

## Level 4 — process sensitivity under identical visible output

Manipulating hidden first-order processing should change a later internal state while the visible answer remains identical.

**Supported exploratorily.**

This is the strongest positive result.

## Level 5 — abstraction over process property

Different mechanisms that induce the same process-level property should converge on the same later representation.

**Compatible with data, not demonstrated.**

## Level 6 — causal mediation

Manipulating the process-sensitive representation should causally change later confidence/correctness judgment.

**Unresolved.**

The scientific experiment could not be run faithfully because of BF16/FP32 precision constraints.

---

# 23. Supported and unsupported claims at sprint end

## Supported

1. Qwen3.6-27B contains J-space directions that causally affect answer-evaluation outputs.
2. The original evaluator directions `97817` and `99973` participate in generic evaluation/output-control machinery.
3. Those directions are not supported as uniformly SELF-specific.
4. Under identical visible outputs, controlled hidden-process degradation can produce a later candidate-state change.
5. This candidate-state change exceeds matched random controls in the quick run.
6. Resetting the hidden process history removes the candidate effect.
7. A second support-matched process mechanism changes the same frozen candidate in the same discovery-selected direction.
8. Greater hidden support damage is associated with reduced later confidence.
9. Generic evaluator-token controls remain nearly flat in PSR, supporting some specificity.
10. The quick PSR result is valid exploratory evidence under its frozen protocol.

## Not supported

1. The original “evaluate” tokens are not established as higher-order representations.
2. SELF wording alone does not identify a higher-order mechanism.
3. Candidate prominence alone does not identify condition-selective causality.
4. The PSR candidate is not proven to represent the same abstract process property across mechanisms.
5. The token label `UIImagePickerController` has no semantic scientific interpretation.
6. The quick run does not provide dataset-wide confirmation.
7. The quick run does not establish the full-answer process estimand.
8. Candidate 75075 is not proven to causally mediate confidence or correctness.
9. The sprint does not prove philosophical higher-order representation or consciousness.
10. The failed mediation implementations are not evidence against causal mediation.

---

# 24. Best final scientific statement

A concise defensible conclusion is:

> **Qwen3.6-27B shows exploratory evidence for a later internal representation that is sensitive to controlled degradation of its own preceding answer process despite identical visible output. The signal generalizes across two independently structured answer-support-reducing mechanisms, exceeds same-layer norm-matched random controls, disappears when the manipulated hidden process history is reset, and covaries with subsequent confidence. This goes beyond a simple output-text evaluator. However, mechanism-level representational equivalence remains unresolved, and candidate-to-judgment causal mediation could not be tested faithfully because BF16 coordinate restoration causes unavoidable orthogonal contamination while an FP32 continuation materially changes downstream quantitative judgment behavior. The evidence therefore supports a process-sensitive readout/representation, but not a demonstrated higher-order representation.**

In compact notation:

```text
Evidence:
P -> R(P)

Suggestive but not established:
P -> M(P)

Not identified:
M(P) -> judgment
```

---

# 25. Why the sprint is scientifically useful despite the unresolved endpoint

The strongest feature of the sprint is the sequence of hypothesis elimination.

The project did not preserve the first exciting interpretation.

Instead:

```text
"evaluation token"
    ->
causal evaluator found

"therefore self-metacognition"
    ->
SELF vs OTHER falsifies this

"maybe just output-text readout"
    ->
same-output/different-process PSR challenges this

"maybe generic perturbation residue"
    ->
same-layer random controls + reset challenge this

"maybe mechanism-specific trace"
    ->
support-matched alternative is compatible with shared process sensitivity,
but equivalence remains unresolved

"now prove mediation"
    ->
precision audits reveal the causal patch is not identifiable cleanly
under the frozen numerical system

stop rather than relax criteria
```

This is a coherent research arc because each experiment was motivated by a failure mode of the previous interpretation.

---

# 26. Relevance beyond this specific model

The research question connects to several broader interpretability and safety topics.

## Confidence and calibration

If model confidence is based on internal process monitoring rather than only answer-text features, that changes how confidence should be interpreted and audited.

## Introspection

Claims that models can “report their own internal states” require distinguishing:

```text
generic reportable features
from
representations of the model's own computation.
```

The sprint provides an operational template for making that distinction more carefully.

## Monitoring and oversight

A process-sensitive internal variable could matter for:

- monitoring hidden reasoning quality;
- detecting internal conflict;
- recognizing likely failures before external verification;
- evaluating whether self-reports contain information unavailable from visible outputs.

## Mechanistic interpretability methodology

The sprint also demonstrates a general principle:

> A probe that is decodable and causally steerable can still be an output-control feature rather than the representation one hoped to find.

And:

> Hidden-process interventions with identical visible outputs are a stronger test than prompt-semantic comparisons alone.

---

# 27. What a stronger future study would do

A follow-up project could improve the evidence in several ways.

## 27.1 Run a full confirmatory PSR profile

Solve the recurrent-autograd memory bottleneck and test:

- larger held-out set;
- full answer gradient;
- layers 36–44;
- multiple frozen candidates;
- preregistered equivalence thresholds.

## 27.2 Independent replication set

Use a fresh item set not used for candidate discovery or the original 8-item held-out test.

## 27.3 Mediation-friendly numerical regime

Use a model/configuration where the natural candidate-coordinate changes are cleanly intervention-representable.

For example, a smaller model might permit consistent FP32 execution from the start, avoiding a precision-regime switch.

## 27.4 Coarser BF16-safe state intervention

A full natural residual swap at layer 42 could test whether the broader late state causally mediates the judgment difference without claiming specificity to token 75075.

This would establish a weaker causal localization:

```text
P -> late layer-42 state -> judgment
```

but not:

```text
candidate 75075 direction -> judgment.
```

## 27.5 Causal abstraction tests

If multiple interventions induce the same process-level property, formal causal-abstraction or distributed-alignment methods could test whether a shared abstract variable explains the downstream state better than mechanism identity.

---

# 28. Research-quality lessons from the sprint

Several methodological lessons emerged.

## 28.1 Separate probe evidence from mechanism evidence

```text
decodable != used
causally steerable != semantically what the probe label suggests
```

## 28.2 Preserve negative and failed campaigns

Failed support matching, failed smoke gates, and precision incompatibilities were retained as diagnostics rather than silently overwritten.

## 28.3 Use fail-closed engineering

Many apparent scientific outcomes would have been uninterpretable without:

- token hash checks;
- suffix-only transcript construction;
- recurrent-kernel parity;
- cache storage isolation;
- hook lifetime assertions;
- support reproduction;
- reset parity;
- CUDA memory trend checks.

## 28.4 Do not turn smoke tests into population inference

A 65% held-out match threshold accidentally became an effective 100% threshold on a two-item smoke set.

The protocol was corrected so smoke tests verified reproducibility of already measured quantities rather than making population-level scientific claims.

## 28.5 Aggregate matching is insufficient

The entropy-based alternative looked plausible in aggregate but failed badly item-by-item.

Item-level functional matching was crucial.

## 28.6 Stop when the intervention is not identifiable

The final mediation study is an important example.

Relaxing BF16 leakage thresholds or FP32 equivalence thresholds after seeing failure would have weakened the causal interpretation.

Stopping preserved the scientific meaning of the project.

---

# 29. Research narrative in one paragraph

The sprint began by finding J-space “evaluation” directions that could causally flip a model's retrospective judgments. A matched SELF-vs-OTHER experiment then showed that these directions were generic evaluation/output-control features rather than robust self-specific representations. This motivated a stronger same-output/different-hidden-process experiment: the exact factual answer was held fixed while the internal answer process was degraded. After several protocol and systems failures—recurrent-kernel mismatches, chat-template prefix changes, entropy-control support-matching failure, and full-profile GPU memory limits—a reduced but preregistered quick PSR run succeeded. It found a frozen layer-42 candidate whose activity tracked hidden process degradation, generalized across two support-matched structured interventions, exceeded same-layer random controls, disappeared under reset, and covaried with confidence. The final candidate-to-judgment mediation experiment was then blocked by a genuine numerical-identifiability problem: BF16 could not realize the tiny coordinate patch without large orthogonal movement, while FP32 continuation changed downstream judgment magnitudes too much. The final result is therefore positive exploratory evidence for a process-sensitive internal representation, with higher-order causal mediation unresolved rather than falsified.

---

# 30. Source artifact map

The report is synthesized from the sprint's experiment and protocol artifacts, especially:

## Global steering

```text
assets/full_run_global/
20260903T083743899864Z_qwen-qwen3-6-27b_global/
```

Key outputs:

```text
results.md
trial_summary.csv
intervention_results.csv
adaptive_paths.jsonl
run_manifest.json
```

## SELF vs OTHER

Token 97817 run:

```text
assets/self_v_external/
20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/
```

Token 99973 run:

```text
assets/self_v_external_token99973/
20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/
```

Key outputs included:

```text
results.md
trial_summary.csv
paired_results.csv
paired_summary.csv
intervention_results.csv
```

## Process-Sensitive Replay implementation / failure history

Key documents:

```text
Process-Sensitive Replay: Implementation Plan
Process-Sensitive Replay Failure History
Process-Sensitive Replay: Quick vs Full Profile
```

Campaigns discussed:

```text
psr
psr-v2
psr-v3
psr-v4
psr-v5
psr-v6
psr-v7
psr-v8
psr-v9
```

## Successful quick PSR

```text
assets/psr-quick-v3/
assets/psr_quick-v3-discovery/
```

Key outputs:

```text
analyze/RESULTS.md
analyze/analysis_report.json
analyze/plot_manifest.json
heldout/support_match_summary.json
heldout_support_match.csv
heldout_effects.csv
frozen_protocol.json
alpha_grid_diagnostics.json
beta_grid_diagnostics.json
candidate_discovery.json
gate_status.json
cuda_memory.jsonl
```

Frozen candidate:

```text
token_id = 75075
layer = 42
orientation = -1
```

The original completed analysis gate recorded the frozen binary direction artifact:

```text
directions/layer42_token75075_8515912d78e8.pt
```

## Final mediation precision diagnostics

The final causal-mediation follow-up generated engineering diagnostics rather than a valid mediation result:

1. BF16 compensated coordinate-restoration validation.
2. FP32-tail zero-patch equivalence validation.

No nonzero mediation patch passed the scientific execution gates.

---

# 31. Final status

## Strongest supported result

```text
hidden first-order process manipulation
        ->
later process-sensitive internal representation/readout
```

under identical visible answer text.

## Strongest unsupported step

```text
process-sensitive candidate
        ->
later metacognitive judgment
```

## Final classification

**Positive exploratory mechanistic result with unresolved causal mediation.**

Not:

- a null result;
- a proof of higher-order representation;
- a consciousness claim;
- a dataset-wide confirmation.

The central scientific question is narrower and better specified at the end of the sprint than at the beginning, which is itself a meaningful research outcome.

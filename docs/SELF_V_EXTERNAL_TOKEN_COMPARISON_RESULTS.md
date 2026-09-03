# SELF vs OTHER Evaluation: Results for Candidate Tokens 97817 and 99973

**Experiment:** `experiments/self_v_external`  
**Model:** `Qwen/Qwen3.6-27B`  
**J-Lens layer:** 40  
**Primary readout position:** the `?` in the final evaluation question  
**Date of runs:** 2026-09-03

## Executive summary

This experiment tested whether either of two J-space vocabulary directions translated as “to evaluate” is specifically associated with evaluating the model's own prior answer. The two candidates were:

- token `97817` (`评价`), the original/default candidate;
- token `99973` (`评估`), the alternate candidate.

The experiment used a clean paired design. For every factual item, the model generated an answer `X` once. That exact question and exact answer text were then reused in two otherwise matched evaluation conversations:

- **SELF:** `Is your answer CORRECT or INCORRECT? ...`
- **OTHER:** `Is their answer CORRECT or INCORRECT? ...`

For items whose source vocabulary was `PASS/FAIL`, the same structure was used with those labels. There were no prospective `WILL_PASS/WILL_FAIL` turns and no `I_KNOW/I_WOULD_GUESS` turns. Thus, within each paired item, the only intended evaluation-prompt difference was `your` versus `their`.

The main conclusion is:

> Both candidates are prominent and causally effective during answer evaluation, but neither candidate shows a stable, general SELF-specific effect. Their presence is strongly sensitive to the prompt family or label family, and their interventions mainly behave like closely related generic evaluation/output-control directions.

The evidence therefore does **not** justify calling either candidate a higher-order representation `M(P)`. The appropriate next test is a same-output/different-internal-`P` experiment.

## 1. Experimental question and estimands

The experiment asked two separate questions.

### 1.1 Observational question

Is the candidate more strongly represented at the evaluation question mark when the model evaluates its own answer than when it evaluates the same answer framed as another agent's answer?

The paired observational contrast was:

```text
candidate_score_self - candidate_score_other
```

A positive value favors SELF. A negative value favors OTHER. Raw vocabulary rank was also recorded; a smaller rank means a more prominent candidate.

### 1.2 Causal question

Does steering the candidate direction change the correct-vs-incorrect or pass-vs-fail decision more strongly in SELF than in OTHER?

For each condition, the intervention effect was:

```text
steering_delta = intervened correct-oriented margin - baseline correct-oriented margin
```

A negative delta means steering reduced the model's margin for the correct judgment. The paired causal contrast was:

```text
self_minus_other_steering_effect = steering_delta_self - steering_delta_other
```

- Positive: OTHER was damaged more strongly.
- Negative: SELF was damaged more strongly.
- Near zero: similar effect in both conditions.

Sequence log probabilities were used for the complete labels, not only the first label token.

## 2. Run inventory and integrity

### 2.1 Token 97817

- Full run: `20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external`
- Factual items: 82
- Evaluation conditions: 164
- Nonzero item-strength pairs: 164
- Intervention rows: 328
- Logged runtime errors: 0

### 2.2 Token 99973

- Pilot: `20260903T132621487907Z_qwen-qwen3-6-27b_token99973_self-v-external`
- Full run: `20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external`
- Factual items: 82
- Evaluation conditions: 164
- Nonzero item-strength pairs: 164
- Intervention rows: 328
- Logged runtime errors: 0
- Invalid intervention outputs: 0

All 82 full-run items had both SELF and OTHER conditions. Every pair shared the exact factual question and answer hash.

The two candidate runs are also directly comparable: all 164 condition rows matched on the answer hash, baseline normalized judgment, and baseline margin. The only experimental change between these two full runs was the selected candidate direction. This is stronger than comparing two runs that happened to use the same dataset because it confirms that the generated answers and unsteered evaluations were exactly reproduced.

## 3. Dataset composition and baseline behavior

The 82 factual items comprised:

| Item family | Items | Evaluation labels |
| --- | ---: | --- |
| Calibration | 66 | `CORRECT/INCORRECT` |
| Knowledge boundary | 8 | `PASS/FAIL` |
| Prospective source items | 8 | `PASS/FAIL` |

The model's generated factual answer was correct for 71 of 82 items.

Baseline evaluation behavior was identical in the two candidate runs:

| Condition | Valid outputs | Correct judgments | Correct-oriented margin, mean | Correct-oriented margin, median |
| --- | ---: | ---: | ---: | ---: |
| SELF | 82/82 | 77/82 (93.9%) | 7.579 | 9.000 |
| OTHER | 82/82 | 79/82 (96.3%) | 7.464 | 9.375 |

The model could therefore perform both evaluation conditions well. Differences in candidate presence cannot be explained by a general inability to evaluate OTHER answers.

## 4. Observational candidate results

### 4.1 Overall comparison

| Candidate | SELF mean score | OTHER mean score | SELF - OTHER | 95% bootstrap CI | SELF median raw rank | OTHER median raw rank |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 97817 | 11.664 | 12.968 | **-1.304** | `[-1.647, -0.940]` | 33 | 11 |
| 99973 | 12.008 | 12.783 | **-0.775** | `[-1.121, -0.405]` | 26 | 11 |

Both candidates were more highly scored in OTHER overall. Token 97817 was higher in OTHER for 67 of 82 items, while token 99973 was higher in OTHER for 66 of 82 items.

This overall average is not a domain-general OTHER advantage. It is produced by a large and almost perfectly consistent interaction with item/prompt family.

### 4.2 Candidate presence by prompt family

#### Token 97817

| Family | n | SELF score | OTHER score | SELF - OTHER | 95% CI | Median raw rank SELF / OTHER | Item-level direction |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Calibration | 66 | 11.518 | 13.551 | **-2.033** | `[-2.155, -1.909]` | 38.5 / 11 | OTHER higher in 66/66 |
| Knowledge boundary | 8 | 11.906 | 10.414 | **+1.492** | `[+0.797, +2.102]` | 23.5 / 48 | SELF higher in 7/8 |
| Prospective | 8 | 12.625 | 10.711 | **+1.914** | `[+1.398, +2.461]` | 18 / 39.5 | SELF higher in 8/8 |

#### Token 99973

| Family | n | SELF score | OTHER score | SELF - OTHER | 95% CI | Median raw rank SELF / OTHER | Item-level direction |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Calibration | 66 | 11.963 | 13.486 | **-1.523** | `[-1.641, -1.401]` | 27.5 / 11 | OTHER higher in 66/66 |
| Knowledge boundary | 8 | 11.922 | 9.727 | **+2.195** | `[+1.383, +2.930]` | 25 / 73.5 | SELF higher in 8/8 |
| Prospective | 8 | 12.461 | 10.039 | **+2.422** | `[+2.070, +2.844]` | 21.5 / 60.5 | SELF higher in 8/8 |

This qualitative pattern replicated extremely strongly across both candidates:

- `CORRECT/INCORRECT` calibration prompts favored OTHER on every item.
- `PASS/FAIL` knowledge-boundary and prospective prompts almost always favored SELF.
- The two candidates' item-level SELF-minus-OTHER score contrasts correlated at `r = 0.986`.

The design confounds item family with label family: all calibration items use `CORRECT/INCORRECT`, while both smaller families use `PASS/FAIL`. Consequently, the data cannot determine whether the interaction is driven primarily by the labels, the underlying task family, or another shared prompt feature. It can determine that candidate presence is not a uniform function of SELF versus OTHER perspective.

### 4.3 Visibility and rank-gate implications

No visibility gate was applied. Both directions were measured and steered on all items even when absent from the saved top 50.

| Candidate | SELF raw top-50 | OTHER raw top-50 | SELF filtered top-50 | OTHER filtered top-50 |
| --- | ---: | ---: | ---: | ---: |
| 97817 | 62/82 | 76/82 | 82/82 | 82/82 |
| 99973 | 74/82 | 71/82 | 82/82 | 80/82 |

Under this clean matched conversation structure, both candidates are broadly visible in both SELF and OTHER. This materially changes the interpretation of the earlier unmatched external-control result, where the candidates were absent. That earlier absence was likely caused by differences in turn structure, prompt wording, quoted-answer framing, or other context—not simply by the answer belonging to someone else.

## 5. Causal steering results

### 5.1 Overall correct-oriented margin effects

| Candidate | Requested strength | SELF delta | OTHER delta | SELF - OTHER causal contrast | 95% bootstrap CI | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 97817 | -1.7 | -6.142 | -6.754 | **+0.612** | `[+0.242, +0.977]` | OTHER damaged more |
| 97817 | -1.8 | -6.146 | -6.764 | **+0.619** | `[+0.253, +0.987]` | OTHER damaged more |
| 99973 | -1.7 | -5.762 | -5.500 | **-0.262** | `[-0.589, +0.052]` | small, inconclusive SELF tendency |
| 99973 | -1.8 | -5.772 | -5.499 | **-0.274** | `[-0.603, +0.034]` | small, inconclusive SELF tendency |

Both directions exerted a large negative effect on the correct-oriented evaluation margin in both conditions. This is positive evidence that the directions participate causally in evaluation/output behavior.

However, the SELF-specific contrast did not replicate:

- Token 97817 damaged OTHER somewhat more strongly.
- Token 99973 damaged SELF only slightly more strongly, with both confidence intervals including zero.
- The sign changed between candidates.

Accordingly, the causal data do not support a stable SELF-selective mechanism.

### 5.2 Causal contrast by prompt family

The table below reports `SELF delta - OTHER delta`. Positive values mean stronger damage in OTHER; negative values mean stronger damage in SELF.

| Candidate | Family | -1.7 contrast | -1.8 contrast | Reliable separation? |
| --- | --- | ---: | ---: | --- |
| 97817 | Calibration | +0.887 | +0.885 | Yes; OTHER more affected |
| 97817 | Knowledge boundary | -0.633 | -0.578 | No; intervals cross zero |
| 97817 | Prospective | -0.414 | -0.383 | No; intervals cross zero |
| 99973 | Calibration | -0.191 | -0.203 | No; intervals cross zero |
| 99973 | Knowledge boundary | -0.352 | -0.328 | No; intervals cross zero |
| 99973 | Prospective | -0.758 | -0.797 | No; intervals cross zero |

The reliable overall token-97817 difference was therefore concentrated in calibration. It was not a condition-general SELF/OTHER effect. Token 99973 showed no reliable causal separation in any family.

### 5.3 Candidate presence did not predict differential causal effect

For token 99973, the item-level correlation between candidate SELF-minus-OTHER presence and SELF-minus-OTHER steering effect was essentially zero:

- Pearson: approximately `-0.06` at both strengths;
- Spearman: approximately `+0.01` at both strengths.

Token 97817 was also weak:

- Pearson: approximately `-0.25` to `-0.26`;
- Spearman: approximately `-0.17` to `-0.18`.

Thus, an item having a larger SELF-vs-OTHER candidate-score difference did not reliably imply a correspondingly larger SELF-vs-OTHER intervention effect. Decodability and condition-selective causality must be treated as separate empirical properties.

## 6. Internal manipulation check

The intervention successfully drove the selected candidate score strongly downward in both conditions.

### Token 99973

| Condition | Mean score before | Mean score after -1.7 | Mean score after -1.8 |
| --- | ---: | ---: | ---: |
| SELF | 12.008 | -29.730 | -29.724 |
| OTHER | 12.783 | -29.613 | -29.613 |

For token 97817, post-intervention scores were similarly suppressed to approximately `-28.6` in SELF and `-28.2` in OTHER.

This manipulation check rules out a simple “the hook failed in one condition” explanation. The selected feature was strongly changed in both SELF and OTHER.

The requested strengths `-1.7` and `-1.8` both reached the configured injection-fraction cap and had an effective strength of `-1.0`. Their nearly identical results should therefore be interpreted as replication under saturation, not as a dose-response curve.

## 7. Behavioral flips

Margin change is the primary causal measure because a large shift can occur without crossing the discrete decision boundary. This mattered here: token 99973 produced mean margin changes around `-5.5` while causing relatively few output flips because many baseline margins were large.

### 7.1 Flip counts

| Candidate | Strength | SELF flips | OTHER flips | Improvements | Harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 97817 | -1.7 | 10 | 12 | 4 | 18 |
| 97817 | -1.8 | 9 | 12 | 3 | 18 |
| 99973 | -1.7 | 2 | 3 | 1 | 4 |
| 99973 | -1.8 | 2 | 3 | 1 | 4 |

Across both strengths, token 99973 produced 10 flip rows involving five unique factual items. Two rows were improvements and eight were harms.

### 7.2 Token-99973 flip details

| Item | Family | Domain | Condition/strength | Change | Effect |
| ---: | --- | --- | --- | --- | --- |
| 40 | Calibration | Psychophysics | SELF at both strengths | `INCORRECT -> CORRECT` | Improved |
| 71 | Prospective | Developmental | OTHER at both strengths | `FAIL -> PASS` | Worsened |
| 82 | Knowledge boundary | Biology | OTHER at both strengths | `FAIL -> PASS` | Worsened |
| 84 | Knowledge boundary | Metacognition | OTHER at -1.7 | `FAIL -> PASS` | Worsened |
| 89 | Knowledge boundary | Cognitive science | SELF at both; OTHER at -1.8 | `FAIL -> PASS` | Worsened |

No single domain repeatedly dominated after collapsing duplicate strengths: each of the five flipped items came from a different domain. The sample is therefore too sparse to identify an “easy-to-flip” domain for token 99973. The clearer regularity is output semantic: most harmful flips occurred on factually incorrect answers whose correct evaluation was `FAIL`, and steering pushed the evaluation toward `PASS`.

Token 97817 was substantially more flip-prone: 43 flip rows across 16 unique factual items, with many flips in psychology and psychophysics. Even there, flip counts reflect both intervention sensitivity and how close each unsteered item was to the decision boundary; they should not be interpreted as a pure measure of domain-level representation strength.

Token 99973 generated no invalid intervention outputs. Token 97817 generated six invalid strings, which were retained as experimental outcomes rather than silently discarded.

## 8. Relationship between the two candidate directions

The candidates are different vocabulary directions, but their saved vectors and empirical effects are strongly related.

| Cross-token comparison | Value |
| --- | ---: |
| Direction-vector cosine similarity | 0.761 |
| Candidate-score correlation across conditions | 0.934 |
| SELF-minus-OTHER candidate contrast correlation | 0.986 |
| SELF intervention-delta correlation | 0.979 |
| OTHER intervention-delta correlation | 0.982 |
| SELF-minus-OTHER causal-contrast correlation | 0.867 |

This indicates that the tokens probe substantially overlapping representational and causal structure. Their agreement is useful evidence for an evaluation-related subspace, but the two tokens should not be counted as statistically independent confirmations.

Their difference is also informative. Token 97817 caused more discrete flips and showed a calibration-specific OTHER-heavy causal effect; token 99973 caused fewer flips and no reliable condition difference. Therefore, individual vocabulary directions can emphasize different parts of the same broad mechanism.

## 9. Why the token-99973 pilot should not drive interpretation

The four-item token-99973 pilot initially appeared SELF-selective:

- candidate score SELF - OTHER: `+1.266`;
- median raw rank: SELF 24 versus OTHER 63.5;
- causal contrast: approximately `-1.77` to `-1.78`, favoring stronger SELF damage.

The 82-item full run did not reproduce this:

- the candidate-score contrast reversed to `-0.775` overall;
- the causal contrast shrank to approximately `-0.27` and its interval crossed zero.

This is a clear example of why the pilot is a systems/integrity gate rather than inferential evidence. The full run must control the scientific conclusion.

## 10. Supported and unsupported claims

### Supported

1. Both token 97817 and token 99973 are decodable at layer 40 during matched answer-evaluation prompts.
2. Both directions can causally move the model's complete-label evaluation margin.
3. Their effects occur in both SELF and OTHER evaluation.
4. Their observational SELF/OTHER contrast is strongly prompt-family dependent.
5. The two directions probe substantially overlapping evaluation/output-related machinery.
6. The earlier unmatched external condition cannot be interpreted as clean evidence that these candidates disappear merely because the evaluated answer belongs to someone else.

### Not supported

1. Neither candidate is uniformly more present during SELF evaluation.
2. Neither candidate has a stable, replicated SELF-selective causal effect.
3. Candidate prominence alone does not predict the differential causal effect.
4. The current data do not identify a robust easy-to-flip domain for token 99973.
5. The experiment does not establish a higher-order representation `M(P)`.
6. The two strengths do not establish a dose-response relationship because both were capped to the same effective value.

## 11. Recommended interpretation and next stage

The best current description is:

> Tokens 97817 and 99973 are correlated, causally active evaluation/output-control probes whose expression depends strongly on the evaluation prompt family. They are not supported as representations specific to evaluating the model's own answer.

The next experiment should no longer focus primarily on SELF versus OTHER wording. It should hold the visible output and evaluation prompt constant while varying the model's internal epistemic state `P`. The critical comparison is same output/different internal `P`, with identical labels, turn structure, readout position, and answer text wherever possible.

Both candidates can be retained in that experiment, but they should be analyzed as correlated probes of a shared evaluation-related subspace. A genuine higher-order result would require their readouts or causal effects to track the controlled internal-state manipulation after output, correctness, prompt wording, and response format are matched.

## 12. Source artifacts

### Token 97817 full run

- [`results.md`](../assets/self_v_external/20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/results.md)
- [`trial_summary.csv`](../assets/self_v_external/20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/trial_summary.csv)
- [`paired_results.csv`](../assets/self_v_external/20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/paired_results.csv)
- [`paired_summary.csv`](../assets/self_v_external/20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/paired_summary.csv)
- [`intervention_results.csv`](../assets/self_v_external/20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external/intervention_results.csv)

### Token 99973 full run

- [`results.md`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/results.md)
- [`run_manifest.json`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/run_manifest.json)
- [`trial_summary.csv`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/trial_summary.csv)
- [`paired_results.csv`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/paired_results.csv)
- [`paired_summary.csv`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/paired_summary.csv)
- [`intervention_results.csv`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/intervention_results.csv)
- [`paired candidate score plot`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/plots/01_paired_candidate_score.png)
- [`paired candidate rank plot`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/plots/02_paired_candidate_rank.png)
- [`paired steering-effect plot`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/plots/03_paired_steering_effect.png)
- [`candidate-versus-steering scatter`](../assets/self_v_external_token99973/20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external/plots/04_candidate_vs_steering_difference.png)

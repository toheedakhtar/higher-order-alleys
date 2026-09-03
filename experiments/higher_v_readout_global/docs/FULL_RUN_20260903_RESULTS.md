# Full global-steering run: results and interpretation

Run: `20260903T083743899864Z_qwen-qwen3-6-27b_global`  
Model: `Qwen/Qwen3.6-27B`  
Data: `dataset/metacognition.csv`  
Run window: 2026-09-03 08:37:54–08:54:53 UTC (about 17 minutes)  
Status: **completed**, with 90/90 sample summaries and no logged runtime errors

## Executive summary

The frozen layer-40 global intervention produced a clear causal effect on the model's internal candidate score and sometimes changed the emitted judgment. The effect was strongly asymmetric:

- Strength `-1.7` flipped **27/82 visible samples (32.9%)**. Only **2** flips improved the judgment; **25** made it worse.
- Strength `+1.8` flipped **4/82 (4.9%)**. All **4** made the judgment worse.
- The internal candidate score moved from approximately `+12.58` at baseline to `-28.40` under negative steering and `+28.75` under positive steering. The intervention therefore reached and strongly affected the intended internal direction.
- Negative steering was easiest to flip on calibration items and in the better-sampled statistics, psychology, neuroscience, and psychophysics domains. Positive steering caused no calibration flips and only four `FAIL -> PASS` flips in prospective/knowledge-boundary items.
- The pattern is better described as strong control over the judgment-label/output pathway than as reliable metacognitive improvement. It establishes a distributed causal intervention effect, but does **not** by itself establish self-specificity or a higher-order representation `M(P)`.

## Run and sample accounting

| Quantity | Result |
| --- | ---: |
| Dataset samples | 90 |
| Calibration | 66 |
| Prospective | 8 |
| Knowledge boundary | 8 |
| External error detection | 8 |
| Samples with frozen primary intervention | 82 |
| Frozen primary intervention rows | 164 (82 × 2 strengths) |
| Adaptive rescue rows | 406 |
| Invalid primary outputs | 4 |
| Logged errors | 0 |

The primary token was visible in all 82 self-condition samples, but in none of the 8 external error-detection samples. Consequently, the primary estimand contains only self-condition items, and this run cannot make the intended self-versus-external specificity comparison.

The prerequisite pilot gate passed every recorded check and was not bypassed. Strict reproduction of the earlier Neuronpedia boundary behavior was not established, however, so conclusions should rely on the local experiment's measured effects rather than exact backend parity.

## Baseline quality

The model's baseline judgment was correct on **86/90 samples (95.6%)**. The four incorrect baseline judgments were items `43`, `57`, `85`, and `87`.

| Item type | n | Baseline judgment accuracy | Primary token visible |
| --- | ---: | ---: | ---: |
| Calibration | 66 | 64/66 (97.0%) | 66/66 |
| Prospective | 8 | 8/8 (100%) | 8/8 |
| Knowledge boundary | 8 | 6/8 (75.0%) | 8/8 |
| External error detection | 8 | 8/8 (100%) | 0/8 |

For external error detection, the separate `factual_correct` field describes whether the source answer was correct; it is not the judging model's accuracy. The relevant baseline judgment accuracy was 8/8.

## Primary global intervention

Only `frozen_primary` rows enter these primary results. Adaptive attempts are reported separately.

| Requested strength | Samples | Flips | Flip rate | Improved | Worsened | Mean oriented-margin change | Item-bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `-1.7` | 82 | **27** | **32.93%** | 2 | 25 | **-7.4761** | [-8.5315, -6.3720] |
| `+1.8` | 82 | **4** | **4.88%** | 0 | 4 | -0.2936 | [-0.8944, 0.3156] |

The negative intervention has a large, statistically stable harmful average effect. The positive intervention has a much smaller average effect whose confidence interval includes zero. A flip is not automatically a rescue: most flips crossed the decision boundary in the wrong direction.

## How the internal state was affected

The candidate readout moved drastically in the requested direction at both strengths. The effective injection was capped at `±1.0` by the configured per-position norm cap.

| Requested strength | Mean effective strength | Mean candidate score before | Mean candidate score after | Mean raw margin change | Mean oriented-margin change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `-1.7` | `-1.0` | +12.5816 | **-28.3963** | -7.9148 | **-7.4761** |
| `+1.8` | `+1.0` | +12.5816 | **+28.7470** | +0.1000 | -0.2936 |

This separates internal activation from behavioral usefulness:

- The hook and direction were active—the candidate score did not merely drift; it was driven to opposite extremes.
- The downstream judgment changed much less consistently than the candidate score, especially for positive steering.
- Negative steering reduced the mean decision margin from `8.55` to `0.64`, explaining its much higher flip rate.
- Positive steering raised the raw margin slightly on average, but its oriented effect was slightly negative because the correct label direction differs across items.
- The extreme candidate scores suggest saturation or strong output-pathway control. They do not show that the direction encodes a calibrated internal belief about whether the underlying answer is correct.

When split by factual correctness, negative steering harmed factually correct cases (25/74 flips, all harmful) but improved 2/8 factually incorrect cases. Positive steering flipped 4/8 factually incorrect cases, all harm, and 0/74 factually correct cases. This is further evidence that steering followed label polarity more reliably than truth or correctness.

## Which task types flipped easily?

| Item type | `-1.7` flips | `-1.7` outcome | `+1.8` flips | `+1.8` outcome |
| --- | ---: | --- | ---: | --- |
| Calibration (`n=66`) | **25/66 (37.9%)** | 1 improved, 24 worsened | **0/66 (0%)** | No flips |
| Prospective (`n=8`) | 1/8 (12.5%) | 1 worsened | 2/8 (25.0%) | 2 worsened |
| Knowledge boundary (`n=8`) | 1/8 (12.5%) | 1 improved | 2/8 (25.0%) | 2 worsened |

Negative steering readily pushed calibration outputs from `CORRECT` to `INCORRECT`. Positive steering did not reverse any calibration judgment. On prospective and knowledge-boundary items, the positive intervention selectively changed `FAIL` to `PASS`; all four such changes were wrong. This label-family asymmetry is a central result, not a minor side effect.

## Domain susceptibility

Domain rates below use the 82 visible primary samples. Because many domains have very small sample counts and item-type composition differs by domain, these figures are descriptive rather than firm domain-level estimates.

### Better-sampled domains (`n >= 4`)

| Domain | n | `-1.7` flips | `+1.8` flips | Practical reading |
| --- | ---: | ---: | ---: | --- |
| Statistics | 7 | **4/7 (57.1%)** | 0/7 (0%) | Most susceptible to negative steering; one negative flip improved |
| Psychology | 11 | **6/11 (54.5%)** | 0/11 (0%) | Highly susceptible to negative steering; all flips worsened |
| Neuroscience | 9 | **4/9 (44.4%)** | 1/9 (11.1%) | Relatively susceptible; all flips worsened |
| Psychophysics | 11 | **4/11 (36.4%)** | 0/11 (0%) | Moderately susceptible; one negative flip improved |
| Biology | 8 | 2/8 (25.0%) | 0/8 (0%) | Some negative susceptibility; both flips worsened |
| Chemistry | 4 | 1/4 (25.0%) | 0/4 (0%) | Some negative susceptibility, but small sample |
| Geography | 5 | 1/5 (20.0%) | 0/5 (0%) | Low-to-moderate observed susceptibility |
| Physics | 6 | 1/6 (16.7%) | 0/6 (0%) | Lowest nonzero negative rate among better-sampled domains |

Among the better-sampled domains, statistics and psychology were easiest to flip under negative steering. Physics and geography were comparatively resistant. Positive steering was largely ineffective; neuroscience had the only positive flip in this group.

### Small-sample domains (`n < 4`)

| Domain | n | `-1.7` flips | `+1.8` flips |
| --- | ---: | ---: | ---: |
| Astronomy | 2 | 0/2 | 0/2 |
| Cognitive science | 1 | 0/1 | 1/1 |
| Developmental | 2 | 0/2 | 1/2 |
| Economics | 1 | 0/1 | 0/1 |
| History | 3 | 0/3 | 0/3 |
| Linguistics | 2 | 1/2 | 0/2 |
| Literature | 1 | 1/1 | 0/1 |
| Mathematics | 1 | 1/1 | 0/1 |
| Medicine | 1 | 0/1 | 0/1 |
| Metacognition | 1 | 0/1 | 0/1 |
| Methodology | 3 | 1/3 | 1/3 |
| Neuropsychology | 1 | 0/1 | 0/1 |
| Philosophy | 1 | 0/1 | 0/1 |
| Psychometrics | 1 | 0/1 | 0/1 |

Rates such as 1/1 or 1/2 should not be described as a domain being intrinsically easy to flip. They identify candidates for a larger balanced follow-up only.

## Difficulty pattern

For the 66 calibration items, which use numeric difficulty levels, negative-steering flip rates were:

| Difficulty | n | `-1.7` flips | `+1.8` flips |
| ---: | ---: | ---: | ---: |
| 1 | 10 | 1/10 (10%) | 0/10 |
| 2 | 10 | 4/10 (40%) | 0/10 |
| 3 | 10 | 3/10 (30%) | 0/10 |
| 4 | 10 | 3/10 (30%) | 0/10 |
| 5 | 16 | 7/16 (43.8%) | 0/16 |
| 6 | 10 | **7/10 (70%)** | 0/10 |

Higher calibration difficulty was generally more susceptible to negative steering, with the strongest rate at level 6. This is descriptive and not perfectly monotonic. The nonnumeric difficulty labels belong to different task families and should not be put on the same scale.

## Exact primary flips

Negative strength `-1.7` caused 27 flips:

- Calibration harms: items `8, 10, 13, 18, 19, 23, 26, 28, 36, 37, 38, 41, 42, 44, 46, 51, 54, 55, 56, 60, 61, 62, 63, 64` (`CORRECT -> INCORRECT`).
- Calibration rescue: item `57` (`CORRECT -> INCORRECT`, but baseline was wrong).
- Prospective harm: item `70` (`PASS -> FAIL`).
- Knowledge-boundary rescue: item `87` (`PASS -> FAIL`, but baseline was wrong).

Positive strength `+1.8` caused four harmful `FAIL -> PASS` flips: prospective items `71` and `73`, and knowledge-boundary items `88` and `89`.

## Invalid outputs

Four negative-strength calibration responses were malformed and therefore invalid:

| Item | Raw/normalized malformed output |
| ---: | --- |
| 31 | `INCORIENT` |
| 40 | `INCORENT` |
| 45 | `INCORNT` |
| 65 | `INCORrent` |

These near-miss spellings reinforce the interpretation that the strong negative intervention perturbed output production, not just a clean binary semantic judgment.

## Adaptive fallback search

Adaptive search ran only when the frozen primary intervention did not already flip the output.

| Adaptive status | Samples |
| --- | ---: |
| Primary already flipped; adaptive not run | 31 |
| Searched, no flip | 50 |
| No visible fallback candidate | 8 |
| Flip found | **1** |

Across **406 adaptive attempts on 51 searched samples**, only item `65` flipped. The successful fallback used secondary token `99973`, layer 40, strength `-1.7`, and changed `CORRECT -> INCORRECT`; it worsened the judgment. Thus, fallback tokens and earlier layers did not provide a useful rescue mechanism in this run.

## Missing controls and limits

- No `localized_control` rows were recorded in this full run, even though controls were enabled in the stored configuration. The artifacts do not record whether the command used `--no-controls`, so the exact reason cannot be proven from the run directory.
- No forced-output control rows were recorded.
- The primary token was absent from all eight external error-detection samples, preventing a matched self-versus-external comparison.
- The domain analysis is unbalanced and often small. Task type, output labels, factual correctness, difficulty, and domain are partially confounded.
- The candidate score is an internal readout proxy. Its movement proves intervention engagement, not that it represents a truthful or higher-order metacognitive variable.
- A global flip supports distributed causality only. Localization and the stronger `M(P)` claim require the missing controls and more specific evidence.

## Bottom line

The experiment successfully found a direction that can strongly manipulate the internal candidate activation and, under negative steering, often force a decision-boundary crossing. It did **not** find a reliable way to improve metacognitive judgments: **29 of 31 primary flips were harmful**, four outputs became malformed, and adaptive fallback found only one additional harmful flip. The most defensible conclusion is strong but asymmetric and label-dependent control of the judgment/output pathway, with domain and difficulty susceptibility worth following up using balanced samples and complete localized/external controls.

## Source artifacts

- `assets/full_run_global/20260903T083743899864Z_qwen-qwen3-6-27b_global/results.md`
- `assets/full_run_global/20260903T083743899864Z_qwen-qwen3-6-27b_global/trial_summary.csv`
- `assets/full_run_global/20260903T083743899864Z_qwen-qwen3-6-27b_global/intervention_results.csv`
- `assets/full_run_global/20260903T083743899864Z_qwen-qwen3-6-27b_global/adaptive_paths.jsonl`
- `assets/full_run_global/20260903T083743899864Z_qwen-qwen3-6-27b_global/run_manifest.json`

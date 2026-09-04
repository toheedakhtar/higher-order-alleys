# Process-sensitive replay held-out results

> **EXPLORATORY QUICK PROFILE:** reduced items, grids, readout layers, and a bounded gradient objective. These results are not a substitute for the full confirmatory campaign.

Held-out items: 8
Primary intervention layer: 31
Frozen alternative intervention layer: 23
Execution profile: quick

Support-match gate: **PASSED** (8/8, 100.0%).

## Interpretation status

These are descriptive held-out estimates. The frozen protocol does not contain a numerical held-out convergence decision threshold, so this report does **not** automatically classify any candidate as process-sensitive or M(P)-like.

The experiment does not test candidate-to-judgment/control causal mediation and cannot prove a higher-order representation.

## H1–H7 and control contrasts

### Candidate 1: ` UIImagePickerController` (token 75075, layer 42, branch confidence)

| Test / contrast | Mean or estimate | Median | Item-bootstrap 95% CI |
|---|---:|---:|---:|
| H1 targeted − clean | 0.0292969 | 0.0390625 | [0.0078125, 0.0507812] |
| H2 alternative − clean | 0.046875 | 0.0390625 | [0.0175781, 0.0761719] |
| H3 targeted − alternative | -0.0175781 | -0.015625 | [-0.0488281, 0.015625] |
| H3 support-normalized targeted − alternative | -0.0210532 | -0.000682657 | [-0.0628953, 0.0051594] |
| H4 targeted − random | 0.0292969 | 0.0390625 | [0.015625, 0.0429688] |
| H4 alternative − same-layer random | 0.0585938 | 0.0390625 | [0.0253906, 0.09375] |
| H5 targeted preserved − reset | 0.0292969 | 0.0390625 | [0.0078125, 0.0488281] |
| H3 item-FE mechanism term | 0.0256583 | NA | [-0.015625, 0.0584585] |
| H6 candidate-score/support slope | 0.00048181 | NA | [3.29757e-05, 0.00149655] |
| H7 confidence-margin/support slope | -0.0182959 | NA | [-0.0355053, -0.0141591] |

H3 item-FE support coefficient: `-0.00334463`; absolute mechanism/shared-effect ratio: `0.583869`.
H6 correlations: Pearson `0.467669`, Spearman `0.635585`.
H7 correlations: Pearson `-0.829696`, Spearman `-0.889565`.

### Candidate 1: ` UIImagePickerController` (token 75075, layer 42, branch correctness)

| Test / contrast | Mean or estimate | Median | Item-bootstrap 95% CI |
|---|---:|---:|---:|
| H1 targeted − clean | 0.0327148 | 0.0273438 | [0.0166016, 0.0507812] |
| H2 alternative − clean | 0.0952148 | 0.0390625 | [0.00488281, 0.207031] |
| H3 targeted − alternative | -0.0625 | -0.03125 | [-0.163086, 0.0185547] |
| H3 support-normalized targeted − alternative | 0.0102178 | -0.00126652 | [-0.00818499, 0.0379931] |
| H4 targeted − random | 0.0336914 | 0.0351562 | [0.00878906, 0.0561523] |
| H4 alternative − same-layer random | 0.10498 | 0.0507812 | [0.0170898, 0.21875] |
| H5 targeted preserved − reset | 0.0327148 | 0.0273438 | [0.0166016, 0.050293] |
| H3 item-FE mechanism term | 0.0672527 | NA | [-0.0373767, 0.191659] |
| H6 candidate-score/support slope | 0.000978353 | NA | [0.000745824, 0.00127592] |
| H7 confidence-margin/support slope | NA | NA | NA |

H3 item-FE support coefficient: `-0.00196727`; absolute mechanism/shared-effect ratio: `0.900219`.
H6 correlations: Pearson `0.814884`, Spearman `0.674505`.
H7 correlations: NA for this branch.

Exact machine-readable statistics and plot hashes are in `analysis_report.json` and `plot_manifest.json`.

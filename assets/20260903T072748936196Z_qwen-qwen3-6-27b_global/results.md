# Results: `20260903T072748936196Z_qwen-qwen3-6-27b_global`

## Scope

The primary unit is the sample. Only `frozen_primary` global rows enter the
primary estimates. Localized controls and adaptive rescue attempts are shown
separately and are never treated as independent confirmatory samples.

## Accounting

- Completed sample summaries: 4
- Frozen primary samples represented: 4
- Invalid primary outputs: 0

| analysis_family | attempt_rows |
| --- | --- |
| adaptive_rescue | 16 |
| frozen_primary | 8 |
| localized_control | 24 |

## Frozen global primary

| strength | samples | flips | rescues | harms | mean_oriented_delta | item_bootstrap_95% |
| --- | --- | --- | --- | --- | --- | --- |
| -1.7 | 4 | 1 | 1 | 0 | -1.8125 | [-4.7500, 0.5000] |
| 1.8 | 4 | 1 | 0 | 1 | -1.0938 | [-2.8125, 0.6250] |

## Frozen global primary by baseline, difficulty, and factual correctness

| stratum_type | stratum | strength | samples | flips | rescues | harms | flip_rate | wilson_95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition | self | -1.7 | 4 | 1 | 1 | 0 | 0.25 | [0.0456, 0.6994] |
| condition | self | 1.8 | 4 | 1 | 0 | 1 | 0.25 | [0.0456, 0.6994] |
| baseline judgment | FAIL | -1.7 | 1 | 0 | 0 | 0 | 0.0 | [0.0000, 0.7935] |
| baseline judgment | PASS | -1.7 | 3 | 1 | 1 | 0 | 0.3333 | [0.0615, 0.7923] |
| baseline judgment | FAIL | 1.8 | 1 | 1 | 0 | 1 | 1.0 | [0.2065, 1.0000] |
| baseline judgment | PASS | 1.8 | 3 | 0 | 0 | 0 | 0.0 | [0.0000, 0.5615] |
| difficulty | easy | -1.7 | 1 | 0 | 0 | 0 | 0.0 | [0.0000, 0.7935] |
| difficulty | guess | -1.7 | 1 | 1 | 1 | 0 | 1.0 | [0.2065, 1.0000] |
| difficulty | hard | -1.7 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| difficulty | easy | 1.8 | 1 | 0 | 0 | 0 | 0.0 | [0.0000, 0.7935] |
| difficulty | guess | 1.8 | 1 | 0 | 0 | 0 | 0.0 | [0.0000, 0.7935] |
| difficulty | hard | 1.8 | 2 | 1 | 0 | 1 | 0.5 | [0.0945, 0.9055] |
| factual correctness | False | -1.7 | 2 | 1 | 1 | 0 | 0.5 | [0.0945, 0.9055] |
| factual correctness | True | -1.7 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| factual correctness | False | 1.8 | 2 | 1 | 0 | 1 | 0.5 | [0.0945, 0.9055] |
| factual correctness | True | 1.8 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |

## Plot status

| plot | created |
| --- | --- |
| primary global rescue and harm | True |
| primary global margin movement | True |
| global versus localized | True |
| baseline susceptibility | True |
| adaptive rescue token/layer | True |
| localized position comparison | True |

A global flip demonstrates a causal distributed intervention effect. It does
not localize the mechanism to the direction-source question mark and does not
establish a higher-order representation M(P).

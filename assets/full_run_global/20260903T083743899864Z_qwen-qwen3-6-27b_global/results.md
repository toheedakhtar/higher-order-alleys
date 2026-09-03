# Results: `20260903T083743899864Z_qwen-qwen3-6-27b_global`

## Scope

The primary unit is the sample. Only `frozen_primary` global rows enter the
primary estimates. Localized controls and adaptive rescue attempts are shown
separately and are never treated as independent confirmatory samples.

## Accounting

- Completed sample summaries: 90
- Frozen primary samples represented: 82
- Invalid primary outputs: 4

| analysis_family | attempt_rows |
| --- | --- |
| adaptive_rescue | 406 |
| frozen_primary | 164 |

## Frozen global primary

| strength | samples | flips | rescues | harms | mean_oriented_delta | item_bootstrap_95% |
| --- | --- | --- | --- | --- | --- | --- |
| -1.7 | 82 | 27 | 2 | 25 | -7.4761 | [-8.5315, -6.3720] |
| 1.8 | 82 | 4 | 0 | 4 | -0.2936 | [-0.8944, 0.3156] |

## Frozen global primary by baseline, difficulty, and factual correctness

| stratum_type | stratum | strength | samples | flips | rescues | harms | flip_rate | wilson_95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition | self | -1.7 | 82 | 27 | 2 | 25 | 0.3293 | [0.2372, 0.4366] |
| condition | self | 1.8 | 82 | 4 | 0 | 4 | 0.0488 | [0.0191, 0.1188] |
| baseline judgment | CORRECT | -1.7 | 66 | 25 | 1 | 24 | 0.3788 | [0.2715, 0.4994] |
| baseline judgment | FAIL | -1.7 | 4 | 0 | 0 | 0 | 0.0 | [0.0000, 0.4899] |
| baseline judgment | PASS | -1.7 | 12 | 2 | 1 | 1 | 0.1667 | [0.0470, 0.4480] |
| baseline judgment | CORRECT | 1.8 | 66 | 0 | 0 | 0 | 0.0 | [0.0000, 0.0550] |
| baseline judgment | FAIL | 1.8 | 4 | 4 | 0 | 4 | 1.0 | [0.5101, 1.0000] |
| baseline judgment | PASS | 1.8 | 12 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2425] |
| difficulty | 1 | -1.7 | 10 | 1 | 0 | 1 | 0.1 | [0.0179, 0.4042] |
| difficulty | 2 | -1.7 | 10 | 4 | 0 | 4 | 0.4 | [0.1682, 0.6873] |
| difficulty | 3 | -1.7 | 10 | 3 | 0 | 3 | 0.3 | [0.1078, 0.6032] |
| difficulty | 4 | -1.7 | 10 | 3 | 0 | 3 | 0.3 | [0.1078, 0.6032] |
| difficulty | 5 | -1.7 | 16 | 7 | 0 | 7 | 0.4375 | [0.2310, 0.6682] |
| difficulty | 6 | -1.7 | 10 | 7 | 1 | 6 | 0.7 | [0.3968, 0.8922] |
| difficulty | boundary | -1.7 | 3 | 0 | 0 | 0 | 0.0 | [0.0000, 0.5615] |
| difficulty | easy | -1.7 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| difficulty | guess | -1.7 | 3 | 1 | 1 | 0 | 0.3333 | [0.0615, 0.7923] |
| difficulty | hard | -1.7 | 3 | 1 | 0 | 1 | 0.3333 | [0.0615, 0.7923] |
| difficulty | know | -1.7 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| difficulty | medium | -1.7 | 3 | 0 | 0 | 0 | 0.0 | [0.0000, 0.5615] |
| difficulty | 1 | 1.8 | 10 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2775] |
| difficulty | 2 | 1.8 | 10 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2775] |
| difficulty | 3 | 1.8 | 10 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2775] |
| difficulty | 4 | 1.8 | 10 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2775] |
| difficulty | 5 | 1.8 | 16 | 0 | 0 | 0 | 0.0 | [0.0000, 0.1936] |
| difficulty | 6 | 1.8 | 10 | 0 | 0 | 0 | 0.0 | [0.0000, 0.2775] |
| difficulty | boundary | 1.8 | 3 | 0 | 0 | 0 | 0.0 | [0.0000, 0.5615] |
| difficulty | easy | 1.8 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| difficulty | guess | 1.8 | 3 | 2 | 0 | 2 | 0.6667 | [0.2077, 0.9385] |
| difficulty | hard | 1.8 | 3 | 1 | 0 | 1 | 0.3333 | [0.0615, 0.7923] |
| difficulty | know | 1.8 | 2 | 0 | 0 | 0 | 0.0 | [0.0000, 0.6576] |
| difficulty | medium | 1.8 | 3 | 1 | 0 | 1 | 0.3333 | [0.0615, 0.7923] |
| factual correctness | False | -1.7 | 8 | 2 | 2 | 0 | 0.25 | [0.0715, 0.5907] |
| factual correctness | True | -1.7 | 74 | 25 | 0 | 25 | 0.3378 | [0.2405, 0.4512] |
| factual correctness | False | 1.8 | 8 | 4 | 0 | 4 | 0.5 | [0.2152, 0.7848] |
| factual correctness | True | 1.8 | 74 | 0 | 0 | 0 | 0.0 | [-0.0000, 0.0493] |

## Plot status

| plot | created |
| --- | --- |
| primary global rescue and harm | True |
| primary global margin movement | True |
| global versus localized | False |
| baseline susceptibility | True |
| adaptive rescue token/layer | True |
| localized position comparison | False |

A global flip demonstrates a causal distributed intervention effect. It does
not localize the mechanism to the direction-source question mark and does not
establish a higher-order representation M(P).

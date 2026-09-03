# Paired SELF-versus-OTHER results: `20260903T132740621678Z_qwen-qwen3-6-27b_token99973_self-v-external`

## Pairing audit

- Paired factual items: 82
- Paired item-strength rows: 164
- Exact question/answer prefix shared in every pair: yes
- Candidate: vocabulary token 99973 at layer 40, measured without a rank gate
- Steering delta: change in the correct-oriented label-sequence log-probability margin

## Candidate presence

| SELF mean score | OTHER mean score | SELF − OTHER | SELF median raw rank | OTHER median raw rank |
| ---: | ---: | ---: | ---: | ---: |
| 12.0076 | 12.7828 | -0.7752 | 26.0 | 11.0 |

## Paired causal effects

| Strength | n | SELF delta | OTHER delta | SELF − OTHER effect | Paired bootstrap 95% CI | SELF flips | OTHER flips |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| -1.8 | 82 | -5.7724 | -5.4989 | -0.2735 | [-0.6027, 0.0340] | 2 | 3 |
| -1.7 | 82 | -5.7620 | -5.4999 | -0.2621 | [-0.5887, 0.0516] | 2 | 3 |

## Interpretation boundary

Similar candidate presence and causal effects support a generic evaluator/readout. A reliably larger and selective SELF effect supports a self-evaluation-selective evaluator candidate. Neither result establishes a higher-order representation M(P); that requires a later same-output/different-internal-P experiment.

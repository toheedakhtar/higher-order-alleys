# Paired SELF-versus-OTHER results: `20260903T132621487907Z_qwen-qwen3-6-27b_token99973_self-v-external`

## Pairing audit

- Paired factual items: 4
- Paired item-strength rows: 8
- Exact question/answer prefix shared in every pair: yes
- Candidate: vocabulary token 99973 at layer 40, measured without a rank gate
- Steering delta: change in the correct-oriented label-sequence log-probability margin

## Candidate presence

| SELF mean score | OTHER mean score | SELF − OTHER | SELF median raw rank | OTHER median raw rank |
| ---: | ---: | ---: | ---: | ---: |
| 12.1094 | 10.8438 | 1.2656 | 24.0 | 63.5 |

## Paired causal effects

| Strength | n | SELF delta | OTHER delta | SELF − OTHER effect | Paired bootstrap 95% CI | SELF flips | OTHER flips |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| -1.8 | 4 | -4.0511 | -2.2709 | -1.7802 | [-2.7802, -0.8750] | 0 | 0 |
| -1.7 | 4 | -4.0223 | -2.2552 | -1.7671 | [-2.8765, -0.7656] | 0 | 0 |

## Interpretation boundary

Similar candidate presence and causal effects support a generic evaluator/readout. A reliably larger and selective SELF effect supports a self-evaluation-selective evaluator candidate. Neither result establishes a higher-order representation M(P); that requires a later same-output/different-internal-P experiment.

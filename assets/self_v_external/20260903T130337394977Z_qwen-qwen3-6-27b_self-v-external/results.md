# Paired SELF-versus-OTHER results: `20260903T130337394977Z_qwen-qwen3-6-27b_self-v-external`

## Pairing audit

- Paired factual items: 82
- Paired item-strength rows: 164
- Exact question/answer prefix shared in every pair: yes
- Candidate: vocabulary token 97817 at layer 40, measured without a rank gate
- Steering delta: change in the correct-oriented label-sequence log-probability margin

## Candidate presence

| SELF mean score | OTHER mean score | SELF − OTHER | SELF median raw rank | OTHER median raw rank |
| ---: | ---: | ---: | ---: | ---: |
| 11.6639 | 12.9680 | -1.3041 | 33.0 | 11.0 |

## Paired causal effects

| Strength | n | SELF delta | OTHER delta | SELF − OTHER effect | Paired bootstrap 95% CI | SELF flips | OTHER flips |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| -1.8 | 82 | -6.1456 | -6.7642 | 0.6186 | [0.2528, 0.9869] | 9 | 12 |
| -1.7 | 82 | -6.1420 | -6.7536 | 0.6116 | [0.2416, 0.9773] | 10 | 12 |

## Interpretation boundary

Similar candidate presence and causal effects support a generic evaluator/readout. A reliably larger and selective SELF effect supports a self-evaluation-selective evaluator candidate. Neither result establishes a higher-order representation M(P); that requires a later same-output/different-internal-P experiment.

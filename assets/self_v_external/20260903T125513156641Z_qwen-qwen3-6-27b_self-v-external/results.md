# Paired SELF-versus-OTHER results: `20260903T125513156641Z_qwen-qwen3-6-27b_self-v-external`

## Pairing audit

- Paired factual items: 4
- Paired item-strength rows: 8
- Exact question/answer prefix shared in every pair: yes
- Candidate: vocabulary token 97817 at layer 40, measured without a rank gate
- Steering delta: change in the correct-oriented label-sequence log-probability margin

## Candidate presence

| SELF mean score | OTHER mean score | SELF − OTHER | SELF median raw rank | OTHER median raw rank |
| ---: | ---: | ---: | ---: | ---: |
| 12.0156 | 11.3125 | 0.7031 | 25.5 | 39.5 |

## Paired causal effects

| Strength | n | SELF delta | OTHER delta | SELF − OTHER effect | Paired bootstrap 95% CI | SELF flips | OTHER flips |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| -1.8 | 4 | -3.9516 | -2.7175 | -1.2341 | [-2.8279, 0.0943] | 0 | 1 |
| -1.7 | 4 | -3.9493 | -2.7020 | -1.2472 | [-2.7629, 0.0368] | 0 | 1 |

## Interpretation boundary

Similar candidate presence and causal effects support a generic evaluator/readout. A reliably larger and selective SELF effect supports a self-evaluation-selective evaluator candidate. Neither result establishes a higher-order representation M(P); that requires a later same-output/different-internal-P experiment.

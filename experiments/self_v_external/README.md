# Matched SELF-versus-OTHER J-Lens experiment

This experiment tests whether vocabulary token `97817` (the selected
"to evaluate" direction) is more present and more causally effective when the
model evaluates its own prior answer than when the same answer is attributed
to another party.

## Paired protocol

For each selected item, the runner generates factual answer `X` exactly once:

```text
User:      [factual question]
Assistant: [answer X]
```

It then constructs two independent evaluation inputs with the exact same
question, byte-identical answer `X`, roles, labels, and message count:

```text
SELF:  Is your answer CORRECT or INCORRECT? Reply exactly CORRECT or INCORRECT.
OTHER: Is their answer CORRECT or INCORRECT? Reply exactly CORRECT or INCORRECT.
```

For rows whose original judgment labels are `PASS/FAIL`, both prompts use
`PASS/FAIL` instead. The only change between the final evaluation messages is
`your` versus `their`.

The experiment uses 82 factual items:

- 66 calibration items with `CORRECT/INCORRECT`
- 8 prospective-source factual questions with `PASS/FAIL`
- 8 knowledge-boundary-source factual questions with `PASS/FAIL`

The prospective `WILL_PASS/WILL_FAIL` and knowledge-boundary
`I_KNOW/I_WOULD_GUESS` turns are never included. The eight prewritten
error-detection/student-answer rows are excluded because they do not provide a
question whose answer can be generated once and paired under both ownership
conditions.

## Measurement and intervention

- Candidate: vocabulary token `97817` (`评价`)
- Layer: 40
- Readout position: the `?` in the final evaluation question
- Full saved readout: top 50 J-space tokens at every available lens layer
- Exact candidate measurement: score, raw vocabulary rank, and word-filtered
  rank at layer 40, even when the candidate is outside the top 50
- Visibility gate: none
- Global steering strengths: baseline `0`, then `-1.7×` and `-1.8×`
- Because both requested magnitudes exceed the configured 1.0 residual-norm
  cap, their effective injections may be identical; requested strengths remain
  separate in every output row for exact protocol reproduction.
- Steering scope: all non-BOS prompt positions and generated positions, using
  the existing per-position residual-norm cap
- Label score: teacher-forced sequence log probability, supporting both
  `CORRECT/INCORRECT` and `PASS/FAIL`

`steering_delta_self` and `steering_delta_other` are changes in the
correct-oriented label-sequence margin. Positive values improve the correct
judgment margin; negative values harm it.

## Commands

Run all commands from the repository root.

Static validation does not load the model:

```powershell
uv run python -m experiments.self_v_external.runner --phase validate
```

Run the four-item pilot:

```powershell
uv run python -m experiments.self_v_external.runner --phase pilot
```

Run all 82 paired items after the pilot passes:

```powershell
uv run python -m experiments.self_v_external.runner `
  --phase full `
  --pilot-run experiments/self_v_external/results/PILOT_RUN_ID
```

Use a different output parent directory:

```powershell
uv run python -m experiments.self_v_external.runner `
  --phase pilot `
  --output-root assets/self_v_external
```

Resume an interrupted full run:

```powershell
uv run python -m experiments.self_v_external.runner `
  --phase full `
  --pilot-run experiments/self_v_external/results/PILOT_RUN_ID `
  --resume-run experiments/self_v_external/results/INTERRUPTED_RUN_ID
```

Regenerate analysis without loading the model:

```powershell
uv run python -m experiments.self_v_external.analysis `
  experiments/self_v_external/results/RUN_ID
```

Run CPU tests:

```powershell
uv run python -m unittest experiments.self_v_external.test_self_v_external
```

## Primary paired outputs

`paired_results.csv` has one row per item and nonzero strength, including:

- `candidate_score_self`
- `candidate_rank_self`
- `candidate_score_other`
- `candidate_rank_other`
- `self_minus_other_candidate_score`
- `steering_delta_self`
- `steering_delta_other`
- `self_minus_other_steering_effect`

Raw and word-filtered ranks, raw and correct-oriented margins, baseline and
intervened labels, flips, improvements, harms, effective capped strength, and
answer identity hashes are also retained.

## Artifacts

Each run contains:

```text
run_manifest.json
config.json
README_run.md
experiment.log
events.jsonl
raw_runs.jsonl
readouts.jsonl
tokenizations.jsonl
adaptive_paths.jsonl
errors.jsonl
trial_summary.csv
intervention_results.csv
paired_results.csv
paired_summary.csv
results.md
plots/
  01_paired_candidate_score.png
  02_paired_candidate_rank.png
  03_paired_steering_effect.png
  04_candidate_vs_steering_difference.png
checkpoints/directions/
checkpoints/residuals/
pilot_summary.json  # pilot only
```

## Interpretation limit

- Similar candidate presence and causal effect in both conditions supports a
  generic evaluator/readout interpretation.
- A reliably stronger and selectively causal SELF effect supports a
  self-evaluation-selective evaluator candidate.
- Neither outcome establishes a higher-order representation `M(P)`. That
  requires the later same-output/different-internal-`P` experiment.

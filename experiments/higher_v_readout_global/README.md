# Qwen3.6-27B global J-Lens metacognition experiment

This is a clean implementation of the Neuronpedia-global migration. It does
not import or modify `higher_v_readout/`.

The frozen primary condition uses:

```text
mode:                   neuronpedia_global
direction source:       final judgment question mark
candidate:              评价 / vocabulary token 97817
layer:                  40
requested strengths:    0, -1.7, +1.8
effective cap:          1.0 residual norm
application scope:      every non-BOS prompt position and generated tokens
cache policy:           disabled; complete prefix recomputed
```

The question mark selects the readout direction. It is not the global
intervention target. Single-position question-mark and preceding-token
interventions are retained as a separate mechanistic control.

## Design guarantees

- The baseline is generated and scored once per sample. Zero-strength rows are
  not duplicated in the intervention table.
- Only `frozen_primary` rows enter primary estimates and confidence intervals.
- `localized_control` and `adaptive_rescue` rows remain separately labelled.
- Global steering uses each position's own residual norm, a 1.0 norm cap,
  actual-BOS skipping, and generated-token steering.
- Requested and cap-effective strengths are both recorded. Thus `-1.7` and
  `-1.8` are visibly identical in exact capped mode.
- Candidate identity is the vocabulary token ID plus direction hash. Token
  97817 is primary; token 99973 is the secondary candidate.
- Every readout records both raw vocabulary rank and a word-filtered rank using
  the public `jlens.vis` meaningful-token rule. The filtered rank drives fallback.
- Direction-source position and intervention scope are independent fields.
- Rescues and harms are distinguished from arbitrary label changes.
- Full runs are append-only and can resume at completed item boundaries.

## Installation

From the repository root:

```powershell
uv sync
```

The Qwen3.6-27B phases require a CUDA host with sufficient VRAM. The CPU test
suite uses synthetic models and performs no downloads.

## 1. CPU tests

```powershell
uv run python -m unittest experiments.higher_v_readout_global.test_global_experiment
```

## 2. Static validation

This validates the configuration, all 90 dataset rows, exact final-question
preservation, candidate identities, rank policy, strengths, and gate IDs
without loading model weights:

```powershell
uv run python -m experiments.higher_v_readout_global.runner --phase validate
```

## 3. Exported parity smoke

The parity phase uses the exact 231 prompt token IDs from the item-73 steering
export and runs item 72 as a stable-PASS control:

```powershell
uv run python -m experiments.higher_v_readout_global.runner --phase parity
```

`parity_summary.json` separates:

- `strict_backend_parity`: whether local exported-polarity output exactly
  matches Neuronpedia's `FAIL -> PASS`; and
- `research_gate_passed`: exact prompt/baseline/identity/scope checks, material
  PASS-margin movement, local-polarity flip, and the stable-PASS control.

If bfloat16 produces the previously observed PASS/FAIL tie, strict parity is
recorded as incomplete rather than silently treated as a match.

## 4. Frozen pilot

```powershell
uv run python -m experiments.higher_v_readout_global.runner `
  --phase pilot `
  --parity-run experiments/higher_v_readout_global/results/PARITY_RUN_ID
```

The pilot checks baseline PASS/FAIL coverage, easy/hard items, factual
correct/incorrect items, valid outputs, distinct global/local families,
sample-unique primary rows, a single baseline per sample, and empty errors.
Gate manifests remain valid after relocating the repository: location-only
dataset paths are ignored, but the dataset SHA-256 and all experimental
settings must still match exactly.

## 5. Full 90-row run

```powershell
uv run python -m experiments.higher_v_readout_global.runner `
  --phase full `
  --pilot-run experiments/higher_v_readout_global/results/PILOT_RUN_ID
```

The full phase always runs all 90 dataset rows. It also runs four forced-output
controls unless `--no-controls` is supplied. A gate bypass is available through
`--skip-gate-requirement` and is permanently recorded in the manifest.

Resume an interrupted full run:

```powershell
uv run python -m experiments.higher_v_readout_global.runner `
  --phase full `
  --pilot-run experiments/higher_v_readout_global/results/PILOT_RUN_ID `
  --resume-run experiments/higher_v_readout_global/results/INTERRUPTED_RUN_ID
```

## Analysis

Analysis runs automatically at the end of model phases. It can be regenerated
without loading Qwen:

```powershell
uv run python -m experiments.higher_v_readout_global.analysis `
  experiments/higher_v_readout_global/results/RUN_ID
```

The report produces:

1. Per-sample primary global rescue and harm rates.
2. Primary global PASS/FAIL-style margin movement.
3. Paired global versus localized effects.
4. Baseline margin versus steering susceptibility.
5. Separately labelled adaptive token/layer results.
6. Question-mark versus preceding-token results for localized controls only.

Bootstrap intervals resample item-level means, never adaptive attempts.

## Output contract

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
primary_stratified_summary.csv
results.md
plots/
checkpoints/directions/
checkpoints/residuals/
parity_summary.json   # parity only
pilot_summary.json    # pilot only
```

The intervention table contains raw and normalized outputs, factual and
judgment correctness, requested and effective strengths, source and target
positions, application scope, cap, generated-token flag, raw and filtered
ranks, candidate identity, margins, oriented margins, flips, rescues, and harms.

## Interpretation limit

A global flip supports a causal distributed evaluator/readout or output-process
effect. It does not localize the mechanism to the source question mark and does
not establish `M(P)`. The same-text/different-hidden-trajectory mediation study
remains a separate next-stage experiment.

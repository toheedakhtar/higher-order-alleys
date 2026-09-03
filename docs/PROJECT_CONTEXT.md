# Qwen3.6-27B J-Lens metacognition experiment: project context

> Repository relocation note: this document describes the predecessor layout
> and is retained as historical research context. The current runnable package
> is `experiments/higher_v_readout_global/`; use the repository-root
> `README.md` for current commands and paths.

Last updated: 2026-09-02

## September 2026 protocol revision

The execution code has been revised after the completed 10-primary-item run
described below. That run remains valid as a historical artifact, but its protocol
and output schema are now legacy behavior.

The current runner now:

- executes all 90 rows in `metacognition.csv`, including 66 calibration rows and
  item 65;
- preserves the exact final metacognitive question from each CSV prompt;
- records factual correctness with the original case-insensitive regex scorer;
- enables both `evaluate` vocabulary candidates as separate identities;
- uses an objective top-k appearance rule to select layer 40 or the first two
  eligible appearance layers;
- falls back to the second candidate only after the first candidate produces no
  question-mark flip;
- tests only the question mark and its immediately preceding meaningful token at
  strengths 0, -1.7, -1.8, and +1.8;
- records every adaptive attempt and an explicit terminal no-flip status;
- supports resuming at completed-item boundaries; and
- produces five focused flip, margin, position, correctness, and token/layer
  plots instead of the previous ten-plot suite.

Because the configuration and schema changed, old smoke runs cannot authorize a
new full run. Create a new smoke run first. Current operational instructions are
in `README.md`. The diagnosis and implementation plan for migrating the main
runner from the historical localized intervention to a separate
Neuronpedia-global condition is in
`docs/ORIGINAL_SETUP_GLOBAL_STEERING_MIGRATION.md`. The separate
same-text/different-hidden-process design required to test `P -> M(P)` is in
`docs/NEXT_STAGE_FIRST_HIGHER_ORDER.md`.

This is a detailed handoff document for the work completed so far in
`code/higher_v_readout/`. It records the research goal, implementation
decisions, commands, debugging history, completed run, interpretation, and the
most important limitations. It is intended to provide enough context to resume
the project without reconstructing the entire development conversation.

## 1. Current status

The repository now contains:

- A utility for loading Qwen3.6-27B and an already-fitted Jacobian Lens from
  Hugging Face, inspecting the lens, and optionally fitting a new lens.
- A reproducible multi-turn metacognition experiment.
- Token-aligned layer-40 readout and activation steering.
- Self-evaluation, external-evaluation, forced-output, lexical, position, and
  baseline-saturation controls.
- Append-only raw logs, tidy CSV outputs, manifests, residual/direction
  checkpoints, analysis code, and plots.
- CPU unit tests for the experiment's local logic.
- One completed 10-primary-prompt cloud-GPU run.

The completed run does **not** support the primary layer-40 candidate as a
robust, self-specific metacognitive mechanism. The most defensible current
interpretation is a weak and nonspecific readout/output-preparation effect.
This is useful negative evidence, not a failed implementation.

The experiment was deliberately designed not to claim that the feature is a
higher-order representation, written as M(P). Even a repeatable causal
self-evaluation effect would only make it a candidate for a later experiment.

## 2. Research question

The motivating distinction is:

```text
First-order computation:
    P = computation that produces a factual answer

Generic retrospective evaluator:
    P -> r(P) -> PASS/FAIL judgment

Possible higher-order representation:
    P -> M(P)
```

The present experiment asks what the manually observed J-space feature is most
consistent with:

1. A robust causal evaluator.
2. Generic machinery for evaluating any answer.
3. Late PASS/FAIL or output-selection machinery.
4. Prompt/lexical machinery.
5. A self-evaluation-selective mechanism worth studying further.

It cannot distinguish a true representation of the model's hidden first-order
process from a post-hoc evaluation of the visible answer text. A future
experiment must manipulate P while holding the observable question and answer
text constant.

### Conservative interpretation ladder

- Level 0: the feature appears in J-space. This establishes decodability only.
- Level 1: its score is associated with correctness or judgment. This is
  correlational evidence only.
- Level 2: steering consistently changes retrospective judgment. This makes it
  a causal evaluator/readout candidate, but not M(P).
- Level 3: the effect is larger for self-evaluation than external evaluation,
  survives wording and position controls, and is not explained by forced output
  generation. This would support a self-evaluation-selective causal mechanism,
  still not M(P).

The completed run securely reaches Level 0. It shows small causal logit changes,
but they are inconsistent, nonspecific, and do not produce primary behavioral
flips. Therefore it does not establish a robust Level-2 self-evaluator and does
not approach Level 3.

## 3. Repository map

### `fit_jlens.py`

Purpose:

- Load the official `Qwen/Qwen3.6-27B` checkpoint.
- By default, download and load Neuronpedia's already-fitted Jacobian Lens.
- Apply the lens to a prompt and perform numerical/top-token checks.
- Optionally create a self-contained interactive J-Lens HTML readout.
- Fit a new Jacobian Lens only when `--mode fit` is explicitly requested.

Important clarification: the normal/default workflow does **not** refit the
lens. It combines the Qwen base model with a separately downloaded fitted lens.

Default lens artifact:

```text
Repository: neuronpedia/jacobian-lens
File: qwen3.6-27b/jlens/Salesforce-wikitext/
      Qwen3.6-27B_jacobian_lens_n1000.pt
```

### `run_metacognition_jlens.py`

This is the main experiment runner. It:

- Loads and validates the dataset.
- Reconstructs real multi-turn conversations instead of sending the
  concatenated CSV prompt.
- Runs deterministic factual and metacognitive generations.
- Finds the target punctuation using actual tokenizer offsets.
- Captures residual-stream states and full J-Lens readouts.
- Builds and saves stable steering directions.
- Recomputes downstream states with KV caching disabled.
- Runs baseline and intervention conditions.
- Writes an append-only experimental record.
- Invokes the analysis stage after a completed full run.

### `jlens_analysis_.py`

This script analyzes an existing result directory without loading Qwen again.
It reads the CSV and JSONL artifacts, creates `results.md`, and produces plots
for dose response, self/external comparison, feature scores, layer profiles,
flips, saturation, and lexical robustness.

### `experiment_config.json`

The configuration controls:

- Model and revision.
- Dtype and device mapping.
- Lens repository, file, and revision.
- Dataset path.
- Random seed.
- Top-k readout size.
- Steering strengths.
- Candidate labels/layers and whether each candidate is enabled.
- Generation limits and deterministic decoding.
- Number of lexical, position, forced-output, and external controls.
- The exact smoke-test item IDs.
- The maximum and prohibited interpretation claims.

There is currently no `--strengths` command-line argument. Change strengths in
this JSON file, or create another config and pass it using `--config PATH`.

### `test_fit_jlens.py`

Despite the historical filename, this contains CPU tests for the fitter,
experiment runner, and analysis logic. It covers, among other things:

- Tensor and layer plumbing.
- Token/character alignment.
- Full next-token vocabulary-logit shape.
- The regression that caused the first cloud smoke-run failure.
- Candidate identity and direction persistence.
- Steering hook behavior.
- Static validation and output analysis.

As of 2026-09-02:

```text
13 tests run
13 passed
```

The verified command is:

```powershell
cd code/higher_v_readout
uv run python -m unittest test_fit_jlens.py
```

### `results/`

Each smoke or full invocation creates a new timestamped directory. Existing
runs are not overwritten.

## 4. Model and lens implementation

### Base model

Configured model:

```text
Qwen/Qwen3.6-27B
```

The completed run resolved this to:

```text
6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
```

Observed architecture:

- Runtime class: `Qwen3_5ForConditionalGeneration`
- Model type: `qwen3_5_text`
- Decoder blocks: 64
- Residual dimension: 5,120
- Parameters: 27,356,728,560

### Jacobian Lens

Completed-run lens metadata:

- Hugging Face repository: `neuronpedia/jacobian-lens`
- Lens revision:
  `0731326edff4ae730ffc5356fe1a4728c748b3a6`
- Upstream implementation commit:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`
- Fitted source layers: decoder outputs 0 through 62.
- Matrices were kept resident on the A100 during the run.

The final model layer can be directly unembedded, so it does not require a
transport Jacobian in the same sense as lower source layers.

### Pretrained versus fitting

Use the downloaded lens:

```powershell
cd code
uv run python higher_v_readout/fit_jlens.py `
  --mode pretrained `
  --output-dir higher_v_readout/jlens_runs/qwen3.6-27b-pretrained
```

Fit a new pilot lens only when needed:

```powershell
cd code
uv run python higher_v_readout/fit_jlens.py `
  --mode fit `
  --n-prompts 1 `
  --source-layers 0,16,32,48,62 `
  --dim-batch 8 `
  --output-dir higher_v_readout/jlens_runs/qwen3.6-27b-pilot
```

The pretrained lens is the appropriate choice for the current metacognition
experiment because the candidate was identified in that fitted J-space.

## 5. Dataset and trial selection

Dataset:

```text
datasets/MMB/metacognition.csv
```

Observed counts:

- Calibration: 66
- Prospective: 8
- Knowledge boundary: 8
- Error detection: 8

The primary self-evaluation pool contains:

```text
8 prospective + 8 knowledge_boundary = 16 primary items
```

The `--n-prompts` argument limits only this primary self-evaluation pool.
It does not mean the entire experiment has only that many generated prompts.
External, forced-output, lexical, and position controls are additional.

Defaults:

- `--n-prompts 10`: deterministic test-sized subset of 10/16 primary items.
- `--n-prompts 16`: complete primary set.
- Seed 42 controls deterministic subset selection.

The eight error-detection rows form the primary external-evaluation control.
Calibration rows are not part of this experiment.

Factual scoring preserves the original notebook semantics:

```python
re.search(answer_key, response, re.IGNORECASE)
```

This supports regular expressions and case-insensitive patterns such as
`Dunning.*Kruger` and `(?i)freud`. The answer keys are not silently changed.
Malformed factual outputs are retained and marked invalid rather than repaired
with a trial-specific extra prompt.

## 6. Primary multi-turn protocol

The CSV prompt is not passed directly to the model. Each primary item becomes a
three-turn conversation.

### Prospective items

Turn 1 asks the model, before seeing the factual question, to output exactly:

```text
WILL_PASS or WILL_FAIL
```

The model only receives the question domain at this point. The runner saves the
generated label and the sequence-log-probability margin:

```text
logP(WILL_PASS) - logP(WILL_FAIL)
```

Turn 2 asks for the factual answer only.

Turn 3 asks:

```text
Did your answer PASS or FAIL? Reply with exactly PASS or FAIL.
```

### Knowledge-boundary items

Turn 1 asks the model to output exactly:

```text
I_KNOW or I_WOULD_GUESS
```

Turn 2 uses identical wording for all knowledge-boundary items so the model
does not merely repeat its Turn-1 label.

Turn 3 uses the same PASS/FAIL prompt as prospective items.

### Sequence scoring

The runner computes teacher-forced sequence log likelihood for both possible
labels. It does not assume that a label is one tokenizer token.

The main retrospective margin is:

```text
logP(PASS) - logP(FAIL)
```

Positive means the model favors PASS; negative means it favors FAIL.

## 7. Exact intervention location

All primary readouts and interventions happen in Turn 3, after the factual
answer exists but before the model generates PASS or FAIL.

The primary location is the token containing the question mark in:

```text
Did your answer PASS or FAIL?
```

The implementation uses the real tokenizer's character offsets. It saves:

- Token index.
- Token ID.
- Decoded token string.
- Character span.
- Neighboring tokens.
- Conversation position.

It does not assume that one Unicode character is one token.

Layer numbering is zero-based. UI layer 40 and code layer 40 both mean the
output of decoder block index 40.

Position controls also test:

- The final prompt punctuation immediately before generation.
- A meaningful token before the judgment question mark.

## 8. Candidate feature identity

Manual inspection identified two displayed Chinese concepts:

- Primary: `评估`, visually the second “to evaluate”.
- Secondary: `评价`, visually the first “to evaluate”.

The display text is not treated as a stable identity. The upstream lens exposes
vocabulary-aligned entries rather than a stable semantic feature registry.

For the completed run, the primary candidate is:

```text
Display label: 评估
Vocabulary token ID: 99973
Feature ID used by the experiment: vocab_token:99973
Layer: 40
Direction SHA-256:
be5d13a5b3822fe4117f5d036dd73708ffd47d3bf8fc1f49007610a6a3307b11
```

The direction tensor is saved in the run's
`checkpoints/directions/` directory.

The current config enables only the primary candidate. The secondary candidate
is present but disabled to reduce compute. Therefore plot 10, candidate
comparison, was correctly skipped in the completed run.

Some Windows-rendered files display the Chinese labels as mojibake such as
`è¯„ä¼°`. This is an encoding/display issue. The stable identity for the
primary result is token ID 99973 plus the saved vector hash, not that corrupted
display string.

## 9. Steering implementation

The upstream `jlens` package does not supply the required activation-steering
API, so the runner implements a local forward hook.

The direction is:

```text
direction = normalize(J_layer.T @ lm_head.weight[token_id])
```

The intervention is:

```text
delta = strength * prompt_mean_residual_norm * direction
```

It is applied at:

- One configured decoder block output.
- One tokenizer-aligned prompt position.
- During every necessary complete-prefix recomputation.

This is called **steering**, not ablation. It adds or subtracts a direction; it
does not project out the component.

Current configured strengths:

```json
[0, -1.7, -1.8]
```

Strength zero is the baseline. The completed experiment therefore tested only
two closely spaced negative steering levels.

### KV-cache handling

KV caching is disabled:

```json
"use_cache": false
```

For every generated or teacher-forced token, the complete prefix is recomputed,
and the hook is reapplied at the original position. This prevents stale
downstream key/value states from bypassing the intervention.

### Full readout

At the selected token, the experiment saves:

- Top 50 vocabulary entries for every fitted layer.
- Candidate score and rank at every layer.
- Selected-position residual checkpoints.
- Candidate direction metadata.
- Layer-indexing convention.

This makes later readout hypotheses testable without repeating the expensive
forward pass.

## 10. Controls

### External evaluation

The eight error-detection items ask the model to judge a student's answer using
the same PASS/FAIL vocabulary. The same candidate, layer, positions, and
strengths are used.

This tests whether the feature is generic evaluation machinery rather than
self-specific machinery.

### Forced output

Synthetic prompts request a specific label without requiring evaluation:

- PASS
- FAIL
- CORRECT
- INCORRECT

If steering strongly changes these margins, the candidate may participate in
output selection, label production, instruction following, or general
confidence rather than metacognitive evaluation.

### Lexical robustness

Four representative self-evaluation items are rerun with:

- PASS/FAIL
- RIGHT/WRONG
- YES/NO

A self-evaluation mechanism should not depend completely on one exact label
pair or prompt template.

### Position control

Four representative items, two factually correct and two incorrect, compare:

- The primary judgment question mark.
- Final prompt punctuation.
- A token before the question mark.

This helps distinguish an earlier evaluation computation from late output
preparation.

### Baseline saturation

Absolute baseline PASS/FAIL margin is compared with absolute steering-induced
change. Very confident judgments may resist flips even if logits move.

## 11. Run phases and command-line behavior

### Static validation

This checks configuration, dataset schema/counts, strengths, candidates, and
selected item IDs without loading the 27B weights:

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase validate `
  --n-prompts 10
```

### Smoke phase

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase smoke
```

The smoke phase always uses the two configured item IDs:

```json
["66", "87"]
```

They must produce exactly two valid completed primary trials with one correct
and one incorrect factual result. The smoke checks also require:

- Correct question-mark alignment.
- Saved feature identity.
- Readout and steering records.
- Raw reconstruction data.
- Scoring metadata.
- Required output files.
- KV cache disabled.
- No logged errors.

### Why full runs require `--smoke-run`

The smoke directory is not reused as experimental data, and it is not where the
new full results are written.

It is a provenance/safety gate. Before allocating the full run, the runner
verifies that:

- The referenced run really used `--phase smoke`.
- All smoke checks succeeded.
- The model/revision/dtype and experiment configuration are compatible.
- The evidence is recorded in the new full run's manifest.

The full run still creates all of its own files because it is a separate,
append-only experiment. Its manifest contains the prior smoke run ID, path,
checks, and manifest hash.

An explicit `--skip-smoke-requirement` bypass exists. Using it is recorded in
the manifest and reduces the reproducibility guarantee.

### Full 10-primary test-sized run

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase full `
  --n-prompts 10 `
  --smoke-run higher_v_readout/results/SMOKE_RUN_ID
```

### Complete 16-primary run

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase full `
  --n-prompts 16 `
  --smoke-run higher_v_readout/results/SMOKE_RUN_ID
```

### Change output directory

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase smoke `
  --output-root /path/to/results
```

`--output-root` is the parent directory. The runner still creates a unique
timestamped RUN_ID subdirectory inside it.

### Debug primary trials without controls

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase full `
  --n-prompts 2 `
  --no-controls `
  --smoke-run higher_v_readout/results/SMOKE_RUN_ID
```

## 12. Initial cloud smoke-run failure and fix

The first cloud smoke run loaded the model and lens successfully, resolved the
primary candidate, and then failed while scoring the first prospective and
knowledge-boundary labels.

Observed exception:

```text
IndexError: index 54 is out of bounds for dimension 0 with size 0
```

It arose in `sequence_logprob()` when indexing next-token log probabilities.
The issue was tensor-shape handling: the Qwen/J-Lens wrapper's unembedding
already represented one position's vocabulary logits, but the scoring path
handled it as though additional batch/sequence dimensions still had to be
selected. That could collapse or empty the expected vocabulary vector.

The fix:

- Capture the final decoder-block residual at the last prefix token.
- Unembed that residual exactly once.
- Normalize a legal `[1, vocab]` result to `[vocab]`.
- Require the final value to be a nonempty one-dimensional vocabulary vector.
- Raise a clear runtime error for scalar, empty, or otherwise invalid shapes.

Regression tests now verify:

- Next-token logits remain the complete vocabulary vector.
- Sequence scoring can index a real candidate token.
- Scalar logits are rejected by the explicit shape guard.

A later smoke run completed all checks and was referenced by the completed full
run.

## 13. Output format

A normal run directory contains:

```text
run_manifest.json
experiment.log
events.jsonl
raw_runs.jsonl
trial_summary.csv
intervention_results.csv
jlens_readouts.jsonl
tokenizations.jsonl
errors.jsonl
config.json
results.md
plots/
checkpoints/directions/
checkpoints/residuals/
README_run.md
```

Roles:

- `run_manifest.json`: exact environment, revisions, configuration, feature
  identity, smoke provenance, and final status.
- `experiment.log`: human-readable chronological log.
- `events.jsonl`: append-only machine-readable event stream.
- `raw_runs.jsonl`: prompts, histories, generations, scores, and condition
  metadata sufficient to reconstruct trials.
- `trial_summary.csv`: one tidy baseline summary row per trial/variant.
- `intervention_results.csv`: one row per item, condition, feature, strength,
  and intervention.
- `jlens_readouts.jsonl`: top-k readouts across layers.
- `tokenizations.jsonl`: exact tokens, offsets, and selected position.
- `errors.jsonl`: complete exceptions and tracebacks; never silently dropped.
- `config.json`: frozen run configuration.
- `results.md` and `plots/`: derived analysis.
- `checkpoints/`: stable directions and selected residual states.

The locally copied completed run does not currently include
`experiment.log`, although its manifest says the run completed and
`errors.jsonl` is empty. This may mean the log was omitted while copying the
cloud outputs. Preserve this as a provenance caveat.

## 14. Completed full run

Run directory:

```text
code/higher_v_readout/results/
20260902T073212605106Z_qwen-qwen3-6-27b_55ebbae/
```

Core metadata:

- Status: completed.
- Phase: full.
- Primary prompts: 10.
- Start: 2026-09-02 07:32:26 UTC.
- End: 2026-09-02 07:37:54 UTC.
- Duration: approximately 5 minutes 28 seconds.
- Logged errors: 0.
- GPU: NVIDIA A100-SXM4-80GB.
- GPU memory: 79.25 GiB.
- CUDA runtime: 13.0.
- Dtype: bfloat16.
- Python: 3.11.15.
- PyTorch: 2.13.0.
- Transformers: 5.15.1.
- J-Lens package: 0.1.0.
- Seed: 42.
- Thinking mode: disabled with `enable_thinking=False`.
- Decoding: manual greedy argmax, temperature 0, sampling disabled.
- KV cache: disabled.
- Candidate: token 99973, layer 40.
- Strengths: 0, -1.7, -1.8.
- Secondary candidate: disabled.

The repository was marked dirty in the run manifest. The recorded repository
commit was:

```text
55ebbaea8bc592cd071178890e331c039fd2b0c1
```

The manifest specifically recorded a modified experiment config and untracked
result/assets paths. Exact frozen config and hashes inside the run directory
should therefore be used for reproduction rather than assuming the Git commit
alone describes the run.

### Prior successful smoke evidence

The full run recorded:

```text
Smoke run:
20260902T072709833664Z_qwen-qwen3-6-27b_55ebbae
```

All eleven configured smoke checks were true, and the full run did not bypass
the smoke requirement.

## 15. Baseline behavioral results

### Aggregate

- Primary self-evaluation trials: 10.
- Valid factual responses: 10.
- Malformed factual responses: 0.
- Factual accuracy: 6/10 = 60%.
- Valid retrospective responses: 10.
- Retrospective self-evaluation accuracy: 9/10 = 90%.
- External-evaluation accuracy: 8/8 = 100%.

### Prospective monitoring

All five sampled prospective trials predicted `WILL_PASS`.
Only three of the five factual answers were correct.

This is weak prospective discrimination and indicates overconfidence:

```text
Predicted WILL_PASS: 5/5
Actually passed:      3/5
```

### Knowledge-boundary monitoring

Four of five outputs were `I_KNOW`; one was `I_WOULD_GUESS`.

Mapping `I_KNOW` to an expected correct answer and
`I_WOULD_GUESS` to an expected incorrect answer, only two of five
pre-assessments aligned with factual correctness:

- The sole `I_WOULD_GUESS` item was answered correctly.
- Two `I_KNOW` items were answered incorrectly.
- Two `I_KNOW` items were answered correctly.

This again shows poor pre-answer knowledge calibration in the small sample.

### Retrospective monitoring

After seeing its own answer in the conversation, Qwen classified nine of ten
answers correctly as PASS or FAIL. The only miss was item 88: its factual answer
was incorrect but it retrospectively said PASS.

The strong external score, 8/8, is important. It shows that the model can
evaluate visible answer content very well even when the answer belongs to a
student. Therefore high retrospective accuracy by itself is not evidence of
introspection into the model's own hidden process.

### Per-primary-item summary

| Item | Type | Pre-assessment | Factually correct | Retrospective | Retro correct | Baseline margin | Candidate score |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| 67 | prospective | WILL_PASS | yes | PASS | yes | 8.375 | 12.9375 |
| 83 | knowledge boundary | I_WOULD_GUESS | yes | PASS | yes | 5.000 | 11.5000 |
| 66 | prospective | WILL_PASS | yes | PASS | yes | 7.875 | 13.0625 |
| 87 | knowledge boundary | I_KNOW | no | FAIL | yes | -0.125 | 10.7500 |
| 71 | prospective | WILL_PASS | no | FAIL | yes | -1.250 | 13.8750 |
| 88 | knowledge boundary | I_KNOW | no | PASS | no | 0.250 | 11.0000 |
| 68 | prospective | WILL_PASS | yes | PASS | yes | 7.125 | 13.7500 |
| 86 | knowledge boundary | I_KNOW | yes | PASS | yes | 6.000 | 11.3125 |
| 73 | prospective | WILL_PASS | no | FAIL | yes | -0.500 | 13.3750 |
| 82 | knowledge boundary | I_KNOW | yes | PASS | yes | 6.500 | 11.5625 |

## 16. Primary steering results

The intervention strongly changed the targeted internal coordinate:

```text
Mean candidate score before steering: approximately +12.312
Mean candidate score after -1.7:      approximately -29.512
Mean candidate score after -1.8:      approximately -29.525
```

Thus the steering hook was active. The behavioral null is not explained by the
direction simply failing to be applied.

However, the nearly identical post-steering scores for -1.7 and -1.8 indicate
that both strengths drove the coordinate into essentially the same extreme
regime. This is likely an oversteered or saturated strength range and is not a
useful dose-response calibration.

### Main PASS/FAIL effects

| Condition | Strength | N | Flips | Mean delta margin | Median delta | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Self | -1.7 | 10 | 0 | -0.0375 | 0.0000 | [-0.1625, 0.0750] |
| Self | -1.8 | 10 | 0 | -0.0125 | 0.0000 | [-0.1375, 0.1000] |
| External | -1.7 | 8 | 0 | 0.0156 | 0.0625 | [-0.1406, 0.1563] |
| External | -1.8 | 8 | 0 | -0.0156 | 0.0625 | [-0.1406, 0.0937] |
| Forced output | -1.7 | 4 | 0 | -0.0936 | -0.1562 | [-1.1876, 1.0003] |
| Forced output | -1.8 | 4 | 0 | -0.0936 | -0.1875 | [-1.2499, 1.0627] |

Important observations:

- There were zero primary self PASS/FAIL flips across 20 nonzero-steering rows.
- Signed self effects were close to zero.
- The median self effect was exactly zero at both strengths.
- Effects differed in sign between items.
- Confidence intervals include zero.
- The stronger negative setting did not create a stronger aggregate effect.
- External effects were comparable to self effects.

Mean absolute effects, which measure movement without allowing signs to cancel:

```text
Self:          0.1375
External:      0.1563
Forced output: 1.1251
```

The forced-output margin response was roughly eight times the self effect.

## 17. Forced-output interpretation

Forced-output prompts contain no evaluation task; they simply tell the model
which label to emit.

Steering effects:

- Forced PASS moved about 0.94 to 1.00 toward FAIL.
- Forced FAIL moved 0.625 toward PASS.
- Forced CORRECT moved about 1.44 to 1.50 toward INCORRECT.
- Forced INCORRECT moved about 1.38 to 1.50 toward CORRECT.

This pattern weakens the margin favoring the requested label and pulls the
decision toward the opposite label/less certain state. It is not a consistent
PASS-direction or a consistent first-label direction.

Because the forced baselines were confident, no forced label actually flipped.
Nevertheless, the much larger logit effect is evidence that the direction is
more coupled to late output execution, instruction compliance, or generic
confidence than specifically to self-evaluation correctness.

There are only four forced prompts, so this is descriptive rather than a
population-level conclusion.

## 18. Lexical and position results

The generated `results.md` steering table summarizes only the main self,
external, and forced conditions. It says zero flips for those main conditions.
There were nevertheless four flip rows in the control variants.

They represent two items changing at both negative strengths:

1. Item 71, factually incorrect:
   - RIGHT/WRONG baseline: WRONG.
   - After steering: RIGHT.
   - Flip occurred at both -1.7 and -1.8.

2. Item 87, factually incorrect:
   - Final-punctuation PASS/FAIL baseline: FAIL.
   - After steering: PASS.
   - Flip occurred at both -1.7 and -1.8.

Every observed behavioral flip therefore made the evaluation less accurate.

No primary question-mark PASS/FAIL judgment flipped.

Lexical average effects also differed:

- PASS/FAIL centered close to zero.
- RIGHT/WRONG shifted positively and was the largest lexical effect.
- YES/NO shifted slightly positively.

This is not lexical robustness. The causal result depends on output vocabulary
and prompt construction.

The only PASS/FAIL control flip occurred at final prompt punctuation on item 87,
whose baseline margin was only -0.125. This is consistent with a near-threshold,
late output-selection effect. It is not strong evidence for computation at the
earlier primary judgment-question position.

## 19. Readout and layer-profile results

The candidate is clearly decodable, but its score does not consistently track
factual correctness.

At layer 40:

- Prospective correct mean candidate score: approximately 13.25.
- Prospective incorrect mean: approximately 13.63.
- Knowledge-boundary correct mean: approximately 11.46.
- Knowledge-boundary incorrect mean: approximately 10.88.

The relationship reverses across the two task types. Overall correct and
incorrect means are very similar. The much larger difference is between prompt
families: prospective items have higher scores than knowledge-boundary items.
This suggests prompt/task-template sensitivity rather than a clean correctness
code.

Candidate ranks on the 10 primary rows ranged from 7 to 29. The feature is
visible but not generally the top J-space entry.

Across layers:

- Candidate scores rise broadly through model depth.
- Self-correct, self-incorrect, and external profiles mostly overlap.
- Layer 40 does not show a unique separation of correct and incorrect trials.
- The largest raw scores appear later than layer 40, around the high-50s.

Raw late-layer magnitude alone is not a reason to move the intervention layer;
the fitted transport, residual scale, causal effect, and controls must be
revalidated for any new layer.

## 20. Overall scientific interpretation

### Evidence that is genuinely present

- The layer-40 vocabulary-aligned `评估` direction is decodable.
- Its stable token identity and exact direction were recorded.
- Direct intervention can drive its local candidate score extremely negative.
- The intervention changes downstream output-label logits by small amounts in
  self and external conditions and larger amounts in forced-output controls.
- Qwen performs strong post-answer self and external evaluation on this sample.

### Evidence that is absent or contrary to the hypothesis

- No primary self PASS/FAIL output flips.
- No consistent signed self effect across items.
- No monotonic difference between -1.7 and -1.8.
- No self effect larger than external evaluation.
- Forced-output effects are much larger than self effects.
- The correctness association reverses across task types.
- Correct/incorrect/external layer profiles overlap.
- Effects do not survive lexical controls consistently.
- The only PASS/FAIL flip occurs at a late position, not the primary question
  mark.
- Every observed control flip changes an incorrect answer's judgment in the
  wrong direction.

### Current best conclusion

The primary candidate should not currently be described as a self-specific
metacognitive representation.

A cautious summary is:

> Token 99973 (`评估`) at layer 40 is a decodable J-Lens direction with weak,
> inconsistent downstream effects. The observed effects are at least as
> compatible with generic evaluation, output preparation, label compliance, or
> prompt-specific machinery as with retrospective self-evaluation.

This experiment does not establish M(P).

## 21. Limitations of the completed run

1. Only 10 of 16 primary items were used.
2. There were only five prospective and five knowledge-boundary primaries.
3. Only two nonzero strengths were tested.
4. Both strengths were negative and very close together.
5. There were no positive-direction interventions.
6. Candidate scores after both strengths were nearly identical, suggesting
   saturation.
7. The secondary `评价` candidate was disabled.
8. Forced-output control has only one prompt per requested label.
9. Lexical and position controls use only four representative items.
10. Several label baselines are highly confident, making flips difficult.
11. The model uses bfloat16; small logit differences are visibly quantized.
12. The repository was dirty at run time.
13. The local copy is missing `experiment.log`.
14. This experiment does not hold answer text fixed while manipulating the
    hidden first-order computation.

Because of these limitations, the result should be treated as a mechanistic
pilot and falsification of a strong simple hypothesis, not a final population
estimate.

## 22. Recommended next run

Before moving to the more ambitious M(P) experiment:

1. Run all 16 primary prompts.
2. Add smaller negative strengths to find the nonsaturated regime.
3. Add matched positive strengths.
4. Keep zero baseline.
5. Inspect candidate score after steering at every strength and require a
   graded internal dose response before interpreting behavioral dose response.
6. Enable and test `评价` separately; never merge the two candidates.
7. Add norm-matched random-direction or unrelated-token controls.
8. Increase forced-output examples beyond one per label.
9. Increase lexical and position-control item counts.
10. Predefine the primary outcome as paired margin movement plus flip direction,
    not only the number of flips.
11. Report self-minus-external paired effects rather than relying only on
    separate group means.
12. Preserve a clean Git commit or patch and copy every run artifact, including
    `experiment.log`.

Only if a candidate shows a consistent, graded self effect that exceeds
external and forced-output controls and survives lexical/position changes
should the next-stage hidden-process experiment be prioritized.

## 23. Analysis/reproduction commands

Recreate plots and `results.md` without loading Qwen:

```powershell
cd code
uv run python higher_v_readout/jlens_analysis_.py `
  higher_v_readout/results/20260902T073212605106Z_qwen-qwen3-6-27b_55ebbae
```

Validate the current configuration:

```powershell
cd code
uv run python higher_v_readout/run_metacognition_jlens.py `
  --phase validate `
  --n-prompts 16
```

Run tests:

```powershell
cd code/higher_v_readout
uv run python -m unittest test_fit_jlens.py
```

## 24. Most important files for resuming work

- `README.md`: operational commands and implementation summary.
- `experiment_config.json`: current experiment definition.
- `run_metacognition_jlens.py`: complete execution logic.
- `jlens_analysis_.py`: derived statistics and plots.
- `test_fit_jlens.py`: local regression tests.
- Completed run `run_manifest.json`: exact cloud environment and hashes.
- Completed run `trial_summary.csv`: baseline trial-level outcomes.
- Completed run `intervention_results.csv`: all causal comparisons.
- Completed run `raw_runs.jsonl`: full reconstructed conversations.
- Completed run `jlens_readouts.jsonl`: top-k layer readouts.
- Completed run `results.md`: automatically generated concise report.

When there is a conflict between this handoff note and a frozen run artifact,
the run's `config.json`, `run_manifest.json`, raw JSONL, and CSV files are the
authoritative record of what actually executed.

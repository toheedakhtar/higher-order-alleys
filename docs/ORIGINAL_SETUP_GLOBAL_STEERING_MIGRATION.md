# Original full-dataset setup: diagnosis and global-steering migration

Last updated: 2026-09-03

## Purpose

This document explains why the completed 90-row experiment did not reproduce
the manual Neuronpedia flip, what the later exports and standalone runs reveal,
and how to revise the main experiment before running the full dataset again.

The central correction is:

> Clicking the final `?` in Neuronpedia selected the J-Lens readout direction.
> It did not restrict the intervention to that token. Neuronpedia applied the
> selected direction globally across the prompt at layer 40.

The old full run is therefore a valid test of a localized, single-position
intervention, but it is not a failed replication of the Neuronpedia global
intervention.

## Relevant artifacts

- Historical full run:
  `assets/20260902T130025610376Z_qwen-qwen3-6-27b_23db742/`
- Neuronpedia baseline export:
  `assets/jlens-chat-qwen3_6-27b-2026-09-02T18-03-27-041Z.json`
- Neuronpedia steering export:
  `assets/jlens-chat-qwen3_6-27b-2026-09-02T18-16-46-541Z-steer.json`
- Standalone item-73 run:
  `assets/manual_runs/20260902T183506098924Z_items73/`
- Standalone four-item run:
  `assets/manual_runs/20260902T183745858435Z_items73-72-70-71/`
- Exact-reproduction runner: `reproduce_neuronpedia_manual.py`
- Full-dataset runner: `run_metacognition_jlens.py`

## What the historical full run actually tested

The old runner constructed a unit J-Lens vocabulary direction as:

```text
normalize(J_l.T @ lm_head.weight[token_id])
```

For every intervention it then added:

```text
strength * mean_prompt_residual_norm * direction
```

to one residual-stream cell: either the final judgment question mark or the
preceding token. KV caching was disabled and the complete prefix was
recomputed.

The completed run reported:

- 90 dataset rows completed;
- 82 self-evaluation rows;
- 984 nonzero intervention attempts;
- zero question-mark flips;
- one preceding-token flip, which worsened judgment accuracy; and
- no logged runtime errors.

This result should now be stated as:

> Localized steering of these J-Lens directions at one punctuation-adjacent
> residual cell was not a robust behavioral intervention in this run.

It must not be stated as evidence that the Neuronpedia global manipulation does
not work.

## What Neuronpedia actually did

The steering export contains the authoritative configuration:

```text
token:           评价
type:            JACOBIAN_LENS
layers:          [40]
strength:        -1.7
ablate:          false
mode:            steer
steerGenerated:  true
prompt_len:      231
baseline output: FAIL
steered output:  PASS
```

The final judgment `?` is token position 214. It is the position from which the
`评价` direction was chosen. The steering configuration contains no target
position.

Neuronpedia's public implementation applies the layer write hook to the whole
residual tensor. For each prompt position `h`, it computes:

```text
injected = strength * ||h|| * unit_direction
```

It caps the injection norm at `1.0 * ||h||`, skips actual BOS positions, and,
when `steerGenerated` is true, applies the intervention during generated-token
forward passes as well.

The export confirms this behavior empirically: all 231 layer-40 prompt
readouts changed between its baseline and steered streams, not only position
214.

## Primary defects in the original setup

### 1. Intervention scope was different

The main experiment steered one token. Neuronpedia steered every prompt token.
This is the main reason the historical run cannot reproduce the UI result.

### 2. Position meaning was misinterpreted

The old schema used `position_control` as the causal intervention location. In
the Neuronpedia workflow, position 214 is instead the source of the displayed
readout used to choose a direction.

The revised schema must separate:

```text
direction_source_position
intervention_scope
```

For `neuronpedia_global`, the first may be `question_mark`, while the second is
`all_prompt_positions_and_generated_tokens`.

### 3. Residual scaling was different

The old code used one mean norm for the entire prompt. Neuronpedia uses the
norm of each position independently. Global steering therefore requires a
different tensor-valued scale.

### 4. The Neuronpedia strength is capped

With a unit direction and a cap of one residual norm, strengths whose absolute
value exceeds 1 have the same intended injection magnitude:

```text
-1.7 -> effective magnitude -1.0
-1.8 -> effective magnitude -1.0
+1.8 -> effective magnitude +1.0
```

Consequently, `-1.7` and `-1.8` are not a meaningful dose comparison in exact
global-parity mode. Minor differences can still appear from bfloat16 rounding.

Keep `-1.7` because it is the exported Neuronpedia setting. Keep `+1.8` as a
polarity control. If a real dose-response analysis is wanted, add magnitudes
inside the cap, such as `0.25`, `0.5`, `0.75`, and `1.0`.

### 5. Generated-token steering was missing as an explicit concept

Neuronpedia recorded `steerGenerated: true`. The original experiment only
described a fixed prompt position. Global mode must continue the write hook on
generated tokens. This matters especially for outputs longer than one token.

### 6. Candidate identity and ordering were confusing

The historical manifest reversed the visual descriptions of the two Chinese
tokens. The stable identities are:

```text
评估: token ID 99973, visually first "to evaluate"
评价: token ID 97817, Neuronpedia manual target, visually second
```

The IDs and direction hashes were saved, so the old data remain recoverable.
The new full run must make token ID 97817 the explicit primary candidate rather
than relying on ambiguous display order.

### 7. Neuronpedia and local ranks used different filtering

Neuronpedia's visible list filters non-word tokens. The local raw vocabulary
rank includes punctuation and special tokens. On item 73, `评价` is rank 4 in
the filtered Neuronpedia list but rank 9 in the local unfiltered list.

Record both raw rank and word-filtered rank. Decide in advance which rank drives
the appearance fallback; use the filtered rank for UI parity.

### 8. Exact prompt state was initially different

The first standalone reproductions locally reconstructed whitespace, markdown,
and chat turns and targeted token position 206. The export establishes a
231-token prefix with the final judgment question mark at position 214.

The standalone parity test now consumes those exact token IDs. The full dataset
cannot depend on a Neuronpedia export for every sample, but it must preserve the
CSV turn text exactly and log the rendered text, token IDs, and selected source
position.

### 9. Backend precision still affects the boundary case

On item 73, the local global `-1.7` intervention moved the PASS-minus-FAIL
margin from `-2.5` to exactly `0.0`; both next-token logits were `17.25`, and
greedy decoding selected `FAIL`. Neuronpedia reported approximately
`P(PASS)=0.4110` and `P(FAIL)=0.3627`, producing `PASS`.

This is a small remaining parity gap, plausibly associated with backend,
caching, kernel, or bfloat16 numerical differences. It is not evidence that the
global intervention had no effect: locally it moved the margin by `+2.5`.

Before calling the local implementation an exact clone, compare cached versus
full-prefix execution and bfloat16 versus float32 scoring on item 73.

## Evidence from the four-item global development run

The standalone global run used the corrected scope and cap:

| Item | Difficulty/outcome | Baseline | Global -1.7 | Global +1.8 fallback |
| --- | --- | ---: | ---: | ---: |
| 73 | hard, factual incorrect | FAIL (-2.500) | FAIL (0.000) | PASS (+0.750) |
| 72 | factual correct | PASS (+1.375) | PASS (+0.875) | PASS (+5.125) |
| 70 | factual incorrect | FAIL (-1.375) | FAIL (-1.563) | PASS (+0.875) |
| 71 | factual incorrect | FAIL (-2.000) | FAIL (-1.938) | PASS (+1.125) |

The consistent local `+1.8` flips on all three baseline-FAIL items and stability
of the baseline-PASS item are promising development evidence. They are not yet
confirmatory because these prompts were used to discover and debug the method.

The sign should not be silently changed and then described as exact
Neuronpedia parity. Use these labels:

- `neuronpedia_exported_polarity`: `-1.7`;
- `local_effective_polarity`: `+1.8`; and
- `zero`: baseline.

## Current implementation status

Already implemented:

- `SteeringSpec` supports backward-compatible `single_position` and
  `all_positions` scopes.
- Global scope supports per-position norms, the 1.0 cap, BOS skipping, and
  generated-token steering.
- `reproduce_neuronpedia_manual.py` exposes
  `--intervention-mode neuronpedia_global`.
- The standalone parser validates the steering export and records whether the
  local baseline and intervention match exported `FAIL -> PASS`.
- The localized mode remains available unchanged.

Still required:

- The full-dataset orchestration in `run_metacognition_jlens.py` still builds
  ordinary `SteeringSpec` objects in `run_attempt`, so it defaults to
  single-position behavior.
- Its adaptive loop still treats question mark versus preceding token as the
  primary intervention comparison.
- The full-run output schema and analysis do not yet distinguish source
  position from intervention scope.

## Required changes to the full-dataset runner

### 1. Add explicit intervention modes to configuration

Recommended configuration shape:

```json
{
  "interventions": {
    "primary_mode": "neuronpedia_global",
    "modes": ["neuronpedia_global", "single_position"],
    "global_strengths": [0.0, -1.7, 1.8],
    "localized_strengths": [0.0, -1.7, -1.8, 1.8],
    "global_cap_fraction": 1.0,
    "steer_generated": true,
    "primary_candidate_token_id": 97817
  }
}
```

Do not overload the old `strengths` and `position_control` fields across both
modes; their meanings differ.

### 2. Centralize steering-spec construction

Move or generalize the standalone `build_steering_spec` helper so both runners
use one implementation. `run_attempt` should receive an `intervention_mode`
and construct:

- a global spec with all-position scope, per-position norm, cap, prompt length,
  BOS positions, and generated-token steering; or
- a localized spec with the existing selected position and mean-norm scale.

### 3. Separate global and localized loops

For each sample, run one pre-specified primary global condition before any
adaptive search. Do not duplicate global attempts for question mark and
preceding-token positions; global steering has no target position.

Run localized question-mark and preceding-token conditions as a separate
mechanistic control family. Their results answer whether a compact local site
is sufficient, not whether the Neuronpedia intervention works.

### 4. Freeze primary versus adaptive analyses

The primary estimand must use one frozen candidate, layer, polarity, and mode
per sample. Candidate/layer sweeps are descriptive rescue analyses and must not
be pooled as independent trials when computing confidence intervals.

Recommended primary condition:

```text
mode:                 neuronpedia_global
direction source:     final judgment question mark
candidate:            评价 / token 97817
layer:                40 when visible under the frozen rule
strengths:            0, -1.7, +1.8
cap:                  1.0
steer generated:      true
```

If layer 40 fails the predeclared visibility rule, record that fact before
trying appearance-layer fallbacks. Report fallback results separately.

### 5. Expand the intervention output schema

Every row should include at least:

```text
intervention_mode
intervention_scope
direction_source_selector
direction_source_position
applied_prompt_position_count
steer_generated
max_injection_fraction
requested_strength
effective_strength_after_cap
feature_label
feature_id / token_id
layer
raw_rank
word_filtered_rank
baseline_output_raw
baseline_output_normalized
intervened_output_raw
intervened_output_normalized
baseline_valid
intervened_valid
baseline_margin
intervened_margin
delta_margin
factual_correct
expected_judgment
baseline_judgment_correct
intervened_judgment_correct
flipped
flip_effect: improved / worsened / changed / no_flip
```

This makes PASS/FAIL visible directly and prevents factual correctness from
being confused with retrospective-judgment correctness.

### 6. Update event and manifest metadata

The manifest must describe the actual intervention, including the global scope,
per-position scale, cap, generated-token behavior, model/lens revisions, dtype,
and cache policy. Each `intervention_applied` event should log the same fields.

### 7. Make analysis mode-aware

Replace the old global position comparison with focused outputs:

1. Primary global rescue and harm rates, split by baseline PASS/FAIL,
   difficulty, and factual correctness, with item-level confidence intervals.
2. PASS-minus-FAIL margin change under `-1.7` and `+1.8` global steering.
3. Paired global-versus-localized effects on the same samples.
4. Baseline margin versus susceptibility to steering.
5. Descriptive token/layer fallback results, clearly separated from the frozen
   primary analysis.

Question-mark versus preceding-token plots apply only to `single_position`.

## Validation gates before another 90-row run

### Unit tests

Retain all existing tests and add tests that verify:

- every non-skipped prompt position is changed in global mode;
- per-position norms are used;
- injections are capped at one residual norm;
- actual BOS positions are unchanged;
- generated positions are steered when enabled;
- localized mode still changes only its selected cell;
- global attempts are not duplicated across position controls;
- `-1.7` and `-1.8` have the same effective capped magnitude;
- raw and normalized PASS/FAIL values are always written; and
- adaptive attempts do not enter the primary CI as independent samples.

### Parity smoke test

Use item 73 with the steering export and item 72 as a stable-PASS control.

Strict item-73 checks:

- exact prompt contains 231 token IDs;
- source question mark is position 214;
- baseline is `FAIL`;
- selected candidate is token 97817 at layer 40;
- global scope and generated-token steering are logged;
- all eligible prompt positions receive the hook;
- candidate/readout movement has the expected direction; and
- local baseline matches the exported baseline.

Treat exported `-1.7 -> PASS` as the strict parity target. If bfloat16 still
produces the observed exact tie, record the parity check as incomplete rather
than hiding it. A practical research smoke gate may additionally require that
`-1.7` materially increases the PASS margin and that the frozen local polarity
produces a valid flip, but that condition must be named local replication, not
exact backend parity.

### Small frozen pilot

After parity smoke, run a small set containing baseline-PASS and baseline-FAIL,
easy and hard, and factually correct and incorrect samples. Confirm:

- no invalid judgment outputs;
- no missing correctness fields;
- global and localized rows have distinct mode labels;
- one baseline is reused rather than counted repeatedly;
- attempt counts match the configured design; and
- analysis completes with valid nonnegative confidence intervals.

Only then authorize the full dataset.

## Full-run interpretation rules

- Estimate flip rates per sample, not per adaptive attempt.
- Separate rescues (`incorrect judgment -> correct`) from harms
  (`correct judgment -> incorrect`).
- Stratify by baseline judgment and baseline margin; easy questions may be
  saturated and structurally unable to show the same rescue effect.
- Treat the four manually examined prompts as development examples. Exclude
  them from a confirmatory subset or label the complete 90-row analysis as
  exploratory.
- A global flip establishes a causal distributed intervention effect. It does
  not localize the mechanism to the question mark.
- A global effect can still reflect broad prompt/output-selection machinery.
  It does not by itself establish a higher-order representation `M(P)`.
- The separate first-order/higher-order experiment in
  `NEXT_STAGE_FIRST_HIGHER_ORDER.md` remains necessary.

## Implementation checklist

- [ ] Add intervention-mode configuration to `experiment_config.json`.
- [ ] Wire `neuronpedia_global` into the main `run_attempt` path.
- [ ] Keep `single_position` as a separately labelled control.
- [ ] Separate direction-source position from application scope.
- [ ] Make token 97817 the explicit primary Neuronpedia candidate.
- [ ] Record raw and filtered candidate ranks.
- [ ] Add scope, cap, generated-token, and effective-strength fields.
- [ ] Remove global duplication across question-mark/preceding-token controls.
- [ ] Separate frozen primary rows from adaptive rescue rows.
- [ ] Update plots and item-level CI calculations.
- [ ] Add global-mode and schema regression tests.
- [ ] Pass item-73/item-72 parity smoke.
- [ ] Pass a small frozen pilot.
- [ ] Run all 90 rows into a new output directory; do not resume the historical
      localized run under the new protocol.


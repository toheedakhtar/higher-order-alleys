# Process-Sensitive Replay Experiment: Implementation Plan

The experiment is implementable, and the design is substantially stronger than the prior SELF/OTHER study.

Two implementation refinements are essential:

1. Qwen3.6-27B is a 64-block hybrid model, not a plain KV-only transformer. Its preserved state contains both full-attention K/V tensors and linear-attention convolution/recurrent states.
2. Candidate discovery and held-out evaluation must be separate, hash-gated phases so held-out data cannot influence candidate or strength selection.

## Proposed experiment directory

```text
experiments/process_sensitive_replay/
├── __init__.py
├── README.md
├── experiment_config.json
├── protocol.py
├── cache_state.py
├── gradient_intervention.py
├── replay.py
├── jlens_readout.py
├── discovery.py
├── analysis.py
├── runner.py
├── test_protocol.py
├── test_cache_state.py
├── test_gradient_intervention.py
├── test_discovery.py
└── test_analysis.py
```

The name `process_sensitive_replay` avoids assuming beforehand that the experiment will discover an `M(P)` representation.

Existing factual extraction, regex scoring, normalization, tokenizer alignment, model loading, manifests, and candidate-direction logic will be reused from the current packages.

The specification says `datasets/MMB/metacognition.csv`; the actual validated repository path is `dataset/metacognition.csv`, which will be used and content-hashed.

## Exact causal flow

```text
Step 0: generate factual answer X once
                    │
                    ▼
        freeze canonical answer tokens
                    │
        ┌───────────┼───────────────┬──────────────────┐
        ▼           ▼               ▼                  ▼
      CLEAN      TARGETED          RANDOM          SUPPORT-MATCHED
 teacher-force   teacher-force    teacher-force     ALTERNATIVE
 exact X         exact X          exact X           teacher-force X
 no hook         -gradient hook   orthogonal hook   distinct mechanism
        │           │               │                  │
        ▼           ▼               ▼                  ▼
       preserved hybrid cache/state per condition
        │           │               │                  │
        ├───────────┴───────────────┴──────────────────┤
        │                           │
        ▼                           ▼
 correctness branch          confidence branch
 independently cloned        independently cloned
 from post-answer state       from post-answer state
```

For reset:

```text
TARGETED_STRONG_PRESERVED
    uses the perturbed stored state

TARGETED_STRONG_RESET
    discards it and reconstructs the identical tokens cleanly
```

Every condition within a meta branch will have identical question, answer, turn delimiters, meta prompt, and tokenizer IDs through the point where meta output begins.

## Tokenization and replay contract

Step 0 will use greedy decoding with `enable_thinking=False` and a hard maximum of 256 answer tokens. Before generating the answer bank, the runner will verify the rendered generation prefix on exactly three examples: one calibration, one prospective, and one knowledge-boundary item. Each must end in Qwen's closed empty thinking block (`<think>\n\n</think>\n\n`), and the rendered-token hashes and suffix IDs will be written to `answer_bank/thinking_mode_verification.json`. Failure is fail-closed.

The answer bank will save both the generated token IDs and decoded text. Generated answer-content token IDs are immutable. A model-produced valid terminal token is logged as the original terminal ID, but the invisible assistant-turn delimiter in the canonical replay transcript is normalized to the single `<|im_end|>` token emitted by the completed chat template. The original and canonical terminal IDs, whether they were already identical, and their hashes remain logged. This normalization never changes answer text or answer-content IDs.

The full chat-template rendering will then be reconstructed and checked to ensure the answer content re-tokenizes to the same IDs and the canonical post-answer transcript matches the rendered conversation. If generation reaches the 256-token cap without any configured valid turn terminator, the truncated content is retained only in raw diagnostics: the item is marked invalid, receives no canonical turn terminator, and is excluded from both discovery and held-out evaluation rather than being treated as canonical X.

If decoding and re-tokenization are not stable:

- the item is marked invalid;
- it remains in raw logs;
- it is excluded from primary analysis;
- the runner does not silently substitute a different answer.

The canonical hashes will cover:

- factual question tokens;
- answer content tokens;
- post-answer transcript tokens;
- correctness-branch prompt tokens;
- confidence-branch prompt tokens.

Conditions will be compared by token hashes, not merely decoded strings.

Turn 3 uses suffix-only construction. The canonical factual history is never
rerendered after its `post_answer_token_ids` and hybrid cache/state have been
frozen, because Qwen's chat template may conditionally rewrite an earlier
assistant thinking prefix when another user turn is added. For each meta
branch, the runner renders only a standalone Turn-3 user turn plus the frozen
no-thinking assistant-generation prefix, prepends the single required newline,
and appends those suffix IDs to the preserved factual prefix/cache.

Before advancing the cache, the runner must prove all of the following exactly:

- the stored question rendering, answer text, and canonical `<|im_end|>`
  reconstruct the frozen `post_answer_token_ids` without rerendering history;
- the frozen prefix ends at the canonical assistant-turn boundary;
- tokenizing `<|im_end|>` together with the new suffix equals the concatenation
  of the independently frozen boundary token and suffix IDs;
- tokenizing the final stored-render-plus-suffix transcript equals exactly
  `post_answer_token_ids + suffix_token_ids`;
- prefix, suffix, boundary, and final-transcript token hashes agree across all
  matched conditions, and the `?` capture index is identical.

Any changed prefix token, invalid chat boundary, hash mismatch, or concatenated
transcript mismatch halts the phase fail-closed as `invalid_cache_state`.

## Answer-support gradient

I recommend freezing:

```text
PROCESS_LAYER = 31
```

This is the zero-based 32nd block, the end of the first half of the 64-block model. Under the installed Qwen configuration, it is also a full-attention block. `META_READOUT_LAYER=40` remains independent.

For canonical prefix length `q` and answer tokens `X₁ … Xₙ`:

```text
logits[q - 1] predict X₁
logits[q]     predict X₂
...
logits[q+n-2] predict Xₙ
```

A differentiable token-by-token recurrent gradient pass will compute:

```text
S = Σ log P(Xₜ | Q, X<t)
gᵢ = ∂S / ∂hᵢ
```

where `hᵢ` is the layer-31 block output at each answer-predicting position.

This gradient pass must use the same Qwen one-token recurrent kernels, token positions, and causal cache semantics as ordinary experimental replay. Only autograd-breaking cache mutations may be functionalized: convolution-state storage is cloned immediately before an in-place recurrent update, and recurrent-state `copy_` writes are replaced by graph-preserving assignment in the gradient-only cache. Model equations, kernels, tokens, intervention definitions, and the ordinary preserved-state cache are unchanged.

The gradient hook is registered only for the declared answer-predicting positions. Before a gradient is accepted, the pass runs an ordinary cached replay alongside it and requires, at the existing `1e-5` absolute and relative tolerances:

- full-vocabulary per-token logit parity at every intended position;
- total answer-sequence support parity;
- layer-31 residual-state parity at every intended position;
- finite, nonzero answer-support and entropy gradients at every intended position;
- an exact hook-position list with no activity outside the declared factual-answer positions.

Every comparison and maximum absolute/relative difference is logged. Any failure halts the phase fail-closed. The earlier full-sequence no-cache gradient route is prohibited because Qwen3.6 selects a chunked delta-rule kernel there while experimental replay uses its recurrent kernel.

The gradient and clean residual norms will then be detached and frozen. During sequential cached replay:

```text
targeted deltaᵢ = -alpha × ||hᵢ_clean|| × normalize(gᵢ)
```

Using the clean norm prevents cumulative perturbations from changing the nominal dose.

The last answer token will still be consumed into the state after its log probability is measured, but the process hook will be disabled because it does not predict another answer token. Assistant turn-ending tokens and all Turn-3 tokens are also processed with the hook disabled.

## Qwen hybrid-state handling

This needs custom infrastructure rather than assuming ordinary `past_key_values`.

The state utility will clone, hash, and audit:

- full-attention keys and values;
- linear-attention convolution states;
- linear-attention recurrent states;
- initialization flags and sequence/cumulative lengths;
- cache positions and layer types.

Because the hook is on the output of layer 31:

- state at layers 0–31 should remain unchanged;
- at least one downstream layer 32–63 state must change;
- the runner will fail if only the transient activation changes but no downstream persistent state differs.

Independent branches will use deep tensor clones. Object identity and tensor storage pointers will be checked so correctness scoring, confidence scoring, and generation cannot mutate one another’s state.

## Conditions

The six primary conditions will be:

1. `clean_preserved`
2. `targeted_weak_preserved`
3. `targeted_strong_preserved`
4. `random_strong_preserved`
5. `support_matched_alternative_preserved`
6. `targeted_strong_reset`

`random_strong_preserved` will use the exact per-position norm of the strong targeted intervention.

For each position:

```text
r ~ seeded Gaussian
r⊥ = r - projection_of_r_onto_g
random delta = matched_target_norm × normalize(r⊥)
```

Seeds will be derived deterministically from the campaign seed, item ID, and global token position.

`clean_reset` will be mandatory in smoke and periodic held-out integrity items. Reset reconstruction will use the same incremental replay engine with interventions disabled, avoiding a chunked-prefill versus recurrent-decoding numerical confound.

## Support-matched alternative perturbation

The norm-matched random condition controls for generic perturbation energy, but it does not control for the functional size of the disruption. A second perturbation family will therefore be calibrated to reproduce the answer-support drop of `targeted_strong_preserved` through a different residual-space direction.

The recommended alternative objective is predictive-distribution entropy rather than direct answer suppression. On the clean gradient replay, compute:

```text
A = Σₜ H[P(next token | Q, X<t)]
aᵢ = ∂A / ∂hᵢ
```

At each answer-predicting position, remove the component parallel to the answer-support gradient:

```text
ĝᵢ = normalize(gᵢ)
aᵢ⊥ = aᵢ - (aᵢ · ĝᵢ) ĝᵢ
uᵢ = normalize(aᵢ⊥)
alternative deltaᵢ = beta × ||hᵢ_clean|| × uᵢ
```

This makes the alternative intervention an entropy-increasing/uncertainty-inducing manipulation rather than another direct step down `-g`. The absolute cosine between `uᵢ` and `gᵢ` must be at most the configured orthogonality tolerance, proposed as `0.10`. If the projected entropy gradient is non-finite or too small, use a deterministic seeded Gaussian direction projected orthogonally to `gᵢ`, and record the fallback explicitly.

The alternative condition will use:

- the same process layer;
- the same answer-predicting positions;
- the same clean residual-norm reference;
- the exact same teacher-forced answer tokens;
- the same state-preservation path into Turn 3;
- a single global `beta`, never a per-item fitted magnitude.

Because an approximately orthogonal direction has little first-order effect on answer support, it may require a larger perturbation norm than the targeted gradient. The alternative norm will therefore be logged separately and bounded. If support matching requires non-finite states or a median total perturbation norm more than four times the targeted condition, the support-matched-control gate fails rather than accepting a broadly destructive intervention.

## Dataset split

The answer-bank phase will first generate X for all 82 items without computing any meta readout.

A deterministic seed-42 allocator will then select 16 discovery items:

- 12 calibration
- 2 prospective-source
- 2 knowledge-boundary-source

Within each family, it will include both factually correct and incorrect answers whenever available. Exact IDs and answer hashes will be frozen in `split_manifest.json` and the run-local resolved config.

All remaining valid items become held-out, up to 66 when every answer is valid. Any item invalidated during answer-bank construction is listed separately in `excluded_invalid_item_ids` and belongs to neither split.

The discovery loader will reject held-out IDs. The held-out runner will refuse to start without a frozen candidate/strength file whose hashes match the dataset, answer bank, model, lens, split, and code configuration.

## Strength discovery

Only the 16 discovery items will test:

```text
alpha = 0.01, 0.02, 0.05, 0.10, 0.20
```

Proposed deterministic selection rule:

- Weak: smallest alpha with median support drop ≥0.75 nat and positive support drop on at least 12/16 valid items.
- Strong: smallest alpha with median support drop in `[2, 4]` nats and no non-finite activations, logits, or cache states.
- If no strong alpha reaches 2 nats, select the largest valid alpha below 4 nats.
- If the grid overshoots or fails to manipulate support reliably, the discovery run fails its strength gate. Expanding the grid requires a new declared discovery run.

This prevents silent per-item or held-out tuning.

After freezing `STRONG`, discovery will calibrate the alternative perturbation on a separate declared `beta` grid, initially:

```text
beta = 0.05, 0.10, 0.20, 0.40, 0.80
```

For each beta, define the paired support mismatch:

```text
support_mismatchᵢ =
    support_drop_alternativeᵢ - support_drop_targeted_strongᵢ
```

Select the single global beta that minimizes median absolute support mismatch, subject to:

- finite activations, logits, and cache/recurrent states;
- positive median alternative support drop;
- median alternative support drop within 25% of the targeted-strong median;
- median absolute paired mismatch no greater than `max(0.5 nat, 25% of the targeted-strong discovery median)`;
- the four-times perturbation-norm ceiling;
- the configured gradient-direction cosine threshold.

The chosen beta, tolerance, direction construction, fallback rule, and support-matching diagnostics are frozen in `frozen_protocol.json`. If no beta passes, discovery ends with `support_match_gate_failed`. The grid cannot be expanded inside the same campaign.

On held-out items, beta is never adapted. Support-match quality is evaluated as an experimental result and as a fail-closed validity gate. Before held-out execution, the protocol will freeze the aggregate discovery criteria above plus a minimum item-level match fraction of `0.65`. Held-out support matching passes only if the alternative has a positive median support drop, its median drop is within 25% of the targeted-strong median, its median absolute paired mismatch is no greater than `max(0.5 nat, 25% of the targeted-strong held-out median)`, and at least 65% of valid held-out items pass the item-specific tolerance. An item counts as support-matched only when `support_drop_targeted > 0` and its absolute paired mismatch is within `max(0.5 nat, 0.25 * abs(support_drop_targeted))`. Failure marks the held-out campaign invalid and blocks any process-property or `M(P)`-like interpretation.

## Meta branches and readout

Each post-answer state will be cloned into two branches:

- Correctness: `CORRECT` versus `INCORRECT`.
- Process confidence: `HIGH_CONFIDENCE` versus `LOW_CONFIDENCE`.

For each branch:

1. Render only the new Turn-3 suffix and incrementally append its exact token
   IDs to a storage-disjoint clone of the preserved factual state.
2. Capture residuals when the `?` token itself is processed.
3. Continue to the output boundary.
4. Clone again for:
   - greedy label generation;
   - first label sequence score;
   - second label sequence score.

This prevents one label-scoring pass from mutating the cache used by another.

J-Lens readout will cover layers `36–44`, including layer 40:

- top 100 vocabulary directions per layer;
- exact scores/ranks for tokens `97817` and `99973`;
- exact scores/ranks for frozen candidates;
- complete residual and direction metadata.

Searching every vocabulary direction at every layer is feasible on 16 discovery items, but saving all scores for every held-out condition is not. Discovery all-vocabulary scores will therefore be stored in a compact tensor artifact, while held-out JSONL stores top-100 plus explicit candidates.

## Candidate discovery rule

Candidate search will use the confidence branch as the primary discovery branch. Correctness will remain a secondary/readout branch, avoiding two-branch fishing.

Eligible vocabulary directions must:

- pass the existing meaningful-word filter;
- not be special tokens;
- not be tokens appearing in either meta prompt or response labels;
- not be `97817` or `99973`, which remain controls;
- have finite, nontrivial score variance.

For every token/layer on discovery, calculate:

- item-centered relationship with support drop across clean, weak, and strong;
- targeted-strong minus clean effect;
- support-matched-alternative minus clean effect;
- targeted-strong minus random effect;
- targeted-preserved minus reset effect;
- agreement between targeted-strong and support-matched-alternative after accounting for their realized support drops;
- item-level sign consistency.

Candidates will be oriented using the discovery slope sign. An eligible candidate must move in the same oriented direction under targeted-strong and support-matched-alternative, while the norm-matched random response is smaller or less structured. Candidates whose targeted and support-matched responses strongly diverge after conditioning on support drop will be excluded as perturbation-mechanism-specific traces.

The remaining candidates will be ranked by a frozen equal-weight rank aggregation of the metrics above and deduplicated by direction-vector cosine similarity. Up to three token/layer candidates will be frozen.

Semantic translation is annotation only and cannot affect ranking.

## Held-out inference

For each frozen candidate, the 66-item analysis will test:

- H1: oriented targeted-strong minus clean effect.
- H2: oriented support-matched-alternative minus clean effect in the same direction as H1.
- H3: targeted-strong and support-matched-alternative produce similar candidate responses after accounting for their realized support drops.
- H4: targeted-strong minus random effect.
- H5: oriented targeted-preserved minus reset effect.
- H6: item-centered candidate-score slope against support drop.
- H7: item-centered confidence-margin slope against support drop.

Statistics will use the item as the resampling unit:

- paired mean and median;
- item-bootstrap 95% intervals;
- item-centered regression slope;
- Pearson and Spearman correlations;
- no discovery statistics presented as confirmation.

Support matching itself will be tested first using the paired held-out difference:

```text
support_drop_targeted_strong - support_drop_alternative
```

Held-out support-match reporting will include the mean signed mismatch, median signed mismatch, mean absolute mismatch, and median absolute mismatch. It will also report the item-level match fraction:

```text
mean_i I[
    support_drop_targeted_i > 0
    and
    abs(support_drop_alternative_i - support_drop_targeted_i)
    <= max(0.5 nat, 0.25 * abs(support_drop_targeted_i))
]
```

The denominator and counts of matched and unmatched valid held-out items will be reported explicitly. The bootstrap interval and aggregate mismatch summaries will be compared with the frozen discovery tolerance, but they will not substitute for item-level coverage. Candidate convergence will then be evaluated in two complementary ways:

1. Paired support-normalized response:

   ```text
   response_targeted =
       oriented(candidate_targeted - candidate_clean)
       / support_drop_targeted

   response_alternative =
       oriented(candidate_alternative - candidate_clean)
       / support_drop_alternative
   ```

2. An item-fixed-effect model over the targeted and alternative rows:

   ```text
   candidate_delta =
       beta_support × support_drop
       + beta_mechanism × I[alternative]
       + item_effect
   ```

Evidence for convergence requires a nonzero support relationship in the discovery-frozen direction and a small mechanism-specific term relative to the shared targeted/alternative effect. Bootstrap intervals will resample whole items, keeping both mechanisms together. Raw differences will always be shown alongside normalized effects so division by small support drops cannot hide instability.

A candidate will only be called process-sensitive if H1–H6 replicate in the predeclared direction, the support-matched alternative converges with the targeted response, reset approaches clean, and both structured effects are distinguishable from the norm-matched random control. H7 provides the stronger metacognitive-monitoring result.

## Required plots

1. **Manipulation check:** support drop for clean, targeted weak, targeted strong, support-matched alternative, and norm-matched random.
2. **Confidence versus first-order support:** item-level confidence margin against support drop, grouped by condition.
3. **Candidate score versus support drop:** held-out candidate scores and item-centered trend.
4. **Targeted versus random:** paired candidate-score effects relative to clean.
5. **Preserved versus reset:** targeted-strong preserved versus identical-transcript reset for candidate score and confidence margin.
6. **Clean versus perturbed with identical text:** paired clean and targeted-strong candidate scores.
7. **Generic evaluator controls:** tokens `97817` and `99973` across all process conditions.
8. **Layer profile:** frozen candidate scores across layers 36–44 for clean, targeted, support-matched alternative, random, and reset.
9. **Held-out effect summary:** targeted-clean, alternative-clean, targeted-random, targeted-reset, and targeted-alternative effects with item-bootstrap intervals.
10. **Five-condition candidate comparison:** candidate score for `CLEAN`, `TARGETED_STRONG`, `SUPPORT_MATCHED_ALTERNATIVE`, `RANDOM_NORM_MATCHED`, and `RESET`, showing individual held-out items and paired means.
11. **Mechanism convergence relative to support drop:** targeted versus alternative support-normalized candidate response, with the identity line and item-paired connections.
12. **Held-out item-level support matching:** paired targeted and alternative support drops for every held-out item, with the frozen item-specific tolerance band and each item marked as matched or unmatched. The panel annotation will show matched count, valid-item denominator, match fraction, and mean/median mismatch summaries.

Plots 10–12 are critical for the new control. Plot 10 shows whether both functionally matched perturbations converge in candidate space; Plot 11 tests whether apparent convergence remains after accounting for realized support reduction; Plot 12 prevents good aggregate matching from hiding poor item-level matching. The held-out summary table will include the per-item targeted drop, alternative drop, signed mismatch, absolute mismatch, item-specific tolerance, and match indicator, followed by the aggregate mean, median, matched count, denominator, and match fraction.

## Run phases and gates

```text
validate
    static schemas, dataset, layers, split algorithm

answer_bank
    verify no-thinking rendering on three examples; generate X once with a
    256-token hard cap; canonicalize only the invisible turn terminator;
    freeze content/transcript hashes and valid-item split

pre_discovery_smoke
    2–4 discovery items; replay/token parity, hybrid-cache cloning,
    state isolation, hook lifetime, gradient sign/indexing, reset mechanics,
    and alpha/beta-grid machinery; does not require a frozen beta

discovery
    alpha calibration, alternative beta support matching,
    and all-vocabulary candidate search

freeze
    write immutable strengths/candidates/selection rule

post_freeze_smoke
    re-run all critical gates with the frozen protocol, including support
    matching, reset parity, cache/state integrity, and hook lifetime

heldout
    66 items; refuses access unless post_freeze_smoke passed fail-closed

analyze
    regenerate reports without loading Qwen only for a valid campaign;
    emit gate-failure diagnostics only for an invalid campaign
```

A campaign directory will contain phase-specific append-only artifacts so resume cannot mix completed discovery and held-out rows.

### Fail-closed gate enforcement

The gates are executable prerequisites, not advisory warnings. Every phase writes a machine-readable `gate_status.json` containing the gate name, frozen thresholds, measurements, pass/fail status, reason, and hashes of the inputs it validated. The runner may create the next phase only when every prerequisite status is `passed` and its hashes match. A missing, stale, skipped, or failed gate is treated as failure; there is no command-line override that converts it to a valid run.

Three gate families are causally critical:

1. **Support-match gate**
   - Discovery halts without writing a usable `frozen_protocol.json` if no global alternative strength passes the frozen matching criteria.
   - Pre-discovery engineering smoke validates the alpha/beta-grid machinery but does not require or select a frozen beta.
   - Post-freeze critical smoke halts immediately if the frozen alternative strength fails its support-match assertion.
   - Held-out evaluation never retunes the strength. Once the paired held-out support data are complete, the aggregate and item-level criteria above are evaluated. Failure sets campaign status to `invalid_support_match`.
2. **Reset-parity gate**
   - `clean_preserved` and independently reconstructed `clean_reset` must have identical transcript/token hashes and cache topology, and must agree within frozen absolute/relative numerical tolerances for answer support, post-answer state, Turn-3 logits, confidence/correctness margins, and candidate readouts.
   - This is mandatory in both smoke phases and on the predeclared periodic held-out integrity items. Any failure halts the active phase and sets campaign status to `invalid_reset_parity`.
3. **Cache/state-integrity gate**
   - The perturbation must produce a persistent downstream state difference before Turn 3; all expected Qwen hybrid-state components must exist with valid shapes, positions, layer types, and finite values.
   - Condition and meta-branch states must be deep, storage-disjoint clones. Hashes of every source state must remain unchanged after any sibling branch runs. The factual-process hook invocation count must be zero during Turn 3.
   - Turn 3 must be suffix-only: the frozen factual history may not be rerendered. Exact prefix, suffix, boundary, and final concatenated-transcript token/hash parity is mandatory across conditions, and any invalid chat boundary fails closed.
   - These assertions execute per item and condition. Any failure immediately halts the active phase and sets campaign status to `invalid_cache_state`.

An invalid campaign may retain append-only raw records and produce a clearly watermarked gate-diagnostic report, including the item-level support-match plot/table needed to locate the failure. It must not produce confirmatory candidate-effect summaries, run hypothesis tests as though the campaign were valid, or enter the results-interpretation path. Resume may continue only from the last clean phase boundary after the defect is fixed under a new campaign/protocol hash; partial invalid outputs cannot be merged into a valid campaign.

## Additional artifacts

Alongside the requested outputs, I would add:

```text
answer_bank.jsonl
split_manifest.json
state_audits.jsonl
frozen_protocol.json
gate_status.json
discovery_vocab_scores.pt
heldout_effects.csv
heldout_support_match.csv
```

`frozen_protocol.json` is the critical handoff between discovery and held-out evaluation.

## Test strategy

CPU tests will cover:

- autoregressive predictor-position indexing;
- stable answer token reconstruction;
- exact preservation of generated answer-content IDs while canonicalizing only the invisible assistant-turn terminator;
- logging of original and canonical terminal IDs;
- invalidation and split exclusion when no valid termination occurs by the 256-token cap;
- `enable_thinking=False` rendering with a closed empty thinking block on one example from each item family;
- targeted intervention sign using finite differences;
- exact targeted/random norm matching;
- random-gradient orthogonality;
- alternative/answer-gradient orthogonality;
- entropy-gradient projection and deterministic fallback;
- deterministic global beta selection;
- rejection of per-item alternative-strength tuning;
- support-match tolerance and perturbation-norm gates;
- held-out item-level support-match indicator, count, denominator, and fraction calculations;
- fail-closed phase transitions and rejection of missing, stale, skipped, or failed gate records;
- invalid-campaign exclusion from confirmatory analysis and interpretation;
- hook activation only at declared factual positions;
- complete hybrid-cache cloning and storage independence;
- reset reconstruction;
- reset-parity failure propagation;
- branch independence;
- transcript/hash equality;
- immutable-prefix suffix-only Turn-3 construction, exact boundary/final token
  parity, and rejection of invalid chat boundaries;
- discovery/held-out access barriers;
- candidate-ranking determinism;
- item-level bootstrap behavior.

The pre-discovery engineering smoke must prove all applicable infrastructure properties below without requiring a frozen beta. After discovery freezes the protocol, the post-freeze critical smoke must re-run the complete list and additionally prove frozen-beta support matching:

- recurrent gradient replay has per-token full-logit, total-support, and intervention-layer residual parity with ordinary cached replay at the frozen tolerance;
- answer-support and entropy gradients are finite and nonzero at every intended position;
- gradient and intervention hooks fire exactly at the declared factual-answer positions and nowhere else;
- targeted support is reduced;
- the alternative direction satisfies the frozen cosine threshold;
- in post-freeze smoke, the alternative support drop approximately matches targeted strong on smoke items using the discovery-frozen beta;
- targeted and alternative caches are distinct from clean and from each other;
- the alternative perturbation remains disabled during Turn 3;
- targeted downstream state differs;
- reset removes the difference;
- clean preserved and clean reset agree within frozen tolerance;
- meta readout is taken from the preserved state;
- Turn-3 suffix construction preserves every frozen factual prefix token and
  passes exact boundary/final transcript hash parity across conditions;
- no factual-process hook fires during either meta branch.

The smoke reports must show per-position targeted, random, and alternative norms; targeted/alternative gradient cosine; support mismatch when a frozen beta exists; reset-parity measurements; hybrid-state digests; and factual-prefix, Turn-3 suffix, prefix-suffix boundary, and final concatenated-transcript hashes. A support-match failure in post-freeze smoke, or a reset-parity, orthogonality, state-preservation, suffix-integrity, branch-isolation, or hook-lifetime failure in either smoke phase, is critical and stops the run. The runner must return a nonzero exit status and must not write a phase-success marker. Discovery may start only after pre-discovery engineering smoke passes. Held-out may start only after post-freeze critical smoke passes fail-closed.

## Results interpretation

The reporting layer will distinguish gate-invalid diagnostics from valid-campaign conclusions. If any critical gate fails, the campaign is marked invalid and only the corresponding diagnostic report is emitted. The substantive outcomes below are considered only when the support-match, reset-parity, and cache/state-integrity gates all pass:

1. **No later effect beyond random:** evidence favors ordinary first-order/readout behavior for the tested mechanism.
2. **Targeted and random produce similar later effects:** evidence favors generic hidden-state contamination or causal persistence.
3. **Targeted and support-matched alternative reduce support similarly but produce different later candidate states:** evidence favors perturbation-specific traces rather than a representation of shared process reliability.
4. **Targeted and support-matched alternative reduce support similarly, converge on the same later candidate response, exceed the norm-matched random response, replicate on held-out items, and disappear under reset:** stronger evidence for a process-sensitive representation of answer reliability or conflict.
5. **Outcome 4 plus convergent reductions in confidence margin:** stronger evidence for process-sensitive metacognitive monitoring despite identical visible answers.

Outcome 4 may only be described as "evidence for a process-property / process-sensitive representation" or as "an `M(P)`-like candidate." It must not be described as a proven higher-order representation. Convergence alone does not establish that the candidate causally controls the later judgment.

## Main remaining scientific caveat

Even a successful result would not yet show that the candidate causes downstream monitoring. It would establish:

```text
P manipulation → process-sensitive candidate/readout
```

not:

```text
candidate → judgment
```

A later causal mediation/restoration experiment establishing `candidate -> judgment/control` would be required before making a stronger higher-order claim.

If support matching fails at either the aggregate or required item-coverage level, reset parity fails, or any cache/state-integrity assertion fails, the campaign is invalid and receives diagnostics only. If all validity gates pass but the alternative fails to reproduce the targeted candidate response on held-out items, that is a valid negative result and cannot support a process-property/process-sensitive interpretation or an `M(P)`-like candidate. Under no outcome in this experiment should the result be called a proven higher-order representation.

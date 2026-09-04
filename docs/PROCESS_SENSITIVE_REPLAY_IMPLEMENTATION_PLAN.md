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

The model and tokenizer are pinned to resolved commit
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The J-Lens checkpoint is pinned
to repository commit `0731326edff4ae730ffc5356fe1a4728c748b3a6` and file
SHA-256 `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.
Those identities and the installed Torch, Transformers, J-Lens, and
Hugging Face Hub versions are part of every phase's base protocol hash and
gate inputs. A change makes prior gates stale and fails closed.
Accordingly, the real campaign's `validate` phase and all subsequent phases
must run in the same unchanged CUDA software environment; CPU validation is a
separate disposable engineering check and cannot seed the CUDA campaign.

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

The full chat-template rendering will then be reconstructed and checked to ensure the answer content re-tokenizes to the same IDs and the canonical post-answer transcript matches exactly. Because Qwen's completed-turn template places one newline after `<|im_end|>` while the frozen factual state intentionally stops at `<|im_end|>`, the completed template must equal the canonical stored factual rendering plus exactly that one independently tokenized suffix separator; arbitrary trailing tokens are forbidden. If generation reaches the 256-token cap without any configured valid turn terminator, the truncated content is retained only in raw diagnostics: the item is marked invalid, receives no canonical turn terminator, and is excluded from both discovery and held-out evaluation rather than being treated as canonical X.

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
- finite, nonzero answer-support gradients at every intended position for the
  primary layer and each predeclared alternative layer;
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

The identical relative invariant applies to the frozen alternative layer:
layers through that intervention layer remain clean and at least one later
persistent layer must change.

Independent branches will use deep tensor clones. Object identity and tensor storage pointers will be checked so correctness scoring, confidence scoring, and generation cannot mutate one another’s state.

## Conditions

The seven primary conditions for `psr-v8` will be:

1. `clean_preserved`
2. `targeted_weak_preserved`
3. `targeted_strong_preserved`
4. `random_strong_preserved`
5. `support_matched_alternative_preserved`
6. `alternative_random_preserved`
7. `targeted_strong_reset`

`random_strong_preserved` will use the exact per-position norm of the strong
targeted intervention at layer 31. `alternative_random_preserved` will use the
exact per-position norm of the alternative targeted intervention at its frozen
earlier layer. Both random controls are deterministic Gaussian directions
projected orthogonally to the answer-support gradient at their own layer.

For each position:

```text
r ~ seeded Gaussian
r⊥ = r - projection_of_r_onto_g
random delta = matched_target_norm × normalize(r⊥)
```

Seeds will be derived deterministically from the campaign seed, item ID, and global token position.

`clean_reset` will be mandatory in smoke and periodic held-out integrity items. Reset reconstruction will use the same incremental replay engine with interventions disabled, avoiding a chunked-prefill versus recurrent-decoding numerical confound.

## Support-matched alternative perturbation (`psr-v8`)

`psr-v7` established that the same-layer entropy/orthogonal perturbation could
match the aggregate support drop but not the frozen paired-item mismatch gate.
It remains a valid failed-gate diagnostic campaign. `psr-v8` replaces only
that failed alternative with a different-layer, same-objective intervention.

The primary intervention remains at zero-based layer 31. The alternative layer
is selected from the predeclared set `{15, 19, 23}`. The pinned Qwen
architecture identifies all four as full-attention blocks; the alternatives
are respectively 16, 12, and 8 blocks before layer 31 and all are well below
meta readout layer 40. Layer 27 is excluded because its four-block separation
is not substantial enough for this comparison.

At each candidate alternative layer, compute the identical clean objective:

```text
S = sum_t log P(X_t | Q, X_<t)
g_alt_i = dS / dh_alt_i
delta_alt_i = -beta * ||h_alt_i_clean|| * normalize(g_alt_i)
```

The alternative uses the same answer-predicting positions, factual question,
exact teacher-forced answer IDs, recurrent replay path, preserved hybrid-state
path, and Turn-3 suffixes. Discovery freezes one alternative layer and one
global beta; per-item strength fitting is forbidden.

Alternative-layer selection may use only support-match quality, finite
state/cache integrity, and the existing four-times median perturbation-norm
ceiling. It may not use J-space/candidate activity, confidence, correctness,
or held-out data. `alternative_random_preserved` is generated at the selected
alternative layer, is orthogonal to that layer's answer-support gradient, and
exactly matches the alternative targeted norm at every position.

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
alpha = 0.01, 0.02, 0.05, 0.10, 0.11, 0.125, 0.15, 0.20
```

This alpha grid remains unchanged from `psr-v7`. That campaign selected weak
alpha `0.10` and strong alpha `0.11`, then failed the support-matched
alternative gate. No `psr-v7` measurements are imported into `psr-v8`.

Proposed deterministic selection rule:

- Weak: smallest alpha with median support drop ≥0.75 nat and positive support drop on at least 12/16 valid items.
- Strong: smallest alpha with median support drop in `[2, 4]` nats and no non-finite activations, logits, or cache states.
- If no strong alpha reaches 2 nats, select the largest valid alpha below 4 nats.
- If the grid overshoots or fails to manipulate support reliably, the discovery run fails its strength gate. Expanding the grid requires a new declared discovery run.

This prevents silent per-item or held-out tuning.

After freezing `STRONG`, discovery evaluates every predeclared alternative
layer against the following predeclared global-beta grid:

```text
alternative_layer = 15, 19, 23
beta = 0.05, 0.08, 0.10, 0.11, 0.125, 0.15, 0.20, 0.30, 0.40
```

For each beta, define the paired support mismatch:

```text
support_mismatchᵢ =
    support_drop_alternativeᵢ - support_drop_targeted_strongᵢ
```

Select the single `(alternative_layer, global_beta)` pair that minimizes
median absolute support mismatch, subject to:

- finite activations, logits, and cache/recurrent states;
- positive median alternative support drop;
- median alternative support drop within 25% of the targeted-strong median;
- median absolute paired mismatch no greater than `max(0.5 nat, 25% of the targeted-strong discovery median)`;
- the four-times perturbation-norm ceiling;

Exact ties are resolved by ascending alternative layer and then ascending
beta. No candidate/J-space or meta-output quantity participates.

The chosen alternative layer, beta, same-objective construction, and complete
support-matching diagnostics are frozen in `frozen_protocol.json`. If no
layer/beta pair passes, discovery ends with `support_match_gate_failed`. The
layer or strength grid cannot be expanded inside the same campaign.

Each discovery alpha row and the aggregate per-alpha median, positive-item
count, finiteness, weak eligibility, and strong eligibility are flushed and
hashed before the alpha selector executes. After alpha passes, the same rule
applies to beta rows and their complete support-match diagnostics before beta
selection. Thus a failed strength gate preserves the exact measurements that
caused failure while still withholding the phase-success marker and all frozen
strength/candidate artifacts. Diagnostic persistence never relaxes a gate.

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
- targeted-strong minus its layer-31 random effect;
- alternative-targeted minus its own earlier-layer random effect;
- targeted-preserved minus reset effect;
- agreement between targeted-strong and support-matched-alternative after accounting for their realized support drops;
- item-level sign consistency.

The executable definition is frozen as follows. For each item, scores and
support drops for clean, targeted-weak, and targeted-strong are centered within
item. Their pooled through-origin slope defines the item-centered support
relationship; its sign orients that token/layer candidate. Targeted,
alternative, both same-layer structured-minus-random effects, and
targeted-preserved-minus-reset effects are then multiplied by this orientation.

Support-adjusted agreement is computed by fitting one pooled through-origin
slope from realized targeted/alternative support drops to their respective
candidate effects. The agreement metric is the negative mean absolute paired
difference between the targeted and alternative residuals. The corresponding
divergence ratio divides that mismatch by the sum of the mean absolute targeted
and alternative effects.

An eligible candidate must have score variance greater than `1e-8`, a nonzero
support slope, positive oriented targeted and alternative mean effects, a
positive oriented targeted-preserved-minus-reset effect, and for each
mechanism either a positive oriented structured-minus-own-random effect or
greater structured than pooled-random item-sign consistency. Its
support-adjusted divergence ratio must be no greater than `1.0`.

Eligible candidates receive deterministic descending ordinal ranks for eight
metrics: absolute item-centered support slope, oriented targeted effect,
oriented alternative effect, oriented targeted-minus-primary-random,
oriented alternative-minus-alternative-random,
oriented targeted-preserved-minus-reset, support-adjusted agreement, and
structured item-sign consistency. The unweighted mean of those eight ranks is
the aggregate rank; exact ties retain ascending flattened layer/token order.
Ranked candidates are greedily deduplicated using an absolute effective
direction cosine ceiling of `0.9`, where the saved effective direction is
`normalize(J_l.T @ lm_head.weight[token_id])`. Up to the first three surviving
token/layer candidates are frozen, including their orientation, vector, tensor
hash, and file hash.

All full-vocabulary scores for both meta branches are stored in float16 with
finite-value validation in `discovery_vocab_scores.pt`; ranking converts the
frozen confidence-branch tensor back to float32. The exact score tensor,
support-drop matrix, eligibility mask, metric tensors, and complete ranked
index are saved, so selection can be reconstructed without held-out data.

Semantic translation is annotation only and cannot affect ranking.

## Held-out inference

For each frozen candidate, the 66-item analysis will test:

- H1: oriented targeted-strong minus clean effect.
- H2: oriented support-matched-alternative minus clean effect in the same direction as H1.
- H3: targeted-strong and support-matched-alternative produce similar candidate responses after accounting for their realized support drops.
- H4: targeted-strong minus its layer-31 random and support-matched-alternative
  minus its frozen earlier-layer random effects.
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

The descriptive pattern relevant to a future explicitly frozen classification
rule is: H1–H6 replicate in the predeclared direction, the support-matched
alternative converges with the targeted response, reset approaches clean, and
both structured effects are distinguishable from their own same-layer random
controls. This experiment does not automatically apply the label
process-sensitive or `M(P)`-like because no numerical convergence decision
threshold has been approved. H7 provides the stronger metacognitive-monitoring
result.

## Required plots

1. **Manipulation check:** support drop for clean, primary targeted, primary
   random, alternative targeted, and alternative random.
2. **Confidence versus first-order support:** item-level confidence margin against support drop, grouped by condition.
3. **Candidate score versus support drop:** held-out candidate scores grouped
   by mechanism and intervention layer.
4. **Mechanism comparisons:** primary targeted versus alternative targeted;
   primary targeted versus primary random; and alternative targeted versus
   alternative random.
5. **Preserved versus reset:** targeted-strong preserved versus identical-transcript reset for candidate score and confidence margin.
6. **Clean versus perturbed with identical text:** paired clean and targeted-strong candidate scores.
7. **Generic evaluator controls:** tokens `97817` and `99973` across all process conditions.
8. **Layer profile:** frozen candidate scores across layers 36–44 for clean,
   both targeted mechanisms, both own-layer random controls, and reset.
9. **Held-out effect summary:** targeted-clean, alternative-clean, both
   structured-minus-own-random contrasts, targeted-reset, and
   targeted-alternative effects with item-bootstrap intervals.
10. **Mechanism/control candidate comparison:** candidate score for `CLEAN`,
    primary targeted/random, alternative targeted/random, and `RESET`, showing
    individual held-out items and paired means.
11. **Mechanism convergence relative to support drop:** targeted versus alternative support-normalized candidate response, with the identity line and item-paired connections.
12. **Held-out item-level support matching:** paired targeted and alternative support drops for every held-out item, with the frozen item-specific tolerance band and each item marked as matched or unmatched. The panel annotation will show matched count, valid-item denominator, match fraction, and mean/median mismatch summaries.
13. **Support versus mechanism residual:** the item-fixed-effect shared-support
    component at the mean realized shared support drop beside the residual
    alternative-layer mechanism coefficient, both in candidate-score units.

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
   - `clean_preserved`, independently reconstructed `clean_reset`, and `targeted_strong_reset` after its perturbed state is discarded must have identical transcript/token hashes and cache topology, and must agree within frozen absolute/relative numerical tolerances for answer support/post-answer state and symmetrically for Turn-3 cache state, logits, confidence/correctness margins, residuals, and candidate readouts.
   - This is mandatory in both smoke phases and on the predeclared periodic held-out integrity items. Any failure halts the active phase and sets campaign status to `invalid_reset_parity`.
3. **Cache/state-integrity gate**
   - The perturbation must produce a persistent downstream state difference before Turn 3; all expected Qwen hybrid-state components must exist with valid shapes, positions, layer types, finite values, and true `is_conv_states_initialized` / `is_recurrent_states_initialized` flags for every recurrent state slot.
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
- exact primary-targeted/primary-random and
  alternative-targeted/alternative-random norm matching;
- same-layer random-gradient orthogonality;
- identical answer-support objective and negative-gradient sign at primary and
  alternative layers;
- deterministic global alternative-layer/beta selection;
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
- answer-support gradients at the primary and every tested alternative layer
  are finite and nonzero at every intended position;
- gradient and intervention hooks fire exactly at the declared factual-answer positions and nowhere else;
- targeted support is reduced;
- the alternative targeted direction is the normalized negative
  answer-support gradient at its own layer, while its same-layer random control
  satisfies the frozen orthogonality threshold;
- in post-freeze smoke, the alternative support drop approximately matches targeted strong on smoke items using the discovery-frozen beta;
- primary-targeted, alternative-targeted, and both random-control caches are
  distinct from clean and are storage-disjoint;
- the alternative perturbation remains disabled during Turn 3;
- targeted downstream state differs;
- reset removes the difference;
- clean preserved and clean reset agree within frozen tolerance;
- meta readout is taken from the preserved state;
- Turn-3 suffix construction preserves every frozen factual prefix token and
  passes exact boundary/final transcript hash parity across conditions;
- no factual-process hook fires during either meta branch.

The smoke reports must show the selected/tested intervention layers,
per-position norms for both targeted mechanisms and their own random controls,
same-layer random-gradient cosine, support mismatch when a frozen beta exists,
reset-parity measurements, hybrid-state digests, and factual-prefix, Turn-3
suffix, prefix-suffix boundary, and final concatenated-transcript hashes. A
failed post-freeze support-match gate must first write its phase-local full
trial log, aggregate report, and item-level matching measurements, then halt
without a success marker. A support-match failure in post-freeze smoke, or a
reset-parity, orthogonality, state-preservation, suffix-integrity,
branch-isolation, or hook-lifetime failure in either smoke phase, is critical
and stops the run. The runner must return a nonzero exit status and must not
write a phase-success marker. Discovery may start only after pre-discovery
engineering smoke passes. Held-out may start only after post-freeze critical
smoke passes fail-closed.

## Results interpretation

The reporting layer will distinguish gate-invalid diagnostics from valid-campaign conclusions. If any critical gate fails, the campaign is marked invalid and only the corresponding diagnostic report is emitted. The substantive outcomes below are considered only when the support-match, reset-parity, and cache/state-integrity gates all pass:

The held-out report will present H1–H7 and all targeted/alternative/random/reset
contrasts descriptively. Because this protocol does not freeze a numerical
held-out convergence decision threshold, it will not automatically classify a
candidate as process-sensitive or `M(P)`-like. Such classification requires an
explicitly approved and frozen decision rule; none is introduced here.

1. **No later effect beyond the corresponding same-layer random control:**
   evidence favors ordinary first-order/readout behavior for that mechanism.
2. **A targeted mechanism and its own random control produce similar later
   effects:** evidence favors layer-local generic hidden-state contamination or
   causal persistence.
3. **Targeted and support-matched alternative reduce support similarly but produce different later candidate states:** evidence favors perturbation-specific traces rather than a representation of shared process reliability.
4. **Targeted and support-matched alternative reduce support similarly,
   converge on the same later candidate response, each exceeds its own
   same-layer random response, replicate on held-out items, and disappear under
   reset:** stronger evidence for a process-sensitive representation of answer
   reliability or conflict.
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

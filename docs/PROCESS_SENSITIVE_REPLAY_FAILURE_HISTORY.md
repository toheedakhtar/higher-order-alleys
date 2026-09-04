# Process-Sensitive Replay Failure History

Last updated: 2026-09-04

## Scope and status language

This document records the known failures encountered while developing and
running the `process_sensitive_replay` experiment, from the original `psr`
attempt through `psr-v7`. It also records implementation gaps found during the
pre-`psr-v8` audit. It is intentionally separate from the scientific results:
no campaign listed here reached held-out evaluation.

The labels below have distinct meanings:

- **Environmental/operational failure**: the host, dependency, network, or
  phase lifecycle prevented execution.
- **Implementation failure**: a smoke assertion exposed behavior inconsistent
  with the frozen protocol.
- **Scientific gate failure**: the implementation ran, but a predeclared
  manipulation criterion failed. This is valid negative diagnostic evidence,
  not a software failure.
- **Expected exclusion**: a predefined item-validity rule worked as intended;
  this is not a campaign failure.

## Chronological ledger

| Campaign | Phase | Classification | Observed failure | Resolution and disposition |
|---|---|---|---|---|
| `psr` | `answer_bank` | Environmental | The first model/tokenizer download attempt ended with `The read operation timed out` while accessing Hugging Face without authentication. | Retried after the required files became available. The original phase remained failed diagnostics. |
| `psr` | `answer_bank` retry | Operational | Two immediate retries were rejected with `phase answer_bank already started in this campaign; partial or completed phase outputs cannot be reused`. | This was the append-only phase guard working as designed. A fresh/clean campaign boundary was needed rather than reusing a partial phase. |
| `psr` | `answer_bank` | Protocol/data-path failure | With `max_answer_tokens=48`, all 66 calibration items were valid, but only 1/8 prospective and 0/8 knowledge-boundary items were valid. Fifteen items lacked a valid turn terminator, and one prospective item also had unstable chat reconstruction. Split construction then failed with `cannot select 2 balanced rows from 1`. | Raised the cap to 256; preserved generated answer-content IDs; canonicalized only the invisible assistant terminator to `<|im_end|>`; logged original and canonical terminal IDs; added fail-closed cap exclusion and three-item thinking-mode verification. Started `psr-v2`. |
| `psr-v2` | `pre_discovery_smoke` | Implementation | Failed `invalid_cache_state`: `clean cached support does not match clean gradient-pass support`. | The full-sequence no-cache gradient path selected a different Qwen hybrid execution regime than experimental recurrent replay. It was replaced only for gradient computation with differentiable, token-by-token recurrent replay using the experimental token positions and recurrent kernels. Per-token logits, total support, intervention-layer residuals, gradients, and hook scope became mandatory parity gates. Started `psr-v3`. |
| `psr-v3` | `pre_discovery_smoke` | Implementation | Failed with `Turn-3 rendering changed the canonical post-answer token prefix`. | Replaced full-transcript rerendering with suffix-only Turn-3 construction. The frozen factual prefix and hybrid state are now preserved, and prefix, suffix, boundary, and concatenated-transcript hashes are asserted. Started `psr-v4`. |
| `psr-v4` | — | No failure | `validate`, `answer_bank`, and the complete pre-discovery engineering smoke passed. | This was the first clean engineering-smoke milestone. It is included here to mark where the recurrent-gradient and suffix-construction defects were cleared. |
| `psr-v5` | `discovery` entry | Operational | An attempted discovery invocation was rejected because the phase had already started and partial phase output could not be reused. | Re-established a clean phase boundary before rerunning. The phase guard correctly prevented accidental append/reuse. |
| `psr-v5` | `discovery` | Selection implementation/configuration | After all 16 discovery items completed, selection failed: `alpha_strength_gate_failed: WEAK must be strictly smaller than STRONG`. | Made discovery write its complete alpha diagnostics before selection and began a fresh diagnostic campaign. |
| `psr-v5` | `freeze` | Secondary orchestration failure | A freeze attempt after failed discovery raised a missing `discovery/strength_grid.jsonl` error. | Freeze remained blocked. This was not an independent scientific result; it was a downstream symptom of the failed discovery phase. |
| `psr-v6` | `answer_bank` | Environmental | One invocation failed with `No module named 'jlens'`. | Corrected the CUDA-host environment/import path and reran under the same campaign lifecycle rules; `answer_bank` and pre-discovery smoke subsequently passed. |
| `psr-v6` | `discovery` | Grid/selection failure | The frozen grid jumped from `alpha=0.10` to `0.20`. Median targeted support drop was `1.86398` nat at `0.10` and `5.69379` nat at `0.20`, so no point occupied the strong target range `[2, 4]` nat. Fallback selection chose `0.10`, equal to weak, and the strict-order gate failed. | Added the predeclared intermediate values `0.11`, `0.125`, and `0.15`; retained the full failed grid diagnostics; started `psr-v7`. See [`psr-v6` alpha diagnostics](../assets/psr-v6/discovery/alpha_grid_diagnostics.json) and [`gate_status.json`](../assets/psr-v6/discovery/gate_status.json). |
| `psr-v6` | `freeze` | Secondary orchestration failure | A freeze attempt after failed discovery raised a missing `discovery/beta_grid.jsonl` error. | Freeze remained blocked. The missing file was a consequence of discovery halting before beta selection, not an additional experimental result. |
| `psr-v7` | `answer_bank` entry | Operational | The first invocation found pre-existing rows and failed with `answer_bank.jsonl already contains data; start a new campaign rather than appending`. | The append-safety assertion prevented duplicate/mixed answer-bank rows. The subsequently clean run produced 82 answers, with 16 discovery and 57 held-out-valid items. |
| `psr-v7` | `discovery` | **Scientific gate failure** | Alpha calibration succeeded: weak `0.10`, strong `0.11`, with strong median support drop `2.07785` nat. No same-layer entropy/orthogonal beta passed paired support matching. At `beta=0.20`, the aggregate alternative median (`2.46601` nat) looked close, but median item-paired mismatch was `3.36568` nat versus a `0.51946`-nat tolerance. Other betas missed aggregate and paired criteria; `0.80` also violated the norm-ratio ceiling. | Discovery correctly halted as `invalid_support_match`. `psr-v7` is retained as a valid failed-gate diagnostic campaign and supports no process-property or `M(P)`-like claim. See [`beta_grid_diagnostics.json`](../assets/psr-v7/discovery/beta_grid_diagnostics.json) and [`gate_status.json`](../assets/psr-v7/discovery/gate_status.json). |
| `psr-v7` | `freeze` | Secondary orchestration/diagnostic failure | A freeze command issued after invalid discovery reported missing `discovery_vocab_scores.pt` instead of first reporting the failed prerequisite gate. | Freeze ordering was corrected to validate discovery status before loading discovery products. Future attempts report the original `invalid_support_match` prerequisite directly. No frozen protocol was produced. |
| `psr-v8` | `pre_discovery_smoke` entry | Operational | A repeated invocation was rejected with `phase pre_discovery_smoke already started in this campaign; partial or completed phase outputs cannot be reused`. | The append-only phase guard worked as intended. This message is separate from the subsequent discovery OOM. |
| `psr-v8` | `discovery` | Implementation/resource-lifetime failure | The alpha grid completed, followed by 12/16 beta-grid items. The next item failed while requesting only 20 MiB: the 94.97-GiB GPU had 5.56 MiB free, with 90.92 GiB allocated and 3.36 GiB reserved but unallocated by PyTorch. | Inspection found redundant primary-gradient computation in beta-only passes and a temporary recurrent-cache bound-method cycle capable of retaining autograd state. Disposable replay/cache lifetimes were also implicit. The correction removes the redundant pass, breaks the cycle, explicitly releases all disposable caches, and adds hash-bound per-item memory measurements plus a fail-closed trend gate. `psr-v8` remains failed diagnostics; start fresh as `psr-v9`. |

The append-only error streams preserved locally for the last two campaigns are
[`psr-v6/errors.jsonl`](../assets/psr-v6/errors.jsonl) and
[`psr-v7/errors.jsonl`](../assets/psr-v7/errors.jsonl). Earlier campaign details
above come from the CUDA console transcripts supplied during development; the
corresponding run directories remain failed diagnostics on that host.

## Answer-bank attrition after the 256-token correction

From `psr-v2` onward, answer-bank construction completes with 82 attempted
answers, 16 discovery items, 57 valid held-out items, and 9 excluded items. In
the preserved `psr-v7` bank, three prospective items (`70`, `71`, `73`) and six
knowledge-boundary items (`84`–`89`) still reached the 256-token cap without a
valid turn termination. They were marked invalid and excluded exactly as the
protocol requires.

This attrition is a dataset limitation worth reporting, but it is not a silent
truncation or a failed gate: the runner does not canonicalize those incomplete
answers or admit them to discovery/held-out analysis.

## Implementation audit gaps found before `psr-v8`

The following lower-level gaps were found by code audit before any held-out run.
They did not generate scientific results and were corrected before authorizing
`psr-v8`:

1. Model/tokenizer loading still referred to floating `main`, and phase hashes
   did not bind all resolved revisions, the J-Lens SHA, and runtime package
   versions. The exact Qwen commit and complete environment identity are now
   gate-bound.
2. A failed post-freeze smoke could raise before saving its complete phase-local
   diagnostic report. Diagnostics are now written before fail-closed exit.
3. Hybrid-cache integrity checks did not explicitly require valid
   `is_conv_states_initialized` and `is_recurrent_states_initialized` flags.
   Both flags are now asserted.
4. Exact answer-bank transcript equality was not explicit enough, targeted-reset
   parity was not symmetric with clean-reset, and the protocol-required
   finite-difference gradient-sign check was absent. All three assertions were
   added.
5. Analysis omitted the alternative-minus-random contrast and could label a
   candidate process-sensitive without an explicitly frozen convergence
   decision rule. The statistic was added, and `RESULTS.md` generation is now
   descriptive and conservative.

## Current campaign boundary

`psr-v8` failed during discovery because live CUDA allocations accumulated
across beta-grid items. It produced no frozen protocol. `psr-v9` is the required
fresh campaign and retains the same different-layer, same-objective alternative,
aggregate support-match criteria, held-out 65% item-level criterion, and every
other fail-closed scientific gate. Only execution lifetime and monitoring were
changed; no adaptive beta search was introduced.

The local CPU/unit suite passes, but this does not authorize discovery or
held-out execution. The required order remains:

1. fresh CUDA `validate` and `answer_bank`;
2. pre-discovery engineering smoke;
3. discovery and freeze;
4. post-freeze critical smoke with frozen strengths;
5. held-out only after every critical post-freeze gate passes.

At the time of this record, there is no held-out evidence for a
process-property/process-sensitive representation and no `M(P)`-like or
higher-order representation claim.

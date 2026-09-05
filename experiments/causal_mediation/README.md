# Candidate mediation: precision qualification

**Status: mixed-precision qualification implemented; real CUDA equivalence
validation and approval pending. The eight-item experiment remains unavailable.**
The Blackwell BF16-native smoke reproduced support exactly but failed leakage
with certified infeasibility. That approach is closed under its frozen criteria.

See [the current FP32-tail policy and run command](../../docs/MIXED_PRECISION_MEDIATION_POLICY.md).
`mixed_smoke` exposes only items 0 and 2 and never computes patched judgments.
The remaining BF16 sections below are retained as the historical implementation
record; they do not supersede the mixed-policy qualification requirements.

The intended follow-up uses the same eight exploratory quick-run held-out
items, not an independent replication set or full-profile confirmation. The
candidate is the exact frozen token-75075/layer-42 vector, orientation -1 for
reporting only. No candidate rediscovery or behavioral selection occurs here.

## Proposed precision policy

`precision.py` defines `bf16_compensated_v1_proposed`. Its input is only a BF16
recipient residual, a frozen unit vector, and a scalar coordinate difference.
There is no API for supplying judgments or behavioral effect sizes.

The following separate patch budgets were selected from geometric fidelity
requirements, not fitted to outcomes or enlarged to admit failing probes:

| Quantity | Proposed requirement |
|---|---|
| Nonzero coordinate error | `abs(realized - intended) <= 0.01 * abs(intended)` |
| Orthogonal leakage L2 | `leakage <= 0.10 * abs(intended)` |
| Random-control realized norm mismatch | At most 1% of the realized candidate-patch norm |
| Random-control projection onto candidate | At most 1% of the realized candidate-patch norm |
| Zero-coordinate SHAM | Bitwise unchanged BF16 residual |
| Existing sham/logit/cache/engineering parity | Original `atol=rtol=1e-5`, unchanged |

There is no absolute floor that would silently accept complete erasure of a
small nonzero patch. Ten-percent orthogonal amplitude bounds off-axis energy
to 1% of the squared intended coordinate change. This is a proposed scientific
fidelity criterion, not a statement that 10% contamination is always harmless.
Every failed item remains visible; no patch is selected by its behavioral effect.

## Exact deterministic algorithm

1. Verify finite BF16 input and the frozen vector's unit norm. Preserve the
   vector's exact saved float32 values; do not renormalize or replace it.
2. Copy numerical inputs to CPU float64 for geometry calculations. This does
   not change model compute, weights, attention, or cache precision.
3. A zero target immediately returns a bitwise copy. For a negative target,
   reflect the recipient and target signs, solve the positive case, and reflect
   back. This makes neighbor tie-breaking independent of patch polarity.
4. For target `delta` and vector `w`, the ideal point is
   `h + delta*w/dot(w,w)`. The denominator accounts for the saved unit vector's
   tiny finite-precision norm error without modifying the vector.
5. Evaluate nearest BF16 rounding by comparing the rounded value and its two
   adjacent BF16 neighbors in float64, independently in every dimension.
6. Bracket the target projection using `q(lambda)=BF16(h+lambda*w)`, doubling
   the upper bound at most 64 times. Use 80 deterministic bisection steps.
   This is the discrete minimum-norm Lagrangian proposal; its projection is a
   monotone step function.
7. Start compensation from three proposals: nearest ideal rounding and the
   lower/upper bracketing proposals. At each of at most 512 correction steps,
   consider both adjacent BF16 values in every dimension. Require strictly
   improved coordinate error. If a move reaches the error budget, choose the
   smallest increase in squared norm. Otherwise prefer non-overshooting moves
   with the smallest squared-norm cost per unit projection, permitting an
   improving overshoot only when no non-overshooting move exists.
8. Perform up to 512 neighbor-refinement steps that decrease squared
   orthogonal leakage while preserving the coordinate-error budget. Ties use
   negative-neighbor-first, then ascending coordinate index. Decreases must
   exceed `1e-30` to avoid numerical zero-cost cycles.
9. Among coordinate-feasible proposals select minimum orthogonal leakage, then
   coordinate error, then total norm. If none is feasible, retain the proposal
   with minimum coordinate error for diagnostics only. This is a bounded local
   search, **not a claim of a globally optimal compensated patch**.
10. Return the final BF16-representable residual and complete diagnostics.
    Apply neither a failed candidate proposal nor a failed random proposal.

No post-realization scaling or float32 model tail is used.

## Independent impossibility certificate

Let `d=q-h`, `e=dot(w,d)-delta`, and let `d_perp` be the component of `d`
orthogonal to `w`. For every representable residual `q`:

```
||q - (h + delta*w/||w||²)||² = e²/||w||² + ||d_perp||²
```

Coordinatewise nearest BF16 rounding supplies the global minimum of the left
side. If it exceeds the combined squared coordinate/leakage budgets, **no BF16
residual can satisfy both requirements**, regardless of the compensation
algorithm. A small float64 safety margin prevents spurious certificates from
numerical roundoff. Failure without a certificate is labeled solver failure,
not mathematical impossibility.

## Matched random controls

Seeds are SHA-256-derived from `[42, item_id, branch, donor_condition]`.
A deterministic float64 Gaussian vector is projected orthogonally to the exact
candidate and normalized. Both polarities use the **same realization algorithm**
with coordinate target `+/- realized_candidate_patch_L2`. They must also pass
the separate realized-norm and candidate-projection checks above. The full
float64 random vector hash and seed are logged. Geometry checks use the actual
BF16 changes, not the intended unquantized vectors. Future behavioral analysis
must retain both polarities and use their mean as specified in the protocol.

## Measurements and J-Lens relationship

Each proposal records intended and realized coordinate changes, absolute and
relative coordinate error, total patch norm, norm error, orthogonal leakage,
leakage/intended-coordinate ratio, dtype, policy hash, solver status, and the
independent lattice bound. Random rows additionally report realized projection
onto the original candidate and realized norm mismatch.

The installed J-Lens `transport` computes `J @ h`. Its `unembed` then casts to
the LM-head dtype and applies the final normalization before the LM head
(and a model-specific softcap if present). Thus the saved effective direction
`normalize(J.T @ W[token])` is not generally the gradient of the normalized
display score, and its raw projection is not the displayed score. The
intervention remains residual-coordinate restoration.

Real numerical smoke records both raw projection and J-Lens score before/after.
Synthetic probes explicitly mark real-model J-Lens scores unavailable. A
synthetic-weight Qwen/J-Lens integration test checks the normalization
relationship using the installed implementations.

## Run synthetic qualification

From the repository root, using the existing environment:

```powershell
.venv/Scripts/python.exe -m unittest experiments.causal_mediation.test_precision experiments.causal_mediation.test_precision_forward -v
.venv/Scripts/python.exe -m experiments.causal_mediation.precision_probe --output assets/precision-synthetic-new.json
```

The probe uses the exact frozen vector with a fixed synthetic residual-scale
grid and patch targets. These scales are not estimates of real smoke residuals.
All cases and random controls are reported, including failures. Output is
exclusive-create; use a fresh filename for each diagnostic run.

## Run the two numerical smoke items on the original CUDA host

```bash
python -m experiments.causal_mediation.precision_smoke \
  --run-dir assets/psr-mediation-precision-v1 \
  --upstream assets/psr-quick-v3 \
  --direction assets/psr_quick_big_files/directions/layer42_token75075_8515912d78e8.pt
```

The smoke IDs are fixed to `0, 2`, the first two items in the frozen held-out
order. They are distinct from the upstream discovery smoke IDs `1, 14`.
No alternate item list, backend, compute dtype, or tolerance override is offered.

The smoke runner:

- Verifies the original frozen protocol, candidate file/tensor, answer bank,
  split, pinned model/tokenizer/lens identities, runtime versions, upstream
  replay source, and original support-reference hashes. Windows CRLF-to-LF
  transfer recovery is explicit, hash-checked, and read-only for text artifacts;
  binary hashes must match byte-for-byte.
- Computes clean/primary/alternative answer states once per item, using the
  original recurrent gradient and replay implementations. Requires both smoke
  support drops to reproduce their machine-readable held-out values at `1e-5`.
- Uses storage-disjoint suffix-only branches. Saves only numerical residuals
  and geometry. It does not generate labels or score confidence/correctness
  sequences. Boundary logits are compared only for sham engineering parity.
- Tests restoration and reverse transplant geometry, both random polarities,
  and true sham. Only geometrically qualified nonzero proposals undergo a causal
  forward check, with their output logits discarded without effect analysis.
- Returns BF16 at layer 42, runs later layers/suffix tokens from that write,
  checks one hook invocation, rejects float32 writes, checks cache integrity,
  and releases caches and temporary states explicitly.
- Runs one process/worker, logs per-item CUDA allocated/reserved/peak memory,
  and uses the original systematic-growth thresholds with a pre-item baseline.
- Writes diagnostics and a nonzero exit on engineering or precision failure.
  Even successful numerical smoke is labeled **pending review** and creates
  no mediation success marker or eight-item execution path.

## Required next decision

Inspect both real numerical smoke items before approving the proposed precision
criterion. If their natural candidate changes cannot meet the coordinate and
leakage budgets, stop. Do not loosen global parity, omit items, amplify targets,
or switch to a mixed-precision tail automatically. The complete causal runner
remains pending this numerical qualification and review.

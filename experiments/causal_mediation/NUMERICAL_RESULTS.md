# BF16 compensation: numerical qualification status

Date: 2026-09-05. **Synthetic qualification completed; real two-item smoke not
executed. No eight-item mediation experiment was run.**

The exact frozen vector's file and tensor hashes were verified. Model weights,
downstream compute dtype, attention backend, cache dtype, and the upstream
`atol=rtol=1e-5` parity settings were not changed. There is no float32 tail.

## Algorithm and proposed criteria

The deterministic algorithm is specified step-by-step in [README.md](README.md)
and implemented in [precision.py](precision.py): BF16 Lagrangian bracketing,
discrete-neighbor projection compensation, then leakage minimization within the
coordinate budget. All geometry runs on CPU float64; proposed model residuals
are BF16. Failed proposals are never applied to the model.

Proposed nonzero-patch limits are **1% relative coordinate error** and **10%
orthogonal-leakage/intended-coordinate ratio**. Random controls use the same
algorithm, target the realized candidate-patch norm, and additionally require
at most 1% realized norm mismatch and 1% realized projection leakage onto the
candidate. These separate limits do not relax global parity. SHAM is bitwise
unchanged.

No confidence/correctness outcomes or mediation effect sizes were supplied to
the solver, used to select its algorithm, or used to choose these limits.

## Synthetic results with the exact frozen vector

The fixed matrix contains 12 residual-scale/coordinate-target combinations,
one zero-residual case, and one sham. Its residual scale is the multiplier on a
seeded Gaussian BF16 vector, **not a measured model-residual scale**.

| Synthetic residual scale | Intended change | Relative coordinate error | Orthogonal leakage / intended change | Qualification |
|---|---:|---:|---:|---|
| 0.01 | 0.001 | 1.000% | 73.410% | Certified infeasible |
| 0.01 | 0.01 | 1.000% | 11.965% | Certified infeasible |
| 0.01 | 0.1 | 0.214% | 1.179% | Passed |
| 0.01 | 1.0 | 0.0484% | 0.201% | Passed |
| 1.0 | 0.001 | 1.000% | 364.311% | Certified infeasible |
| 1.0 | 0.01 | 1.000% | 185.226% | Certified infeasible |
| 1.0 | 0.1 | 1.000% | 73.883% | Certified infeasible |
| 1.0 | 1.0 | 1.000% | 11.886% | Certified infeasible |
| 100.0 | 0.001 | 0.977% | 1075.896% | Certified infeasible |
| 100.0 | 0.01 | 0.999% | 606.396% | Certified infeasible |
| 100.0 | 0.1 | 1.000% | 365.254% | Certified infeasible |
| 100.0 | 1.0 | 1.000% | 187.842% | Certified infeasible |
| Zero residual | 0.01 | 0.0619% | 0.163% | Passed |
| SHAM | 0 | Exactly zero | Exactly zero | Passed |

Values shown as 1.000% are rounded; all achieved coordinate errors were within
the proposed 1% limit. **4/14 candidate proposals passed both limits, including
sham; 10/14 failed leakage qualification.** Eight of the 28 random proposals
passed all their numerical criteria. Every candidate and both random polarities
remain in the [machine-readable report](diagnostics/synthetic_precision_final.json),
which includes absolute errors, realized changes, norms, and lower bounds.

For example, at residual scale 1 and target 0.001, naive rounding loses 99.19%
of the target coordinate. Compensation reduces that error below 1%, but its
off-axis movement is approximately 3.64 times the intended change. The
independent lattice certificate proves that **even an optimal BF16 procedure**
would require at least approximately 99.62% orthogonal leakage at the 1%
coordinate-error limit in this example. A better optimizer therefore cannot
make this geometry satisfy the proposed 10% leakage budget.

This does not prove failure on the actual natural smoke differences. Those
recipient/donor residuals have not been obtained on the CUDA host.

## Engineering validation

Eleven focused CPU tests passed, covering exact frozen identity, deterministic
compensation, sign symmetry, bitwise sham, norm-matched random realization,
lattice-bound checks, rejected contaminated proposals, and invalid inputs.
The integration tests use a tiny 64-block Qwen with synthetic BF16 weights and
the installed Qwen/J-Lens code. They verify one causal layer-42 hook, downstream
cache recomputation, unchanged source/cache dtypes, hook removal on failure,
and sham equivalence at the original tolerance. They also demonstrate that the
normalized J-Lens score differs from the raw residual projection.

The synthetic report marks real-model J-Lens scores unavailable; they are not
invented. The numerical CUDA smoke runner will report both before/after values.

## Pending real smoke and decision

The workspace reports `torch=2.13.0+cpu`, CUDA unavailable, and zero CUDA
devices. The local smoke invocation correctly failed before model loading;
see [the guard diagnostic](diagnostics/cpu_guard_check/gate_status.json).

The smoke runner is restricted to held-out IDs **0 and 2**. It requires original
support-drop reproduction and preserves the original replay and numerical
execution identity. It does not generate labels or score their sequence
probabilities. Its nonzero-patch output logits are discarded, so mediation
effects cannot influence qualification.

**Do not run all eight items or implement a float32 fallback on the basis of
these synthetic results.** Real numerical smoke and review are still required.
If its natural patches fail, stop and report the measurements and certificates.
The full mediation implementation remains pending that decision.

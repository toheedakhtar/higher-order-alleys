# Provenance experiment report

Run: `assets\provenance-v1\20261016T201632_holdout`
Matched holdout items: 34
Run gates passed: True

## Rungs

- **R1_content_exclusion**: FAIL; `ci={"mean": 0.159737, "lo": -0.213093, "hi": 0.509747, "positive_fraction": 0.8178, "n": 34}`
- **R2_damage_exclusion**: PASS; `ci={"mean": 0.759747, "lo": 0.363767, "hi": 1.222712, "positive_fraction": 1.0, "n": 34}`
- **R3_causal_mediation**: FAIL; `restore_ci={"mean": -0.000464, "lo": -0.001599, "hi": 0.000724, "positive_fraction": 0.2212, "n": 34}`; `contrast_ci={"mean": -0.001142, "lo": -0.002714, "hi": 0.00018, "positive_fraction": 0.047, "n": 34}`

Convergence diagnostic (P1 vs P2, non-blocking): compatible with convergence; `ci={"mean": 0.224391, "lo": -0.14672, "hi": 0.591932, "positive_fraction": 0.8809, "n": 34}`

## Claim

R1/R2/R3 chain not earned on this run

R3 depends on patch-fidelity and sham-parity gates recorded in `status.json`.

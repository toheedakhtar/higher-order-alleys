# Research Sprint Executive Brief
## First-Order Readout vs Higher-Order / Process-Sensitive Representation in Qwen3.6-27B

### Question

Does an LLM merely read out uncertainty/evaluation information from its ordinary task computation, or does it form a later representation of properties of its own preceding process?

Operationally:

```text
ordinary readout: P -> r(P)

higher-order candidate: P -> M(P)
```

The sprint sought increasingly strong evidence for the second interpretation.

### Experiment 1 — evaluator steering

J-Lens identified two layer-40 “evaluation” directions:

```text
97817 = 评价
99973 = 评估
```

They were causally powerful. In the main global run, negative steering flipped 27/82 judgments.

But 25/27 of those flips worsened the judgment, outputs sometimes became malformed, and effects were strongly label-dependent.

**Update:** causal evaluator/output-control feature, not higher-order evidence.

### Experiment 2 — SELF vs OTHER

The exact same factual answer was evaluated under:

```text
your answer
their answer
```

Neither evaluator direction showed a stable general SELF-specific causal effect.

Both were active in SELF and OTHER and were strongly affected by prompt/label family.

**Update:** the simple self-evaluation interpretation was falsified.

### Experiment 3 — Process-Sensitive Replay

The exact visible answer was held fixed while the hidden answer process was perturbed.

Quick exploratory profile:

```text
primary mechanism: layer 31
alternative mechanism: layer 23
same answer-support objective
same exact answer text
same-layer random controls
reset control
confidence + correctness branches
```

A frozen layer-42 candidate was discovered:

```text
token 75075
orientation -1
```

On 8 held-out items:

- support matching passed 8/8;
- primary and alternative structured perturbations changed the candidate;
- each beat its own random control;
- reset returned the candidate to clean;
- candidate activity tracked support damage;
- confidence decreased as support damage increased.

**Update:** exploratory evidence for:

```text
P -> process-sensitive later representation/readout
```

under identical visible outputs.

Mechanism convergence was compatible with a shared process-property representation, but not proven equivalent.

### Final mediation attempt

The missing test was:

```text
candidate -> judgment
```

A natural-size layer-42 candidate-coordinate restoration experiment was designed.

It could not be executed faithfully.

**BF16-native patching:**  
Coordinate accuracy was achievable, but unavoidable BF16 quantization produced ~90–249% orthogonal leakage; certified lower bounds showed ~79–99% leakage was unavoidable.

**FP32-tail fallback:**  
Patch geometry became excellent, but the precision switch itself changed judgment margins/log-probabilities by up to ~0.204 nats and process contrasts by up to ~0.195 nats.

No valid nonzero candidate mediation patch was run.

### Final result

Supported exploratorily:

```text
P -> R(P)
```

where `R(P)` is a later process-sensitive representation/readout.

Not established:

```text
P -> M(P) -> judgment
```

The mediation arrow is **unresolved because of intervention precision / identifiability constraints**, not falsified.

### Strongest defensible statement

> Qwen3.6-27B contains a later internal signal that tracks controlled degradation of its preceding answer process despite identical visible output, survives matched random and reset controls, generalizes across two support-matched structured perturbations, and covaries with subsequent confidence. This is stronger than a simple output-text evaluator, but it does not establish a higher-order representation because candidate-level causal mediation remains unidentified.

### Research arc

```text
evaluation-looking probe
    ->
causal evaluator
    ->
not SELF-specific
    ->
same-output hidden-process sensitivity
    ->
random/reset controlled process-sensitive candidate
    ->
causal mediation attempted
    ->
precision audit blocks invalid intervention
```

The sprint's main value is therefore not a sensational positive claim, but a progressively sharpened and falsification-driven mechanistic result.

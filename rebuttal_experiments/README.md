# Rebuttal experiments

This directory contains isolated evaluation harnesses added for the rebuttal.
They read the original codebase, checkpoints, and caches but never modify
original result artifacts.

## Experiment 1: same-model soft-gate ablation

`exp1_gate_ablation.py` loads the canonical learned RELLIS Stage-2 checkpoint
and rolls out `CoefEnergyNetMaterial` through
`integrate_surrogate_material`. It does not use the legacy heuristic
`route_aware_stage2` dynamic controller.

The paired arms differ only in the multiplier applied to the model's learned
soft-risk coefficient:

- `gate_on`: `lam_soft_used = feasibility_gate * lam_soft_learned`
- `gate_off`: `lam_soft_used = lam_soft_learned`

The learned `lam_hard` is active and identical in both arms. The local gate
tests uniformly sampled straight-ray primitives for hard clearance,
goal progress, and risk improvement over the nominal local scaffold ray.

CPU smoke test:

```bash
python rebuttal_experiments/exp1_gate_ablation.py \
  --max-episodes 2 \
  --event-type delayed_required_escape \
  --dt 0.01 \
  --device cpu \
  --out rebuttal_experiments/results/exp1_gate_ablation_smoke
```

Outputs include immutable-input hashes and configuration (`config.json`),
paired per-episode results, paired differences, full step traces, event specs,
and an aggregate summary with intervention invariants. The `0.01` controller
timestep ensures these smoke episodes span event onset and corridor opening;
with `0.04`, the continuous learned rollout reaches the goal before event
onset. These smoke-test values are engineering validation, not paper-ready
statistical estimates.

## Experiment 8: fixed semantic APF comparison

The dynamic evaluator's former `neural_potential_field` method is now
accurately named `semantic_apf`. It is a fixed, discrete semantic artificial
potential field: there is no network, training, or learned parameter. The old
name remains an input-only compatibility alias; new artifacts always use the
canonical name.

Regression checks:

```bash
python -m pytest -q rebuttal_experiments/test_exp8_semantic_apf.py
```

The paper-scale delayed-required-escape run and its deterministic summary are
in `results/exp8_semantic_apf_delayed/`. See that directory's `REPORT.md` and
`provenance.json` for exact commands, 100-paired-episode bootstrap intervals,
input hashes, the preliminary historical eight-event comparison, and
limitations.

## Experiment 2: gate witness versus executed trajectory

Experiment 2 postprocesses the instrumented gate-on traces:

```bash
python rebuttal_experiments/exp2_gate_trajectory_agreement.py \
  --exp1-results rebuttal_experiments/results/exp1_gate_ablation_100 \
  --out rebuttal_experiments/results/exp2_gate_trajectory_agreement
```

For each gate-positive decision it compares the selected primitive with the
subsequent learned-field path at equal arc length. The primitive is an
activation witness, not a commanded trajectory. Reported metrics include
directional cosine, endpoint and cross-track deviations, clearance/hard-contact
agreement, and predicted versus realized risk improvement, with regime and
pre/post-opening breakdowns.

## Experiment 3: activation frequency and flicker

```bash
python rebuttal_experiments/exp3_gate_activation_analysis.py \
  --exp1-results rebuttal_experiments/results/exp1_gate_ablation_100 \
  --out rebuttal_experiments/results/exp3_gate_activation
```

This reports activation rates by regime and dynamic-event phase, episode-level
false pre-activation and post-opening non-activation, contiguous activation
runs, transition and one-step flicker counts, and learned coefficient
distributions conditioned on the gate state. The checkpoint-driven harness has
no cooldown, latch, or hysteresis.

## Experiment 4: primitive-count sensitivity

```bash
python rebuttal_experiments/exp4_k_sensitivity.py \
  --out rebuttal_experiments/results/exp4_k_sensitivity
```

This runs `K ∈ {4, 8, 16, 32}` on the same 100 episodes using only the
gate-on arm, verifies that gate-off dynamics are exactly K-invariant on a
preserved smoke pair, computes gate–trajectory agreement for each K, and
reports episode-bootstrap intervals and paired differences against `K=16`.

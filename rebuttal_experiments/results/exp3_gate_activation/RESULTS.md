# Experiment 3: Gate activation analysis

This analysis uses the `gate_on` learned-controller trajectories. The gate is recomputed independently at every step; this harness contains **no cooldown, latch, or hysteresis state**.

The blocked pre-opening phase is `event_step ≤ step < opening_step`. Activation there is counted as false pre-activation. Ordinary behavior before event onset is reported separately.

## Headline episode rates

- Episodes with any activation: 76/100 (0.760)
- False-preactivation episodes: 17/100 (0.170)
- Post-opening no-activation episodes: 27/100 (0.270)
- Activation runs: 145 (mean/median/max length 3.393/2.000/17 steps)
- Isolated one-step runs: 49/145 (0.338)
- Gate transitions: 283 (138 off→on, 145 on→off)

## Frequency and temporal stability

| Group | Episodes | Steps | Active | Activation rate | Episodes active | Runs | Transitions | Trans./100 steps | One-step runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 100 | 6334 | 492 | 0.078 | 76 | 145 | 283 | 4.777 | 49 |
| regime=R1 | 55 | 3433 | 304 | 0.089 | 42 | 88 | 171 | 5.336 | 30 |
| regime=R2 | 24 | 1463 | 94 | 0.064 | 18 | 28 | 54 | 3.733 | 9 |
| regime=R3 | 21 | 1438 | 94 | 0.065 | 16 | 29 | 58 | 4.508 | 10 |
| phase=pre_event | 100 | 2854 | 103 | 0.036 | 23 | 31 | 50 | 1.860 | 14 |
| phase=blocked_pre_opening | 100 | 1000 | 57 | 0.057 | 17 | 18 | 24 | 2.667 | 4 |
| phase=post_opening | 100 | 2480 | 332 | 0.134 | 73 | 99 | 197 | 9.090 | 34 |
| regime=R1,phase=pre_event | 55 | 1542 | 79 | 0.051 | 15 | 22 | 35 | 2.375 | 9 |
| regime=R1,phase=blocked_pre_opening | 55 | 550 | 36 | 0.065 | 11 | 12 | 17 | 3.434 | 4 |
| regime=R1,phase=post_opening | 55 | 1341 | 189 | 0.141 | 41 | 57 | 114 | 9.972 | 20 |
| regime=R2,phase=pre_event | 24 | 659 | 9 | 0.014 | 3 | 3 | 3 | 0.471 | 2 |
| regime=R2,phase=blocked_pre_opening | 24 | 240 | 8 | 0.033 | 3 | 3 | 4 | 1.852 | 0 |
| regime=R2,phase=post_opening | 24 | 564 | 77 | 0.137 | 16 | 22 | 43 | 7.881 | 7 |
| regime=R3,phase=pre_event | 21 | 653 | 15 | 0.023 | 5 | 6 | 12 | 2.096 | 3 |
| regime=R3,phase=blocked_pre_opening | 21 | 210 | 13 | 0.062 | 3 | 3 | 3 | 1.587 | 0 |
| regime=R3,phase=post_opening | 21 | 575 | 66 | 0.115 | 16 | 20 | 40 | 8.160 | 7 |

## Learned coefficient distributions by gate state

The gate changes only the multiplier applied to learned `lam_soft`; `lam_hard` remains active in both states.

| Gate state | Coefficient | n | Mean | Std | P10 | Median | P90 |
|---|---|---:|---:|---:|---:|---:|---:|
| active | lam_soft | 492 | 0.07445 | 0.02434 | 0.04329 | 0.06972 | 0.10800 |
| active | lam_hard | 492 | 0.04745 | 0.02383 | 0.01921 | 0.04081 | 0.08037 |
| inactive | lam_soft | 5842 | 0.06029 | 0.02802 | 0.02571 | 0.05642 | 0.10135 |
| inactive | lam_hard | 5842 | 0.03528 | 0.02531 | 0.00824 | 0.02878 | 0.07315 |

## Validation

```json
{
  "synthetic_run_segmentation_test": true,
  "input_contains_both_exp1_arms": true,
  "analysis_uses_gate_on_only": true,
  "binary_gate_values": true,
  "unique_contiguous_steps_per_episode": true,
  "phase_partition_matches_total_steps": true,
  "run_lengths_sum_to_active_steps": true,
  "episode_count_matches_exp1_gate_on": true,
  "all_coefficients_finite": true,
  "cooldown_or_latch_present": false
}
```

## Provenance

```json
{
  "input_results": "rebuttal_experiments/results/exp1_gate_ablation_100",
  "step_traces_sha256": "df3bf06dfb7bfed13b2e8dd5c4d921c7cc69b0efa23c836c4dea06d2ab8c883d",
  "per_episode_metrics_sha256": "8ca107e5ff7d08e53f9139641eec2925f1b3d2ebd61eca49c8eb801d49c7d05a",
  "config_sha256": "8c0013dd97382b5fd5c64bf33c4b3d85e1579670596da27ca7694bd0773e647f",
  "analyzed_arm": "gate_on",
  "gate_semantics": "binary decision recomputed independently every step",
  "cooldown_latch_hysteresis": "none in checkpoint-driven harness",
  "phase_definitions": {
    "pre_event": "step < event_step",
    "blocked_pre_opening": "event_step <= step < opening_step",
    "post_opening": "step >= opening_step"
  },
  "isolated_flicker_definition": "active run of exactly one controller step"
}
```

# Experiment 2: Gate witness–trajectory agreement

The sampled primitive is an **activation witness**, not a trajectory command. The learned Hamiltonian field still determines the executed motion. This audit measures whether those two objects agree over an arc-length-matched short horizon.

Only gate-positive decisions from the learned `gate_on` rollouts are included. Aggregate metrics use decisions with at least 95% of the 12-cell horizon observed.

| Group | Gate + | Complete | Cosine ↑ | Endpoint dev. m ↓ | Cross-track m ↓ | Clearance agree ↑ | Hard disagreement ↓ | Pred. Δrisk | Realized Δrisk | Sign agree ↑ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 157 | 145 | 0.726 | 4.031 | 1.836 | 0.559 | 0.193 | 0.1431 | 0.0350 | 0.538 | 0.439 |
| regime=R1 | 100 | 92 | 0.658 | 4.607 | 2.082 | 0.641 | 0.141 | 0.1327 | 0.0095 | 0.457 | 0.137 |
| regime=R2 | 32 | 29 | 0.910 | 2.257 | 0.972 | 0.414 | 0.172 | 0.1693 | 0.1619 | 0.931 | 0.908 |
| regime=R3 | 25 | 24 | 0.760 | 3.968 | 1.939 | 0.417 | 0.417 | 0.1512 | -0.0207 | 0.375 | 0.426 |
| phase=before_event | 20 | 20 | 0.884 | 2.649 | 1.105 | 0.000 | 0.550 | 0.1291 | 0.0931 | 0.850 | 0.707 |
| phase=pre_opening | 11 | 11 | 0.999 | 0.241 | 0.114 | 0.091 | 0.818 | 0.1084 | -0.1482 | 0.273 | 0.714 |
| phase=post_opening | 126 | 114 | 0.672 | 4.640 | 2.131 | 0.702 | 0.070 | 0.1489 | 0.0425 | 0.509 | 0.418 |
| regime=R1,phase=before_event | 12 | 12 | 0.916 | 2.188 | 0.892 | 0.000 | 0.250 | 0.1018 | 0.0748 | 0.750 | 0.760 |
| regime=R1,phase=pre_opening | 9 | 9 | 0.999 | 0.252 | 0.109 | 0.111 | 0.778 | 0.1197 | -0.0952 | 0.333 | 0.627 |
| regime=R1,phase=post_opening | 79 | 71 | 0.572 | 5.568 | 2.533 | 0.817 | 0.042 | 0.1395 | 0.0117 | 0.423 | 0.023 |
| regime=R2,phase=before_event | 5 | 5 | 0.929 | 2.327 | 0.967 | 0.000 | 1.000 | 0.1548 | 0.1293 | 1.000 | 0.995 |
| regime=R2,phase=pre_opening | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| regime=R2,phase=post_opening | 27 | 24 | 0.906 | 2.242 | 0.973 | 0.500 | 0.000 | 0.1723 | 0.1687 | 0.917 | 0.910 |
| regime=R3,phase=before_event | 3 | 3 | 0.679 | 5.029 | 2.189 | 0.000 | 1.000 | 0.1952 | 0.1058 | 1.000 | 0.850 |
| regime=R3,phase=pre_opening | 2 | 2 | 1.000 | 0.189 | 0.137 | 0.000 | 1.000 | 0.0578 | -0.3865 | 0.000 | -1.000 |
| regime=R3,phase=post_opening | 20 | 19 | 0.748 | 4.198 | 2.089 | 0.526 | 0.263 | 0.1541 | -0.0022 | 0.316 | 0.077 |
| phase=pre_opening,crosses_opening=0 | 2 | 2 | 0.998 | 0.436 | 0.231 | 0.500 | 0.000 | 0.1611 | 0.1694 | 1.000 | 1.000 |
| phase=pre_opening,crosses_opening=1 | 9 | 9 | 1.000 | 0.198 | 0.087 | 0.000 | 1.000 | 0.0967 | -0.2188 | 0.111 | 0.720 |

## Validation

```json
{
  "all_input_rows_gate_on": true,
  "all_decisions_gate_positive": true,
  "selected_primitives_meet_clearance_threshold": true,
  "predicted_improvements_meet_gate_margin": true,
  "complete_alignment_in_unit_interval": true,
  "nonnegative_deviations": true,
  "num_gate_positive_decisions": 157,
  "num_complete_horizon_decisions": 145,
  "num_episodes_with_gate_positive": 39
}
```

## Provenance

```json
{
  "input_results": "rebuttal_experiments/results/exp4_k_sensitivity/raw/k4",
  "step_traces_sha256": "7c4b43795c46cf1253f34954a36cb3cb3268b2b8f892862615d06f12ae677332",
  "per_episode_metrics_sha256": "a365fac99d9d46709288acfe8ddeb887cfebeb25c5e63fee6e614c98b0149b60",
  "config_sha256": "45220edd90d5f76b83295edcf35eb239bf640631f1a4420e2aef0d68dc246af1",
  "bev_manifest_sha256": "744e0c92c290b8235d7395226e4e7a15c6c1ecb5be42aa1ab588ce34d2975ccf",
  "grid_resolution_m_per_cell": 0.5,
  "primitive_horizon_cells": 12,
  "actual_horizon_definition": "subsequent learned-field path through the first observed controller endpoint reaching/exceeding equal grid-cell arc length",
  "complete_horizon_threshold": 0.95,
  "actual_clearance_sampling": "minimum dynamic-map SDF sampled at executed controller endpoints",
  "phase_definition": {
    "before_event": "decision_step < event_step",
    "pre_opening": "event_step <= decision_step < opening_step",
    "post_opening": "decision_step >= opening_step"
  },
  "primitive_semantics": "activation witness, not trajectory command"
}
```

Interpretation caution: agreement indicates that a local witness and the resulting field motion are directionally compatible. It does not turn the witness into a safety certificate or prove that the integrator executes the sampled primitive.

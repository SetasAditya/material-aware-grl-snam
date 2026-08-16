# Experiment 2: Gate witness–trajectory agreement

The sampled primitive is an **activation witness**, not a trajectory command. The learned Hamiltonian field still determines the executed motion. This audit measures whether those two objects agree over an arc-length-matched short horizon.

Only gate-positive decisions from the learned `gate_on` rollouts are included. Aggregate metrics use decisions with at least 95% of the 12-cell horizon observed.

| Group | Gate + | Complete | Cosine ↑ | Endpoint dev. m ↓ | Cross-track m ↓ | Clearance agree ↑ | Hard disagreement ↓ | Pred. Δrisk | Realized Δrisk | Sign agree ↑ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 572 | 516 | 0.689 | 4.549 | 2.061 | 0.548 | 0.275 | 0.1347 | 0.0055 | 0.521 | 0.355 |
| regime=R1 | 361 | 328 | 0.681 | 4.585 | 2.066 | 0.558 | 0.256 | 0.1243 | 0.0021 | 0.503 | 0.229 |
| regime=R2 | 104 | 93 | 0.687 | 4.623 | 2.096 | 0.602 | 0.237 | 0.1611 | 0.0470 | 0.591 | 0.652 |
| regime=R3 | 107 | 95 | 0.717 | 4.351 | 2.011 | 0.463 | 0.379 | 0.1445 | -0.0237 | 0.516 | 0.263 |
| phase=before_event | 129 | 129 | 0.692 | 4.590 | 1.964 | 0.318 | 0.357 | 0.0938 | 0.0267 | 0.519 | 0.723 |
| phase=pre_opening | 71 | 71 | 0.714 | 4.236 | 1.934 | 0.070 | 0.887 | 0.1247 | -0.1951 | 0.056 | -0.137 |
| phase=post_opening | 372 | 316 | 0.682 | 4.602 | 2.130 | 0.750 | 0.104 | 0.1536 | 0.0418 | 0.627 | 0.494 |
| regime=R1,phase=before_event | 101 | 101 | 0.701 | 4.473 | 1.904 | 0.327 | 0.317 | 0.0869 | 0.0195 | 0.475 | 0.586 |
| regime=R1,phase=pre_opening | 45 | 45 | 0.706 | 4.252 | 1.939 | 0.111 | 0.844 | 0.1244 | -0.1594 | 0.089 | 0.018 |
| regime=R1,phase=post_opening | 215 | 182 | 0.664 | 4.729 | 2.187 | 0.797 | 0.077 | 0.1451 | 0.0325 | 0.621 | 0.313 |
| regime=R2,phase=before_event | 10 | 10 | 0.743 | 4.195 | 1.721 | 0.000 | 1.000 | 0.1679 | 0.1160 | 0.900 | 0.947 |
| regime=R2,phase=pre_opening | 10 | 10 | 0.705 | 4.644 | 2.140 | 0.000 | 0.900 | 0.1187 | -0.2088 | 0.000 | -0.637 |
| regime=R2,phase=post_opening | 84 | 73 | 0.677 | 4.679 | 2.142 | 0.767 | 0.041 | 0.1660 | 0.0725 | 0.630 | 0.833 |
| regime=R3,phase=before_event | 18 | 18 | 0.610 | 5.469 | 2.432 | 0.444 | 0.222 | 0.0913 | 0.0176 | 0.556 | 0.926 |
| regime=R3,phase=pre_opening | 16 | 16 | 0.744 | 3.938 | 1.793 | 0.000 | 1.000 | 0.1293 | -0.2867 | 0.000 | -0.581 |
| regime=R3,phase=post_opening | 73 | 61 | 0.742 | 4.130 | 1.944 | 0.590 | 0.262 | 0.1641 | 0.0330 | 0.639 | 0.346 |
| phase=pre_opening,crosses_opening=0 | 9 | 9 | 0.718 | 4.044 | 1.853 | 0.556 | 0.111 | 0.1232 | 0.0206 | 0.222 | 0.651 |
| phase=pre_opening,crosses_opening=1 | 62 | 62 | 0.714 | 4.264 | 1.946 | 0.000 | 1.000 | 0.1249 | -0.2264 | 0.032 | -0.251 |

## Validation

```json
{
  "all_input_rows_gate_on": true,
  "all_decisions_gate_positive": true,
  "selected_primitives_meet_clearance_threshold": true,
  "predicted_improvements_meet_gate_margin": true,
  "complete_alignment_in_unit_interval": true,
  "nonnegative_deviations": true,
  "num_gate_positive_decisions": 572,
  "num_complete_horizon_decisions": 516,
  "num_episodes_with_gate_positive": 80
}
```

## Provenance

```json
{
  "input_results": "rebuttal_experiments/results/exp4_k_sensitivity/raw/k32",
  "step_traces_sha256": "6b008315613e40da43480579b7363a5e692f863107bd7200fd04ba7b8fcf983d",
  "per_episode_metrics_sha256": "075fa7b25ea63c760938679cf7046a2e78e1d447feb58fb274bdf5a6faa59e40",
  "config_sha256": "f5d3fe6faf34739d2ce1edb81fa31c9b9102a2744b6c326810c7b6d23eb2ba97",
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

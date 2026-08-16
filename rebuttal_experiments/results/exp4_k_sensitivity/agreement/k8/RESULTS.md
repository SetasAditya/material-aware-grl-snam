# Experiment 2: Gate witness–trajectory agreement

The sampled primitive is an **activation witness**, not a trajectory command. The learned Hamiltonian field still determines the executed motion. This audit measures whether those two objects agree over an arc-length-matched short horizon.

Only gate-positive decisions from the learned `gate_on` rollouts are included. Aggregate metrics use decisions with at least 95% of the 12-cell horizon observed.

| Group | Gate + | Complete | Cosine ↑ | Endpoint dev. m ↓ | Cross-track m ↓ | Clearance agree ↑ | Hard disagreement ↓ | Pred. Δrisk | Realized Δrisk | Sign agree ↑ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 399 | 361 | 0.712 | 4.430 | 2.044 | 0.579 | 0.233 | 0.1307 | 0.0038 | 0.529 | 0.413 |
| regime=R1 | 238 | 220 | 0.695 | 4.568 | 2.090 | 0.595 | 0.195 | 0.1223 | -0.0007 | 0.500 | 0.338 |
| regime=R2 | 80 | 70 | 0.751 | 3.968 | 1.838 | 0.614 | 0.214 | 0.1486 | 0.0491 | 0.629 | 0.569 |
| regime=R3 | 81 | 71 | 0.725 | 4.456 | 2.103 | 0.493 | 0.366 | 0.1390 | -0.0268 | 0.521 | 0.391 |
| phase=before_event | 77 | 77 | 0.751 | 4.120 | 1.775 | 0.286 | 0.273 | 0.0935 | 0.0300 | 0.532 | 0.664 |
| phase=pre_opening | 53 | 53 | 0.741 | 4.064 | 1.893 | 0.057 | 0.906 | 0.1100 | -0.2066 | 0.057 | -0.104 |
| phase=post_opening | 269 | 231 | 0.692 | 4.617 | 2.168 | 0.797 | 0.065 | 0.1478 | 0.0433 | 0.636 | 0.541 |
| regime=R1,phase=before_event | 56 | 56 | 0.749 | 4.106 | 1.741 | 0.286 | 0.196 | 0.0876 | 0.0224 | 0.500 | 0.462 |
| regime=R1,phase=pre_opening | 32 | 32 | 0.746 | 3.954 | 1.843 | 0.094 | 0.844 | 0.1033 | -0.1645 | 0.094 | 0.044 |
| regime=R1,phase=post_opening | 150 | 132 | 0.660 | 4.913 | 2.298 | 0.848 | 0.038 | 0.1416 | 0.0291 | 0.598 | 0.416 |
| regime=R2,phase=before_event | 7 | 7 | 0.859 | 2.948 | 1.226 | 0.000 | 1.000 | 0.1448 | 0.1089 | 0.857 | 0.989 |
| regime=R2,phase=pre_opening | 8 | 8 | 0.702 | 4.705 | 2.171 | 0.000 | 1.000 | 0.1245 | -0.2252 | 0.000 | -0.585 |
| regime=R2,phase=post_opening | 65 | 55 | 0.744 | 3.991 | 1.867 | 0.782 | 0.000 | 0.1526 | 0.0813 | 0.691 | 0.804 |
| regime=R3,phase=before_event | 14 | 14 | 0.707 | 4.761 | 2.185 | 0.429 | 0.214 | 0.0917 | 0.0208 | 0.500 | 0.975 |
| regime=R3,phase=pre_opening | 13 | 13 | 0.755 | 3.939 | 1.844 | 0.000 | 1.000 | 0.1177 | -0.2990 | 0.000 | 0.040 |
| regime=R3,phase=post_opening | 54 | 44 | 0.722 | 4.512 | 2.153 | 0.659 | 0.227 | 0.1604 | 0.0385 | 0.682 | 0.465 |
| phase=pre_opening,crosses_opening=0 | 5 | 5 | 0.814 | 3.061 | 1.451 | 0.600 | 0.000 | 0.1284 | 0.0646 | 0.400 | 0.812 |
| phase=pre_opening,crosses_opening=1 | 48 | 48 | 0.734 | 4.168 | 1.939 | 0.000 | 1.000 | 0.1081 | -0.2349 | 0.021 | -0.304 |

## Validation

```json
{
  "all_input_rows_gate_on": true,
  "all_decisions_gate_positive": true,
  "selected_primitives_meet_clearance_threshold": true,
  "predicted_improvements_meet_gate_margin": true,
  "complete_alignment_in_unit_interval": true,
  "nonnegative_deviations": true,
  "num_gate_positive_decisions": 399,
  "num_complete_horizon_decisions": 361,
  "num_episodes_with_gate_positive": 66
}
```

## Provenance

```json
{
  "input_results": "rebuttal_experiments/results/exp4_k_sensitivity/raw/k8",
  "step_traces_sha256": "8cdf309f459488352bd436a529525751bfe8ed8379a29f42dcfbc8aaeb86ba44",
  "per_episode_metrics_sha256": "c4cbd738b2ffc43b400e3ee8088118b9fea29a947b6d3417dbf73bd47d3373e4",
  "config_sha256": "0fc6ec8c8ac4b3187c1bd602b95688a78e6362f71d460f0fa373f93163d7d790",
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

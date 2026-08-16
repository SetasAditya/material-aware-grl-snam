# Experiment 2: Gate witness–trajectory agreement

The sampled primitive is an **activation witness**, not a trajectory command. The learned Hamiltonian field still determines the executed motion. This audit measures whether those two objects agree over an arc-length-matched short horizon.

Only gate-positive decisions from the learned `gate_on` rollouts are included. Aggregate metrics use decisions with at least 95% of the 12-cell horizon observed.

| Group | Gate + | Complete | Cosine ↑ | Endpoint dev. m ↓ | Cross-track m ↓ | Clearance agree ↑ | Hard disagreement ↓ | Pred. Δrisk | Realized Δrisk | Sign agree ↑ | Pearson r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 492 | 444 | 0.705 | 4.458 | 2.039 | 0.556 | 0.259 | 0.1348 | 0.0088 | 0.529 | 0.369 |
| regime=R1 | 304 | 277 | 0.700 | 4.473 | 2.037 | 0.574 | 0.227 | 0.1222 | 0.0032 | 0.505 | 0.239 |
| regime=R2 | 94 | 84 | 0.692 | 4.584 | 2.081 | 0.607 | 0.238 | 0.1620 | 0.0554 | 0.619 | 0.642 |
| regime=R3 | 94 | 83 | 0.735 | 4.279 | 2.003 | 0.446 | 0.386 | 0.1493 | -0.0196 | 0.518 | 0.293 |
| phase=before_event | 103 | 103 | 0.706 | 4.540 | 1.954 | 0.301 | 0.340 | 0.0911 | 0.0247 | 0.505 | 0.706 |
| phase=pre_opening | 57 | 57 | 0.779 | 3.776 | 1.775 | 0.088 | 0.877 | 0.1250 | -0.1986 | 0.053 | -0.171 |
| phase=post_opening | 332 | 284 | 0.690 | 4.564 | 2.123 | 0.743 | 0.106 | 0.1527 | 0.0446 | 0.634 | 0.525 |
| regime=R1,phase=before_event | 79 | 79 | 0.705 | 4.488 | 1.917 | 0.316 | 0.278 | 0.0832 | 0.0159 | 0.456 | 0.508 |
| regime=R1,phase=pre_opening | 36 | 36 | 0.809 | 3.400 | 1.612 | 0.139 | 0.806 | 0.1213 | -0.1564 | 0.083 | -0.045 |
| regime=R1,phase=post_opening | 189 | 162 | 0.673 | 4.704 | 2.190 | 0.796 | 0.074 | 0.1415 | 0.0324 | 0.623 | 0.354 |
| regime=R2,phase=before_event | 9 | 9 | 0.710 | 4.610 | 1.894 | 0.000 | 1.000 | 0.1623 | 0.1080 | 0.889 | 0.919 |
| regime=R2,phase=pre_opening | 8 | 8 | 0.702 | 4.705 | 2.171 | 0.000 | 1.000 | 0.1245 | -0.2252 | 0.000 | -0.585 |
| regime=R2,phase=post_opening | 77 | 67 | 0.688 | 4.566 | 2.096 | 0.761 | 0.045 | 0.1665 | 0.0819 | 0.657 | 0.828 |
| regime=R3,phase=before_event | 15 | 15 | 0.705 | 4.773 | 2.183 | 0.400 | 0.267 | 0.0895 | 0.0211 | 0.533 | 0.961 |
| regime=R3,phase=pre_opening | 13 | 13 | 0.744 | 4.247 | 1.982 | 0.000 | 1.000 | 0.1355 | -0.2990 | 0.000 | -0.397 |
| regime=R3,phase=post_opening | 66 | 55 | 0.741 | 4.152 | 1.958 | 0.564 | 0.273 | 0.1688 | 0.0353 | 0.636 | 0.398 |
| phase=pre_opening,crosses_opening=0 | 7 | 7 | 0.848 | 2.815 | 1.349 | 0.714 | 0.000 | 0.1102 | 0.0416 | 0.286 | 0.842 |
| phase=pre_opening,crosses_opening=1 | 50 | 50 | 0.769 | 3.911 | 1.835 | 0.000 | 1.000 | 0.1271 | -0.2322 | 0.020 | -0.251 |

## Validation

```json
{
  "all_input_rows_gate_on": true,
  "all_decisions_gate_positive": true,
  "selected_primitives_meet_clearance_threshold": true,
  "predicted_improvements_meet_gate_margin": true,
  "complete_alignment_in_unit_interval": true,
  "nonnegative_deviations": true,
  "num_gate_positive_decisions": 492,
  "num_complete_horizon_decisions": 444,
  "num_episodes_with_gate_positive": 76
}
```

## Provenance

```json
{
  "input_results": "rebuttal_experiments/results/exp4_k_sensitivity/raw/k16",
  "step_traces_sha256": "cc21bf3fc96534990804d9224917fab4b2b4fdb6e9f978c1ea13eb7e0a92f079",
  "per_episode_metrics_sha256": "f1417d10366c5471b517fb8174bf49b71f5caaa15c2f3dd8f01504362f49ae4c",
  "config_sha256": "66a8ae6589852c18298b61f1e1610257e49311fb157c4ab4668d2ab0ea35e58a",
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

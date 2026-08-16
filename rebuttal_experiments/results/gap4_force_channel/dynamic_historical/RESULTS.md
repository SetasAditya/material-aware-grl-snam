# Gap 4: paired historical route-aware versus risk-loss-only

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| aggregate | `route_aware_stage2` | `risk_loss_only` | success | 0.610 | 1.000 | -0.390 [-0.490, -0.290] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | post_event_cvar_violation | 0.840 | 1.793 | -0.953 [-1.066, -0.834] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | event_window_cvar_violation | 0.845 | 1.793 | -0.948 [-1.061, -0.828] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | hard_contact | 0.080 | 0.920 | -0.840 [-0.910, -0.770] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | path_length_ratio | 1.386 | 0.976 | 0.410 [0.338, 0.484] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | stuck | 0.480 | 0.000 | 0.480 [0.380, 0.580] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | reaction_delay | 2.850 | 9.740 | -6.890 [-10.640, -3.670] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `risk_loss_only` | route_deviation_delay | 13.100 | 26.170 | -13.070 [-18.980, -7.290] | 1.000 | 100 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| aggregate | `route_aware_stage2` | `risk_loss_only` | 100 | 0 | 0 | True |

# Experiment 6: delayed-required paired uncertainty

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | false_pre_activation | 0.380 | 0.370 | 0.010 [-0.040, 0.060] | 0.283 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | suppression | 0.620 | 0.630 | -0.010 [-0.060, 0.040] | 0.281 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | success | 0.610 | 0.620 | -0.010 [-0.050, 0.030] | 0.249 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | stuck | 0.480 | 0.460 | 0.020 [-0.020, 0.060] | 0.092 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | post_event_cvar_violation | 0.840 | 0.855 | -0.015 [-0.034, 0.004] | 0.943 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | hard_contact | 0.080 | 0.060 | 0.020 [0.000, 0.050] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | hard_hazard_length_m | 0.059 | 0.040 | 0.019 [0.000, 0.044] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | stale_exposure | 0.783 | 0.848 | -0.065 [-0.163, 0.021] | 0.923 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | reaction_delay | 2.850 | 3.110 | -0.260 [-0.610, 0.050] | 0.942 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | route_deviation_delay | 13.040 | 13.680 | -0.640 [-2.870, 1.310] | 0.729 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | false_pre_activation | 0.380 | 0.950 | -0.570 [-0.670, -0.470] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | suppression | 0.620 | 0.050 | 0.570 [0.470, 0.670] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | success | 0.610 | 0.480 | 0.130 [0.030, 0.230] | 0.994 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stuck | 0.480 | 0.590 | -0.110 [-0.210, -0.010] | 0.984 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.840 | 0.695 | 0.145 [0.085, 0.202] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.080 | 0.040 | 0.040 [-0.010, 0.100] | 0.046 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.059 | 0.062 | -0.002 [-0.088, 0.066] | 0.478 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 0.783 | 0.688 | 0.094 [-0.214, 0.397] | 0.268 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 2.850 | 2.260 | 0.590 [-0.310, 1.480] | 0.100 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | route_deviation_delay | 13.040 | 1.960 | 11.080 [7.510, 15.120] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | false_pre_activation | 0.380 | 0.990 | -0.610 [-0.700, -0.510] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | suppression | 0.620 | 0.010 | 0.610 [0.510, 0.700] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | success | 0.610 | 0.030 | 0.580 [0.480, 0.670] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | stuck | 0.480 | 0.980 | -0.500 [-0.600, -0.400] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | post_event_cvar_violation | 0.840 | 0.463 | 0.377 [0.334, 0.421] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | hard_contact | 0.080 | 0.000 | 0.080 [0.030, 0.140] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | hard_hazard_length_m | 0.059 | 0.000 | 0.059 [0.021, 0.105] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | stale_exposure | 0.783 | 0.409 | 0.373 [0.146, 0.625] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | reaction_delay | 2.850 | 1.710 | 1.140 [0.420, 1.910] | 0.001 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | route_deviation_delay | 13.040 | 0.280 | 12.760 [9.200, 16.780] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | false_pre_activation | 0.380 | 0.920 | -0.540 [-0.640, -0.440] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | suppression | 0.620 | 0.080 | 0.540 [0.440, 0.640] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | success | 0.610 | 0.200 | 0.410 [0.300, 0.520] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | stuck | 0.480 | 0.810 | -0.330 [-0.440, -0.220] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | post_event_cvar_violation | 0.840 | 0.503 | 0.337 [0.298, 0.378] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | hard_contact | 0.080 | 0.000 | 0.080 [0.030, 0.140] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | hard_hazard_length_m | 0.059 | 0.000 | 0.059 [0.021, 0.106] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | stale_exposure | 0.783 | 0.595 | 0.188 [-0.070, 0.472] | 0.079 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | reaction_delay | 2.850 | 2.270 | 0.580 [-0.240, 1.430] | 0.082 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | route_deviation_delay | 13.040 | 1.190 | 11.850 [8.400, 15.950] | 0.000 | 100 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | 100 | 0 | 0 | True |

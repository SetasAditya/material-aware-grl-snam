# Experiment 6: unified three-event paired uncertainty

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | success | 0.917 | 0.923 | -0.007 [-0.023, 0.013] | 0.176 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | stuck | 0.133 | 0.117 | 0.017 [-0.007, 0.040] | 0.062 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | event_window_cvar_violation | 0.667 | 0.674 | -0.007 [-0.012, -0.001] | 0.987 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | post_event_cvar_violation | 0.667 | 0.674 | -0.007 [-0.013, -0.001] | 0.990 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | hard_contact | 0.027 | 0.027 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | hard_hazard_length_m | 0.027 | 0.021 | 0.006 [0.001, 0.015] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | stale_exposure | 1.637 | 1.871 | -0.233 [-0.495, 0.019] | 0.966 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | reaction_delay | 6.843 | 7.303 | -0.460 [-1.600, 1.000] | 0.775 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | path_length_ratio | 1.055 | 1.053 | 0.002 [-0.014, 0.013] | 0.353 | 100 |
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | compute_ms | 1.988 | 1.537 | 0.451 [0.292, 0.617] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | success | 0.780 | 0.800 | -0.020 [-0.060, 0.020] | 0.092 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | stuck | 0.220 | 0.210 | 0.010 [-0.030, 0.060] | 0.245 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | event_window_cvar_violation | 0.588 | 0.588 | 0.000 [-0.007, 0.011] | 0.514 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | post_event_cvar_violation | 0.588 | 0.587 | 0.000 [-0.007, 0.011] | 0.516 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | hard_contact | 0.080 | 0.080 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | hard_hazard_length_m | 0.082 | 0.063 | 0.019 [0.002, 0.045] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | stale_exposure | 1.352 | 1.667 | -0.315 [-0.623, 0.007] | 0.972 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | reaction_delay | 7.180 | 7.850 | -0.670 [-2.190, 1.400] | 0.776 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | path_length_ratio | 1.155 | 1.154 | 0.001 [-0.024, 0.019] | 0.444 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | compute_ms | 3.713 | 3.050 | 0.663 [0.315, 1.084] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | success | 0.970 | 0.980 | -0.010 [-0.030, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | stuck | 0.180 | 0.130 | 0.050 [0.010, 0.100] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | event_window_cvar_violation | 0.561 | 0.565 | -0.004 [-0.006, -0.003] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | post_event_cvar_violation | 0.561 | 0.565 | -0.004 [-0.006, -0.003] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | stale_exposure | 1.085 | 1.292 | -0.207 [-0.427, -0.042] | 0.998 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | reaction_delay | 5.120 | 5.950 | -0.830 [-1.710, -0.170] | 0.998 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | path_length_ratio | 1.035 | 1.020 | 0.016 [0.006, 0.027] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | compute_ms | 2.015 | 1.278 | 0.738 [0.474, 1.023] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | success | 1.000 | 0.990 | 0.010 [0.000, 0.030] | 0.632 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | stuck | 0.000 | 0.010 | -0.010 [-0.030, 0.000] | 0.635 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | event_window_cvar_violation | 0.853 | 0.870 | -0.016 [-0.033, -0.001] | 0.981 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | post_event_cvar_violation | 0.853 | 0.870 | -0.016 [-0.033, -0.001] | 0.981 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | stale_exposure | 2.476 | 2.654 | -0.178 [-0.587, 0.269] | 0.801 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | reaction_delay | 8.230 | 8.110 | 0.120 [-1.470, 2.280] | 0.483 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | path_length_ratio | 0.974 | 0.985 | -0.011 [-0.032, 0.001] | 0.926 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | compute_ms | 0.236 | 0.284 | -0.048 [-0.161, 0.009] | 0.637 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | success | 0.917 | 0.567 | 0.350 [0.260, 0.440] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stuck | 0.133 | 0.507 | -0.373 [-0.467, -0.280] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.667 | 0.595 | 0.073 [0.051, 0.096] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.667 | 0.593 | 0.075 [0.054, 0.097] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.027 | 0.023 | 0.003 [-0.017, 0.023] | 0.307 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.027 | 0.030 | -0.003 [-0.025, 0.018] | 0.583 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.637 | 1.002 | 0.636 [0.113, 1.164] | 0.007 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 6.843 | 2.803 | 4.040 [1.500, 7.027] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.055 | 1.408 | -0.353 [-0.427, -0.282] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | compute_ms | 1.988 | 0.184 | 1.804 [1.331, 2.331] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | success | 0.780 | 0.540 | 0.240 [0.130, 0.350] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | stuck | 0.220 | 0.540 | -0.320 [-0.440, -0.200] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.588 | 0.551 | 0.037 [0.004, 0.084] | 0.011 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.588 | 0.551 | 0.036 [0.003, 0.083] | 0.015 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.080 | 0.070 | 0.010 [-0.050, 0.070] | 0.305 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.082 | 0.091 | -0.008 [-0.075, 0.054] | 0.601 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.352 | 1.048 | 0.303 [-0.210, 0.844] | 0.124 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 7.180 | 3.360 | 3.820 [0.960, 7.160] | 0.003 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.155 | 1.431 | -0.277 [-0.379, -0.173] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | compute_ms | 3.713 | 0.184 | 3.529 [2.307, 4.842] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | success | 0.970 | 0.590 | 0.380 [0.280, 0.480] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | stuck | 0.180 | 0.490 | -0.310 [-0.420, -0.200] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.561 | 0.539 | 0.023 [0.008, 0.039] | 0.001 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.561 | 0.539 | 0.022 [0.007, 0.039] | 0.002 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.085 | 0.714 | 0.371 [-0.106, 0.870] | 0.067 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 5.120 | 2.290 | 2.830 [0.510, 5.710] | 0.007 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.035 | 1.383 | -0.347 [-0.422, -0.276] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | compute_ms | 2.015 | 0.184 | 1.832 [1.278, 2.431] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | success | 1.000 | 0.570 | 0.430 [0.330, 0.530] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | stuck | 0.000 | 0.490 | -0.490 [-0.590, -0.390] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.853 | 0.695 | 0.158 [0.120, 0.196] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.853 | 0.688 | 0.165 [0.129, 0.202] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 2.476 | 1.242 | 1.233 [0.555, 1.940] | 0.001 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 8.230 | 2.760 | 5.470 [2.410, 9.110] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 0.974 | 1.409 | -0.435 [-0.509, -0.365] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.236 | 0.184 | 0.052 [0.050, 0.055] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | success | 0.917 | 1.000 | -0.083 [-0.110, -0.057] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | stuck | 0.133 | 0.000 | 0.133 [0.103, 0.167] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.667 | 0.794 | -0.127 [-0.164, -0.090] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.667 | 0.794 | -0.127 [-0.164, -0.091] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | hard_contact | 0.027 | 0.200 | -0.173 [-0.207, -0.140] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.027 | 0.447 | -0.420 [-0.505, -0.336] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | stale_exposure | 1.637 | 2.538 | -0.900 [-1.325, -0.514] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | reaction_delay | 6.843 | 10.300 | -3.457 [-5.690, -1.623] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | path_length_ratio | 1.055 | 0.959 | 0.096 [0.070, 0.124] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | compute_ms | 1.988 | 0.052 | 1.936 [1.467, 2.465] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | success | 0.780 | 1.000 | -0.220 [-0.300, -0.140] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | stuck | 0.220 | 0.000 | 0.220 [0.140, 0.300] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.588 | 0.918 | -0.330 [-0.440, -0.222] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.588 | 0.918 | -0.330 [-0.442, -0.220] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | hard_contact | 0.080 | 0.400 | -0.320 [-0.410, -0.230] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.082 | 0.920 | -0.838 [-1.089, -0.594] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | stale_exposure | 1.352 | 2.196 | -0.844 [-1.259, -0.463] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | reaction_delay | 7.180 | 10.300 | -3.120 [-5.500, -1.040] | 0.998 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | path_length_ratio | 1.155 | 0.959 | 0.196 [0.122, 0.276] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | compute_ms | 3.713 | 0.052 | 3.661 [2.454, 4.993] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | success | 0.970 | 1.000 | -0.030 [-0.070, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | stuck | 0.180 | 0.000 | 0.180 [0.110, 0.260] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.200 | -0.200 [-0.280, -0.130] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.421 | -0.421 [-0.605, -0.249] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | stale_exposure | 1.085 | 2.149 | -1.064 [-1.533, -0.627] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | reaction_delay | 5.120 | 10.300 | -5.180 [-8.050, -2.820] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | path_length_ratio | 1.035 | 0.959 | 0.077 [0.052, 0.109] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | compute_ms | 2.015 | 0.053 | 1.963 [1.408, 2.567] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.853 | 0.897 | -0.044 [-0.064, -0.026] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.853 | 0.897 | -0.044 [-0.064, -0.026] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | stale_exposure | 2.476 | 3.268 | -0.792 [-1.367, -0.282] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | reaction_delay | 8.230 | 10.300 | -2.070 [-4.510, 0.090] | 0.968 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | path_length_ratio | 0.974 | 0.959 | 0.016 [0.011, 0.021] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | compute_ms | 0.236 | 0.052 | 0.184 [0.181, 0.186] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | success | 0.917 | 0.120 | 0.797 [0.737, 0.850] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | stuck | 0.133 | 0.887 | -0.753 [-0.810, -0.693] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | event_window_cvar_violation | 0.667 | 0.463 | 0.205 [0.179, 0.232] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | post_event_cvar_violation | 0.667 | 0.461 | 0.207 [0.181, 0.233] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | hard_contact | 0.027 | 0.007 | 0.020 [0.000, 0.040] | 0.015 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | hard_hazard_length_m | 0.027 | 0.009 | 0.018 [-0.008, 0.044] | 0.086 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | stale_exposure | 1.637 | 0.384 | 1.254 [0.802, 1.739] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | reaction_delay | 6.843 | 1.470 | 5.373 [2.883, 8.380] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | path_length_ratio | 1.055 | 2.002 | -0.947 [-1.046, -0.850] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | compute_ms | 1.988 | 1.867 | 0.121 [-0.359, 0.641] | 0.333 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | success | 0.780 | 0.080 | 0.700 [0.600, 0.790] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | stuck | 0.220 | 0.920 | -0.700 [-0.790, -0.600] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | event_window_cvar_violation | 0.588 | 0.455 | 0.133 [0.091, 0.187] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | post_event_cvar_violation | 0.588 | 0.454 | 0.134 [0.092, 0.186] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | hard_contact | 0.080 | 0.020 | 0.060 [0.000, 0.120] | 0.014 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | hard_hazard_length_m | 0.082 | 0.028 | 0.054 [-0.022, 0.136] | 0.084 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | stale_exposure | 1.352 | 0.373 | 0.979 [0.529, 1.472] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | reaction_delay | 7.180 | 1.520 | 5.660 [2.740, 9.120] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | path_length_ratio | 1.155 | 2.051 | -0.896 [-1.019, -0.774] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | compute_ms | 3.713 | 1.822 | 1.891 [0.693, 3.194] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | success | 0.970 | 0.110 | 0.860 [0.790, 0.920] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | stuck | 0.180 | 0.910 | -0.730 [-0.820, -0.640] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | event_window_cvar_violation | 0.561 | 0.446 | 0.115 [0.093, 0.139] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | post_event_cvar_violation | 0.561 | 0.443 | 0.118 [0.095, 0.141] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | stale_exposure | 1.085 | 0.328 | 0.756 [0.349, 1.219] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | reaction_delay | 5.120 | 1.290 | 3.830 [1.610, 6.580] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | path_length_ratio | 1.035 | 1.982 | -0.947 [-1.053, -0.843] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | compute_ms | 2.015 | 1.867 | 0.148 [-0.422, 0.745] | 0.324 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | success | 1.000 | 0.170 | 0.830 [0.750, 0.900] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | stuck | 0.000 | 0.830 | -0.830 [-0.900, -0.750] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | event_window_cvar_violation | 0.853 | 0.487 | 0.366 [0.335, 0.398] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | post_event_cvar_violation | 0.853 | 0.485 | 0.369 [0.338, 0.400] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | stale_exposure | 2.476 | 0.449 | 2.026 [1.413, 2.700] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | reaction_delay | 8.230 | 1.600 | 6.630 [3.670, 10.090] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | path_length_ratio | 0.974 | 1.973 | -0.999 [-1.112, -0.883] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | compute_ms | 0.236 | 1.912 | -1.676 [-1.829, -1.523] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | success | 0.917 | 0.273 | 0.643 [0.557, 0.730] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | stuck | 0.133 | 0.733 | -0.600 [-0.687, -0.513] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | event_window_cvar_violation | 0.667 | 0.487 | 0.180 [0.153, 0.208] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | post_event_cvar_violation | 0.667 | 0.487 | 0.181 [0.152, 0.209] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | hard_contact | 0.027 | 0.013 | 0.013 [-0.007, 0.033] | 0.075 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | hard_hazard_length_m | 0.027 | 0.018 | 0.009 [-0.013, 0.034] | 0.214 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | stale_exposure | 1.637 | 0.872 | 0.766 [0.327, 1.242] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | reaction_delay | 6.843 | 3.607 | 3.237 [1.077, 5.873] | 0.001 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | path_length_ratio | 1.055 | 1.554 | -0.499 [-0.578, -0.421] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | compute_ms | 1.988 | 2.905 | -0.917 [-1.394, -0.388] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | success | 0.780 | 0.260 | 0.520 [0.410, 0.630] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | stuck | 0.220 | 0.750 | -0.530 [-0.640, -0.420] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | event_window_cvar_violation | 0.588 | 0.479 | 0.109 [0.066, 0.161] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | post_event_cvar_violation | 0.588 | 0.479 | 0.109 [0.066, 0.162] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | hard_contact | 0.080 | 0.040 | 0.040 [-0.020, 0.100] | 0.070 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | hard_hazard_length_m | 0.082 | 0.054 | 0.028 [-0.040, 0.102] | 0.211 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | stale_exposure | 1.352 | 0.721 | 0.631 [0.188, 1.112] | 0.002 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | reaction_delay | 7.180 | 3.250 | 3.930 [1.410, 6.930] | 0.001 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | path_length_ratio | 1.155 | 1.571 | -0.417 [-0.512, -0.322] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | compute_ms | 3.713 | 2.906 | 0.808 [-0.396, 2.171] | 0.108 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | success | 0.970 | 0.270 | 0.700 [0.610, 0.790] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | stuck | 0.180 | 0.740 | -0.560 [-0.670, -0.440] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | event_window_cvar_violation | 0.561 | 0.458 | 0.103 [0.082, 0.125] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | post_event_cvar_violation | 0.561 | 0.457 | 0.104 [0.083, 0.126] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | stale_exposure | 1.085 | 0.958 | 0.126 [-0.340, 0.629] | 0.319 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | reaction_delay | 5.120 | 4.140 | 0.980 [-1.200, 3.530] | 0.212 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | path_length_ratio | 1.035 | 1.570 | -0.534 [-0.632, -0.437] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | compute_ms | 2.015 | 2.908 | -0.893 [-1.456, -0.292] | 0.998 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | success | 1.000 | 0.290 | 0.710 [0.620, 0.800] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | stuck | 0.000 | 0.710 | -0.710 [-0.800, -0.620] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | event_window_cvar_violation | 0.853 | 0.525 | 0.329 [0.288, 0.369] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | post_event_cvar_violation | 0.853 | 0.524 | 0.329 [0.288, 0.369] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | stale_exposure | 2.476 | 0.937 | 1.539 [0.934, 2.169] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | reaction_delay | 8.230 | 3.430 | 4.800 [2.260, 7.920] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | path_length_ratio | 0.974 | 1.522 | -0.548 [-0.638, -0.459] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | compute_ms | 0.236 | 2.902 | -2.666 [-2.695, -2.634] | 1.000 | 100 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| aggregate | `route_aware_stage2` | `stage2_expected_cost` | 300 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `stage2_expected_cost` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `stage2_expected_cost` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `stage2_expected_cost` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `dwa_semantic` | 300 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `stage1` | 300 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `fixed_coeff_stage2` | 300 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `fixed_coeff_stage2` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `fixed_coeff_stage2` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `fixed_coeff_stage2` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `blackbox_cvar` | 300 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `blackbox_cvar` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `blackbox_cvar` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `blackbox_cvar` | 100 | 0 | 0 | True |

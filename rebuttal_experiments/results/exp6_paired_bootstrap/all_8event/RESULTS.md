# Experiment 6: unified eight-event paired uncertainty

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| aggregate | `route_aware_stage2` | `dwa_semantic` | success | 0.938 | 0.566 | 0.371 [0.282, 0.464] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stuck | 0.111 | 0.510 | -0.399 [-0.490, -0.305] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.717 | 0.619 | 0.097 [0.075, 0.121] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.716 | 0.616 | 0.100 [0.078, 0.123] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.064 | 0.048 | 0.016 [0.003, 0.030] | 0.007 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.103 | 0.075 | 0.028 [-0.004, 0.056] | 0.043 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.685 | 1.008 | 0.677 [0.181, 1.176] | 0.004 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 6.902 | 2.880 | 4.022 [1.650, 6.711] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.038 | 1.413 | -0.376 [-0.447, -0.309] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | success | 0.790 | 0.540 | 0.250 [0.140, 0.360] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | stuck | 0.210 | 0.540 | -0.330 [-0.450, -0.220] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.588 | 0.551 | 0.038 [0.004, 0.084] | 0.011 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.588 | 0.551 | 0.037 [0.004, 0.083] | 0.013 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.080 | 0.070 | 0.010 [-0.050, 0.070] | 0.300 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.082 | 0.091 | -0.008 [-0.074, 0.054] | 0.599 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.352 | 1.048 | 0.303 [-0.224, 0.849] | 0.130 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 7.180 | 3.360 | 3.820 [0.960, 7.160] | 0.002 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.142 | 1.431 | -0.289 [-0.386, -0.187] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | success | 0.970 | 0.590 | 0.380 [0.280, 0.480] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | stuck | 0.180 | 0.490 | -0.310 [-0.420, -0.200] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.561 | 0.539 | 0.023 [0.008, 0.039] | 0.001 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.561 | 0.539 | 0.022 [0.007, 0.038] | 0.001 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.085 | 0.714 | 0.371 [-0.127, 0.879] | 0.075 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 5.120 | 2.290 | 2.830 [0.490, 5.610] | 0.007 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.035 | 1.383 | -0.347 [-0.420, -0.275] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | success | 1.000 | 0.600 | 0.400 [0.310, 0.500] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | stuck | 0.000 | 0.460 | -0.460 [-0.560, -0.360] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.561 | 0.559 | 0.002 [-0.030, 0.025] | 0.415 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.561 | 0.560 | 0.001 [-0.030, 0.025] | 0.419 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.020 | -0.020 [-0.050, 0.000] | 0.862 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.062 | -0.062 [-0.171, 0.000] | 0.866 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.597 | 0.977 | 0.620 [0.093, 1.143] | 0.010 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 8.170 | 3.200 | 4.970 [2.120, 8.270] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 0.974 | 1.389 | -0.415 [-0.488, -0.346] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | success | 0.970 | 0.540 | 0.430 [0.330, 0.530] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | stuck | 0.240 | 0.570 | -0.330 [-0.440, -0.210] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.861 | 0.676 | 0.185 [0.147, 0.225] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.859 | 0.664 | 0.195 [0.156, 0.234] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.799 | 0.915 | 0.884 [0.310, 1.478] | 0.001 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 5.570 | 2.200 | 3.370 [1.010, 6.220] | 0.001 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.070 | 1.432 | -0.362 [-0.440, -0.284] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | success | 0.990 | 0.610 | 0.380 [0.290, 0.470] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | stuck | 0.040 | 0.480 | -0.440 [-0.540, -0.340] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.743 | 0.606 | 0.137 [0.081, 0.202] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.743 | 0.601 | 0.142 [0.084, 0.205] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.350 | 0.220 | 0.130 [0.040, 0.220] | 0.001 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.651 | 0.343 | 0.308 [0.129, 0.494] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.117 | 1.109 | 0.007 [-0.489, 0.502] | 0.488 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 5.740 | 3.980 | 1.760 [-1.250, 4.860] | 0.122 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 0.994 | 1.390 | -0.397 [-0.467, -0.329] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | success | 1.000 | 0.570 | 0.430 [0.330, 0.530] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | stuck | 0.000 | 0.490 | -0.490 [-0.590, -0.390] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.853 | 0.695 | 0.158 [0.121, 0.195] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.853 | 0.688 | 0.165 [0.129, 0.202] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 2.476 | 1.242 | 1.233 [0.545, 1.941] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 8.230 | 2.760 | 5.470 [2.400, 9.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 0.974 | 1.409 | -0.435 [-0.509, -0.365] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | success | 0.780 | 0.500 | 0.280 [0.170, 0.390] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | stuck | 0.220 | 0.560 | -0.340 [-0.460, -0.220] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.902 | 0.720 | 0.183 [0.127, 0.242] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.898 | 0.712 | 0.186 [0.131, 0.245] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.080 | 0.070 | 0.010 [-0.050, 0.070] | 0.308 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.091 | 0.104 | -0.013 [-0.106, 0.067] | 0.601 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 2.134 | 1.223 | 0.911 [0.198, 1.639] | 0.007 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 7.350 | 2.730 | 4.620 [1.570, 8.310] | 0.001 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 1.137 | 1.460 | -0.323 [-0.424, -0.217] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | success | 1.000 | 0.580 | 0.420 [0.320, 0.520] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | stuck | 0.000 | 0.490 | -0.490 [-0.580, -0.390] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | event_window_cvar_violation | 0.664 | 0.609 | 0.055 [0.028, 0.082] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | post_event_cvar_violation | 0.664 | 0.610 | 0.054 [0.026, 0.081] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | stale_exposure | 1.922 | 0.833 | 1.090 [0.511, 1.692] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | reaction_delay | 7.860 | 2.520 | 5.340 [2.460, 8.740] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | path_length_ratio | 0.975 | 1.411 | -0.437 [-0.511, -0.367] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | success | 0.938 | 1.000 | -0.062 [-0.085, -0.041] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | stuck | 0.111 | 0.000 | 0.111 [0.084, 0.139] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.717 | 0.860 | -0.143 [-0.181, -0.106] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.716 | 0.860 | -0.144 [-0.183, -0.106] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | hard_contact | 0.064 | 0.228 | -0.164 [-0.193, -0.135] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.103 | 0.527 | -0.424 [-0.504, -0.346] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | stale_exposure | 1.685 | 2.647 | -0.962 [-1.389, -0.552] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | reaction_delay | 6.902 | 10.300 | -3.398 [-5.642, -1.520] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | path_length_ratio | 1.038 | 0.959 | 0.079 [0.059, 0.101] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | success | 0.790 | 1.000 | -0.210 [-0.300, -0.130] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | stuck | 0.210 | 0.000 | 0.210 [0.130, 0.290] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.588 | 0.918 | -0.330 [-0.439, -0.223] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.588 | 0.918 | -0.330 [-0.438, -0.222] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | hard_contact | 0.080 | 0.400 | -0.320 [-0.420, -0.230] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.082 | 0.920 | -0.838 [-1.096, -0.597] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | stale_exposure | 1.352 | 2.196 | -0.844 [-1.268, -0.457] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | reaction_delay | 7.180 | 10.300 | -3.120 [-5.440, -1.020] | 0.999 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | path_length_ratio | 1.142 | 0.959 | 0.184 [0.113, 0.263] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | success | 0.970 | 1.000 | -0.030 [-0.070, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | stuck | 0.180 | 0.000 | 0.180 [0.110, 0.260] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.200 | -0.200 [-0.280, -0.130] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.421 | -0.421 [-0.611, -0.251] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | stale_exposure | 1.085 | 2.149 | -1.064 [-1.543, -0.634] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | reaction_delay | 5.120 | 10.300 | -5.180 [-7.980, -2.850] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | path_length_ratio | 1.035 | 0.959 | 0.077 [0.051, 0.109] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.561 | 0.568 | -0.007 [-0.009, -0.005] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.561 | 0.568 | -0.007 [-0.009, -0.005] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | stale_exposure | 1.597 | 2.196 | -0.599 [-0.982, -0.255] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | reaction_delay | 8.170 | 10.300 | -2.130 [-4.370, -0.230] | 0.987 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | path_length_ratio | 0.974 | 0.959 | 0.016 [0.011, 0.021] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | success | 0.970 | 1.000 | -0.030 [-0.070, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | stuck | 0.240 | 0.000 | 0.240 [0.160, 0.330] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.861 | 1.010 | -0.149 [-0.222, -0.084] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.859 | 1.010 | -0.150 [-0.224, -0.085] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.320 | -0.320 [-0.410, -0.230] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.695 | -0.695 [-0.914, -0.485] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | stale_exposure | 1.799 | 3.116 | -1.318 [-1.936, -0.729] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | reaction_delay | 5.570 | 10.300 | -4.730 [-7.640, -2.200] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | path_length_ratio | 1.070 | 0.959 | 0.112 [0.079, 0.149] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | success | 0.990 | 1.000 | -0.010 [-0.030, 0.000] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | stuck | 0.040 | 0.000 | 0.040 [0.010, 0.080] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.743 | 0.990 | -0.247 [-0.350, -0.152] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.743 | 0.990 | -0.248 [-0.353, -0.154] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | hard_contact | 0.350 | 0.500 | -0.150 [-0.220, -0.090] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.651 | 1.256 | -0.605 [-0.855, -0.373] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | stale_exposure | 1.117 | 2.196 | -1.079 [-1.549, -0.637] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | reaction_delay | 5.740 | 10.300 | -4.560 [-7.530, -2.030] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | path_length_ratio | 0.994 | 0.959 | 0.035 [0.024, 0.048] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.853 | 0.897 | -0.044 [-0.064, -0.026] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.853 | 0.897 | -0.044 [-0.063, -0.026] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | stale_exposure | 2.476 | 3.268 | -0.792 [-1.347, -0.272] | 0.999 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | reaction_delay | 8.230 | 10.300 | -2.070 [-4.540, 0.030] | 0.973 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | path_length_ratio | 0.974 | 0.959 | 0.016 [0.011, 0.021] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | success | 0.780 | 1.000 | -0.220 [-0.300, -0.140] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | stuck | 0.220 | 0.000 | 0.220 [0.140, 0.300] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.902 | 1.217 | -0.315 [-0.427, -0.199] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.898 | 1.217 | -0.320 [-0.432, -0.206] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | hard_contact | 0.080 | 0.400 | -0.320 [-0.420, -0.230] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.091 | 0.920 | -0.830 [-1.085, -0.585] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | stale_exposure | 2.134 | 3.268 | -1.133 [-1.734, -0.562] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | reaction_delay | 7.350 | 10.300 | -2.950 [-5.430, -0.730] | 0.996 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | path_length_ratio | 1.137 | 0.959 | 0.178 [0.106, 0.256] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | event_window_cvar_violation | 0.664 | 0.709 | -0.045 [-0.059, -0.033] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | post_event_cvar_violation | 0.664 | 0.709 | -0.045 [-0.059, -0.033] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | stale_exposure | 1.922 | 2.791 | -0.868 [-1.374, -0.404] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | reaction_delay | 7.860 | 10.300 | -2.440 [-4.970, -0.240] | 0.986 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | path_length_ratio | 0.975 | 0.959 | 0.016 [0.012, 0.021] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.938 | 0.133 | 0.805 [0.750, 0.855] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.111 | 0.886 | -0.775 [-0.825, -0.721] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.717 | 0.473 | 0.244 [0.218, 0.271] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.716 | 0.473 | 0.243 [0.218, 0.270] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.064 | 0.015 | 0.049 [0.033, 0.065] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.103 | 0.017 | 0.086 [0.058, 0.114] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.685 | 0.394 | 1.291 [0.855, 1.764] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 6.902 | 1.510 | 5.393 [2.912, 8.383] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 1.038 | 2.002 | -0.965 [-1.064, -0.867] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.790 | 0.080 | 0.710 [0.620, 0.800] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.210 | 0.920 | -0.710 [-0.800, -0.620] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.588 | 0.455 | 0.133 [0.092, 0.185] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.588 | 0.454 | 0.134 [0.093, 0.186] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.080 | 0.020 | 0.060 [0.000, 0.120] | 0.015 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.082 | 0.028 | 0.054 [-0.022, 0.134] | 0.080 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.352 | 0.373 | 0.979 [0.516, 1.481] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 7.180 | 1.520 | 5.660 [2.800, 9.240] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 1.142 | 2.051 | -0.908 [-1.026, -0.787] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.970 | 0.110 | 0.860 [0.790, 0.920] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.180 | 0.910 | -0.730 [-0.820, -0.640] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.561 | 0.446 | 0.115 [0.093, 0.139] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.561 | 0.443 | 0.118 [0.095, 0.141] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.085 | 0.328 | 0.756 [0.351, 1.215] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 5.120 | 1.290 | 3.830 [1.570, 6.650] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 1.035 | 1.982 | -0.947 [-1.051, -0.844] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | success | 1.000 | 0.160 | 0.840 [0.770, 0.910] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.000 | 0.870 | -0.870 [-0.930, -0.800] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.561 | 0.474 | 0.087 [0.062, 0.112] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.561 | 0.471 | 0.090 [0.066, 0.114] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.000 | 0.060 | -0.060 [-0.110, -0.020] | 0.998 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.000 | 0.058 | -0.058 [-0.117, -0.015] | 0.998 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.597 | 0.327 | 1.271 [0.802, 1.786] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 8.170 | 1.400 | 6.770 [3.890, 10.200] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 0.974 | 1.977 | -1.003 [-1.105, -0.902] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.970 | 0.140 | 0.830 [0.750, 0.900] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.240 | 0.890 | -0.650 [-0.740, -0.550] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.861 | 0.486 | 0.375 [0.343, 0.407] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.859 | 0.480 | 0.379 [0.347, 0.410] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.799 | 0.456 | 1.343 [0.820, 1.918] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 5.570 | 1.590 | 3.980 [1.720, 6.680] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 1.070 | 1.981 | -0.911 [-1.023, -0.799] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.990 | 0.140 | 0.850 [0.780, 0.920] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.040 | 0.890 | -0.850 [-0.920, -0.780] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.743 | 0.456 | 0.287 [0.227, 0.352] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.743 | 0.454 | 0.288 [0.228, 0.354] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.350 | 0.020 | 0.330 [0.240, 0.420] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.651 | 0.024 | 0.627 [0.441, 0.824] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.117 | 0.371 | 0.746 [0.369, 1.177] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 5.740 | 1.490 | 4.250 [1.870, 7.300] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 0.994 | 2.017 | -1.023 [-1.135, -0.914] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | success | 1.000 | 0.170 | 0.830 [0.750, 0.900] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.000 | 0.830 | -0.830 [-0.900, -0.750] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.853 | 0.487 | 0.366 [0.335, 0.397] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.853 | 0.485 | 0.369 [0.338, 0.399] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 2.476 | 0.449 | 2.026 [1.428, 2.688] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 8.230 | 1.600 | 6.630 [3.650, 10.060] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 0.974 | 1.973 | -0.999 [-1.114, -0.885] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | success | 0.780 | 0.080 | 0.700 [0.610, 0.790] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.220 | 0.920 | -0.700 [-0.790, -0.610] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.902 | 0.497 | 0.405 [0.346, 0.475] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.898 | 0.494 | 0.404 [0.345, 0.475] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.080 | 0.020 | 0.060 [0.000, 0.120] | 0.013 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.091 | 0.028 | 0.062 [-0.018, 0.148] | 0.068 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 2.134 | 0.441 | 1.694 [1.083, 2.369] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 7.350 | 1.580 | 5.770 [2.790, 9.240] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 1.137 | 2.052 | -0.915 [-1.037, -0.796] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | success | 1.000 | 0.180 | 0.820 [0.740, 0.890] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | stuck | 0.000 | 0.860 | -0.860 [-0.920, -0.790] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | event_window_cvar_violation | 0.664 | 0.484 | 0.180 [0.154, 0.207] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | post_event_cvar_violation | 0.664 | 0.498 | 0.166 [0.137, 0.195] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | stale_exposure | 1.922 | 0.410 | 1.513 [0.981, 2.087] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | reaction_delay | 7.860 | 1.610 | 6.250 [3.300, 9.740] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | path_length_ratio | 0.975 | 1.985 | -1.010 [-1.119, -0.903] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | success | 0.938 | 0.902 | 0.035 [0.015, 0.056] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | stuck | 0.111 | 0.194 | -0.083 [-0.109, -0.058] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.717 | 0.792 | -0.075 [-0.094, -0.057] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.716 | 0.791 | -0.075 [-0.094, -0.056] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.064 | 0.062 | 0.001 [-0.007, 0.010] | 0.352 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.103 | 0.037 | 0.066 [0.045, 0.088] | 0.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.685 | 2.330 | -0.645 [-1.004, -0.318] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 6.902 | 9.957 | -3.055 [-5.065, -1.329] | 1.000 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 1.038 | 1.048 | -0.010 [-0.030, 0.009] | 0.843 | 100 |
| aggregate | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | success | 0.790 | 0.630 | 0.160 [0.090, 0.240] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | stuck | 0.210 | 0.370 | -0.160 [-0.240, -0.090] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.588 | 0.593 | -0.004 [-0.057, 0.049] | 0.570 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.588 | 0.592 | -0.004 [-0.058, 0.051] | 0.565 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.080 | 0.060 | 0.020 [0.000, 0.050] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.082 | 0.032 | 0.050 [0.012, 0.101] | 0.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.352 | 1.944 | -0.592 [-0.924, -0.291] | 1.000 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 7.180 | 10.940 | -3.760 [-6.880, -1.100] | 0.999 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 1.142 | 1.244 | -0.102 [-0.173, -0.035] | 0.998 | 100 |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | success | 0.970 | 0.990 | -0.020 [-0.050, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | stuck | 0.180 | 0.130 | 0.050 [0.010, 0.100] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.561 | 0.567 | -0.006 [-0.008, -0.004] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.085 | 1.782 | -0.697 [-1.135, -0.324] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 5.120 | 8.560 | -3.440 [-5.640, -1.620] | 1.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 1.035 | 0.994 | 0.042 [0.022, 0.068] | 0.000 | 100 |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.561 | 0.568 | -0.007 [-0.009, -0.005] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.561 | 0.568 | -0.007 [-0.009, -0.005] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.597 | 2.196 | -0.599 [-0.985, -0.254] | 1.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 8.170 | 10.300 | -2.130 [-4.310, -0.180] | 0.986 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 0.974 | 0.959 | 0.016 [0.011, 0.020] | 0.000 | 100 |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | success | 0.970 | 0.980 | -0.010 [-0.030, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | stuck | 0.240 | 0.260 | -0.020 [-0.070, 0.030] | 0.703 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.861 | 0.907 | -0.046 [-0.067, -0.027] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.859 | 0.906 | -0.047 [-0.068, -0.027] | 1.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.799 | 2.190 | -0.392 [-0.823, 0.005] | 0.973 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 5.570 | 6.710 | -1.140 [-2.490, 0.090] | 0.965 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 1.070 | 1.032 | 0.039 [0.015, 0.068] | 0.001 | 100 |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | success | 0.990 | 0.990 | 0.000 [-0.030, 0.030] | 0.349 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | stuck | 0.040 | 0.420 | -0.380 [-0.480, -0.290] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.743 | 1.182 | -0.439 [-0.565, -0.318] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.743 | 1.182 | -0.440 [-0.569, -0.316] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.350 | 0.380 | -0.030 [-0.090, 0.030] | 0.804 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.651 | 0.232 | 0.420 [0.260, 0.588] | 0.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.117 | 1.514 | -0.397 [-0.788, -0.007] | 0.977 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 5.740 | 11.610 | -5.870 [-8.680, -3.080] | 1.000 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 0.994 | 0.993 | 0.001 [-0.018, 0.019] | 0.470 | 100 |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.853 | 0.897 | -0.043 [-0.063, -0.025] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.853 | 0.897 | -0.043 [-0.064, -0.025] | 1.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 2.476 | 3.268 | -0.792 [-1.344, -0.275] | 0.999 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 8.230 | 10.300 | -2.070 [-4.590, 0.070] | 0.970 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 0.974 | 0.959 | 0.016 [0.011, 0.021] | 0.000 | 100 |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | success | 0.780 | 0.630 | 0.150 [0.080, 0.230] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | stuck | 0.220 | 0.370 | -0.150 [-0.230, -0.080] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.902 | 0.914 | -0.012 [-0.063, 0.037] | 0.684 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.898 | 0.902 | -0.005 [-0.056, 0.045] | 0.572 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.080 | 0.060 | 0.020 [0.000, 0.050] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.091 | 0.032 | 0.059 [0.016, 0.114] | 0.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 2.134 | 2.954 | -0.820 [-1.331, -0.339] | 1.000 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 7.350 | 10.940 | -3.590 [-6.870, -0.760] | 0.994 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 1.137 | 1.244 | -0.108 [-0.176, -0.043] | 0.999 | 100 |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | success | 1.000 | 1.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | stuck | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | event_window_cvar_violation | 0.664 | 0.710 | -0.045 [-0.059, -0.033] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | post_event_cvar_violation | 0.664 | 0.710 | -0.045 [-0.059, -0.033] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | hard_contact | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | hard_hazard_length_m | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | stale_exposure | 1.922 | 2.791 | -0.868 [-1.364, -0.396] | 1.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | reaction_delay | 7.860 | 10.300 | -2.440 [-4.880, -0.240] | 0.986 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | path_length_ratio | 0.975 | 0.959 | 0.016 [0.011, 0.021] | 0.000 | 100 |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | compute_ms | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.000 | 100 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| aggregate | `route_aware_stage2` | `dwa_semantic` | 800 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:crossing_obstacle | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:delayed_escape_opens | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| event:puddle_expansion | `route_aware_stage2` | `dwa_semantic` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `stage1` | 800 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:crossing_obstacle | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:delayed_escape_opens | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| event:puddle_expansion | `route_aware_stage2` | `stage1` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `non_route_directional_stage2` | 800 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:crossing_obstacle | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:delayed_escape_opens | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| event:puddle_expansion | `route_aware_stage2` | `non_route_directional_stage2` | 100 | 0 | 0 | True |
| aggregate | `route_aware_stage2` | `neural_potential_field` | 800 | 0 | 0 | True |
| event:corridor_closes | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:corridor_opens | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:crossing_obstacle | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:delayed_escape_opens | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:moving_obstacle_blocks_detour | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:mud_onset | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:mud_onset_detour_blocked | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |
| event:puddle_expansion | `route_aware_stage2` | `neural_potential_field` | 100 | 0 | 0 | True |

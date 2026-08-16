# Experiment 6: static force-source paired uncertainty

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| static | `stage2_directional_head` | `s2_model_lambda` | correct_activation | 0.884 | 0.365 | 0.519 [0.429, 0.607] | 1.000 | 113 |
| static | `stage2_directional_head` | `s2_model_lambda` | false_activation | 0.129 | 0.752 | -0.623 [-0.643, -0.603] | 1.000 | 600 |
| static | `stage2_directional_head` | `s2_model_lambda` | selectivity_ratio | 1.292 | 1.200 | 0.093 [-0.158, 0.364] | 0.761 | 600 |
| static | `stage2_directional_head` | `s2_model_lambda` | force_perp_norm | 0.086 | 0.016 | 0.070 [0.064, 0.076] | -- | 900 |
| static | `stage2_directional_head` | `s2_model_lambda` | force_norm | 0.160 | 0.020 | 0.139 [0.130, 0.149] | -- | 900 |
| static | `stage2_directional_head` | `s2_model_lambda` | dot_safe | 0.073 | 0.013 | 0.060 [0.053, 0.067] | -- | 900 |
| static | `stage2_directional_head` | `s2_model_lambda` | force_risk_alignment | 0.037 | 0.893 | -0.856 [-0.867, -0.845] | -- | 900 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | correct_activation | 0.884 | 0.566 | 0.317 [0.234, 0.401] | 1.000 | 113 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | false_activation | 0.129 | 0.888 | -0.759 [-0.777, -0.741] | 1.000 | 600 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | selectivity_ratio | 1.292 | 1.123 | 0.169 [-0.060, 0.420] | 0.921 | 600 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | force_perp_norm | 0.086 | 1.291 | -1.205 [-1.240, -1.170] | -- | 900 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | force_norm | 0.160 | 1.543 | -1.383 [-1.422, -1.344] | -- | 900 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | dot_safe | 0.073 | 1.104 | -1.032 [-1.066, -0.998] | -- | 900 |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | force_risk_alignment | 0.037 | 0.715 | -0.678 [-0.690, -0.666] | -- | 900 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | correct_activation | 0.365 | 0.566 | -0.201 [-0.268, -0.139] | 0.000 | 113 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | false_activation | 0.752 | 0.888 | -0.136 [-0.144, -0.128] | 1.000 | 600 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | selectivity_ratio | 1.200 | 1.123 | 0.077 [0.033, 0.122] | 1.000 | 600 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | force_perp_norm | 0.016 | 1.291 | -1.275 [-1.308, -1.243] | -- | 900 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | force_norm | 0.020 | 1.543 | -1.523 [-1.557, -1.488] | -- | 900 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | dot_safe | 0.013 | 1.104 | -1.092 [-1.123, -1.061] | -- | 900 |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | force_risk_alignment | 0.893 | 0.715 | 0.178 [0.168, 0.188] | -- | 900 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| static | `stage2_directional_head` | `s2_model_lambda` | 13553 | 0 | 0 | True |
| static | `stage2_directional_head` | `analytic_fixed_lambda` | 13553 | 0 | 0 | True |
| static | `s2_model_lambda` | `analytic_fixed_lambda` | 13553 | 0 | 0 | True |

Static labels are the exact `force_source` values in the raw artifact. `analytic_fixed_lambda`, `s2_model_lambda`, and `stage2_directional_head` are not labeled as gate-off or route-aware.

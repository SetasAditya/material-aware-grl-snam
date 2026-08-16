# Gap 4: controlled directional force-channel versus learned scalar risk coefficient

Paired episode-cluster bootstrap with 10,000 replicates (seed 27370). Differences are method A minus method B.

| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |
|---|---|---|---|---:|---:|---:|---:|---:|
| static | `stage2_directional_head` | `s2_model_lambda` | correct_activation | 0.884 | 0.365 | 0.519 [0.429, 0.607] | 1.000 | 113 |
| static | `stage2_directional_head` | `s2_model_lambda` | false_activation | 0.129 | 0.752 | -0.623 [-0.643, -0.603] | 1.000 | 600 |
| static | `stage2_directional_head` | `s2_model_lambda` | selectivity_ratio | 1.292 | 1.200 | 0.093 [-0.158, 0.364] | 0.761 | 600 |
| static | `stage2_directional_head` | `s2_model_lambda` | force_perp_norm | 0.086 | 0.016 | 0.070 [0.064, 0.076] | -- | 900 |
| static | `stage2_directional_head` | `s2_model_lambda` | force_norm | 0.160 | 0.020 | 0.139 [0.130, 0.149] | -- | 900 |
| static | `stage2_directional_head` | `s2_model_lambda` | dot_safe | 0.073 | 0.013 | 0.060 [0.053, 0.067] | -- | 900 |

Pairing validation:

| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |
|---|---|---|---:|---:|---:|---|
| static | `stage2_directional_head` | `s2_model_lambda` | 13553 | 0 | 0 | True |

Static labels are the exact `force_source` values in the raw artifact. `analytic_fixed_lambda`, `s2_model_lambda`, and `stage2_directional_head` are not labeled as gate-off or route-aware.

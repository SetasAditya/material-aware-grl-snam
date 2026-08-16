# Gap 4 — risk-loss-only versus explicit force channel

Two complementary comparisons are reported because the repository does not
contain a single dynamic artifact in which the final learned Hamiltonian
controller differs only by adding/removing the force channel.

## Controlled static ablation

The exactly paired static evaluator compares:

- `s2_model_lambda`: the learned scalar risk coefficient; and
- `stage2_directional_head`: an explicit learned directional force output.

The maps, episodes, path samples, and evaluator are shared.

| Metric | Directional force | Scalar risk coefficient | Paired Δ (95% CI) |
|---|---:|---:|---:|
| Correct activation (CAR) | 0.884 | 0.365 | **+0.519 [+.429, +.607]** |
| False activation (FAR) | 0.129 | 0.752 | **-0.623 [-.643, -.603]** |
| Selectivity ratio | 1.292 | 1.200 | +0.093 [-.158, +.364] |
| Perpendicular-force norm | 0.086 | 0.016 | +0.070 [+.064, +.076] |
| Total force norm | 0.160 | 0.020 | +0.139 [+.130, +.149] |
| Safe-direction projection | 0.073 | 0.013 | +0.060 [+.053, +.067] |

This is strong evidence that an explicit directional force output improves the
measured activation/suppression behavior over the scalar coefficient. The
selectivity-ratio interval overlaps zero, so the rebuttal should emphasize the
resolved CAR and FAR differences rather than claim every summary statistic
improves.

## Paired historical dynamic comparison

The available 100-episode delayed-required artifact compares a prerecorded
risk-path follower (`risk_loss_only`) with the hand-coded
`route_aware_stage2` grid controller.

| Metric | Route-aware | Risk-loss-only | Paired Δ (95% CI) |
|---|---:|---:|---:|
| Success | 0.610 | 1.000 | **-0.390 [-.490, -.290]** |
| Post-event violation CVaR | 0.840 | 1.793 | **-0.953 [-1.066, -.834]** |
| Hard-contact episodes | 0.080 | 0.920 | **-0.840 [-.910, -.770]** |
| Path-length ratio | 1.386 | 0.976 | **+0.410 [+.338, +.484]** |
| Stuck | 0.480 | 0.000 | **+0.480 [+.380, +.580]** |

The dynamic evidence shows a sharp safety–completion tradeoff: route-aware
control greatly reduces empirical violation and contact but sacrifices success
and path efficiency.

## Rebuttal-safe conclusion

The controlled evidence supports the narrow structural claim that explicitly
predicting a directional force changes CAR/FAR relative to a scalar
risk-loss/coefficient pathway. It does **not** establish that the final learned
Hamiltonian controller improves navigation outcomes solely because of that
channel. The dynamic comparison is complementary behavioral evidence only:
`route_aware_stage2` is a hand-coded controller and must not be described as
the learned Hamiltonian model or as a one-variable force-channel ablation.

All intervals use 10,000 paired episode-cluster bootstrap replicates with seed
27370. Pairing is exact with zero method-only observations.

## Artifacts

- `static_controlled/paired_bootstrap_results.csv`
- `dynamic_historical/paired_bootstrap_results.csv`
- the corresponding `pairing_validation.csv`, JSON, and Markdown reports


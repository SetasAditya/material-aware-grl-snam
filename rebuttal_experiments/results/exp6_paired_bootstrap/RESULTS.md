# Experiment 6: Paired Statistical Uncertainty

All estimates use a paired episode-cluster bootstrap with 10,000 replicates
and seed 27370. An episode is resampled as one cluster, so all event rows in
the three- and eight-event aggregates receive the same bootstrap
multiplicity. Every requested comparison had exact pairing: there were zero
method-only observations in all four analyses.

The contrast below is Route-aware Stage 2 minus the comparator. A positive
success difference is favorable; a negative violation-CVaR or false
pre-activation difference is favorable.

| Dataset | Comparator | Metric | Route-aware | Comparator | Paired difference (95% CI) |
|---|---|---|---:|---:|---:|
| Delayed required | Expected cost | False pre-activation | 0.380 | 0.370 | +0.010 [-0.040, +0.060] |
| Delayed required | Expected cost | Success | 0.610 | 0.620 | -0.010 [-0.050, +0.030] |
| Delayed required | Expected cost | Post-event violation CVaR | 0.840 | 0.855 | -0.015 [-0.034, +0.004] |
| Delayed required | DWA | False pre-activation | 0.380 | 0.950 | **-0.570 [-0.670, -0.470]** |
| Delayed required | DWA | Success | 0.610 | 0.480 | **+0.130 [+0.030, +0.230]** |
| Delayed required | DWA | Post-event violation CVaR | 0.840 | 0.695 | **+0.145 [+0.085, +0.202]** |
| Three-event aggregate | Expected cost | Success | 0.917 | 0.923 | -0.007 [-0.023, +0.013] |
| Three-event aggregate | Expected cost | Event-window violation CVaR | 0.667 | 0.674 | **-0.007 [-0.012, -0.001]** |
| Three-event aggregate | DWA | Success | 0.917 | 0.567 | **+0.350 [+0.260, +0.440]** |
| Three-event aggregate | DWA | Event-window violation CVaR | 0.667 | 0.595 | **+0.073 [+0.051, +0.096]** |
| Three-event aggregate | Stage 1 (geometry only) | Success | 0.917 | 1.000 | **-0.083 [-0.110, -0.057]** |
| Three-event aggregate | Stage 1 (geometry only) | Event-window violation CVaR | 0.667 | 0.794 | **-0.127 [-0.164, -0.090]** |
| Eight-event aggregate | DWA | Success | 0.938 | 0.566 | **+0.371 [+0.283, +0.464]** |
| Eight-event aggregate | DWA | Event-window violation CVaR | 0.717 | 0.619 | **+0.097 [+0.075, +0.121]** |
| Eight-event aggregate | Stage 1 (geometry only) | Success | 0.938 | 1.000 | **-0.063 [-0.085, -0.041]** |
| Eight-event aggregate | Stage 1 (geometry only) | Event-window violation CVaR | 0.717 | 0.860 | **-0.143 [-0.181, -0.106]** |

## Static force-source results

The original static `force_samples.csv` does **not** contain the final
route-aware source. It contains three exactly paired evaluator sources:
`analytic_fixed_lambda`, `s2_model_lambda`, and
`stage2_directional_head`. The last source is the non-route directional
head used for the black-box-CVaR row; it must not be described as route-aware
or as a clean gate-off model.

| Source A | Source B | Metric | A | B | Paired difference (95% CI) |
|---|---|---|---:|---:|---:|
| Directional head | Learned scalar lambda | CAR | 0.884 | 0.365 | **+0.519 [+0.429, +0.607]** |
| Directional head | Learned scalar lambda | FAR | 0.129 | 0.752 | **-0.623 [-0.643, -0.603]** |
| Directional head | Learned scalar lambda | Selectivity ratio | 1.292 | 1.200 | +0.093 [-0.158, +0.364] |
| Directional head | Fixed lambda | CAR | 0.884 | 0.566 | **+0.317 [+0.234, +0.401]** |
| Directional head | Fixed lambda | FAR | 0.129 | 0.888 | **-0.759 [-0.777, -0.741]** |
| Directional head | Fixed lambda | Selectivity ratio | 1.292 | 1.123 | +0.169 [-0.060, +0.420] |

Thus the static artifact strongly supports the directional head's CAR/FAR
improvement over scalar and fixed coefficients, but its selectivity-ratio
advantage is not resolved by the 95% cluster-bootstrap intervals. A paired
CI for the final route-aware static method requires exporting its raw
per-sample force rows; the existing fold-level summary is insufficient for
an episode-cluster bootstrap.

## Rebuttal-safe interpretation

- The cleanest supported delayed-event result is relative to DWA:
  route-aware control substantially reduces premature activation and improves
  success. It does not reduce DWA's violation CVaR in this artifact.
- Route-aware and expected-cost variants are statistically indistinguishable
  on delayed false pre-activation and success. On the three-event aggregate,
  expected-cost has indistinguishable success while route-aware has a small
  favorable violation-CVaR difference.
- Against Stage 1, route-aware control trades a modest success reduction for
  a substantial reduction in violation CVaR on both the three- and
  eight-event aggregates.
- Against DWA, route-aware control trades a large success gain for higher
  violation CVaR on both aggregates. The rebuttal should state this tradeoff
  instead of claiming uniform dominance.

## Artifact consistency warning

The specifically requested raw delayed-required artifact reports
route-aware false pre-activation **0.380** and success **0.610**. Those values
do not match the current manuscript's 0.180 and 0.810. The manuscript values
must not be paired with the intervals generated here unless the exact raw
rollout artifact underlying 0.180/0.810 is identified and reanalyzed.

## Detailed machine-readable outputs

- `delayed_required/`: primary delayed-required comparisons, including
  expected cost, DWA, fixed coefficients, and black-box CVaR.
- `material_3event/`: aggregate and per-event results for the unified
  three-event artifact.
- `all_8event/`: aggregate and per-event results for the unified eight-event
  artifact.
- `static_force_sources/`: paired static force-source results.

Each directory contains `paired_bootstrap_results.csv`,
`paired_bootstrap_results.json`, `pairing_validation.csv`, and a complete
Markdown table.

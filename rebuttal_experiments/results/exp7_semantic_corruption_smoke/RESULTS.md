# Experiment 7 — Semantic-label corruption robustness

Frozen clean LOSO route-aware heads were evaluated on **15 episodes** from **11 scenes**. This is a pointwise selectivity study; navigation rollout outcomes are not available and are not reported.

## Results

| Corruption | CAR (95% CI) | FAR (95% CI) | SR (95% CI) | Active | No-op |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.460 [0.305, 0.729] | 0.074 [0.026, 0.133] | 3.257 [1.069, 547371.598] | 0.144 | 0.856 |
| 10% | 0.240 [0.102, 0.458] | 0.061 [0.020, 0.096] | 2.262 [0.454, 19213888.186] | 0.100 | 0.900 |
| 20% | 0.220 [0.114, 0.346] | 0.044 [0.009, 0.088] | 3.461 [0.927, 23449568.966] | 0.079 | 0.921 |
| 30% | 0.240 [0.058, 0.469] | 0.052 [0.017, 0.097] | 1.890 [0.544, 12.964] | 0.073 | 0.927 |

## Paired change from the clean map

| Corruption | ΔCAR (95% CI) | ΔFAR (95% CI) | ΔSR (95% CI) |
|---:|---:|---:|---:|
| 0% | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| 10% | -0.220 [-0.368, -0.180] | -0.013 [-0.052, +0.034] | -0.995 [-192336.292, +19213879.630] |
| 20% | -0.240 [-0.429, -0.146] | -0.031 [-0.052, -0.008] | +0.204 [-236687.937, +23449560.922] |
| 30% | -0.220 [-0.554, -0.111] | -0.022 [-0.055, +0.012] | -1.367 [-547368.046, +5.150] |

CIs are percentile cluster-bootstrap intervals (episode is the resampling unit); delta intervals use the same resampled episodes at every corruption level.

## Protocol

For each scene, exactly round(p*N) observed BEV cells are selected by a SHA-256-derived fixed scene seed. Their modal semantic ID is independently sampled from the evaluated scenes' empirical observed-cell distribution, conditioned on differing from the clean ID. Masks are nested across p.

CAR/FAR eligibility and best direction come from the clean ontology map; only model inputs are recomputed from corrupted semantics.

All risk, hard-hazard, SDF, risk-gradient, and SDF-gradient fields are recomputed with the canonical `main` ontology after corruption. The controller weights and fold-specific clean calibration threshold remain frozen.

## Reproduce

```bash
python rebuttal_experiments/exp7_semantic_corruption.py
```

See `provenance.json` for fixed seeds, input hashes, checkpoints, and full configuration.

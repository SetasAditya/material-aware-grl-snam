# Experiment 7 — Semantic-label corruption robustness

Frozen clean LOSO route-aware heads were evaluated on **2250 episodes** from **293 scenes**. This is a pointwise selectivity study; navigation rollout outcomes are not available and are not reported.

## Results

| Corruption | CAR (95% CI) | FAR (95% CI) | SR (95% CI) | Active | No-op |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.733 [0.718, 0.749] | 0.217 [0.206, 0.228] | 1.864 [1.727, 2.023] | 0.287 | 0.713 |
| 10% | 0.489 [0.471, 0.506] | 0.171 [0.162, 0.179] | 1.673 [1.550, 1.821] | 0.209 | 0.791 |
| 20% | 0.397 [0.380, 0.415] | 0.145 [0.137, 0.152] | 1.604 [1.474, 1.751] | 0.175 | 0.825 |
| 30% | 0.347 [0.331, 0.364] | 0.134 [0.126, 0.141] | 1.549 [1.413, 1.699] | 0.158 | 0.842 |

## Paired change from the clean map

| Corruption | ΔCAR (95% CI) | ΔFAR (95% CI) | ΔSR (95% CI) |
|---:|---:|---:|---:|
| 0% | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| 10% | -0.245 [-0.261, -0.228] | -0.047 [-0.053, -0.041] | -0.190 [-0.276, -0.106] |
| 20% | -0.336 [-0.355, -0.317] | -0.072 [-0.079, -0.066] | -0.260 [-0.363, -0.158] |
| 30% | -0.386 [-0.404, -0.366] | -0.083 [-0.091, -0.076] | -0.315 [-0.429, -0.201] |

CIs are percentile cluster-bootstrap intervals (episode is the resampling unit); delta intervals use the same resampled episodes at every corruption level.

## Finding

Corruption makes the frozen gate progressively more conservative: activation falls from 0.287 to 0.158, while CAR falls from 0.733 to 0.347. The lower FAR under corruption is therefore not evidence of improved robustness; it accompanies a large loss of required activations. SR also degrades. These pointwise results do not establish navigation success or safety under corruption.

## Protocol

For each scene, exactly round(p*N) observed BEV cells are selected by a SHA-256-derived fixed scene seed. Their modal semantic ID is independently sampled from the evaluated scenes' empirical observed-cell distribution, conditioned on differing from the clean ID. Masks are nested across p.

CAR/FAR eligibility and best direction come from the clean ontology map; only model inputs are recomputed from corrupted semantics.

All risk, hard-hazard, SDF, risk-gradient, and SDF-gradient fields are recomputed with the canonical `main` ontology after corruption. The controller weights and fold-specific clean calibration threshold remain frozen.

## Reproduce

```bash
python rebuttal_experiments/exp7_semantic_corruption.py
```

See `provenance.json` for fixed seeds, input hashes, checkpoints, and full configuration.

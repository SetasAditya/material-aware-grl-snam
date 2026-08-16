# Gap 3 — ground-truth versus predicted semantics

A lightweight LiDAR-cell semantic-risk predictor was trained on sequences `00000–00002` and evaluated on validation sequence `00003`. The sealed sequence `00004` was not loaded. The comparison uses the same frozen route-aware directional head and the same 450 balanced validation episodes.

This auxiliary predictor is a controlled perception-stress test, not a claim of state-of-the-art RELLIS semantic segmentation.

## Perception quality

- Observed-cell semantic-group accuracy: **0.623**.
- Mean group IoU: **0.400**.
- Train/validation scenes: **179/57**.

## Same-episode control metrics

| Map input | CAR (95% CI) | FAR (95% CI) | SR (95% CI) | Activation |
|---|---:|---:|---:|---:|
| Ground Truth | 0.725 [0.690, 0.756] | 0.196 [0.172, 0.221] | 2.020 [1.715, 2.459] | 0.279 |
| Predicted | 0.701 [0.666, 0.734] | 0.206 [0.181, 0.227] | 1.891 [1.626, 2.233] | 0.292 |

## Paired predicted-minus-ground-truth change

| Metric | Δ (95% CI) |
|---|---:|
| CAR | -0.024 [-0.052, +0.003] |
| FAR | +0.009 [-0.004, +0.021] |
| SR | -0.129 [-0.326, +0.046] |
| activation_rate | +0.014 [+0.002, +0.025] |

Eligibility and correct-direction labels always come from the ground-truth map; only the frozen head's semantic inputs and route context change. CIs use a paired episode-cluster bootstrap.

## Reproduce

```bash
python rebuttal_experiments/exp_predicted_semantics.py
```

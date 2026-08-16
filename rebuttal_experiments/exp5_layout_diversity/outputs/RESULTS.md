# Experiment 5 — Training-layout diversity

> **Scope limitation:** the available `rellis_stagewise_val1500_decision` bundle contains only RELLIS sequence `00000`. These results test held-out scene/layout generalization *within one sequence*, not cross-sequence generalization.

All variants use `CoefEnergyNetMaterial`, the same byte-identical initialization, the same seed/configuration, frozen geometry, and train only the risk encoder and lambda heads with the model's selectivity objective. The final equal-budget checkpoint after 3 epochs is evaluated on the same 180 validation episodes (60 per regime). Validation scenes are excluded from every training subset.

| Train scenes | Train episodes (R1/R2/R3) | CAR ↑ | FAR ↓ | SR ↑ | Train time (s) | Total/trainable params |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 80 (28/27/25) | 0.5000 | 0.8622 | 1.1090 | 6.5 | 207,254/105,682 |
| 25 | 200 (69/66/65) | 0.5000 | 0.8556 | 1.1091 | 8.0 | 207,254/105,682 |
| 50 | 400 (134/134/132) | 0.5000 | 0.8429 | 1.1099 | 13.8 | 207,254/105,682 |
| 100 | 687 (219/240/228) | 0.4815 | 0.8248 | 1.1154 | 24.4 | 207,254/105,682 |

CAR/FAR use force threshold `0.02` and the checkpoint-predicted $\lambda_{\mathrm{soft}},\lambda_{\mathrm{hard}}$ values.

## Interpretation

Increasing training layouts from 10 to 100 reduced held-out FAR from 0.862 to
0.825 and changed SR from 1.109 to 1.115. CAR did not improve (0.500 to
0.481). Thus this short, single-seed study supports only a modest improvement
in suppression with greater within-sequence layout coverage; it does not
establish improved activation or cross-sequence generalization.

Each checkpoint was evaluated on 2,671 force samples: 850 R1, 903 R2, and 918
R3. The CAR denominator contains the 54 R1 samples that satisfy the evaluator's
safe-alternative condition; the FAR denominator contains all 1,821 R2/R3
samples.

Reported training time covers optimizer epochs and the trainer's internal
validation passes, but excludes dataset construction and post-training force
evaluation. Parameter count is fixed across subsets.

The 1-epoch smoke checkpoints are retained separately under `training/*/smoke_1ep/`; they are validation artifacts and are not mixed into the reported equal-budget results.

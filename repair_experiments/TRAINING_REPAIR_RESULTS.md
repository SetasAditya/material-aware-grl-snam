# Behavioral soft-force training repair

## Outcome

The repaired training path is implemented and reproducible, but it does **not**
meet the preregistered validation CAR/effect criterion. It should not be
presented as a successful rescue of the headline result.

The final bounded recall attempt improves learned magnitude separation, but the
CAR remains far below the preregistered 0.65 target:

| Validation diagnostic | Default corrected run | Bounded recall run |
|---|---:|---:|
| Selected epoch | 5 | 6 |
| Active mean `lambda_soft` | 1.360 | 1.595 |
| Inactive mean `lambda_soft` | 0.962 | 1.216 |
| Active median nominal effect | 0.312 m | 0.382 m |
| Active fraction reaching 0.5 m nominal effect | 0.144 | 0.229 |
| CAR at physically admissible threshold | 0.144 | 0.229 |
| R2 activation | 0.036 | 0.089 |
| R3 activation | 0.053 | 0.153 |
| Pooled R2/R3 FAR | 0.044 | 0.121 |

The bounded attempt used the one prespecified stronger configuration:
`active_weight=8`, `effect_weight=12`, and `separation_weight=1`. Its
preregistration-aligned checkpoint selector maximized constrained validation
CAR at every epoch, tie-breaking by lower pooled FAR and then lower validation
loss. The selected threshold was `lambda_soft = 2.0015`.

Both comparisons enforce:

- `lambda_soft >= 2.0`, the minimum that nominally produces 0.5 m over 1.0 s
  at 0.5 m/cell;
- R2 activation at most 0.25;
- R3 activation at most 0.20; and
- pooled R2/R3 FAR at most 0.25.

The safety-output teacher diagnostics are:

| Frozen-head output | Default mean/max drift | Recall mean/max drift |
|---|---:|---:|
| `lambda_hard` | 0.0551 / 0.2920 | 0.0673 / 0.6200 |
| `mu_lat` | 0.00307 / 0.00790 | 0.00314 / 0.01484 |

These are output drifts caused by changing the shared risk representation even
though the hard/mu head parameters remain frozen. They are not a safety
certificate.

## Corrected method

The final training run:

- initializes from the exact historical checkpoint
  `rellis_stage2_decision_mid_ep12/best.pt`;
- freezes the geometry/goal/obstacle/fuser modules and the hard/mu head
  parameters;
- trains `risk_enc` and `lam_soft_head`;
- applies an explicit paired-displacement effect loss and inactive
  suppression;
- distills the original `lambda_hard` and `mu_lat` outputs;
- fits on sequences `00000`–`00002`;
- selects the checkpoint and calibrates magnitude on sequence `00003`; and
- refuses test-named manifests and sequence `00004`.

Training contained 31,517 decision points from 1,350 train episodes.
Validation contained 10,243 points from 450 episodes. The final bounded
attempt selected epoch 6 of 12 with constrained CAR 0.2287.

The immutable epoch files exposed a CPU tensor-aliasing issue in the original
in-memory best-state handoff. Both reported `best.pt` files were rematerialized
from their authoritative `epoch_XXX.pt` files and their validation predictions
were recomputed. The trainer now clones CPU state tensors when selecting future
checkpoints.

## Important unit correction

Earlier exploratory outputs in `behavioral_soft_force_full` used

`0.5 * lambda_soft * horizon_seconds^2`

as if controller coordinates were metres. Controller coordinates are BEV
pixels, so the corrected nominal displacement is

`0.5 * lambda_soft * horizon_seconds^2 * 0.5 m/cell`.

The corrected run also uses a realizable 1.0 s training horizon. Therefore the
minimum coefficient for the 0.5 m nominal-effect target is 2.0. Earlier
head-only and smoke outputs are retained as failed/unit-mismatched diagnostics,
not paper evidence.

The current evaluator's 12 steps at `dt=0.01 s` span only 0.12 s. At
`lambda_soft <= 5`, the maximum soft-only constant-acceleration displacement is
only `0.5 * 5 * 0.12^2 * 0.5 = 0.018 m`. A 0.5 m separation requirement is
analytically impossible on that window; this must be reported or the evaluation
horizon corrected before interpreting trajectory-separation results.

## Artifacts

- Trainer: `repair_experiments/train_behavioral_soft_force.py`
- Corrected checkpoint:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_corrected_full/best.pt`
- Final bounded-recall checkpoint:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_recall_full/best.pt`
- Bounded-recall summary and all per-epoch calibrated operating points:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_recall_full/summary.json`
- Full summary:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_corrected_full/summary.json`
- Per-decision validation predictions:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_corrected_full/validation_predictions.csv`
- Curves:
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_corrected_full/curves.csv`
- Failed frozen-representation capacity control:
  `repair_experiments/outputs/behavioral_soft_force_full/`

The corrected checkpoint strictly loads into `CoefEnergyNetMaterial` with no
missing or unexpected keys. The focused repair/split/controller suite passes:
**19 tests passed**.

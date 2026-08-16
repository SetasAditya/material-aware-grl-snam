# Gaps 1–2 — gate ablation and regime/phase analysis

This report uses the frozen 100-episode, same-checkpoint gate ablation and the corresponding decision traces. No controller parameter, episode, map, event, or seed differs between gate-on and gate-off.

## Gap 1: same-model gate-on/off by regime

The first row below is a **dynamic soft-channel exposure rate**. It is the closest rollout analogue of CAR/FAR, but it is not relabeled as static directional CAR/FAR because the trace does not contain a direction-correctness label at every step.

| Regime | Pairs | Gate-on exposure | Gate-off exposure | Paired Δ (95% CI) | Success on/off | Violation CVaR on/off |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 55 | 0.094 | 1.000 | -0.906 [-0.930, -0.880] | 1.000/1.000 | 2.327/2.330 |
| R2 | 24 | 0.061 | 1.000 | -0.939 [-0.963, -0.913] | 1.000/1.000 | 2.339/2.339 |
| R3 | 21 | 0.076 | 1.000 | -0.924 [-0.961, -0.879] | 1.000/1.000 | 2.434/2.434 |

The gate sharply reduces force exposure in every regime, including R2/R3 where exposure is undesirable. It does not create a resolved outcome benefit: success is identical, and the violation-CVaR changes are tiny relative to their paired intervals. This supports a mechanism/suppression claim, not an efficacy claim.

## Gap 2: activation and gate–execution mismatch by regime

| Regime | Step activation | Episodes with activation | Direction cosine | Clearance agreement | Hard disagreement | Risk-sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 0.089 | 42/55 | 0.700 | 0.574 | 0.227 | 0.505 |
| R2 | 0.064 | 18/24 | 0.692 | 0.607 | 0.238 | 0.619 |
| R3 | 0.065 | 16/21 | 0.735 | 0.446 | 0.386 | 0.518 |

The mismatch is not confined to one regime. R3 is worst on clearance agreement and hard-contact disagreement; R2 has the strongest predicted/realized risk correlation. The primitive remains evidence for activation, not a guarantee of the executed trajectory.

## Phase breakdown

| Phase | Activation | Direction cosine | Clearance agreement | Hard disagreement | Realized risk improvement |
|---|---:|---:|---:|---:|---:|
| pre-event | 0.036 | 0.706 | 0.301 | 0.340 | 0.025 |
| blocked/pre-opening | 0.057 | 0.779 | 0.088 | 0.877 | -0.199 |
| post-opening | 0.134 | 0.690 | 0.743 | 0.106 | 0.045 |

Pre-opening mismatch is strongly affected by horizons that cross the opening event (50/57 decisions). Post-opening is the cleanest phase, but endpoint and risk agreement remain incomplete.

## Reproduce

```bash
python rebuttal_experiments/analyze_gate_gaps.py
```

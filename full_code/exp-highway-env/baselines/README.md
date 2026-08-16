# Highway Baselines

This folder contains a first-pass baseline suite for `exp-highway-env`.

Implemented baselines:

- `constant_velocity`
- `idm`
- `mobil_idm`
- `safe_stop`
- `idm_follow_only`
- `idm_mobil`
- `ppo`
- `sac`
- `ppo_lagrangian`
- `sac_lagrangian`
- `cpo`
- `risk_aware_mpc`
- `chance_constrained_mpc`
- `cbf_qp_filter`
- `s1_model`
- `s2_model`

## Files

- `common.py`
  Shared baseline interface and action helpers.
- `rule_based.py`
  Constant-speed, conservative stopping, and IDM/MOBIL controllers.
- `learned.py`
  Stage 1 / Stage 2 checkpoint adapters, SB3 PPO/SAC wrappers, and generic
  continuous-policy loaders for exported safe-RL actors.
- `mpc.py`
  Receding-horizon risk-aware and chance-constrained controllers.
- `safety.py`
  A lightweight CBF-QP-style safety filter baseline.
- `registry.py`
  Baseline factory.
- `evaluate.py`
  CLI to run the baselines on the existing authored highway scenarios.
- `train_ppo.py`
  Optional SB3 PPO training entrypoint.
- `train_sac.py`
  Optional SB3 SAC training entrypoint.
- `train_ppo_lagrangian.py`
  In-repo PPO-Lagrangian training entrypoint with adaptive safety penalty.
- `train_sac_lagrangian.py`
  In-repo SAC-Lagrangian training entrypoint with adaptive safety penalty.
- `train_cpo.py`
  In-repo practical CPO-style trust-region trainer with rollback to feasible checkpoints.

## Example

Run rule-based and learned baselines:

```bash
cd /mnt/data/adityas/GRL-SNAM/exp-highway-env
python -m baselines.evaluate \
  --baselines constant_velocity idm mobil_idm risk_aware_mpc chance_constrained_mpc cbf_qp_filter s1_model s2_model \
  --s1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --s2-ckpt checkpoints/highway_stage2_mu_lat/best.pt \
  --episodes 5 \
  --out runs/baselines_eval.json
```

If you have a Stable-Baselines3 PPO checkpoint:

```bash
python -m baselines.evaluate \
  --baselines ppo sac \
  --ppo-ckpt checkpoints/highway_ppo_baseline/model.zip
  --sac-ckpt checkpoints/highway_sac_baseline/model.zip
```

Train the safe-RL baselines in repo:

```bash
python -m baselines.train_ppo_lagrangian --out checkpoints/highway_ppo_lagrangian_baseline
python -m baselines.train_sac_lagrangian --out checkpoints/highway_sac_lagrangian_baseline
python -m baselines.train_cpo --out checkpoints/highway_cpo_baseline
```

## Notes

- `idm` is the no-lane-change car-following baseline.
- `mobil_idm` is the lane-changing IDM/MOBIL baseline.
- `idm_mobil` uses highway-env's built-in `IDMVehicle`, which already combines
  IDM longitudinal behavior with MOBIL lane changes.
- `idm_follow_only` disables lane changes on the same `IDMVehicle` so it is a
  true follow-only baseline rather than a duplicate of `idm_mobil`.
- `risk_aware_mpc` is a lightweight in-repo receding-horizon controller, not an
  external MPC package.
- `chance_constrained_mpc` and `cbf_qp_filter` are lightweight in-repo
  benchmark baselines, not external control packages.
- `ppo` and `sac` are optional and require `stable_baselines3`.
- `ppo_lagrangian`, `sac_lagrangian`, and `cpo` now support in-repo `.zip`
  checkpoints from the training scripts above, while still accepting exported
  continuous actor checkpoints in TorchScript or simple PyTorch formats.
- `train_cpo.py` is a practical CPO-style trust-region trainer with rollback to
  the last feasible checkpoint, not a claim of a theorem-faithful canonical
  CPO implementation.

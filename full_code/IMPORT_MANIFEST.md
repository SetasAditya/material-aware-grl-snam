# Full-code import manifest

This directory preserves the paper-relevant source copied from:

```text
/mnt/data/adityas/GRL-SNAM
```

The source Git commit was:

```text
a28f85604a99975c973766cdce541be584420aab
```

The source checkout had local modifications and untracked research code. The
files here were copied from the working tree, not reconstructed from the Git
commit, so those local versions are intentionally preserved.

## Included

- `train_coef_energy.py`, `eval_coef_energy.py`, and `surrogate_robust.py`:
  the original coefficient-energy/Hamiltonian training and evaluation path.
- `train_material.py`, `eval_material.py`, and `eval_topdown_swarm.py`:
  material-aware training and evaluation.
- `src/` and `experiments/`: shared model utilities and earlier experiment
  drivers.
- `scripts/`: dataset generation, DFC2018 evaluation, planners, metrics, and
  learned-policy utilities.
- `exp-highway-env/`: highway data collection, surrogate dynamics, Stage 1/2
  training, CVaR episode costs, evaluation, force diagnostics, baselines, and
  paper-artifact scripts.
- `exp-highway-env/HighwayEnv/`: the local highway-env fork, including the
  custom scenario environment required by the highway experiments.
- `exp-rellis/`: RELLIS BEV construction, stagewise data, directional-force
  training, static/dynamic evaluation, robustness reports, and artifact
  generation. The small source/configuration portions of the upstream
  RELLIS-3D tooling are retained because the local pipeline refers to that
  checkout.

## Intentionally excluded

- Raw DFC2018 and RELLIS-3D datasets.
- BEV/data caches.
- Model checkpoints.
- Generated images, GIFs, videos, and bulk evaluation outputs.
- Python bytecode and `__pycache__` directories.
- Nested Git metadata and dated backup files.
- The duplicate `neurips2026_anonymous_code/` export, whose cleaned contents
  already exist at the repository root.

Some original scripts contain machine-specific example paths. They are
preserved verbatim for provenance and must be replaced with local paths before
running those entry points.

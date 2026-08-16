# Experiment 5: training-layout diversity

This experiment measures how Stage-2 static force selectivity changes as the
number of unique training scenes increases.

It uses the actual `CoefEnergyNetMaterial` model and `MaterialTrainer`
selectivity loss from `/mnt/data/adityas/GRL-SNAM/train_material.py`. It does
not use `train_rellis_directional_force.py` or the separate directional
classifier.

## Important scope limitation

The available `rellis_stagewise_val1500_decision` data contains only sequence
`00000`. This is therefore a **within-sequence held-out-layout** experiment,
not evidence of cross-sequence generalization.

The source manifest's original train and validation splits share nine scenes.
The experiment prevents layout leakage by preserving the original 180
balanced validation episodes and excluding all 28 scenes represented by them
from every training subset. This leaves 100 eligible training scenes. The
10%, 25%, 50%, and 100% scene subsets are deterministic, stratified by regime,
and nested.

No episode or scene tensors are copied. Generated manifests contain absolute
references to the original tensors.

## Reproduce

From the cleaned repository root:

```bash
python rebuttal_experiments/exp5_layout_diversity/run_layout_diversity.py \
  --phase all \
  --smoke-epochs 1 \
  --epochs 3 \
  --threads 4
```

The workflow:

1. Generates and audits deterministic subset manifests.
2. Creates one shared initialization by loading the same Stage-1 geometry
   weights into a fixed-seed `CoefEnergyNetMaterial`.
3. Runs an independent one-epoch smoke test for every subset.
4. Restarts every equal-budget run from the byte-identical shared
   initialization and trains for three epochs on CPU.
5. Evaluates final-checkpoint lambda outputs on the same held-out static
   force/selectivity episodes.
6. Writes `outputs/results.csv`, `outputs/results.json`, and
   `outputs/RESULTS.md`.

CAR is the fraction of eligible R1 force samples whose context force has a
positive projection exceeding `force_eps` along the best safe direction. FAR
is the fraction of R2/R3 samples whose force perpendicular to the geometric
scaffold exceeds `force_eps`. SR is the mean R1 perpendicular force divided by
the mean R2 perpendicular force, matching the repository evaluator.

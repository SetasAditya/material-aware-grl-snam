# Highway Stage 1/2 Experiments

This directory contains the highway-env branch of the project: the protected Stage 1 scaffold, the Stage 2 risk-enrichment pipeline, the paper-evaluation scripts, and the figures/tables used in `exp.tex`.

For the current handoff and paper-state summary, start with [HANDOFF_STAGE1.md](HANDOFF_STAGE1.md).

## Current Snapshot

Best paper-facing checkpoints:

```bash
Stage 1 scaffold:
checkpoints/highway_stage1_default_slow_x4/best.pt

Stage 2 paper checkpoint:
checkpoints/highway_stage2_mu_lat/best.pt
```

Best paper-facing JSONs already generated:

```bash
runs/paper_data/eval_paired_full.json
runs/paper_data/force_diagnostic.json
runs/paper_data/sim2sim_stage2_mu_lat.json
runs/paper_data/eval_stage2_mu_lat_disabled.json
```

Final figure set currently used in `exp.tex`:

```bash
figures/figure2_paired_rollout.pdf
figures/figure3_outcome_summary.pdf
figures_v3/figure4_paired_transitions.pdf
```

Current tables:

```bash
tables/table1_enrichment_ablation.tex
tables/table2_paired_comparison.tex
```

## Headline Highway Results

From `runs/paper_data/eval_paired_full.json`:

- `default`
  - crash `5% -> 0%`
  - off-road `95% -> 0%`
  - mean speed `24.66 -> 22.68 m/s`
  - lane-keep error `5.75 -> 0.17 m`
- `authored_slow_leader`
  - crash `100% -> 0%`
  - lane changes/ep `0.00 -> 1.00`
  - mean speed `23.87 -> 26.31 m/s`
- `authored_slow_leader_boxed`
  - both remain `100%` crash

The honest current claim is selective lateral passing in solvable passing contexts, not universal recovery in boxed geometries.

## Setup

Run commands from:

```bash
cd /mnt/data/adityas/GRL-SNAM/exp-highway-env
```

If `highway_env` is not importable:

```bash
python -m pip install -e HighwayEnv
```

Quick code sanity:

```bash
python -m py_compile \
  train_stage1.py train_stage2.py eval_stage1.py eval_stage2.py \
  eval_force_diagnostic.py sim2sim_consistency.py \
  make_paper_figures.py make_paper_tables.py
```

## Paper Artifact Pipeline

### Rebuild figures from existing JSONs

```bash
python make_paper_figures.py \
  --force-diagnostic runs/paper_data/force_diagnostic.json \
  --paired-eval runs/paper_data/eval_paired_full.json \
  --out figures
```

### Rebuild tables from existing JSONs / histories

```bash
python make_paper_tables.py \
  --paired-eval runs/paper_data/eval_paired_full.json \
  --frozen-history checkpoints/highway_stage2_mu_lat/history.json \
  --unfrozen-history checkpoints/highway_stage2_navscale/history.json \
  --out tables
```

### Full paper pipeline

```bash
bash run_paper_evals.sh
```

This script:

1. runs paired Stage 1 vs Stage 2 evaluation,
2. runs the per-step force diagnostic,
3. regenerates the paper figures,
4. regenerates the paper tables.

Useful overrides:

```bash
EPISODES=20 MAX_STEPS=120 DEVICE=cuda bash run_paper_evals.sh
EPISODES=5  MAX_STEPS=60  DEVICE=cpu  bash run_paper_evals.sh
```

## Important Scripts

Core training / eval:

- `train_stage1.py`
- `train_stage2.py`
- `eval_stage1.py`
- `eval_stage2.py`
- `collect_onpolicy.py`
- `onpolicy_buffer.py`

Paper artifact generation:

- `run_paper_evals.sh`
- `eval_force_diagnostic.py`
- `make_paper_figures.py`
- `make_paper_tables.py`

Matched-ablation wrappers:

- `train_stage2_frozen_geom_stressmix.py`
- `train_stage2_unfrozen_stressmix.py`
- `train_stage2_frozen_geom_stressmix_no_mu_lat.py`

Consistency / support diagnostics:

- `sim2sim_consistency.py`
- `analyze_stage1.py`

## Training Recipes

### Stage 1 scaffold

The paper-facing scaffold is already trained:

```bash
checkpoints/highway_stage1_default_slow_x4/best.pt
```

If you need the older protected IDM-like baseline for historical comparison:

```bash
checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt
```

### Stage 2 paper checkpoint

The current best paper checkpoint is:

```bash
checkpoints/highway_stage2_mu_lat/best.pt
```

It came from:

```bash
python train_stage2_frozen_geom_stressmix.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_mu_lat \
  --epochs 30 --warmup-frac 0.3 --bs 64 \
  --collect-max-steps 120 \
  --closed-loop-episodes 10 --closed-loop-every 10 \
  --device cuda
```

## Long GPU Runs Worth Doing

These are the remaining high-value ablations.

### 1. Matched no-`mu_lat` retrain

```bash
python train_stage2_frozen_geom_stressmix_no_mu_lat.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_no_mu_lat_matched \
  --epochs 30 --bs 64 --lr 3e-4 \
  --device cuda
```

Evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_no_mu_lat_matched/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_no_mu_lat_matched.json
```

### 2. Matched unfrozen-geometry retrain

```bash
python train_stage2_unfrozen_stressmix.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_unfrozen_matched \
  --epochs 30 --bs 64 --lr 3e-4 \
  --device cuda
```

Evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_unfrozen_matched/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_unfrozen_matched.json
```

### 3. Higher-budget Stage 2 rerun

```bash
python train_stage2_frozen_geom_stressmix.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_mu_lat_hi_budget \
  --epochs 40 --bs 64 --lr 3e-4 \
  --collect-episodes 12 \
  --closed-loop-episodes 20 \
  --device cuda
```

Evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_mu_lat_hi_budget/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_mu_lat_hi_budget.json
```

## Notes on Interpretation

- `force_diagnostic.json` is useful support data, but the main paper story now lives in the outcome-level figures.
- `eval_stage2_mu_lat_disabled.json` is an inference-time ablation, not yet a full matched no-`mu_lat` retrain result.
- `table1_enrichment_ablation.tex` is currently based on the frozen `mu_lat` history and the older unfrozen `navscale` history, so it should still be described as provisional.
- `figures_v3/figure4_paired_transitions.pdf` is the preferred Figure 4. The older `figures/figure4_paired_transitions.pdf` is superseded for the paper draft.

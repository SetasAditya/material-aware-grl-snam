#!/usr/bin/env bash
# run_paper_evals.sh — orchestrate evaluation runs for paper figures and tables.
#
# Idempotent: skips runs whose JSONs already exist. To force re-run, delete the
# output JSON first.
#
# Usage:
#   bash run_paper_evals.sh
#
# Configure paths below. All paths are relative to exp-highway-env/.

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit if your paths differ
# ─────────────────────────────────────────────────────────────────────────────

STAGE1_CKPT="checkpoints/highway_stage1_default_slow_x4/best.pt"
STAGE2_CKPT="checkpoints/highway_stage2_mu_lat/best.pt"

# Optional: unfrozen Stage 2 (e.g. the navscale run before freeze-geometry)
# Set to "" if you don't have one — Table 1 will note the missing data.
STAGE2_UNFROZEN_HISTORY="checkpoints/highway_stage2_navscale/history.json"
STAGE2_FROZEN_HISTORY="checkpoints/highway_stage2_mu_lat/history.json"

OUT_DIR="runs/paper_data"
EPISODES="${EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-120}"
DEVICE="${DEVICE:-auto}"
SCENARIOS="default authored_slow_leader authored_slow_leader_boxed"

mkdir -p "$OUT_DIR"

if [ "$DEVICE" = "auto" ]; then
  DEVICE="$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi

echo "[config] episodes=$EPISODES max_steps=$MAX_STEPS device=$DEVICE"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: paired Stage 1 vs Stage 2 eval (Table 2 input)
# ─────────────────────────────────────────────────────────────────────────────

PAIRED_OUT="$OUT_DIR/eval_paired_full.json"
if [ -f "$PAIRED_OUT" ]; then
  echo "[skip] $PAIRED_OUT exists"
else
  echo "[run]  paired Stage 1 vs Stage 2 across $SCENARIOS"
  python eval_stage2.py \
    --ckpt "$STAGE2_CKPT" \
    --stage1-ckpt "$STAGE1_CKPT" \
    --scenarios $SCENARIOS \
    --episodes $EPISODES --max-steps $MAX_STEPS \
    --device "$DEVICE" \
    --out "$PAIRED_OUT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: per-step force diagnostic (Figures 2, 3, 4 input)
# ─────────────────────────────────────────────────────────────────────────────

FORCE_OUT="$OUT_DIR/force_diagnostic.json"
if [ -f "$FORCE_OUT" ]; then
  echo "[skip] $FORCE_OUT exists"
else
  echo "[run]  per-step force diagnostic across $SCENARIOS"
  python eval_force_diagnostic.py \
    --stage1-ckpt "$STAGE1_CKPT" \
    --stage2-ckpt "$STAGE2_CKPT" \
    --scenarios $SCENARIOS \
    --episodes $EPISODES --max-steps $MAX_STEPS \
    --device "$DEVICE" \
    --out "$FORCE_OUT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: render figures
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR="figures"
echo "[run]  generating figures → $FIGURES_DIR/"
python make_paper_figures.py \
  --force-diagnostic "$FORCE_OUT" \
  --paired-eval "$PAIRED_OUT" \
  --out "$FIGURES_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: render tables
# ─────────────────────────────────────────────────────────────────────────────

TABLES_DIR="tables"
echo "[run]  generating tables → $TABLES_DIR/"

UNFROZEN_FLAG=""
if [ -n "$STAGE2_UNFROZEN_HISTORY" ] && [ -f "$STAGE2_UNFROZEN_HISTORY" ]; then
  UNFROZEN_FLAG="--unfrozen-history $STAGE2_UNFROZEN_HISTORY"
else
  echo "  note: unfrozen-Stage-2 history not found at"
  echo "        $STAGE2_UNFROZEN_HISTORY"
  echo "        Table 1 will note the missing data."
fi

python make_paper_tables.py \
  --paired-eval "$PAIRED_OUT" \
  --frozen-history "$STAGE2_FROZEN_HISTORY" \
  $UNFROZEN_FLAG \
  --out "$TABLES_DIR"

echo ""
echo "Done. Paper artifacts:"
echo "  Figures: $FIGURES_DIR/figure2_paired_rollout, figure3_outcome_summary, figure4_paired_transitions"
echo "  Tables:  $TABLES_DIR/table{1,2}_*.{tex,txt}"

#!/usr/bin/env bash
set -euo pipefail

# Render the two complementary highway mechanisms:
#   open adjacent lane -> activate lateral passing;
#   boxed adjacent lane -> suppress passing and apply TTC braking.
WORKSPACE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_ROOT="${GRL_SNAM_SOURCE_ROOT:-/mnt/data/adityas/GRL-SNAM}"
OUT_DIR="${1:-$WORKSPACE_ROOT/rq_visualizations/gifs/behavioral}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-highway-gifs}" \
python "$WORKSPACE_ROOT/full_code/exp-highway-env/render_paired_gif.py" \
  --stage1-ckpt "$SOURCE_ROOT/exp-highway-env/checkpoints/highway_stage1_default_slow_x4/best.pt" \
  --stage2-ckpt "$SOURCE_ROOT/exp-highway-env/checkpoints/highway_stage2_ttc_clear_smooth_g8_t35/best.pt" \
  --scenarios authored_slow_leader authored_slow_leader_boxed \
  --seed 1000 \
  --max-steps 70 \
  --config paper \
  --visualization-only \
  --hide-scenario-title \
  --x-window 60 \
  --out "$OUT_DIR" \
  --ttc-gain 8 \
  --ttc-threshold-s 3.5

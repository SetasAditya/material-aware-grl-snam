# Material-Aware GRL-SNAM

This repository contains the paper-facing implementation, rebuttal experiments,
compact result artifacts, and visualization tools for the material-aware
extension of GRL-SNAM. The method augments an inherited geometry-only
port-Hamiltonian navigation field with an explicit material-risk force, an
always-on hard-hazard force, and a local feasibility-witness gate controlling
when the material force is exposed.

The executed field is

```text
F = F_geom + g(context) * lambda_soft(context) * f_material
             + lambda_hard(context) * f_hazard
```

with highway-specific lateral and time-to-collision channels for deciding
between passing and braking. Raw datasets and large BEV caches are not
redistributed.

## Contents

- `rellis/`: compact reviewer-facing RELLIS-3D BEV construction, static selectivity evaluation,
  RELLIS-Dyn event generation/evaluation, and figure/table artifact scripts.
- `rellis/grl_rellis/`: local BEV, semantic-risk ontology, and dynamic event
  utilities.
- `scripts/baselines/dfc/`: local planner, metric, and model utilities shared
  by the RELLIS and DFC-style evaluations.
- `results/`: compact CSV/JSON outputs used for the main RELLIS-Dyn tables and
  missing-ablation diagnostics.
- `paper_generated/`: generated LaTeX table fragments and selected figures used
  in the submission.
- `full_code/`: imported full implementation, including DFC2018, RELLIS, and
  highway-env code. See `full_code/IMPORT_MANIFEST.md` for provenance and
  `full_code/README.md` for the original implementation entry points.
- `rebuttal_experiments/`: one-factor gate, witness, coefficient, perception,
  and paired-bootstrap experiments with tests and saved summaries.
- `repair_experiments/`: stagewise controller repairs, evaluation code, locked
  splits, and the frozen v7 artifact described in `FROZEN_V7_STATE.md`.
- `rq_visualizations/`: reproducible RQ1--RQ5 figures and closed-loop GIF
  renderers. Generated figures include per-file provenance records.
- `site/`: self-contained visualization page using the generated GIFs.

## Setup

Create a Python environment and install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from this directory with:

```bash
export PYTHONPATH="$PWD:$PWD/rellis"
```

The highway renderer additionally uses the vendored environment under
`full_code/exp-highway-env/HighwayEnv`. Some full experiment reruns require
PyTorch and domain-specific dependencies documented in `full_code/README.md`.

## Data

The raw RELLIS-3D dataset is not redistributed in this bundle. Download the
official RELLIS-3D LiDAR point clouds and semantic labels separately, then place
or symlink them under a local path such as:

```text
data/RELLIS-3D/
```

The scripts expect the standard RELLIS-3D sequence layout and a split file that
lists point-cloud/label pairs. The semantic-to-risk mapping used in the paper is
`rellis/grl_rellis/risk_ontology.yaml`.

## Reproducing The RELLIS Pipeline

Build BEV risk/SDF maps from RELLIS-3D:

```bash
python rellis/build_rellis_bev.py \
  --data-root data/RELLIS-3D \
  --split-file data/pt_val.lst \
  --out cache/rellis_bev \
  --ontology rellis/grl_rellis/risk_ontology.yaml
```

Sample local start-goal pairs and R1/R2/R3 regimes:

```bash
python rellis/sample_rellis_pairs.py \
  --bev-root cache/rellis_bev \
  --out cache/rellis_pairs \
  --target-per-regime 500 \
  --shuffle-scenes \
  --seed 0
```

Evaluate static RELLIS selectivity:

```bash
python rellis/eval_rellis_selectivity.py \
  --bev-root cache/rellis_bev \
  --pairs-root cache/rellis_pairs \
  --out runs/rellis_static_selectivity \
  --max-episodes 1500
```

Evaluate the three-event RELLIS-Dyn material subset:

```bash
python rellis/eval_rellis_dyn.py \
  --bev-root cache/rellis_bev \
  --pairs-root cache/rellis_pairs \
  --out runs/rellis_dyn_3event \
  --event-types mud_onset corridor_closes corridor_opens \
  --methods stage1 risk_loss_only dwa_semantic cbf_safety_filter route_aware_stage2 local_astar_budget mpc_budget oracle_replanner \
  --max-episodes 100 \
  --progress-every 10
```

Evaluate the delayed-required-escape diagnostic:

```bash
python rellis/eval_rellis_dyn.py \
  --bev-root cache/rellis_bev \
  --pairs-root cache/rellis_pairs \
  --out runs/rellis_dyn_delayed_required \
  --event-types delayed_required_escape \
  --methods blackbox_cvar stage2_expected_cost fixed_coeff_stage2 route_aware_stage2 dwa_semantic \
  --max-episodes 100 \
  --progress-every 10
```

Generate RELLIS-Dyn paper tables and figures from completed runs:

```bash
python rellis/make_rellis_dyn_artifacts.py \
  --bev-root cache/rellis_bev \
  --pairs-root cache/rellis_pairs \
  --fast-run runs/rellis_dyn_3event \
  --delayed-required-run runs/rellis_dyn_delayed_required \
  --out generated/results \
  --tex-out generated/tables \
  --paper-figures generated/figures
```

## Included Result Artifacts

The `results/` directory includes compact outputs used to form the reported
tables:

- `results/rellis_missing_ablation_results/static_table1_with_missing_rows.csv`
- `results/rellis_missing_ablation_results/dyn_table2_with_missing_rows.csv`
- `results/rellis_missing_ablation_results/delayed_required_with_missing_rows.csv`
- `results/rellis_missing_ablation_results/delayed_required_false_preact_100.csv`
- `results/rellis_dyn_missing_3event_100/dynamic_main_table.csv`
- `results/rellis_dyn_missing_delayed_required_100/dynamic_main_table.csv`

These files are included so reviewers can inspect the exact aggregate values
without rerunning the full dataset pipeline.

## Rebuttal And Repair Tests

Run the lightweight experiment tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider \
  rebuttal_experiments repair_experiments
```

The frozen controller, retained files, limitations, and hashes are documented
in `FROZEN_V7_STATE.md`. In particular, the repaired v7 controller did not pass
its efficacy stage gate; its artifacts are retained for reproducibility and
must not be presented as a positive held-out result.

## RQ1--RQ5 Figures

Generate all static research-question figures with:

```bash
MPLCONFIGDIR=/tmp/material-aware-mpl \
python rq_visualizations/make_all.py
```

Outputs are written to `rq_visualizations/output/` as PDF, PNG, and provenance
JSON. See `rq_visualizations/README.md` for the exact measured-versus-rendered
scope of every panel.

## Behavioral GIFs

Generate the paired RELLIS-Dyn examples:

```bash
MPLCONFIGDIR=/tmp/material-aware-mpl \
python rq_visualizations/gifs/make_behavioral_rellis_gifs.py
```

Generate the enlarged highway comparisons (open passing lane and boxed lane):

```bash
bash rq_visualizations/gifs/make_behavioral_highway_gifs.sh
```

Generate the same-event comparison against semantic DWA, MPPI, and budgeted
MPC:

```bash
MPLCONFIGDIR=/tmp/material-aware-mpl \
python rq_visualizations/gifs/make_planner_comparison_gif.py
```

The generated GIFs are in `rq_visualizations/gifs/behavioral/`. They show
selected closed-loop examples and should accompany, not replace, the paired
aggregate statistics in the manuscript. Detailed generation notes and claim
boundaries are in `rq_visualizations/gifs/README.md`.

To preview the included visualization page locally:

```bash
python -m http.server 8000 --directory site
```

## Notes On Scope

The compact root pipeline remains the simplest reviewer entry point. The
`full_code/` tree is included for implementation completeness but retains its
original experiment-specific organization. Large external datasets, local
caches, and most training checkpoints remain excluded. Saved visualizations are
qualitative examples; quantitative claims should be taken from paired result
tables and their confidence intervals.

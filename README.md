# Anonymous Submission Code

This folder contains the code and compact result artifacts needed to reproduce
the RELLIS-3D and RELLIS-Dyn experiments reported in the submission. It is a
cleaned reviewer bundle, not a full research checkout. Raw datasets, large
caches, and checkpoints are intentionally not included.

## Contents

- `rellis/`: RELLIS-3D BEV construction, static selectivity evaluation,
  RELLIS-Dyn event generation/evaluation, and figure/table artifact scripts.
- `rellis/grl_rellis/`: local BEV, semantic-risk ontology, and dynamic event
  utilities.
- `scripts/baselines/dfc/`: local planner, metric, and model utilities shared
  by the RELLIS and DFC-style evaluations.
- `results/`: compact CSV/JSON outputs used for the main RELLIS-Dyn tables and
  missing-ablation diagnostics.
- `paper_generated/`: generated LaTeX table fragments and selected figures used
  in the submission.

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

## Notes On Scope

This bundle focuses on the RELLIS-3D static and dynamic material-risk
experiments because they are the primary code-heavy additions in the submission.
The raw RELLIS data, large BEV caches, trained checkpoints, and full LaTeX
source are excluded for size and anonymity. The included scripts are sufficient
to rebuild caches, rerun the local evaluations, and regenerate the RELLIS tables
and figures once the external dataset is available.

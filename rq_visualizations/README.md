# RQ1--RQ5 comparative visualizations

These scripts turn the saved rebuttal artifacts into publication-quality PDF
and PNG figures. They use measured traces and paired results; when a spatial
trajectory was not logged, they use counterfactual distributions rather than
reconstructing or inventing a path. See "Rendering the policy in action" below
for what is reconstructed exactly and what is not available.

Run all figures from the repository root:

```bash
python rq_visualizations/make_all.py
```

Outputs are written to `rq_visualizations/output/`. Each figure has a PDF, a
300-dpi PNG, and a JSON provenance file containing the source paths and hashes.

## Rendering the policy in action

RQ1 and RQ2 render decisions spatially rather than only as summary bars. The
background is the `delayed_required_escape` event field rebuilt by
`common.event_field()`, which calls the simulator's own mask helpers
(`grl_rellis.dyn_events._barrier_mask` / `._ellipse_mask`) on the geometry saved
in `event_specs.csv`, so the drawn shapes match what the rollout applied. Drawn
on top are logged quantities only: executed path, feasibility witness ray,
selected primitive direction, gate decision, and feasible primitive count.

Two limits are deliberate and are stated on the figures:

- **The base risk map is not recoverable.** The per-scene BEV cache
  (`cache/rellis_bev_*`, `cache/rellis_pairs_*`) is absent from this snapshot,
  so only the event-induced structure is exact. The scene-dependent component
  is drawn as neutral ground rather than given an invented value, and the field
  is shown as labelled categorical layers instead of a continuous risk colormap.
- **The gate does not move the robot.** Across all 100 episodes of
  `exp1_gate_ablation_100`, the maximum gate-on vs gate-off path divergence is
  0.33 cells (median 0.0002), so no paired spatial comparison exists to draw.
  The gate modulates soft-force *exposure*, not route. RQ1 panel F reports this
  divergence distribution directly instead of implying a route change.

RQ4 panel B is also spatial: it overlays all 100 logged gate-on rollouts in the
shared BEV frame and highlights the worst-decile episodes by path-integrated
risk exposure, so the CVaR tail is visible as geometry. The 100 episodes span 13
scenes with different event geometry, so no field is drawn behind them; only the
paths are comparable.

RQ3 and RQ5 stay non-spatial because no position data exists for them.
`exp9_soft_coefficient_isolation/rollout_coefficients.csv` (RQ3) has one row per
rollout arm with coefficients only, and `exp7_semantic_corruption`'s
`raw_predictions.csv` (RQ5) is per decision point with no pose. Rendering RQ3
spatially would require re-running `_roll_case`
(`rellis/make_rellis_dyn_artifacts.py`), which needs the missing BEV cache and
trained checkpoints. RQ5 instead uses its 52,113 decision points matched across
all four corruption levels, which is the more direct evidence for that claim.

Individual scripts:

- `rq1_gate_exposure.py`: four-phase delayed-escape storyboard over the
  reconstructed event field, exposure timeline, paired path-divergence
  distribution, and paired exposure intervention.
- `rq2_witness_execution.py`: logged witness rays vs executed steps on the
  field, the witness/execution angle distribution over all gate-positive
  decisions, and measured agreement of direction, clearance, contact, and risk
  sign.
- `rq3_soft_channel.py`: paired one-factor `lambda_s` intervention on DFC2018
  and RELLIS-3D.
- `rq4_adaptation_cvar.py`: cross-domain coefficient distributions, the spatial
  CVaR-tail overlay, and the auditable 100-episode expected-cost/CVaR
  intervention. This intentionally does not use the manuscript's currently
  unsupported `0.180/0.810` values.
- `rq5_perception_robustness.py`: corruption curves, the CAR/FAR selectivity
  ratio, and the matched decision-fate comparison over all 52,113 decision
  points tracked across every corruption level. Activations that were correct
  when clean and those that were false when clean decay almost identically
  (0.414 vs 0.446 retained at 30% corruption), which is the evidence that the
  failure mode is conservative rather than selective.

The scripts accept command-line overrides for their input and output paths;
run any script with `--help` for details.

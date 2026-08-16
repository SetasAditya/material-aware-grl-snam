# Experiment 8: fixed semantic APF baseline

The former `neural_potential_field` baseline is now accurately named `semantic_apf`. It contains no neural network, learned weights, or training. It performs one-step descent over eight neighboring cells using fixed goal, semantic-risk, clearance, and progress terms.

Delayed-required-escape results use 100 paired episodes. Intervals are episode bootstrap 95% confidence intervals (10,000 resamples; seed 27370).

| Method | Success | Pre-open deviation proxy | Violation CVaR | Hard length (m) | Path ratio | Stuck |
|---|---:|---:|---:|---:|---:|---:|
| Semantic APF | 0.110 [0.050, 0.180] | 0.260 [0.180, 0.350] | 0.916 [0.895, 0.935] | 0.033 [0.007, 0.064] | 1.738 [1.659, 1.817] | 0.900 [0.840, 0.950] |
| Route-aware Stage 2 | 0.610 [0.520, 0.710] | 0.380 [0.290, 0.480] | 0.840 [0.808, 0.872] | 0.059 [0.021, 0.107] | 1.386 [1.316, 1.462] | 0.480 [0.380, 0.580] |
| Semantic DWA | 0.480 [0.380, 0.580] | 0.950 [0.900, 0.990] | 0.695 [0.644, 0.750] | 0.062 [0.005, 0.149] | 1.499 [1.432, 1.566] | 0.590 [0.490, 0.690] |

The pre-open deviation measure is only a trajectory proxy: whether the rollout deviates more than 1 m from the nominal route before the escape opens. The APF and DWA have no internal activation gate.

The historical eight-event sweep is included only as a preliminary descriptive comparison. It was generated before the rename, but the controller implementation is the same fixed baseline; copied rows are canonicalized to `semantic_apf`.

Limitations: this is a lightweight grid APF comparison, not an RMP or Geometric Fabric implementation; its weights were not tuned in a nested validation protocol; and it follows the next Stage-1 waypoint, so it is not independent of the nominal scaffold.

Machine-readable results: `aggregate_metrics.csv`, `paired_differences_vs_semantic_apf.csv`, `historical_8event_preliminary.csv`, and `provenance.json`.

## Reproduction

```bash
python full_code/exp-rellis/eval_rellis_dyn.py \
  --bev-root /mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_bev_all_seqbalanced_2500 \
  --pairs-root /mnt/data/adityas/GRL-SNAM/exp-rellis/cache/rellis_pairs_all_seqbalanced_2500_loso \
  --out rebuttal_experiments/results/exp8_semantic_apf_delayed \
  --event-types delayed_required_escape \
  --methods semantic_apf route_aware_stage2 dwa_semantic \
  --max-episodes 100 --progress-every 10

python rebuttal_experiments/summarize_exp8_semantic_apf.py \
  --delayed-run rebuttal_experiments/results/exp8_semantic_apf_delayed \
  --historical-8event-run /mnt/data/adityas/GRL-SNAM/exp-rellis/runs/rellis_dyn_8events_fast_100 \
  --out rebuttal_experiments/results/exp8_semantic_apf_delayed \
  --bootstrap-samples 10000 --seed 27370
```

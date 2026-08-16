# Experiment 4: Primitive-count sensitivity

The same checkpoint, first 100 LOSO episodes, dynamic events, seed, and rollout settings are used for every value of `K`. Only the number of uniformly spaced gate-witness directions changes.

| K | Activation rate [95% CI] | False-pre episode rate | Post-open miss | Success | Violation CVaR | Hard contacts | Cosine | Clearance agree | Hard disagree | CPU ms/step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.025 [0.016, 0.034] | 0.030 | 0.660 | 1.000 | 2.3520 | 12.43 | 0.726 | 0.559 | 0.193 | 7.98 |
| 8 | 0.063 [0.048, 0.080] | 0.150 | 0.380 | 1.000 | 2.3520 | 12.43 | 0.712 | 0.579 | 0.233 | 7.67 |
| 16 | 0.078 [0.062, 0.095] | 0.170 | 0.270 | 1.000 | 2.3520 | 12.43 | 0.705 | 0.556 | 0.259 | 8.02 |
| 32 | 0.090 [0.072, 0.110] | 0.200 | 0.250 | 1.000 | 2.3520 | 12.43 | 0.689 | 0.548 | 0.275 | 8.67 |

## Main finding

Increasing K from 4 to 32 raises activation from 0.025 to 0.090, reduces post-opening misses from 0.660 to 0.250, but increases false-preactivation episodes from 0.030 to 0.200. Success, hard contacts, and violation CVaR are unchanged at reported precision. Meanwhile, directional cosine decreases and hard-contact disagreement rises, so added primitive coverage makes the gate more permissive without improving this checkpoint's outcomes.

## Permissiveness and sharing validation

At an identical state the direction sets are nested (`K=4 ⊂ 8 ⊂ 16 ⊂ 32`), so adding candidates cannot make the witness test less permissive. Closed-loop monotonicity is reported separately because trajectories can diverge after activation.

```json
{
  "monotonicity": {
    "candidate_sets_nested_at_identical_state": true,
    "closed_loop_activation_rate_nondecreasing": true,
    "closed_loop_false_pre_episode_rate_nondecreasing": true,
    "closed_loop_post_open_miss_rate_nonincreasing": true,
    "episodes_with_nondecreasing_activation_rate": 100,
    "n_episodes": 100,
    "caution": "Nested directions imply monotonic permissiveness only at the same state. Closed-loop trajectories diverge after K-dependent gates."
  },
  "gateoff_invariance": {
    "tested_k": [
      4,
      32
    ],
    "tested_episodes": 2,
    "trace_rows_each": 115,
    "compared_non_gate_trace_fields": 20,
    "non_gate_trace_mismatches": 0,
    "outcome_metric_mismatches_excluding_compute_and_gate_diagnostic": 0,
    "validated": true,
    "sharing": "gate-off outcomes are taken once from exp1_gate_ablation_100 (K=16); K changes diagnostic gate computation but cannot enter lam_soft_used=lam_soft_learned or the integrator"
  }
}
```

Gate-off outcomes are shared rather than recomputed for every K. The preserved K=4/K=32 smoke validation compares every non-gate trajectory field and every outcome metric (excluding timing and diagnostic rate) exactly.

## Provenance

```json
{
  "ks": [
    4,
    8,
    16,
    32
  ],
  "max_episodes": 100,
  "bootstrap_reps": 5000,
  "bootstrap_seed": 27370,
  "checkpoint_sha256": "7ccb38ca448f56946f07c882c8efc8d5e193dd085e2d898d26b7044ff96dc090",
  "per_k": {
    "4": {
      "config_sha256": "45220edd90d5f76b83295edcf35eb239bf640631f1a4420e2aef0d68dc246af1",
      "metrics_sha256": "a365fac99d9d46709288acfe8ddeb887cfebeb25c5e63fee6e614c98b0149b60",
      "traces_sha256": "7c4b43795c46cf1253f34954a36cb3cb3268b2b8f892862615d06f12ae677332"
    },
    "8": {
      "config_sha256": "0fc6ec8c8ac4b3187c1bd602b95688a78e6362f71d460f0fa373f93163d7d790",
      "metrics_sha256": "c4cbd738b2ffc43b400e3ee8088118b9fea29a947b6d3417dbf73bd47d3373e4",
      "traces_sha256": "8cdf309f459488352bd436a529525751bfe8ed8379a29f42dcfbc8aaeb86ba44"
    },
    "16": {
      "config_sha256": "66a8ae6589852c18298b61f1e1610257e49311fb157c4ab4668d2ab0ea35e58a",
      "metrics_sha256": "f1417d10366c5471b517fb8174bf49b71f5caaa15c2f3dd8f01504362f49ae4c",
      "traces_sha256": "cc21bf3fc96534990804d9224917fab4b2b4fdb6e9f978c1ea13eb7e0a92f079"
    },
    "32": {
      "config_sha256": "f5d3fe6faf34739d2ce1edb81fa31c9b9102a2744b6c326810c7b6d23eb2ba97",
      "metrics_sha256": "075fa7b25ea63c760938679cf7046a2e78e1d447feb58fb274bdf5a6faa59e40",
      "traces_sha256": "6b008315613e40da43480579b7363a5e692f863107bd7200fd04ba7b8fcf983d"
    }
  },
  "paired_uncertainty": "episode bootstrap; paired K-minus-K16 differences resample the same episode indices",
  "shared_gateoff_reference": {
    "path": "rebuttal_experiments/results/exp1_gate_ablation_100/per_episode_metrics.csv",
    "sha256": "8ca107e5ff7d08e53f9139641eec2925f1b3d2ebd61eca49c8eb801d49c7d05a",
    "role": "single K-invariant gate-off reference; per-K table reports gate-on sensitivity only"
  }
}
```

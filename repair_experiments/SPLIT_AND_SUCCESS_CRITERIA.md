# Frozen RELLIS split and success criteria

Status: **preregistered before repaired-model tuning**. The held-out test
sequence is sealed and has not been evaluated by this split-generation work.

## 1. Leakage-free data split

The repaired-method study uses the canonical five-sequence, balanced pair
manifests from `rellis_pairs_all_seqbalanced_2500_seqXXXXX`. Each sequence
contains 450 episodes: 150 each in R1, R2, and R3.

| Role | RELLIS sequences | Static episodes | Permitted use |
|---|---:|---:|---|
| Train | `00000`, `00001`, `00002` | 1,350 | Fit model parameters |
| Validation | `00003` | 450 | Select checkpoints and calibrate all thresholds |
| Test | `00004` | 450 | One final evaluation after the configuration is frozen |

This is a whole-sequence split. No frame, scene, or sampled episode can occur
in more than one partition. It is stronger than a random episode split because
temporally adjacent frames from one traversal cannot leak across roles.

The choice is fixed by sequence identifier, not by observed method outcomes:
the first three canonical sequences train the model, the fourth performs model
selection, and the fifth is held out. All five source manifests are already
balanced by R1/R2/R3, so no outcome-dependent resampling is needed.

For dynamic evaluation, every base episode is paired with each of the nine
event types frozen in `grl_rellis.dyn_events.MAIN_EVENT_TYPES`. This Cartesian
construction gives equal counts for every event type and, within each event,
equal counts for R1/R2/R3. Event seeds are deterministic hashes of the base
episode and event identity.

Run:

```bash
python repair_experiments/generate_rellis_splits.py
```

The command generates:

- `splits/train_static.json`, `splits/validation_static.json`, and
  `splits/test_static.json`;
- corresponding `*_dynamic.json` indexes;
- `splits/SPLIT_LOCK.json`, which records source and output SHA-256 hashes and
  the zero-overlap audit. It also hashes the canonical 2,500-scene BEV
  manifest and verifies that every sampled pair references a scene in it.

The full episode payload is in each static manifest. A dynamic item references
its `base_episode_uid` in that split's static manifest. Repaired training and
evaluation code must use `episode_uid`, not a row number.

## 2. Test-set sealing policy

Until the configuration is frozen, no experiment may load:

- `test_static.json` or `test_dynamic.json`;
- sequence `00004` BEV tensors;
- any prior outcome file or checkpoint evaluation derived from sequence
  `00004`.

All model architecture, loss weights, checkpoint selection, projection rule,
gate thresholds, hysteresis, dwell time, random seeds, and metric code must be
chosen using train and validation only. Before the one-shot test run, record
their file hashes beside `SPLIT_LOCK.json`. A failed test criterion is reported
as a failed criterion; the test result must not initiate another tuning cycle.

Historical sequence-`00004` results in the repository are not admissible for
model selection in this repaired study.

## 3. Methods frozen for the primary comparison

The primary repaired method is the learned Hamiltonian controller with:

1. a learned, behaviorally non-negligible soft-material force;
2. feasible-cone projection toward the gate-selected primitive;
3. a stateful gate with hysteresis and minimum dwell time.

The matched controls use the same checkpoint, integrator, observations,
episodes, and random seeds:

- **gate off:** apply the same learned soft-force output without the feasibility
  gate or feasible-cone projection;
- **stateless gate:** use the same learned controller and projection without
  temporal hysteresis or dwell;
- **geometry only:** set the soft-material channel to zero.

No heuristic `route_aware_stage2` rollout may be labeled as a learned
Hamiltonian result.

## 4. Primary validation metrics and go/no-go criteria

The following criteria are fixed before tuning. Report paired episode-level
95% bootstrap confidence intervals, clustered by `scene_id`. Point estimates
must satisfy every safety/mechanism criterion and at least four of the five
behavior criteria to justify the one-shot test evaluation.

### Safety and mechanism criteria — all required

| Criterion | Validation threshold |
|---|---:|
| Hard-contact rate versus geometry-only | no worse by more than 2 percentage points |
| Violation-CVaR versus geometry-only | no worse by more than 0.05 absolute |
| Median cosine alignment: selected primitive vs. executed displacement | at least 0.70 |
| Executions retaining the selected primitive's clearance condition | at least 90% |
| Gate-positive executions realizing the predicted risk reduction | at least 70% |

The alignment horizon is the same horizon used to certify the primitive.
Zero-displacement windows count as failed alignment. Clearance and risk
improvement are recomputed from the executed trajectory, not inherited from
the primitive.

### Behavioral criteria — at least four of five required

| Criterion | Validation threshold |
|---|---:|
| Static R1 correct activation rate (CAR) | at least 0.65 |
| Static R2 false activation rate (FAR) | at most 0.25 |
| Static R3 activation rate | at most 0.20 |
| Delayed-required-escape post-opening success | at least 0.70 |
| Delayed-required-escape false pre-activation | at most 0.25 |

An activation is counted from the actual context-force command after all
thresholding, not from a diagnostic predicate.

### Evidence that the gate changes the learned controller

For gate-on versus gate-off, report the paired trajectory endpoint distance
over a fixed 1.0-second behavioral-effect horizon. At the frozen control
interval `dt = 0.01 s`, this is 100 control steps. At least 25% of windows in
which their gate states differ must separate by one BEV cell (0.5 m) or more.
Otherwise the soft channel is still behaviorally negligible and the claimed
gate effect is not supported, even if the headline task metrics happen to
pass.

**Preregistration amendment (before repaired-checkpoint validation or test):**
the original text used the 12-step primitive-certification horizon for this
behavioral-effect check. An analytical unit audit showed that this made the
0.5 m threshold physically unreachable: with `lambda_soft <= 5` and
`dt = 0.01 s`, the maximum soft-only displacement over 12 steps is about
0.018 m. The mechanism checks—primitive/execution alignment, clearance
retention, and realized risk reduction—remain at their original 12-step
primitive horizon. Only the paired endpoint-separation check now uses the
1.0-second horizon. This correction was fixed before inspecting any repaired
checkpoint validation result, and the held-out test remains sealed.

Rollouts that reach the success radius before the 1.0-second endpoint use an
absorbing terminal state: their recorded terminal position is carried forward
to the requested endpoint time. For a failed, timed-out, or otherwise
truncated arm without the full horizon, the paired window remains in the
denominator and is conservatively assigned zero separation. Outputs record
whether an endpoint was directly observed, terminal-carried, or failure
imputed. This terminal-state rule was also fixed before repaired-checkpoint
validation or test.

### Temporal-stability criterion

Relative to the stateless gate, the stateful gate must reduce activation-state
transitions per rollout by at least 30%, while increasing median post-opening
reaction delay by no more than one control step. Immediate hard-hazard override
events are excluded from the dwell-time constraint but included in safety
metrics.

## 5. Reporting rules

- Report the static metrics by R1/R2/R3 and the dynamic metrics by event type;
  do not pool away a failed regime.
- Report denominators, point estimates, and clustered paired confidence
  intervals.
- Keep all failed and timed-out episodes in the denominator.
- Generate tables directly from immutable per-episode CSV/JSONL outputs.
- Report every preregistered criterion, including failures.
- Do not replace the primary thresholds after observing validation or test
  outcomes. Additional analyses must be labeled exploratory.

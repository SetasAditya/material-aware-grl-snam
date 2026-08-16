# Highway Stage 1/2 + NeurIPS Paper Handoff

Date: 2026-05-02

This handoff is the current highway-env and NeurIPS paper-artifact snapshot. It replaces the older `navscale`-centered handoff. The repo now has a stable highway paper checkpoint, generated highway figures/tables, DFC figures imported from the thesis tree, and a NeurIPS draft whose intro, related work, method, and experiments have been updated around the spotlight claim.

## Current Goal

The immediate goal is not more Stage 1 tuning. It is:

1. keep the current paper checkpoint and artifact pipeline stable,
2. use the existing JSONs/figures/tables in the paper draft,
3. optionally improve reviewer-defensibility with longer matched ablation retrains on GPU,
4. resolve the boxed slow-leader failure in a way that strengthens, rather than weakens, the spotlight claim.

For the NeurIPS draft, the immediate goal is also:

1. keep the story centered on context-conditioned Hamiltonian enrichment,
2. highlight that sensed context changes active objectives, barriers/constraints, and effective action degrees of freedom,
3. avoid DFC-only method framing in the main method section,
4. keep the highway-env result honest: the stock paper checkpoint still only guarantees selective lateral passing when a pass corridor exists,
5. evaluate whether boxed recovery should enter the paper as a deployment override result or as a new context-conditioned longitudinal energy term,
6. determine whether the retrained TTC checkpoint is strong enough to replace the stock paper checkpoint in the writeup.

## NeurIPS 2026 Paper Draft Status

Main source:

```bash
Master_s_Thesis/NeurIPS_2026/neurips_2026.tex
```

Active section files:

```bash
Master_s_Thesis/NeurIPS_2026/tex/intro.tex
Master_s_Thesis/NeurIPS_2026/tex/related_works.tex
Master_s_Thesis/NeurIPS_2026/tex/method.tex
Master_s_Thesis/NeurIPS_2026/tex/exp.tex
```

Current compiled PDF:

```bash
Master_s_Thesis/neurips_2026.pdf
```

The draft currently compiles with:

```bash
cd /mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis
TEXINPUTS=NeurIPS_2026//: pdflatex -interaction=nonstopmode -halt-on-error NeurIPS_2026/neurips_2026.tex
```

Known compile warnings:

- hyperref warning from math in a method subsection title;
- overfull boxes in `NeurIPS_2026/tex/exp.tex` around the experiment tables;
- shell-level `dconf` warnings from the environment, harmless for PDF output.

The last successful compile produced an 18-page PDF.

### NeurIPS Story State

The paper framing has been updated to emphasize:

- dynamic, context-conditioned objectives rather than one fixed reward;
- changing active barrier/constraint sets under sensed local fields;
- a product space of force/action factors, where the sensed field determines which coarse degrees of freedom are useful;
- frozen Stage 1 geometry as the scaffold, with Stage 2 adding context/risk/lateral channels;
- interpretability through force decomposition and selective activation, not just lower crash rate.

The method section has been rewritten away from DFC-only language. It now uses `H_ctx` / `F_ctx` for the new context-conditioned Hamiltonian term and force channel, rather than `H_mat` / `F_mat`. The setup permits DFC material context and highway interaction context under the same local-patch abstraction.

Removed from the active method section:

- DFC2018 Houston/oracle-specific method paragraphs;
- placeholder footnotes and Appendix placeholders;
- stale commented duplicate method draft;
- old material-only force terminology in active equations.

The enrichment proposition is now self-contained with an inline proof sketch instead of relying on an Appendix placeholder.

### NeurIPS Figures Currently Used

Method/overview figures imported from the thesis figure tree:

```bash
Master_s_Thesis/figs/teaser_decision_field.png
Master_s_Thesis/figs/ch4_factored_hamiltonian.png
Master_s_Thesis/figs/ch4_three_loops.png
Master_s_Thesis/figs/overview.png
Master_s_Thesis/figs/cum_risk.png
```

Highway figures copied into the NeurIPS figure tree:

```bash
Master_s_Thesis/NeurIPS_2026/figures/figure2_paired_rollout.png
Master_s_Thesis/NeurIPS_2026/figures/figure3_outcome_summary.png
Master_s_Thesis/NeurIPS_2026/figures/figure4_paired_transitions.png
```

Interpretation:

- `figure2_paired_rollout.png`: slow-leader storyboard plus lateral-position / clearance traces; this replaced the earlier uninformative trajectory-only plot.
- `figure3_outcome_summary.png`: scenario-level Stage 1 vs Stage 2 outcome summary.
- `figure4_paired_transitions.png`: paired transition/snapshot figure that makes the highway mechanism interpretable.

The older root-level highway PDFs remain useful for artifact generation, but the NeurIPS draft currently references the PNG copies above.

## DFC Baseline / Eval Status

The DFC side now has a unified baseline evaluator:

```bash
scripts/eval_dfc_baselines.py
```

Supporting code lives in:

```bash
scripts/baselines/dfc/
```

This evaluator now supports both planner baselines and the learned DFC models
under one metric suite.

### DFC Baselines Currently Implemented

Classical / planner baselines:

- `blind_dijkstra`
  - shortest path with no risk awareness
- `geometry_astar`
  - geometry-only A*
- `risk_weighted_astar`
  - geometry-blocked A* with soft-risk traversal cost
- `oracle_astar`
  - hard hazards blocked, soft risk penalized
  - strongest currently implemented planner reference / ceiling

Stronger added baselines:

- `cvar_costmap_astar`
  - local CVaR / tail-risk costmap A* approximation
- `chance_constrained_mpc`
  - receding-horizon chance-constrained / risk-aware MPC-style approximation
- `ppo_lagrangian`
  - lightweight constrained policy-optimization approximation

Our method / ablation:

- `s1_model`
  - Stage 1 geometry-only model
- `s2_model`
  - Stage 2 material-aware model

Important note for the paper:

- `cvar_costmap_astar`, `chance_constrained_mpc`, and `ppo_lagrangian` are
  local faithful approximations implemented in this repo, not imported
  official third-party repos.
- `s1_model` and `s2_model` are the actual project checkpoints.

### DFC Failure Metrics Now Wired

The unified evaluator reports the same paper-facing metrics used in the updated
NeurIPS writeup:

- `hard_hits`
- `risk_exposure`
- `barrier_violation_m`
- `path_length_m`
- `path_length_ratio`
- `oscillation`
- `catastrophic_failure`
- `failure_score`

Per-episode outputs:

- `metrics.json`
- `overview.png`
- `cumrisk.png`

Aggregate outputs:

- `aggregate.csv`
- `summary.json`
- `summary_metrics.png`

### Current DFC Comparison Snapshot

Initial unified comparison artifact:

```bash
/tmp/dfc_ours_eval3
```

This is the first unified `3`-episode test comparison across:

- `blind_dijkstra`
- `risk_weighted_astar`
- `oracle_astar`
- `s1_model`
- `s2_model`

Initial mean failure scores from `summary.json`:

- `blind_dijkstra`: `9979.35`
- `risk_weighted_astar`: `792.25`
- `oracle_astar`: `57.76`
- `s1_model`: `55011.19`
- `s2_model`: `827.04`

Interpretation:

- `s1_model` is clearly not competitive as a safety baseline.
- `s2_model` is a very large improvement over `s1_model`.
- on this small held-out slice, `s2_model` does **not** beat all baselines:
  - it is slightly worse than `risk_weighted_astar` in mean failure score;
  - it is much worse than `oracle_astar`, which avoids hazards by taking very
    long detours.

Important nuance from the per-episode breakdown:

- `s2_model` beats `risk_weighted_astar` on `2/3` held-out episodes;
- both lose a hard tail-risk case (`episode 0007`);
- `oracle_astar` is the only method in this slice that stays fully safe on
  that case, but it does so with a very long route.

That first unified read established:

- Stage 2 materially improves over the Stage 1 ablation;
- Stage 2 is competitive with strong non-oracle baselines;
- Stage 2 is **not yet** clearly best overall on the current held-out slice.

### Updated DFC Comparison After Geometry-Waypoint Retrain

New best DFC Stage 2 checkpoint:

```bash
checkpoints/s2_geom_cvar99_hard10/best.pt
```

Current full held-out comparison artifact:

```bash
/tmp/dfc_geom_cvar99_hard10_eval3_all
```

This is the updated unified `3`-episode test comparison across:

- `blind_dijkstra`
- `geometry_astar`
- `risk_weighted_astar`
- `oracle_astar`
- `cvar_costmap_astar`
- `chance_constrained_mpc`
- `ppo_lagrangian`
- `s1_model`
- `s2_model`

Current mean failure scores from `summary.json`:

- `oracle_astar`: `57.76`
- `s2_model`: `422.50`
- `chance_constrained_mpc`: `661.17`
- `cvar_costmap_astar`: `662.33`
- `ppo_lagrangian`: `718.07`
- `risk_weighted_astar`: `792.25`
- `geometry_astar`: `1129.25`
- `blind_dijkstra`: `9979.35`
- `s1_model`: `55011.19`

Interpretation:

- the geometry-waypoint retrain plus stronger tail-risk weighting materially
  improved Stage 2 again:
  - previous geometry-waypoint Stage 2 was `691.89`;
  - current `s2_model` is `422.50`.
- `s2_model` now beats all **non-oracle** DFC baselines on this held-out slice;
- `oracle_astar` remains better, and should still be treated as a strong
  planner ceiling/reference rather than a normal direct competitor.

Current honest DFC claim:

- Stage 2 now appears to be the strongest **practical / non-oracle** DFC
  method we have implemented in-repo;
- Stage 2 clearly beats the Stage 1 ablation and the added non-oracle planner /
  policy baselines on the current held-out slice;
- Stage 2 still does **not** beat `oracle_astar`.

### DFC Improvement Path Now Landed

The most important DFC train/eval mismatch we found was:

- training always used checkpoint/oracle `stage_exit` local goals;
- fair end-to-end evaluation uses geometry-derived waypoints.

To address that, `train_material.py` now supports:

```bash
--waypoint_mode {oracle,geom}
```

New behavior:

- `oracle`
  - preserves the old training path
- `geom`
  - derives local goals / `o_tgt` / `v_tgt` from a geometry-only route,
    matching the fair end-to-end eval story much more closely

This change is now in place and has already been used in the current best DFC
retrain.

Current best DFC retrain command:

```bash
MPLCONFIGDIR=/tmp/mpl python train_material.py \
  --root data/dfc2018_stagewise \
  --stage 2 \
  --epochs 100 \
  --bs 64 \
  --workers 0 \
  --out checkpoints/s2_geom_cvar99_hard10 \
  --ckpt_s1 checkpoints/s1/best.pt \
  --waypoint_mode geom \
  --w_hard 10 \
  --cvar_alpha 0.99
```

Current best follow-up eval command:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/eval_dfc_baselines.py \
  --split test \
  --max-episodes 3 \
  --planners \
    blind_dijkstra \
    geometry_astar \
    risk_weighted_astar \
    oracle_astar \
    cvar_costmap_astar \
    chance_constrained_mpc \
    ppo_lagrangian \
    s1_model \
    s2_model \
  --ckpt-s1 checkpoints/s1/best.pt \
  --ckpt-s2 checkpoints/s2_geom_cvar99_hard10/best.pt \
  --out /tmp/dfc_geom_cvar99_hard10_eval3_all
```

Expectation setting:

- this path has already produced the best non-oracle DFC result we have so far;
- it should not be assumed to beat `oracle_astar`, which is better treated as
  a strong planner ceiling than a normal direct competitor.

## Best Current Checkpoints

Use these for the paper-facing highway-env story:

```bash
Stage 1 scaffold:
checkpoints/highway_stage1_default_slow_x4/best.pt

Stage 2 paper checkpoint:
checkpoints/highway_stage2_mu_lat/best.pt
```

Notes:

- `highway_stage2_mu_lat/best.pt` remains the best paper checkpoint we have.
- Later snapshots from that run can clear the slow-leader case only by going off-road, so they are worse for the paper.
- The older protected Stage 1 checkpoint `checkpoints/highway_stage1_action_dhat10_afloor0015/best.pt` is still useful historically and for older sim2sim checks, but it is not the Stage 1 scaffold used in the current paper figures/tables.

### Checkpoint Status Update (2026-04-30)

What we verified after the original handoff:

- `checkpoints/highway_stage2_no_mu_lat_matched/best.pt`
  - confirms that `mu_lat` is load-bearing for the passable slow-leader case;
  - `authored_slow_leader` regresses from `0%` Stage 2 crash in the paper checkpoint to `100%` crash;
  - boxed remains `100%` crash.
- `checkpoints/highway_stage2_mu_lat_hi_budget/best.pt`
  - does **not** fix boxed;
  - keeps `authored_slow_leader` good;
  - slightly degrades `default` with some off-road behavior, so it is not a better paper checkpoint than `highway_stage2_mu_lat/best.pt`.
- `checkpoints/highway_stage2_unfrozen_matched/last.pt`
  - can avoid boxed crashes in its final-epoch closed-loop log only by collapsing into a very slow policy;
  - its surrogate validation trajectory error is far outside the best-checkpoint guard, so it should not be treated as the current answer.

Bottom line:

- the best **stock** paper checkpoint is still `checkpoints/highway_stage2_mu_lat/best.pt`;
- the boxed failure is not fixed by the matched no-`mu_lat` retrain, the higher-budget frozen retrain, or the accepted unfrozen best-checkpoint selection;
- the boxed failure looks like a **late longitudinal braking** problem, not a missing lateral-selectivity problem.

### TTC-Retrained Candidate Checkpoint (2026-05-02)

The first training-path TTC checkpoint now exists:

```bash
checkpoints/highway_stage2_ttc_frozen_g8_t4/best.pt
```

This is **not yet** the settled paper checkpoint, but it is the first retrained
candidate that actually bakes the boxed fix into the checkpoint rather than
depending on an eval-time override.

Its significance is:

- the TTC mechanism is now present during on-policy collection, surrogate rollout,
  and closed-loop checkpoint selection;
- the 20-episode authored eval preserves the open slow-leader passing result and
  fixes boxed;
- the main unresolved question is whether broader `default` stability also holds
  on the longer 100-episode check.

## Paper-Facing Highway Results

Primary JSON:

```bash
runs/paper_data/eval_paired_full.json
```

This is the current paired Stage 1 vs Stage 2 result across:

- `default`
- `authored_slow_leader`
- `authored_slow_leader_boxed`

Current headline read for the **stock paper operating point** (`d_hat` inherited from checkpoint config, i.e. `15.0` for `highway_stage2_mu_lat/best.pt`):

- `default`: Stage 2 improves robustness and stays on-road
  - crash `5% -> 0%`
  - off-road `95% -> 0%`
  - mean speed `24.66 -> 22.68 m/s`
  - lane-keep error `5.75 -> 0.17 m`
- `authored_slow_leader`: the load-bearing positive result
  - crash `100% -> 0%`
  - lane changes/ep `0.00 -> 1.00`
  - mean speed `23.87 -> 26.31 m/s`
- `authored_slow_leader_boxed`: both still crash `100%`
  - this is a boundary case, not a success case

Interpretation:

- Stage 2 learns selective lateral passing when a corridor exists.
- The current architecture still does not solve boxed-in longitudinal braking well enough.

### Boxed Failure Mechanism We Verified

The force diagnostic and authored-scenario rollouts now support a more precise explanation of the boxed failure:

- in `authored_slow_leader_boxed`, Stage 2 does **not** invent an inappropriate lane change;
- side risks remain high/similar, so the lateral channel stays weak;
- the failure comes from insufficiently early **longitudinal** braking against the slow leader.

This matters for the paper because it means the boxed issue is not evidence that the model has no context selectivity. The model appears to distinguish "open corridor" from "boxed corridor" already; what it lacks is a second context-conditioned corrective mode for high closing-risk same-lane interactions.

### Deployment-Only Boxed Fix Already Tested

New artifact:

```bash
runs/paper_data/eval_paired_mu_lat_dhat25.json
```

This uses the same checkpoint as the paper result:

```bash
checkpoints/highway_stage2_mu_lat/best.pt
```

but evaluates with:

```bash
--d-hat 25
```

What changes:

- `default`
  - Stage 2 remains `0%` crash and `0%` off-road
  - but slows down from `22.68 -> 19.34 m/s`
- `authored_slow_leader`
  - Stage 2 remains `0%` crash and still lane-changes/passes
  - but slows from `26.31 -> 24.51 m/s`
- `authored_slow_leader_boxed`
  - Stage 2 improves from `100%` crash to `0%` crash
  - no off-road escape, no fake pass
  - mean speed drops to `6.17 m/s`, i.e. it survives by very conservative following

Additional stability check we ran:

- `default`, `100` episodes, `d_hat=25`: `0/100` crash, `0/100` off-road, on-road fraction `1.000`
- `authored_slow_leader`, `20` episodes, `d_hat=25`: `20/20` success, identical across seeds
- `authored_slow_leader_boxed`, `20` episodes, `d_hat=25`: `20/20` success, identical across seeds

Important caveat:

- this is a **deployment override**, not a new checkpoint;
- in paired eval, `--d-hat 25` affects both Stage 1 and Stage 2 runtime, so the resulting JSON is not a clean replacement for the stock paper table;
- it is best interpreted as evidence that the boxed failure is fixable through earlier longitudinal caution, not as the final paper result by itself.

## Other Highway JSON Artifacts Already In Hand

Force/mechanism dump:

```bash
runs/paper_data/force_diagnostic.json
```

Stage 2 sim2sim:

```bash
runs/paper_data/sim2sim_stage2_mu_lat.json
```

Current summary from that file:

- terminal drift at `2.0 s`: median `2.05 m`
- p95 `3.39 m`
- max `3.87 m`

Inference-time `mu_lat` ablation:

```bash
runs/paper_data/eval_stage2_mu_lat_disabled.json
```

Current read:

- `authored_slow_leader`: `0%` crash with `mu_lat` enabled becomes `100%` crash with `mu_lat` disabled
- `authored_slow_leader_boxed`: unchanged at `100%` crash
- `default`: largely unchanged

This is useful and paper-worthy, but it is still an inference-time same-checkpoint ablation, not a clean retrained no-`mu_lat` model.

## What We Think The Paper Should Say Right Now

Until a new method variant is implemented and validated, the honest paper-facing claim remains:

- Stage 2 opens a lateral passing degree of freedom when a pass corridor exists;
- the stock paper checkpoint does **not** solve the boxed case;
- a deployment-time increase in `d_hat` shows the boxed failure is mainly a missing early-braking mechanism, but that override slows the policy and should not be oversold as the final method.

If the current code snapshot had to be written up immediately, that is the correct framing.

### Updated Read After TTC Retrain (20-Episode Check)

New artifact:

```bash
runs/paper_data/eval_stage2_ttc_retrained_ep20.json
```

Checkpoint:

```bash
checkpoints/highway_stage2_ttc_frozen_g8_t4/best.pt
```

What the retrained checkpoint achieves over 20 episodes:

- `default`
  - remains clean: `0%` crash, `0%` off-road
  - mean speed `22.76 m/s`
  - effectively unchanged from the stock paper Stage 2 operating point
- `authored_slow_leader`
  - remains clean: `0%` crash, `0%` off-road
  - still performs `1.0` lane changes/episode
  - mean speed `26.19 m/s`, essentially matching the stock paper checkpoint
- `authored_slow_leader_boxed`
  - improves from `100%` crash in the stock paper checkpoint to `0%` crash
  - no off-road escape, no fake pass
  - mean speed `7.62 m/s`

Interpretation:

- retraining successfully **bakes in** the TTC behavior that we previously only
  had as an eval-time prototype;
- the boxed fix survives training without destroying the open-corridor passing result;
- however, the learned boxed behavior is still very conservative, so the boxed
  improvement is "safe but timid," not "safe and fast."

Important caveat:

- this 20-episode artifact is encouraging, but it is **not by itself** enough
  to declare the retrained TTC checkpoint paper-ready;
- the deciding artifact is the longer `default` stability check
  (`eval_stage2_ttc_retrained_default_ep100.json`), which should be treated as
  the main gate before replacing the stock paper tables/figures.

## Planned Paper Upgrade: Longitudinal TTC Energy

The current leading method direction is **not** "fixed `d_hat=25`".
It is:

- keep the existing Stage 2 lateral channel for passable slow-leader scenes;
- add a context-conditioned **longitudinal TTC / closing-risk energy** for boxed or near-boxed same-lane interactions.

Why this is the preferred spotlight direction:

- fixed `d_hat=25` works, but reads like a global conservatism knob;
- a longitudinal TTC energy is a cleaner method contribution;
- it directly matches the diagnosed failure mode: "boxed and closing too fast to the leader";
- it naturally supports the stronger claim that sensed local context changes which control mode is active:
  - lateral passing when a corridor exists,
  - longitudinal braking when the same-lane TTC risk dominates and lateral escape is poor.

Proposed paper wording direction if TTC works empirically:

- the stock Stage 2 result demonstrates selective lateral passing;
- the TTC-augmented Stage 2 result demonstrates selective activation of **both** lateral passing and longitudinal braking modes from local context.

That is the current best path to a stronger spotlight claim.

## TTC Prototype Status (Initial Analytic Eval-Time Version Landed)

The first analytic TTC prototype is now implemented in the inference/eval path.

Files changed:

- `env_wrapper.py`
  - now exposes `V_neighbors` in the observation dict alongside `C`, `R`, `W`, `mask`;
- `surrogate_integrator.py`
  - now has a shared lateral-probe helper and an optional analytic `F_ttc` force term;
- `eval_stage2.py`
  - now exposes TTC runtime flags and passes them into deployment eval;
- `eval_force_diagnostic.py`
  - now logs `F_ttc`, TTC, closing speed, leader gap, and boxed-gate state.

Design currently implemented:

- detect the nearest same-lane leader ahead using the ego forward axis and a lane-width corridor;
- compute closing speed and TTC from observed neighbour velocity;
- activate a rearward longitudinal force `F_ttc` only when:
  - TTC is low, and
  - both lateral probes look risky enough that the ego is plausibly boxed.

The default path is preserved because:

- TTC is off unless `--ttc-gain > 0`;
- existing checkpoints/configs do not enable TTC by default.

### Initial TTC Eval Result (2026-04-30)

Using the stock paper checkpoint:

```bash
checkpoints/highway_stage2_mu_lat/best.pt
```

and the initial runtime setting:

```bash
--ttc-gain 8.0 --ttc-threshold-s 4.0
```

we verified on matched seeds:

- `default`, `5` episodes
  - TTC path is effectively unchanged relative to stock:
  - crash `0/5 -> 0/5`
  - mean speed `22.16 -> 22.16 m/s`
  - on-road fraction remains `1.000`
- `authored_slow_leader`, `5` episodes
  - pass behavior is preserved:
  - crash `0/5 -> 0/5`
  - lane changes/ep stays `1.00`
  - mean speed changes only slightly: `26.31 -> 26.21 m/s`
- `authored_slow_leader_boxed`, `5` episodes
  - boxed improves from deterministic crash to deterministic survival:
  - crash `5/5 -> 0/5`
  - on-road fraction remains `1.000`
  - no fake pass, no off-road escape
  - but mean speed collapses to `7.57 m/s`

Interpretation:

- the **context gate is doing the right qualitative thing**:
  - open slow-leader behavior is preserved;
  - boxed behavior activates strong longitudinal braking;
- but the first TTC setting is **too conservative in boxed** to be paper-final.

This is still an encouraging result because it is strictly better than fixed `d_hat=25` in one important respect:

- fixed `d_hat=25` slowed `default` materially;
- TTC leaves `default` and open-lane passing essentially unchanged on matched seeds.

The remaining work is to make boxed braking less timid while keeping that selectivity.

## TTC Training Path Status

The TTC training path is now implemented.

Files landed:

- `collect_onpolicy.py`
  - TTC knobs now flow into closed-loop on-policy collection;
- `train_stage2.py`
  - TTC knobs are part of `Stage2Cfg`;
  - old IDM samples are made compatible by injecting zero `V_neighbors` when absent;
  - TTC is used in rollout, validation, collection, and closed-loop checkpoint selection;
- `surrogate_integrator.py`
  - the analytic TTC force is currently treated as a **forward-only** branch
    during training (`F_ttc.detach()`), because the first fully differentiable
    attempt hit a PyTorch backward-time segfault through the piecewise leader
    selection logic.

This design choice is deliberate and acceptable for now:

- it preserves deployed TTC behavior exactly;
- it lets the checkpoint adapt to TTC-conditioned trajectories during training;
- it avoids tying the current retrain to a brittle non-smooth autograd path.

We also ran a one-epoch smoke retrain successfully after this change, confirming:

- on-policy collection works with TTC enabled;
- mixed IDM/on-policy batches still load correctly;
- training/validation/closed-loop eval/checkpoint write all complete.

### Official Convoy Ladder Scenarios Added

To test sequential generalization explicitly, the repo now has official authored convoy variants:

- open:
  - `highway-slow-leader-x2-v0`
  - `highway-slow-leader-x3-v0`
  - `highway-slow-leader-x4-v0`
  - alias: `highway-slow-convoy-v0` = open 4-leader case
- boxed:
  - `highway-slow-leader-boxed-x2-v0`
  - `highway-slow-leader-boxed-x3-v0`
  - `highway-slow-leader-boxed-x4-v0`
  - alias: `highway-slow-convoy-boxed-v0` = boxed 4-leader case

These scenario names are now wired into:

- `eval_stage2.py`
- `render_paired_gif.py`
- `eval_force_diagnostic.py`

#### Breakpoint Result We Observed

Using small matched-seed checks on the official ladder:

- open ladder, stock Stage 2:
  - `x1` works in the original authored setting;
  - `x2/x3/x4` all fail, but importantly they make **one lane change** and survive deep into the episode before crashing.
- boxed ladder, TTC prototype:
  - `x1` works in the original authored boxed setting;
  - `x2/x3/x4` all fail early with **no lane change**.

Interpretation:

- open convoys show **partial** sequential generalization:
  - the policy can initiate the first pass,
  - but does not sustain a repeated overtake strategy across multiple leaders.
- boxed convoys do **not** yet show sequential generalization under the current analytic TTC patch.

## TTC Implementation Plan (Next Pass)

The next implementation pass should proceed in this order:

1. inference-time analytic TTC prototype, no model-head changes yet;
2. expose TTC knobs in `eval_stage2.py` and `eval_force_diagnostic.py`;
3. verify that TTC preserves:
   - `default` robustness,
   - `authored_slow_leader` passing,
   - while improving `authored_slow_leader_boxed`;
4. only then wire TTC into collection/training, ideally frozen-geometry first.

The concrete design we agreed on:

- add a new force term `F_ttc` parallel to `F_lat`;
- estimate the most relevant leader ahead in-lane;
- compute closing speed and TTC;
- gate the TTC term by context so it does not suppress passing when an adjacent corridor is clearly available;
- keep the first prototype analytic and interpretable before considering a learned `lambda_ttc` head.

At the code level, the first patch should touch:

- `env_wrapper.py` — expose neighbour velocities alongside `C`, `R`, `W`, `mask`;
- `surrogate_integrator.py` — add TTC force computation and context gate;
- `eval_stage2.py` — add TTC eval-time flags and pass-through;
- `eval_force_diagnostic.py` — log `F_ttc`, TTC, closing speed, leader gap, and gate state.

## Root Highway Figure and Table Choices

The root highway experiment draft (`/mnt/data/adityas/GRL-SNAM/exp-highway-env/exp.tex`) currently uses:

- Figure 2:
  - `figures/figure2_paired_rollout.pdf`
  - redesigned as a slow-leader storyboard plus lateral-position / clearance traces
- Figure 3:
  - `figures/figure3_outcome_summary.pdf`
- Figure 4:
  - `figures_v3/figure4_paired_transitions.pdf`
  - `figures_v3` is preferred over the older `figures/` and `figures_v2/` variants
- Table 1:
  - `tables/table1_enrichment_ablation.tex`
- Table 2:
  - `tables/table2_paired_comparison.tex`

That root `exp.tex` has already been updated to use these figure paths and to inline the table LaTeX. The NeurIPS draft uses the copied PNG figure paths listed in the NeurIPS section above.

## Paper Artifact Pipeline

Run from this directory:

```bash
cd /mnt/data/adityas/GRL-SNAM/exp-highway-env
```

Smoke / rebuild figures:

```bash
python make_paper_figures.py \
  --force-diagnostic runs/paper_data/force_diagnostic.json \
  --paired-eval runs/paper_data/eval_paired_full.json \
  --out figures
```

Rebuild tables:

```bash
python make_paper_tables.py \
  --paired-eval runs/paper_data/eval_paired_full.json \
  --frozen-history checkpoints/highway_stage2_mu_lat/history.json \
  --unfrozen-history checkpoints/highway_stage2_navscale/history.json \
  --out tables
```

Full end-to-end paper pipeline:

```bash
bash run_paper_evals.sh
```

`run_paper_evals.sh` is idempotent and now supports:

- `EPISODES=...`
- `MAX_STEPS=...`
- `DEVICE=cpu|cuda|auto`

## Long GPU Runs Worth Doing

These were the originally planned high-value highway-env runs. As of 2026-04-30, the matched no-`mu_lat`, higher-budget frozen, and unfrozen-matched families all exist already, so rerunning them is lower priority than the TTC implementation/eval path unless we need reproducibility on a different machine.

### 1. Matched no-`mu_lat` training ablation

Wrapper:

```bash
python train_stage2_frozen_geom_stressmix_no_mu_lat.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_no_mu_lat_matched \
  --epochs 30 --bs 64 --lr 3e-4 \
  --device cuda
```

Then evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_no_mu_lat_matched/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_no_mu_lat_matched.json
```

### 2. Matched unfrozen-geometry ablation

Wrapper:

```bash
python train_stage2_unfrozen_stressmix.py \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --idm-data runs/stage1_data_default_slow_x4 \
  --out checkpoints/highway_stage2_unfrozen_matched \
  --epochs 30 --bs 64 --lr 3e-4 \
  --device cuda
```

Then evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_unfrozen_matched/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_unfrozen_matched.json
```

### 3. Higher-budget Stage 2 candidate

Wrapper:

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

Then evaluate:

```bash
python eval_stage2.py \
  --ckpt checkpoints/highway_stage2_mu_lat_hi_budget/best.pt \
  --stage1-ckpt checkpoints/highway_stage1_default_slow_x4/best.pt \
  --scenarios default authored_slow_leader authored_slow_leader_boxed \
  --episodes 20 --max-steps 120 \
  --device cuda \
  --out runs/paper_data/eval_paired_mu_lat_hi_budget.json
```

## Important Caveats

1. The current enrichment ablation table is still provisional.
   - It uses the frozen `mu_lat` history and the older unfrozen `navscale` history.
   - That comparison is useful, but not a perfectly matched freeze-only ablation.

2. The current `mu_lat` ablation is useful but not maximal.
   - `eval_stage2_mu_lat_disabled.json` is an inference-time ablation.
   - The stronger result is the matched retrained no-`mu_lat` run above.

3. The stock boxed scenario is still a failure case.
   - Do not oversell the stock checkpoint.
   - The honest stock claim is selective lateral passing when a pass corridor exists.

4. `eval_paired_mu_lat_dhat25.json` is encouraging but not yet the paper answer.
   - It demonstrates a stable deployment-time boxed fix by increasing `d_hat`.
   - It also changes the runtime of both Stage 1 and Stage 2 in paired eval.
   - It slows the policy substantially, especially in boxed.
   - Treat it as mechanism evidence and a debugging waypoint, not the final result.

5. The current best paper-upgrade direction is TTC, not more blind checkpoint fishing.
   - The next meaningful contribution is a context-conditioned longitudinal risk mode.
   - If TTC hurts `default` or suppresses the pass case, it should not go into the paper.
   - The `highway_stage2_ttc_frozen_g8_t4` retrain is promising, but should only
     replace the stock paper checkpoint after the longer `default` stability eval
     and the paired paper-style comparison are both checked.

## Minimal “What Do I Use Right Now?” Summary

If you need the current highway-env package quickly:

- checkpoint:
  - `checkpoints/highway_stage2_mu_lat/best.pt`
- retrained TTC candidate:
  - `checkpoints/highway_stage2_ttc_frozen_g8_t4/best.pt`
- main paired results:
  - `runs/paper_data/eval_paired_full.json`
- boxed-fix exploratory variant:
  - `runs/paper_data/eval_paired_mu_lat_dhat25.json`
- TTC retrained 20-episode check:
  - `runs/paper_data/eval_stage2_ttc_retrained_ep20.json`
- mechanism/support:
  - `runs/paper_data/force_diagnostic.json`
  - `runs/paper_data/sim2sim_stage2_mu_lat.json`
  - `runs/paper_data/eval_stage2_mu_lat_disabled.json`
- final paper figures:
  - `figures/figure2_paired_rollout.pdf`
  - `figures/figure3_outcome_summary.pdf`
  - `figures_v3/figure4_paired_transitions.pdf`
- root highway paper source already updated:
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/exp.tex`
- NeurIPS source already updated:
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/NeurIPS_2026/neurips_2026.tex`
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/NeurIPS_2026/tex/intro.tex`
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/NeurIPS_2026/tex/related_works.tex`
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/NeurIPS_2026/tex/method.tex`
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/NeurIPS_2026/tex/exp.tex`
- current NeurIPS PDF:
  - `/mnt/data/adityas/GRL-SNAM/exp-highway-env/Master_s_Thesis/neurips_2026.pdf`

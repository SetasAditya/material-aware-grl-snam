# Comparative GIFs

## Behavioral comparisons (use these in the rebuttal)

These animations use closed-loop trajectories that visibly diverge between
methods. Generate the RELLIS-Dyn examples with:

```bash
MPLCONFIGDIR=/tmp/rq-gif-matplotlib \
python rq_visualizations/gifs/make_behavioral_rellis_gifs.py
```

Generate the open-lane and boxed-lane highway examples with:

```bash
bash rq_visualizations/gifs/make_behavioral_highway_gifs.sh
```

Generate the same-event planner comparison with:

```bash
MPLCONFIGDIR=/tmp/rq-gif-matplotlib \
python rq_visualizations/gifs/make_planner_comparison_gif.py
```

The outputs are in `rq_visualizations/gifs/behavioral/`:

- `rellis_delayed_escape_example_1.gif` and `_2.gif`: four actual policy
  rollouts on the same changing map. They show different pre-opening choices
  and the post-opening route commitment.
- `authored_slow_leader.gif`: the geometry baseline crashes after 19 steps;
  the full policy activates its lateral channel and survives the 70-step
  horizon.
- `authored_slow_leader_boxed.gif`: the same baseline crashes after 19 steps;
  the full policy suppresses the lateral maneuver, applies TTC braking, and
  survives the 70-step horizon.
- `planner_comparison.gif`: one measured moving-obstacle-blocks-detour replay
  comparing semantic DWA, semantic MPPI, budgeted MPC, and our zero-replan
  field. In this selected episode DWA and ours reach the goal while MPPI and
  MPC exhaust the rollout. This is an illustrative counterexample, not a claim
  that the field dominates trajectory optimization on every event.

The highway script deliberately supplies the evaluated TTC operating point
(`gain=8`, `threshold=3.5 s`). Omitting it silently disables the braking term.

## Gate diagnostic (mechanism only; superseded for behavioral claims)

Generate four measured delayed-escape examples:

```bash
MPLCONFIGDIR=/tmp/rq-gif-matplotlib \
python rq_visualizations/gifs/make_gate_advantage_gifs.py
```

The renderer compares the same checkpoint and episode with the feasibility
gate enabled and removed. Spatial backgrounds are reconstructed schematically
from the saved event specification; positions, paths, witness endpoints,
forces, gate states, and timing come from the saved per-step traces.

The demonstrated diagnostic is deliberately narrow: the gate suppresses
premature soft-force exposure while a lower-risk escape is blocked and admits
it after the escape becomes feasible. Since the learned coefficient is small
on this checkpoint, gate-on and gate-off success are both 1.0; these GIFs must
not be captioned as showing a success improvement.

Useful options:

```bash
python rq_visualizations/gifs/make_gate_advantage_gifs.py --count 6
python rq_visualizations/gifs/make_gate_advantage_gifs.py --episodes 59,12,31
python rq_visualizations/gifs/make_gate_advantage_gifs.py --stride 1 --duration-ms 80
```

These older outputs and a selection/claim manifest are written to
`rq_visualizations/gifs/output/`.

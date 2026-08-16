# Repaired checkpoint-driven controller (v1)

`run_v1_dynamic.py` is an isolated replacement for the dynamic learned-policy
evaluation path. It does not modify or overwrite `rebuttal_experiments`.

The primary `repaired` mode:

1. runs the canonical `CoefEnergyNetMaterial` checkpoint at every step;
2. uses the local primitive search as a feasibility witness;
3. applies separate on/off thresholds, evidence persistence, and minimum dwell;
4. immediately suppresses the soft channel at a hard hazard;
5. projects the learned soft force into a cone around the selected feasible
   primitive; and
6. executes the result with the canonical material Hamiltonian integrator.

The projected direction is normalized, making `lambda_soft` the actual
soft-force magnitude. A sufficiently strong local gradient supplies direction;
below the configured confidence threshold, the default policy uses the
gate-selected primitive axis. The `zero` fallback disables the force in flat
regions. Both confidence and fallback use are logged. Only the soft-risk
gradient channels change; the hard SDF force and geometry force are untouched.

Available matched modes are `repaired`, `stateful_unprojected`,
`stateless_projected`, `stateless_unprojected`, `gate_off`, and
`geometry_only`. All use the same checkpoint and integrator.

Repaired/stateless modes apply both the feasibility decision and the checkpoint
metadata field `repair_calibration.lambda_active_threshold`. `gate_off`
removes only the feasibility gate/projection and retains this learned-magnitude
threshold as a matched control. `geometry_only` always sets the soft channel to
zero. Historical checkpoints without repair metadata default to a threshold of
zero; the CLI can override it explicitly.

For stateful modes, learned-magnitude eligibility is included in the
hysteresis evidence: activation requires it, while a drop below threshold is
deactivation evidence subject to the same persistence and minimum dwell.
It is not applied as a second frame-wise switch after the state machine.

Development runs only accept the preregistered `train` and `validation`
manifests. The sealed test split and sequence `00004` are unavailable.

```bash
pytest -q repair_experiments/test_v1_controller.py

python repair_experiments/run_v1_dynamic.py \
  --split train \
  --max-episodes 2 \
  --modes repaired stateless_unprojected gate_off \
  --out repair_experiments/results/v1_dynamic_smoke
```

The output contains the exact configuration/checkpoint hashes,
per-episode metrics, and step traces including raw/effective gate state,
state transitions and reasons, persistence counters, dwell, hard overrides,
selected primitive, projection alignment, learned/used coefficients, and
executed motion.

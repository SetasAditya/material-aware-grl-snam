# Frozen v7 controller state

Frozen: 2026-07-27

## Decision

The frozen controller is
`checkpoint_direct_forward_velocity_tracking_v7`.

The matched 54-item validation subset fixed the reverse-command defect but
did not pass the efficacy stage gate. Full validation was not run, sequence
`00004` remains sealed, and no held-out or acceptance claim is authorized.
The detailed results are in
[`REPAIRED_METHOD_RESULTS.md`](REPAIRED_METHOD_RESULTS.md#l-forward-only-velocity-tracking-and-bounded-safe-hold--directional-fix-efficacy-no-go).

## Retained reproducibility set

- all top-level source, test, and documentation files in
  `repair_experiments/`;
- all locked split definitions in `repair_experiments/splits/`;
- final checkpoint directory
  `repair_experiments/outputs/behavioral_soft_force_risk_encoder_recall_full/`;
- final matched result directory
  `repair_experiments/results/direct_forward_velocity_tracking_v7_validation_subset54/`;
- rebuttal checklists and consolidated Markdown reports at repository root.

The original code, paper, data-facing helpers, and prior consolidated rebuttal
experiments outside `repair_experiments/results/` and
`repair_experiments/outputs/` were not included in cleanup.

## Frozen hashes

| Item | SHA-256 |
|---|---|
| `repair_experiments/v1_controller.py` | `354087cf56f267b94a671f1efd8aa85892206bdca6424e0d4fd846e01736620e` |
| `repair_experiments/velocity_selector.py` | `a4decb7e538199c96198b3e5055f844b31f34218420c96c67d76f332824de4af` |
| `repair_experiments/run_v1_dynamic.py` | `8f4b3e3e1329c9798ef27979077fc1510045afad7d0c412ae7c5a09ed18da87f` |
| `repair_experiments/evaluate_v1.py` | `05f4df9e5def294071ec1de505348746f22f0d46cb1e34e6b7e31e67dc0a88cf` |
| `repair_experiments/evaluate_direct_waypoint.py` | `fc808ee08d2d88bf39430a28a622f5a3c525b33ca173a2b9c1fda1913bb4e5f4` |
| `repair_experiments/evaluation_metrics.py` | `5d108630dfef2787c9e09142e8e70c28bf27fbb45570e3521c9da801c669f9d9` |
| final checkpoint `best.pt` | `327c744dea093a5436e02ec03328690bac71dbbdf57fd7713a13c5735cdcbda6` |
| final result `ARTIFACT_MANIFEST.json` | `16b72ec22ce236cd0403a34531b27436e3f70b9b330d7e58ac851a6ee9d3ae0a` |

Deterministic aggregate hashes, computed from sorted per-file SHA-256 lines:

| Retained tree | Aggregate SHA-256 |
|---|---|
| top-level `repair_experiments/*.{py,md}` | `882cc65ef6d53dcfc22b44d27d0d46df655ec1adc0fcfd7f0fc27b48221aed73` |
| `repair_experiments/splits/` | `8d5fd3eb9b948fc81ad3a5025950d21effe4b2da1ef96f3b213843ab188b3720` |
| final checkpoint directory | `ac174c3ef2356ec338b839bc3ce89d3048589793e502ea54fbe80ea96d47345f` |
| final result directory | `97d60d91d7e6226f379f76a6f8dc7b2c41de76672c4af19652b0eadcd3495d26` |

## Cleanup scope

Cleanup removes generated caches, all smoke artifacts, all v1–v6 controller
result directories, and five superseded checkpoint/training-output
directories. Those directories were untracked and are not recoverable from
this workspace after deletion. Their numerical conclusions and prior artifact
manifest hashes remain consolidated in `REPAIRED_METHOD_RESULTS.md`.

## Post-cleanup verification

- cleanup completed on 2026-07-27;
- `repair_experiments/` decreased from approximately 677 MiB to 116 MiB;
- the only retained result directory is
  `direct_forward_velocity_tracking_v7_validation_subset54`;
- the only retained output directory is
  `behavioral_soft_force_risk_encoder_recall_full`;
- all six files in the final immutable artifact match its manifest;
- the checkpoint and final artifact-manifest hashes match the frozen values;
- the complete `repair_experiments/` suite passes: `82 passed`; and
- the verification run disabled bytecode and pytest cache generation.

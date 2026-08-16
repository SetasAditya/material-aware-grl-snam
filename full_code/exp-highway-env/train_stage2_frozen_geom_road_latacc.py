#!/usr/bin/env python3
"""Stage 2 preset: freeze Stage 1 geometry, train risk enrichment only.

This is the recommended fix after the road/latacc run showed geometry
collapse (val L_traj -> ~44). The preset preserves the Stage 1 scaffold and
trains only:

  - risk_enc
  - lam_soft_head
  - lam_hard_head

It also enables the road-boundary and lateral-acceleration guards.
"""

from __future__ import annotations

import sys

from train_stage2 import main


def _inject_flag(flag: str) -> None:
    if flag not in sys.argv:
        sys.argv.append(flag)


def _inject_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_flag("--freeze-geometry")
    _inject_default("--w-road", "10.0")
    _inject_default("--road-margin", "0.25")
    _inject_default("--road-tau", "0.25")
    _inject_default("--w-latacc", "0.05")
    _inject_default("--latacc-free", "4.0")
    main()

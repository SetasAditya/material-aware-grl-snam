#!/usr/bin/env python3
"""Stage 2 training preset with on-road and lateral-comfort guards.

This is the recommended first retrain after the authored-scenario offroad
failure. It preserves the base Stage 2 pipeline, then adds:

  - road-boundary penalty to discourage offroad lateral escapes
  - lateral-acceleration excess penalty to smooth evasive maneuvers

All regularizer defaults can be overridden by passing the same flags on the
command line.
"""

from __future__ import annotations

import sys

from train_stage2 import main


def _inject_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _inject_default("--w-road", "10.0")
    _inject_default("--road-margin", "0.25")
    _inject_default("--road-tau", "0.25")
    _inject_default("--w-latacc", "0.05")
    _inject_default("--latacc-free", "4.0")
    main()

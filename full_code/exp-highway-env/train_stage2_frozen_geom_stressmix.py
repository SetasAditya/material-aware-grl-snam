#!/usr/bin/env python3
"""Stage 2 preset: frozen geometry with authored stress-scenario collection.

This extends the frozen-geometry road/latacc recipe by adding the fixed
slow-leader scenarios to the on-policy buffer and to closed-loop checkpoint
selection. Use this when default highway-v0 is solved but authored slow-leader
eval still crashes.
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
    _inject_flag("--stress-offroad-terminal")
    _inject_default("--w-road", "10.0")
    _inject_default("--road-margin", "0.25")
    _inject_default("--road-tau", "0.25")
    _inject_default("--w-latacc", "0.05")
    _inject_default("--latacc-free", "4.0")
    _inject_default(
        "--collect-envs",
        "highway-v0,highway-slow-leader-v0,highway-slow-leader-boxed-v0",
    )
    _inject_default(
        "--best-eval-envs",
        "highway-v0,highway-slow-leader-v0,highway-slow-leader-boxed-v0",
    )
    # Three envs means the default 5 episodes/epoch would split as 2/2/1.
    # Use 6 so each env contributes equally unless the caller overrides.
    _inject_default("--collect-episodes", "6")
    main()

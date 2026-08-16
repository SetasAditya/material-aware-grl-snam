#!/usr/bin/env python3
"""Stage 2 preset: frozen geometry, stress-mix collection, no mu_lat.

This is the matched training-time lateral-channel ablation for the spotlight
highway-env run. It uses the same stress-scenario recipe as the frozen
mu_lat-enabled run, but disables the lateral channel during training,
collection, and checkpoint selection.
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
    _inject_flag("--disable-mu-lat")
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
    _inject_default("--collect-episodes", "6")
    main()

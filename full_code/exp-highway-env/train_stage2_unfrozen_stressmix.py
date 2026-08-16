#!/usr/bin/env python3
"""Stage 2 preset: full-model stress-scenario training without freezing.

This mirrors the authored-scenario stress-mix recipe used for the frozen
Stage 2 spotlight run, but leaves the Stage 1 geometry scaffold trainable.
Use it for a matched frozen-vs-unfrozen ablation.
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
    _inject_default("--collect-episodes", "6")
    main()

#!/usr/bin/env python3
"""Stage 2 training preset with an on-road rollout guard.

This is a thin preset over ``train_stage2.py``. It keeps the same CLI but
injects a road-boundary penalty unless the caller explicitly passes the
corresponding flag.
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
    main()

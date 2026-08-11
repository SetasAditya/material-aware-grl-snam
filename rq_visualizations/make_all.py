"""Run all five RQ visualization scripts."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/rq-visualizations-matplotlib")
    for script in [
        "rq1_gate_exposure.py",
        "rq2_witness_execution.py",
        "rq3_soft_channel.py",
        "rq4_adaptation_cvar.py",
        "rq5_perception_robustness.py",
    ]:
        subprocess.run([sys.executable, str(HERE / script)], check=True, env=env)


if __name__ == "__main__":
    main()

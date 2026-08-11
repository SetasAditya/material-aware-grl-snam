"""Shared styling and artifact helpers for the RQ1--RQ5 figures."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "rebuttal_experiments" / "results"
DEFAULT_OUTPUT = ROOT / "rq_visualizations" / "output"

# The BEV patch geometry used by every RELLIS-Dyn episode in these experiments
# (BevConfig defaults: 50 m x 50 m extent at 0.5 m resolution).
SHAPE = (100, 100)

# Base risk assumed under the event overlay. The per-scene BEV cache is not
# present in this repository snapshot, so the scene-dependent component of the
# risk map cannot be recovered; only the event-induced structure is exact.
NEUTRAL_BASE_RISK = 0.35

COLORS = {
    "material": "#2463A6",
    "material_light": "#8EB6DC",
    "geometry": "#666666",
    "dwa": "#D97706",
    "expected": "#8B5CF6",
    "fixed": "#B45309",
    "zero": "#777777",
    "risk": "#C83E4D",
    "safe": "#2E8B57",
    "hazard": "#252525",
    "muted": "#D8DCE2",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        value = float(row.get(key, ""))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def grouped(data: Iterable[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in data:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def parse_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _dyn_events():
    """Import the simulator's own event-geometry helpers.

    Reconstructing the field with these functions rather than re-deriving the
    shapes keeps the drawn geometry identical to what the rollout applied.
    """
    path = ROOT / "rellis"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from grl_rellis import dyn_events

    return dyn_events


def event_field(spec: dict[str, str], step: float) -> dict[str, np.ndarray]:
    """Rebuild the ``delayed_required_escape`` overlay at a given control step.

    Returns the masks the simulator applied at that instant plus a composed
    risk field over `NEUTRAL_BASE_RISK`. The mask geometry is exact; the
    absolute risk values are exact only where the event writes them.
    """
    de = _dyn_events()
    if spec["event_type"] != "delayed_required_escape":
        raise ValueError(f"Unsupported event type: {spec['event_type']}")

    center = parse_vector(spec["center_rc"])
    detour = parse_vector(spec["detour_rc"])
    goal = parse_vector(spec["goal_rc"])
    axis = de._unit(parse_vector(spec["axis_rc"]))
    radius = f(spec, "radius_cells")
    half_len = f(spec, "barrier_half_len_cells")
    half_width = f(spec, "barrier_half_width_cells")
    risk_value = f(spec, "risk_value")
    low = f(spec, "low_risk_value")
    open_step = f(spec, "event_step") + f(spec, "open_delay")

    block = de._barrier_mask(SHAPE, detour, axis, half_len, half_width)
    escape = de._ellipse_mask(
        SHAPE, (detour + goal) / 2.0, radius * 1.45, radius * 0.55, de._unit(goal - detour)
    )

    risk = np.full(SHAPE, NEUTRAL_BASE_RISK, dtype=np.float32)
    hard = np.zeros(SHAPE, dtype=bool)
    mud = np.zeros(SHAPE, dtype=bool)
    closure = np.zeros(SHAPE, dtype=bool)

    if step < open_step:
        risk[escape] = np.minimum(risk[escape], low)
        hard[block] = True
    else:
        risk[block] = np.minimum(risk[block], low)
        risk[escape] = np.minimum(risk[escape], low)
        closure_center = center + (f(spec, "open_delay") + 3.0) * axis
        mud = de._ellipse_mask(SHAPE, closure_center, radius * 1.8, radius * 1.0, axis)
        closure = de._barrier_mask(
            SHAPE, closure_center, axis, half_len * 1.15, half_width * 1.15
        )
        risk[mud] = np.maximum(risk[mud], risk_value)
        hard[closure] = True

    from scipy.ndimage import gaussian_filter

    return {
        "risk": gaussian_filter(np.clip(risk, 0.0, 1.0), sigma=0.75),
        "hard": hard,
        "block": block,
        "escape": escape,
        "mud": mud,
        "closure": closure,
        "open_step": open_step,
    }


FIELD_LAYERS = [
    ("escape", "#2E8B57", 0.45, "low-risk escape corridor"),
    ("mud", "#C83E4D", 0.55, "high-risk mud"),
    ("hard", "#1A1A1E", 0.95, "hard (infeasible)"),
]


def draw_field(ax, field: dict[str, np.ndarray]) -> None:
    """Render the event-induced structure as labelled categorical layers.

    The scene-dependent base risk is not recoverable from the saved artifacts,
    so it is drawn as neutral ground rather than given a fabricated value; only
    the regions the event actually writes are coloured.
    """
    base = np.full((*SHAPE, 3), 0.92, dtype=float)
    ax.imshow(base, origin="upper", interpolation="nearest", zorder=0)
    for key, color, alpha, _ in FIELD_LAYERS:
        mask = field[key]
        if not np.any(mask):
            continue
        rgba = np.zeros((*SHAPE, 4), dtype=float)
        rgba[mask] = (*mpl.colors.to_rgb(color), alpha)
        ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=1)


def field_legend_handles() -> list:
    from matplotlib.patches import Patch

    return [Patch(facecolor=c, alpha=a, label=label) for _, c, a, label in FIELD_LAYERS]


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, inputs: Iterable[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    provenance = {
        "figure": stem,
        "inputs": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in inputs
        ],
    }
    (output_dir / f"{stem}.provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, weight="bold", va="bottom")


def arrow(ax: mpl.axes.Axes, start: np.ndarray, delta: np.ndarray, color: str, label: str | None = None) -> None:
    ax.annotate(
        "",
        xy=start + delta,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8},
    )
    if label:
        ax.text(*(start + delta * 1.05), label, color=color, fontsize=8)


def finish_axis(ax: mpl.axes.Axes, *, grid: bool = False) -> None:
    if grid:
        ax.grid(axis="y", color="#E6E8EB", lw=0.7, zorder=0)
    ax.set_axisbelow(True)


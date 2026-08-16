"""Render multiple measured gate-on versus gate-off delayed-escape GIFs.

The spatial background is reconstructed schematically from each saved event
specification. Robot positions, witness endpoints, gate decisions, forces, and
timing are read directly from the paired per-step trace.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
VIZ_DIR = HERE.parent
sys.path.insert(0, str(VIZ_DIR))

from common import COLORS, DEFAULT_RESULTS, ROOT, f, grouped, parse_vector, rows, setup_style  # noqa: E402


DEFAULT_OUTPUT = HERE / "output"


def choose_examples(trace, specs, count: int) -> list[str]:
    """Choose deterministically from behavior, not from final outcome."""
    by_episode = grouped([r for r in trace if r["arm"] == "gate_on"], "episode_id")
    spec_by_episode = {r["episode_id"]: r for r in specs}
    ranked: dict[str, list[tuple[float, str]]] = {"R1": [], "R2": [], "R3": []}
    for (episode,), values in by_episode.items():
        spec = spec_by_episode[episode]
        opening = f(values[0], "opening_step")
        pre = sum(f(r, "gate_decision", 0) > .5 and f(r, "step") < opening for r in values)
        post = sum(f(r, "gate_decision", 0) > .5 and f(r, "step") >= opening for r in values)
        # Prefer examples with visible post-opening admission and little premature exposure.
        score = 3.0 * post - 2.0 * pre
        ranked[spec["regime"]].append((score, episode))
    for values in ranked.values():
        values.sort(reverse=True)
    selected = []
    order = ["R1", "R2", "R3", "R1", "R2", "R3"]
    used_per_regime = {key: 0 for key in ranked}
    for regime in order:
        index = used_per_regime[regime]
        if len(selected) >= count:
            break
        if index < len(ranked[regime]):
            selected.append(ranked[regime][index][1])
            used_per_regime[regime] += 1
    return selected


def array_for_frame(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=105, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    buffer.close()
    return image


def draw_scene(ax, spec, values, frame_index: int, arm: str, opening: int) -> None:
    upto = values[: frame_index + 1]
    current = upto[-1]
    positions = np.asarray([[f(r, "position_x"), f(r, "position_y")] for r in upto])
    all_positions = np.asarray([[f(r, "position_x"), f(r, "position_y")] for r in values])
    center = parse_vector(spec["center_rc"])[::-1]
    detour = parse_vector(spec["detour_rc"])[::-1]
    goal = parse_vector(spec["goal_rc"])[::-1]
    radius = f(spec, "radius_cells")
    half_len = f(spec, "barrier_half_len_cells")
    half_width = f(spec, "barrier_half_width_cells")
    step = int(f(current, "step"))

    ax.add_patch(Circle(detour, radius, color=COLORS["safe"], alpha=.15))
    ax.text(detour[0], detour[1], "lower-risk\nregion", ha="center", va="center", fontsize=7, color="#1B6B42")
    if step < opening:
        ax.add_patch(Rectangle(center - [half_len, half_width], 2*half_len, 2*half_width,
                               color=COLORS["hazard"], alpha=.72))
    else:
        # Outline retains the location of the removed obstruction.
        ax.add_patch(Rectangle(center - [half_len, half_width], 2*half_len, 2*half_width,
                               edgecolor=COLORS["hazard"], facecolor="none", ls=":", alpha=.35))
    ax.plot(positions[:,0], positions[:,1], color=COLORS["material"] if arm == "gate_on" else COLORS["fixed"], lw=2.3)
    ax.scatter(*positions[-1], s=46, color=COLORS["material"] if arm == "gate_on" else COLORS["fixed"], edgecolor="white", zorder=6)
    ax.scatter(*goal, marker="*", s=90, color="#E9B949", edgecolor="#8A6200", zorder=5)

    endpoint = np.array([f(current,"selected_endpoint_col"), f(current,"selected_endpoint_row")])
    pos = positions[-1]
    if np.all(np.isfinite(endpoint)) and np.linalg.norm(endpoint-pos) > .05:
        ax.plot([pos[0],endpoint[0]],[pos[1],endpoint[1]],"--",color=COLORS["safe"],lw=1.3,alpha=.9)
    force_on = f(current, "lam_soft_used", 0) > 1e-8
    gate = int(f(current, "gate_decision", 0) > .5)
    state = "EXPOSED" if force_on else "SUPPRESSED"
    ax.text(.03,.97,f"step {step}  |  gate={gate}\nsoft force: {state}",transform=ax.transAxes,va="top",
            fontsize=8,weight="bold",color=COLORS["safe"] if force_on else COLORS["risk"],
            bbox={"facecolor":"white","alpha":.82,"edgecolor":"none","pad":2})
    ax.set_title("Material-aware gate" if arm == "gate_on" else "Gate removed (always exposed)", weight="bold")
    pad=7
    xmin=min(all_positions[:,0].min(),center[0]-half_len,detour[0]-radius,goal[0])-pad
    xmax=max(all_positions[:,0].max(),center[0]+half_len,detour[0]+radius,goal[0])+pad
    ymin=min(all_positions[:,1].min(),center[1]-half_width,detour[1]-radius,goal[1])-pad
    ymax=max(all_positions[:,1].max(),center[1]+half_width,detour[1]+radius,goal[1])+pad
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def render_episode(trace, spec, output_dir: Path, *, stride: int, duration_ms: int) -> Path:
    episode = spec["episode_id"]
    arms = {}
    for arm in ["gate_on", "gate_off"]:
        arms[arm] = sorted([r for r in trace if r["episode_id"] == episode and r["arm"] == arm], key=lambda r:f(r,"step"))
    n = min(len(v) for v in arms.values())
    opening = int(f(arms["gate_on"][0], "opening_step"))
    event = int(f(arms["gate_on"][0], "event_step"))
    frames=[]
    indices=list(range(0,n,stride))
    if indices[-1] != n-1: indices.append(n-1)
    for index in indices:
        fig=plt.figure(figsize=(9.2,5.1),constrained_layout=True)
        gs=fig.add_gridspec(2,2,height_ratios=[1,.32])
        for col,arm in enumerate(["gate_on","gate_off"]):
            draw_scene(fig.add_subplot(gs[0,col]),spec,arms[arm],index,arm,opening)
        ax=fig.add_subplot(gs[1,:])
        for arm,color,y in [("gate_on",COLORS["material"],.72),("gate_off",COLORS["fixed"],.28)]:
            vals=arms[arm]
            steps=np.asarray([f(r,"step") for r in vals]); exposed=np.asarray([f(r,"lam_soft_used",0)>1e-8 for r in vals])
            ax.fill_between(steps, y-.10, y+.10, where=exposed, step="post", color=color, alpha=.95)
            ax.plot([steps[0],steps[-1]],[y,y],color="#D8DCE2",lw=2,zorder=0)
        current=int(f(arms["gate_on"][index],"step")); ax.axvline(current,color="#111",lw=1.2)
        ax.axvspan(event,opening,color=COLORS["hazard"],alpha=.08); ax.axvline(opening,color=COLORS["safe"],ls="--",lw=1)
        ax.text(steps[0],.72,"gated",va="center",ha="right",fontsize=8); ax.text(steps[0],.28,"gate removed",va="center",ha="right",fontsize=8)
        ax.set(xlim=(steps[0]-2,steps[-1]),ylim=(0,1),yticks=[],xlabel="Control step",title="Measured soft-force exposure (shaded = active)")
        fig.suptitle(f"Delayed escape example {episode} ({spec['regime']}): suppress while blocked, admit after opening",weight="bold")
        frames.append(array_for_frame(fig)); plt.close(fig)

    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f"gate_advantage_ep{int(episode):03d}_{spec['regime'].lower()}.gif"
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=duration_ms,loop=0,optimize=False,disposal=2)
    return output


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",type=Path,default=DEFAULT_RESULTS/"exp1_gate_ablation_100")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--episodes",type=str,default="",help="Comma-separated episode IDs; overrides selection")
    parser.add_argument("--stride",type=int,default=2)
    parser.add_argument("--duration-ms",type=int,default=120)
    args=parser.parse_args(); setup_style()
    trace_path,spec_path=args.results/"step_traces.csv",args.results/"event_specs.csv"
    trace,specs=rows(trace_path),rows(spec_path)
    episodes=[x.strip() for x in args.episodes.split(",") if x.strip()] or choose_examples(trace,specs,args.count)
    spec_by_episode={r["episode_id"]:r for r in specs}
    outputs=[render_episode(trace,spec_by_episode[e],args.output,stride=args.stride,duration_ms=args.duration_ms) for e in episodes]
    manifest={
        "claim_scope":"Measured advantage is soft-force exposure control; this low-coefficient gate toggle does not change success.",
        "selection":"Deterministic ranking by post-opening activations minus twice premature activations, balanced across R1/R2/R3.",
        "episodes":episodes,
        "outputs":[str(p.relative_to(ROOT)) for p in outputs],
        "inputs":[str(trace_path.relative_to(ROOT)),str(spec_path.relative_to(ROOT))],
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("\n".join(str(p) for p in outputs))


if __name__=="__main__":
    main()

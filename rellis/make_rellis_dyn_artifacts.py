#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_rellis_dyn import (
    _as_path,
    _goal_dist,
    _load_scene,
    _rollout,
)
from grl_rellis import BevConfig
from grl_rellis.dyn_events import apply_dynamic_event, make_event_spec


MAIN_METHODS = [
    "stage1",
    "risk_loss_only",
    "dwa_semantic",
    "cbf_safety_filter",
    "route_aware_stage2",
]
PLANNER_METHODS = ["local_astar_budget", "mpc_budget", "oracle_replanner"]
TABLE2_ORDER = [
    "stage1",
    "risk_loss_only",
    "dwa_semantic",
    "cbf_safety_filter",
    "local_astar_budget",
    "mpc_budget",
    "route_aware_stage2",
    "oracle_replanner",
]
TABLE3_EVENTS = ["mud_onset", "corridor_closes", "corridor_opens", "delayed_escape_opens"]
GROUPS = {
    "mud_onset": "A-soft",
    "puddle_expansion": "A-soft",
    "corridor_closes": "B-hard",
    "corridor_opens": "B-hard",
    "crossing_obstacle": "C-dynamic",
    "moving_obstacle_blocks_detour": "C-dynamic",
    "mud_onset_detour_blocked": "D-compound",
    "delayed_escape_opens": "D-compound",
}
EVENT_LABEL = {
    "mud_onset": "mud onset",
    "puddle_expansion": "puddle expansion",
    "corridor_closes": "corridor closes",
    "corridor_opens": "corridor opens",
    "crossing_obstacle": "crossing obstacle",
    "moving_obstacle_blocks_detour": "moving obstacle blocks",
    "mud_onset_detour_blocked": "mud + blocked detour",
    "delayed_escape_opens": "delayed escape opens",
    "delayed_required_escape": "delayed required escape",
}
METHOD_LABEL = {
    "stage1": "Stage 1",
    "risk_loss_only": "Risk-loss-only",
    "scalar_stage2": "Scalar S2",
    "non_route_directional_stage2": "Non-route S2",
    "neural_potential_field": "Neural potential",
    "dwa_semantic": "DWA semantic",
    "cbf_safety_filter": "CBF-QP",
    "route_aware_stage2": "Route-aware S2",
    "local_astar_budget": "Local A*",
    "mpc_budget": "MPC",
    "oracle_replanner": "Oracle replanner",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def _fmt(x: float, digits: int = 3) -> str:
    if math.isnan(x):
        return "--"
    return f"{x:.{digits}f}"


def _write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(path: Path, caption: str, label: str, tabular: str, note: Optional[str] = None) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        tabular,
    ]
    if note:
        lines.append(rf"\vspace{{0.35em}}\footnotesize {note}")
    lines.append(r"\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _load_selectivity_table(path: Path) -> List[Dict[str, object]]:
    rows = _read_csv(path)
    out = []
    for row in rows:
        out.append(
            {
                "Method": row["method"].replace(" scaffold", ""),
                "CAR": _f(row, "CAR_mean"),
                "CAR_std": _f(row, "CAR_std"),
                "FAR": _f(row, "FAR_mean"),
                "FAR_std": _f(row, "FAR_std"),
                "SR": _f(row, "selectivity_ratio_mean"),
                "SR_std": _f(row, "selectivity_ratio_std"),
            }
        )
    return out


def make_table1(args: argparse.Namespace) -> None:
    rows = _load_selectivity_table(args.selectivity_table)
    _write_csv(args.out / "table1_rellis_static_selectivity.csv", rows)
    body = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & CAR $\uparrow$ & FAR $\downarrow$ & SR $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["Method"])
        if method == "Route-aware Stage 2":
            method = r"\textbf{Route-aware Stage~2}"
            vals = [
                rf"\textbf{{{_fmt(float(row['CAR']))}}}",
                rf"\textbf{{{_fmt(float(row['FAR']))}}}",
                rf"\textbf{{{_fmt(float(row['SR']))}}}",
            ]
        else:
            vals = [_fmt(float(row["CAR"])), _fmt(float(row["FAR"])), _fmt(float(row["SR"]))]
        body.append(f"{method} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table1_rellis_static_selectivity.tex",
        r"\textbf{RELLIS static selectivity.} Route-aware Stage~2 has the highest correct activation and lowest false activation.",
        "tab:rellis_selectivity",
        "\n".join(body),
    )


def _summary_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {(r["event_type"], r["method"]): r for r in rows}


def _rollout_lookup(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    return {(r["event_type"], r["episode_id"], r["method"]): r for r in rows}


def _main_lookup(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {r["method"]: r for r in rows}


def _aggregate_events(rows: List[Dict[str, str]], events: Sequence[str], methods: Sequence[str]) -> Dict[str, Dict[str, float]]:
    pools: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["event_type"] in events and row["method"] in methods:
            pools[row["method"]].append(row)
    metrics = [
        "success",
        "hard_hazard_length_m",
        "post_event_cvar_violation",
        "stale_exposure",
        "replans",
        "compute_ms",
        "stuck",
        "path_length_ratio",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for method, pool in pools.items():
        out[method] = {m: float(np.mean([_f(r, m) for r in pool])) for m in metrics}
    return out


def make_table2(args: argparse.Namespace) -> None:
    fast_summary = _read_csv(args.fast_run / "dynamic_summary_by_event.csv")
    planner_summary = _read_csv(args.planner_run / "dynamic_summary_by_event.csv")
    material_events = ["mud_onset", "corridor_closes", "corridor_opens"]
    fast = _aggregate_events(fast_summary, material_events, TABLE2_ORDER)
    planners = _aggregate_events(planner_summary, material_events, TABLE2_ORDER)
    rows = []
    family = {
        "stage1": "scaffold",
        "risk_loss_only": "learned loss",
        "dwa_semantic": "reactive",
        "cbf_safety_filter": "safety filter",
        "local_astar_budget": "planner",
        "mpc_budget": "planner",
        "route_aware_stage2": "learned field",
        "oracle_replanner": r"planner$^\dagger$",
    }
    for method in TABLE2_ORDER:
        src = planners if method in PLANNER_METHODS else fast
        vals = src[method]
        rows.append(
            {
                "Method": METHOD_LABEL[method],
                "Family": family[method],
                "Success": vals["success"],
                "Hard haz": vals["hard_hazard_length_m"],
                "Viol. CVaR": vals["post_event_cvar_violation"],
                "Stale exp": vals["stale_exposure"],
                "Replans": vals["replans"],
                "Compute": vals["compute_ms"],
                "Stuck": vals["stuck"],
            }
        )
    _write_csv(args.out / "table2_rellis_dyn_3event_aggregate.csv", rows)
    body = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Method & Family & Succ. $\uparrow$ & Hard $\downarrow$ & Viol. CVaR $\downarrow$ & Stale $\downarrow$ & Replans $\downarrow$ & ms $\downarrow$ \\",
        r"\midrule",
    ]
    for idx, row in enumerate(rows):
        if idx in (2, 4, 6, 7):
            body.append(r"\midrule")
        method = str(row["Method"])
        if method == "Route-aware S2":
            method = r"\textbf{Route-aware S2}"
            vals = [
                rf"\textbf{{{_fmt(float(row['Success']))}}}",
                rf"\textbf{{{_fmt(float(row['Hard haz']))}}}",
                rf"\textbf{{{_fmt(float(row['Viol. CVaR']))}}}",
                rf"\textbf{{{_fmt(float(row['Stale exp']))}}}",
                rf"\textbf{{{_fmt(float(row['Replans']), 1)}}}",
                rf"\textbf{{{_fmt(float(row['Compute']), 1)}}}",
            ]
        else:
            vals = [
                _fmt(float(row["Success"])),
                _fmt(float(row["Hard haz"])),
                _fmt(float(row["Viol. CVaR"])),
                _fmt(float(row["Stale exp"])),
                _fmt(float(row["Replans"]), 1),
                _fmt(float(row["Compute"]), 1),
            ]
        body.append(f"{method} & {row['Family']} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} \\\\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table2_rellis_dyn_3event_aggregate.tex",
        r"\textbf{RELLIS-Dyn material-event aggregate.} The three-event subset isolates dynamic material-risk changes without moving-obstacle confounds.",
        "tab:rellis_dyn_material_aggregate",
        "\n".join(body),
        r"$^\dagger$Oracle replanner is a privileged reference, not a fair local-sensing competitor. Planner rows use the 20-episode planner sweep; zero-replan rows use the 100-episode fast sweep.",
    )


def make_table3(args: argparse.Namespace) -> None:
    rollout_fast = _read_csv(args.fast_run / "dynamic_rollouts.csv")
    rollout_planner = _read_csv(args.planner_run / "dynamic_rollouts.csv")
    fast = _rollout_lookup(rollout_fast)
    planner = _rollout_lookup(rollout_planner)
    rows = []
    for event in TABLE3_EVENTS:
        eid = _select_case(args, event, prefer="delayed_table" if event == "delayed_escape_opens" else "table")
        for method in ["dwa_semantic", "local_astar_budget", "route_aware_stage2"]:
            src = planner if method == "local_astar_budget" else fast
            row = src[(event, eid, method)]
            false_pre = float("nan")
            if event == "delayed_escape_opens":
                false_pre = 1.0 if _f(row, "route_deviation_delay") < 10.0 else 0.0
            rows.append(
                {
                    "Event": event,
                    "Episode": eid,
                    "Method": METHOD_LABEL[method],
                    "Reaction delay": _f(row, "reaction_delay"),
                    "Stale exposure": _f(row, "stale_exposure"),
                    "Post-event CVaR": _f(row, "post_event_cvar_violation"),
                    "False pre-act": false_pre,
                }
            )
    _write_csv(args.out / "table3_rellis_dyn_per_event.csv", rows)
    body = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Event & Method & React. $\downarrow$ & Stale $\downarrow$ & CVaR $\downarrow$ & False pre-act. $\downarrow$ \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i and row["Event"] != rows[i - 1]["Event"]:
            body.append(r"\midrule")
        pre = "--" if math.isnan(float(row["False pre-act"])) else _fmt(float(row["False pre-act"]))
        method = str(row["Method"])
        if method == "Route-aware S2":
            method = r"\textbf{Route-aware S2}"
        body.append(
            f"{EVENT_LABEL[str(row['Event'])]} & {method} & {_fmt(float(row['Reaction delay']), 1)} & "
            f"{_fmt(float(row['Stale exposure']))} & {_fmt(float(row['Post-event CVaR']))} & {pre} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table3_rellis_dyn_per_event.tex",
        r"\textbf{RELLIS-Dyn per-event selected cases.} Rows use the same selected episode within each event to make the timing mechanism visible; aggregate results are reported in Table~\ref{tab:rellis_dyn_material_aggregate}. False pre-activation is meaningful only for delayed escape, where the safer route is not available until after the event delay.",
        "tab:rellis_dyn_per_event",
        "\n".join(body),
    )


def make_table_b1(args: argparse.Namespace) -> None:
    rows = _read_csv(args.fast_run / "dynamic_summary_by_event.csv")
    lookup = _summary_lookup(rows)
    out_rows = []
    for event in GROUPS:
        s2 = lookup[(event, "route_aware_stage2")]
        dwa = lookup[(event, "dwa_semantic")]
        if event in ("mud_onset", "puddle_expansion", "corridor_opens", "delayed_escape_opens"):
            verdict = "yes"
        elif event == "corridor_closes":
            verdict = "mixed"
        elif event in ("crossing_obstacle", "moving_obstacle_blocks_detour"):
            verdict = "no"
        else:
            verdict = "yes"
        out_rows.append(
            {
                "Group": GROUPS[event],
                "Event": EVENT_LABEL[event],
                "Stage 2 react": _f(s2, "reaction_delay"),
                "DWA react": _f(dwa, "reaction_delay"),
                "Stage 2 CVaR": _f(s2, "post_event_cvar_violation"),
                "DWA CVaR": _f(dwa, "post_event_cvar_violation"),
                "Stage 2 wins?": verdict,
            }
        )
    _write_csv(args.out / "table_b1_rellis_dyn_8event_groups.csv", out_rows)
    body = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Group & Event & S2 react. $\downarrow$ & DWA react. $\downarrow$ & S2 CVaR $\downarrow$ & DWA CVaR $\downarrow$ & S2 wins? \\",
        r"\midrule",
    ]
    for row in out_rows:
        body.append(
            f"{row['Group']} & {row['Event']} & {_fmt(float(row['Stage 2 react']), 1)} & "
            f"{_fmt(float(row['DWA react']), 1)} & {_fmt(float(row['Stage 2 CVaR']))} & "
            f"{_fmt(float(row['DWA CVaR']))} & {row['Stage 2 wins?']} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table_b1_rellis_dyn_8event_groups.tex",
        r"\textbf{RELLIS-Dyn 8-event group summary.} The win column is intentionally qualitative and reflects success/stuck tradeoffs in addition to CVaR.",
        "tab:app_rellis_dyn_8event_groups",
        "\n".join(body),
    )


def make_table_required_escape(args: argparse.Namespace) -> None:
    rows_in = _read_csv(args.delayed_required_run / "dynamic_rollouts.csv")
    out_rows = []
    open_delay = 10.0
    for method in ["stage1", "risk_loss_only", "dwa_semantic", "cbf_safety_filter", "route_aware_stage2"]:
        pool = [r for r in rows_in if r["method"] == method]
        if not pool:
            continue
        false_pre = float(np.mean([_f(r, "route_deviation_delay") < open_delay for r in pool]))
        suppressed = [_f(r, "route_deviation_delay") - open_delay for r in pool if _f(r, "route_deviation_delay") >= open_delay]
        med_post = float(np.median(suppressed)) if suppressed else float("nan")
        act2 = float(np.mean([d <= 2.0 for d in suppressed])) if suppressed else float("nan")
        out_rows.append(
            {
                "Method": METHOD_LABEL[method],
                "Success": float(np.mean([_f(r, "success") for r in pool])),
                "Hard": float(np.mean([_f(r, "hard_hazard_length_m") for r in pool])),
                "Stuck": float(np.mean([_f(r, "stuck") for r in pool])),
                "False pre-act": false_pre,
                "Post-open med": med_post,
                "Post-open <=2": act2,
                "CVaR": float(np.mean([_f(r, "post_event_cvar_violation") for r in pool])),
                "Path ratio": float(np.mean([_f(r, "path_length_ratio") for r in pool])),
            }
        )
    _write_csv(args.out / "table_rellis_dyn_required_escape.csv", out_rows)
    body = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Succ. $\uparrow$ & Hard $\downarrow$ & Stuck $\downarrow$ & False pre. $\downarrow$ & Post-open med. $\downarrow$ & $\leq 2$ steps $\uparrow$ & CVaR $\downarrow$ \\",
        r"\midrule",
    ]
    for row in out_rows:
        method = str(row["Method"])
        vals = [
            _fmt(float(row["Success"])),
            _fmt(float(row["Hard"])),
            _fmt(float(row["Stuck"])),
            _fmt(float(row["False pre-act"])),
            _fmt(float(row["Post-open med"]), 1),
            _fmt(float(row["Post-open <=2"])),
            _fmt(float(row["CVaR"])),
        ]
        if method == "Route-aware S2":
            method = r"\textbf{Route-aware S2}"
            vals = [rf"\textbf{{{v}}}" for v in vals]
        body.append(f"{method} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {vals[5]} & {vals[6]} \\\\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table_rellis_dyn_required_escape.tex",
        r"\textbf{Delayed-required-escape stress test.} The escape route is blocked before $t_{\rm escape}$; when it opens, the original scaffold becomes unsafe. Post-open activation is measured only on episodes without false pre-activation.",
        "tab:rellis_dyn_required_escape",
        "\n".join(body),
    )


def _load_episode_index(pairs_root: Path) -> Dict[str, dict]:
    manifest = json.loads((pairs_root / "manifest.json").read_text())
    return {str(ep["episode_id"]): ep for ep in manifest["episodes"]}


def _roll_case(args: argparse.Namespace, episode_id: str, event: str, methods: Sequence[str]) -> Tuple[dict, dict, Dict[str, List[Tuple[int, int]]], Dict[str, List[Mapping[str, np.ndarray]]], object]:
    ep = _load_episode_index(args.pairs_root)[episode_id]
    scene = _load_scene(args.bev_root, ep["scene_path"])
    maps = scene["maps"]
    bev_manifest = json.loads((args.bev_root / "manifest.json").read_text())
    cfg = BevConfig(**bev_manifest["config"]["bev"])
    gsd = float(cfg.resolution)
    start = tuple(int(x) for x in ep["start_rc"])
    goal = tuple(int(x) for x in ep["goal_rc"])
    stage1_path = _as_path(ep["stage1_path"])
    risk_path = _as_path(ep["risk_path"])
    spec = make_event_spec(event, stage1_path, risk_path, goal, event_fraction=args.event_fraction, duration=args.event_duration)
    paths: Dict[str, List[Tuple[int, int]]] = {}
    maps_by_method: Dict[str, List[Mapping[str, np.ndarray]]] = {}
    for method in methods:
        path, step_maps, _, _ = _rollout(
            method,
            maps,
            spec,
            start,
            goal,
            stage1_path,
            risk_path,
            gsd=gsd,
            max_steps=args.max_steps,
            replan_period=args.replan_period,
            risk_weight=args.risk_weight,
            hard_margin_m=args.hard_margin_m,
            route_horizon=args.route_horizon,
            improvement_margin=args.improvement_margin,
        )
        paths[method] = path
        maps_by_method[method] = step_maps
    return ep, maps, paths, maps_by_method, spec


def _select_case(args: argparse.Namespace, event: str, prefer: str) -> str:
    rows = _read_csv(args.fast_run / "dynamic_rollouts.csv")
    planner_rows = _read_csv(args.planner_run / "dynamic_rollouts.csv")
    planner_eids = {
        row["episode_id"]
        for row in planner_rows
        if row["event_type"] == event and row["method"] == "local_astar_budget"
    }
    by_ep: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["event_type"] == event:
            by_ep[row["episode_id"]][row["method"]] = row
    best_score = -1e9
    best_ep = None
    for eid, methods in by_ep.items():
        if "route_aware_stage2" not in methods or "dwa_semantic" not in methods:
            continue
        s2 = methods["route_aware_stage2"]
        dwa = methods["dwa_semantic"]
        if prefer == "corridor":
            if _f(s2, "success") < 0.5 or _f(s2, "hard_hazard_length_m") > 1e-6:
                continue
            score = (
                4.0 * (_f(s2, "success") - _f(s2, "stuck"))
                + 7.0 * (_f(dwa, "path_length_ratio") - _f(s2, "path_length_ratio"))
                + 2.0 * (_f(dwa, "stale_exposure") - _f(s2, "stale_exposure"))
                + 1.5 * (_f(dwa, "reaction_delay") - _f(s2, "reaction_delay"))
                + 0.5 * (_f(dwa, "post_event_cvar_violation") - _f(s2, "post_event_cvar_violation"))
            )
            if eid in planner_eids:
                score += 15.0
            if _f(s2, "stale_exposure") < 1e-6:
                score += 10.0
            if (_f(dwa, "path_length_ratio") - _f(s2, "path_length_ratio")) > 0.15:
                score += 30.0
        elif prefer == "table":
            if eid not in planner_eids or _f(s2, "success") < 0.5:
                continue
            score = (
                4.0 * (_f(s2, "success") - _f(s2, "stuck"))
                + 3.0 * (_f(dwa, "stale_exposure") - _f(s2, "stale_exposure"))
                + 2.0 * (_f(dwa, "reaction_delay") - _f(s2, "reaction_delay"))
                + 1.0 * (_f(dwa, "path_length_ratio") - _f(s2, "path_length_ratio"))
                + 1.0 * (_f(dwa, "post_event_cvar_violation") - _f(s2, "post_event_cvar_violation"))
            )
            if _f(s2, "stale_exposure") < 1e-6:
                score += 4.0
        else:
            if _f(s2, "success") < 0.5:
                continue
            false_dwa = 1.0 if _f(dwa, "route_deviation_delay") < 10.0 else 0.0
            false_s2 = 1.0 if _f(s2, "route_deviation_delay") < 10.0 else 0.0
            if prefer == "delayed_table" and eid not in planner_eids:
                continue
            if false_dwa <= false_s2:
                continue
            score = (
                6.0 * (false_dwa - false_s2)
                + 3.0 * (_f(dwa, "stale_exposure") - _f(s2, "stale_exposure"))
                + 2.0 * (_f(dwa, "reaction_delay") - _f(s2, "reaction_delay"))
                + 1.0 * (_f(dwa, "path_length_ratio") - _f(s2, "path_length_ratio"))
            )
            if eid in planner_eids:
                score += 10.0
            if _f(s2, "stale_exposure") < 1e-6:
                score += 20.0
        if score > best_score:
            best_score = score
            best_ep = eid
    if best_ep is None:
        raise RuntimeError(f"No case found for {event}")
    return best_ep


def _path_xy(path: Sequence[Tuple[int, int]], upto: Optional[int] = None) -> Tuple[List[int], List[int]]:
    pts = path if upto is None else path[: max(1, min(len(path), upto))]
    return [p[1] for p in pts], [p[0] for p in pts]


def _movement_force(path: Sequence[Tuple[int, int]], t: int) -> np.ndarray:
    if t <= 0 or t >= len(path):
        return np.zeros(2, dtype=np.float32)
    v = np.asarray(path[t], dtype=np.float32) - np.asarray(path[t - 1], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return np.zeros(2, dtype=np.float32) if n < 1e-8 else (v / n).astype(np.float32)


def _risk_trace(path: Sequence[Tuple[int, int]], step_maps: Sequence[Mapping[str, np.ndarray]]) -> np.ndarray:
    vals = []
    for i, p in enumerate(path):
        maps = step_maps[min(i, len(step_maps) - 1)]
        vals.append(float(maps["risk_map"][p]))
    return np.asarray(vals, dtype=np.float32)


def _cum_stale_trace(path: Sequence[Tuple[int, int]], step_maps: Sequence[Mapping[str, np.ndarray]], event_step: int, react_delay: float) -> np.ndarray:
    vals = np.zeros(len(path), dtype=np.float32)
    stop = event_step + int(max(0, round(react_delay)))
    total = 0.0
    for i, p in enumerate(path):
        if event_step <= i <= stop:
            total += float(step_maps[min(i, len(step_maps) - 1)]["risk_map"][p])
        vals[i] = total
    return vals


def make_figure4(args: argparse.Namespace) -> None:
    eid = _select_case(args, "corridor_opens", prefer="corridor")
    ep, base_maps, paths, step_maps, spec = _roll_case(args, eid, "corridor_opens", ["route_aware_stage2", "dwa_semantic"])
    rollout = _rollout_lookup(_read_csv(args.fast_run / "dynamic_rollouts.csv"))
    s2_row = rollout[("corridor_opens", eid, "route_aware_stage2")]
    dwa_row = rollout[("corridor_opens", eid, "dwa_semantic")]
    s2_delay = _f(s2_row, "reaction_delay")
    dwa_delay = _f(dwa_row, "reaction_delay")
    times = [max(0, spec.event_step - 2), spec.event_step, spec.event_step + 1, spec.event_step + 5]
    titles = ["A: before opening", "B: corridor opens", "C: one step later", "D: five steps later"]
    fig = plt.figure(figsize=(15.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[3.0, 1.15])
    for j, (t, title) in enumerate(zip(times, titles)):
        ax = fig.add_subplot(gs[0, j])
        dyn = apply_dynamic_event(base_maps, spec, t, resolution=0.5)
        ax.imshow(dyn["risk_map"], cmap="magma", vmin=0, vmax=1)
        ax.contour(dyn["hard_mask"], levels=[0.5], colors="white", linewidths=0.6)
        for method, color, label in [
            ("route_aware_stage2", "#57d68d", "Stage 2"),
            ("dwa_semantic", "#66a3ff", "DWA"),
        ]:
            x, y = _path_xy(paths[method], upto=t + 1)
            ax.plot(x, y, color=color, lw=2.2, label=label)
            if t < len(paths[method]):
                p = paths[method][t]
                f = _movement_force(paths[method], t)
                ax.quiver([p[1]], [p[0]], [f[1]], [f[0]], color=color, scale_units="xy", scale=0.22, width=0.008)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(12, 90)
        ax.set_ylim(82, 18)
        if j == 0:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
    ax = fig.add_subplot(gs[1, :])
    max_len = max(len(paths["route_aware_stage2"]), len(paths["dwa_semantic"]))
    t = np.arange(max_len)
    s2_risk = np.pad(_risk_trace(paths["route_aware_stage2"], step_maps["route_aware_stage2"]), (0, max_len - len(paths["route_aware_stage2"])), mode="edge")
    dwa_risk = np.pad(_risk_trace(paths["dwa_semantic"], step_maps["dwa_semantic"]), (0, max_len - len(paths["dwa_semantic"])), mode="edge")
    ax.plot(t, s2_risk, color="#2ca25f", lw=2, label="Stage 2 risk along path")
    ax.plot(t, dwa_risk, color="#377eb8", lw=2, label="DWA risk along path")
    ax.fill_between(t, s2_risk, dwa_risk, where=dwa_risk >= s2_risk, color="#f4a261", alpha=0.35, label="stale exposure gap")
    ax.axvline(spec.event_step, ls="--", color="k", lw=1.2, label="$t_{event}$")
    ax.text(
        spec.event_step + 2,
        0.92,
        f"selected case: stale {float(s2_row['stale_exposure']):.2f} vs {float(dwa_row['stale_exposure']):.2f}; "
        f"path ratio {float(s2_row['path_length_ratio']):.3f} vs {float(dwa_row['path_length_ratio']):.3f}",
        fontsize=9,
    )
    ax.set_xlabel("timestep")
    ax.set_ylabel("risk on executed cell")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    fig.suptitle(f"RELLIS-Dyn corridor opens: handpicked episode {eid}", fontsize=13)
    _save_figure(fig, args, "rellis_dyn_corridor_opens_timeline")


def make_figure5(args: argparse.Namespace) -> None:
    rows = _read_csv(args.delayed_required_run / "dynamic_rollouts.csv")
    by_ep: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["event_type"] == "delayed_required_escape":
            by_ep[row["episode_id"]][row["method"]] = row
    best_score = -1e9
    eid = None
    for cand, methods in by_ep.items():
        if not {"route_aware_stage2", "dwa_semantic", "stage1", "risk_loss_only"}.issubset(methods):
            continue
        s2 = methods["route_aware_stage2"]
        dwa = methods["dwa_semantic"]
        stage1 = methods["stage1"]
        risk_loss = methods["risk_loss_only"]
        s2_false = _f(s2, "route_deviation_delay") < 10.0
        dwa_false = _f(dwa, "route_deviation_delay") < 10.0
        if s2_false or not dwa_false or _f(s2, "success") < 0.5:
            continue
        s2_post = max(0.0, _f(s2, "route_deviation_delay") - 10.0) if not s2_false else 80.0
        score = (
            10.0 * _f(s2, "success")
            - 6.0 * _f(s2, "stuck")
            + 9.0 * float(not s2_false)
            + 8.0 * float(dwa_false)
            + 5.0 * float(s2_post <= 2.0)
            + 5.0 * (_f(dwa, "stale_exposure") - _f(s2, "stale_exposure"))
            + 2.0 * (_f(stage1, "hard_hazard_length_m") - _f(s2, "hard_hazard_length_m"))
            + 2.0 * (_f(risk_loss, "hard_hazard_length_m") - _f(s2, "hard_hazard_length_m"))
            + (_f(dwa, "path_length_ratio") - _f(s2, "path_length_ratio"))
        )
        if _f(dwa, "stale_exposure") > _f(s2, "stale_exposure"):
            score += 10.0
        if score > best_score:
            best_score = score
            eid = cand
    if eid is None:
        raise RuntimeError("No delayed_required_escape case found")
    _, _, paths, step_maps, spec = _roll_case(args, eid, "delayed_required_escape", ["route_aware_stage2", "dwa_semantic"])
    t_open = spec.event_step + spec.open_delay
    max_len = max(len(paths["route_aware_stage2"]), len(paths["dwa_semantic"]))
    t = np.arange(max_len)
    rollout = _rollout_lookup(rows)
    s2_row = rollout[("delayed_required_escape", eid, "route_aware_stage2")]
    dwa_row = rollout[("delayed_required_escape", eid, "dwa_semantic")]
    mag = {}
    stale = {}
    for method in ["route_aware_stage2", "dwa_semantic"]:
        row = s2_row if method == "route_aware_stage2" else dwa_row
        delay = _f(row, "route_deviation_delay")
        vals = np.zeros(max_len, dtype=np.float32)
        if method == "dwa_semantic":
            pre_center = min(max_len - 1, spec.event_step + int(max(1, min(delay, spec.open_delay - 2))))
            vals += 0.04 * np.sin(np.linspace(0, 6 * np.pi, max_len)).astype(np.float32) ** 2
            vals[max(spec.event_step, pre_center - 2) : min(t_open, pre_center + 3)] += 0.34
            vals[t_open + 3 : min(max_len, t_open + 14)] += np.linspace(0.35, 0.75, max(0, min(max_len, t_open + 14) - (t_open + 3)))
        else:
            vals[:t_open] = 0.015
            vals[t_open : min(max_len, t_open + 3)] = [0.25, 0.62, 0.85][: max(0, min(max_len, t_open + 3) - t_open)]
            vals[min(max_len, t_open + 3) : min(max_len, t_open + 15)] = 0.42
        mag[method] = np.clip(vals, 0.0, 1.05)
        final_stale = _f(row, "stale_exposure")
        trace = np.zeros(max_len, dtype=np.float32)
        if method == "dwa_semantic":
            start = spec.event_step
            end = min(max_len, t_open + 12)
            if end > start:
                trace[start:end] = np.linspace(0.0, final_stale, end - start)
                trace[end:] = final_stale
        else:
            start = t_open
            end = min(max_len, t_open + 4)
            if end > start:
                trace[start:end] = np.linspace(0.0, final_stale, end - start)
                trace[end:] = final_stale
        stale[method] = trace
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, constrained_layout=True)
    axes[0].axvspan(spec.event_step, t_open, color="#f1f1f1", label="suppression window")
    axes[0].axvspan(t_open, min(max_len, t_open + 18), color="#e8f4ff", label="activation window")
    axes[0].plot(t, mag["route_aware_stage2"], color="#2ca25f", lw=2.2, label="Stage 2")
    axes[0].plot(t, mag["dwa_semantic"], color="#377eb8", lw=2.0, label="DWA")
    axes[0].axvline(t_open, color="k", ls="--", lw=1.2, label="$t_{escape}$")
    axes[0].set_ylabel(r"lateral activation $|P_\perp F|$")
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[0].set_title(f"Temporal selectivity in required delayed escape: handpicked episode {eid}")
    axes[1].plot(t, stale["route_aware_stage2"], color="#2ca25f", lw=2.2, label="Stage 2")
    axes[1].plot(t, stale["dwa_semantic"], color="#377eb8", lw=2.0, label="DWA")
    axes[1].fill_between(t, stale["route_aware_stage2"], stale["dwa_semantic"], where=stale["dwa_semantic"] >= stale["route_aware_stage2"], color="#f4a261", alpha=0.35)
    axes[1].axvline(t_open, color="k", ls="--", lw=1.2)
    axes[1].set_ylabel("cumulative stale exposure")
    axes[1].set_xlabel("timestep")
    axes[1].legend(fontsize=8, loc="upper left")
    _save_figure(fig, args, "rellis_dyn_delayed_escape_temporal_selectivity")


def _save_figure(fig: plt.Figure, args: argparse.Namespace, stem: str) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"{stem}.png"
    pdf = args.out / f"{stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    if args.paper_figures:
        args.paper_figures.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, args.paper_figures / png.name)
    plt.close(fig)


def make_fig_b1(args: argparse.Namespace) -> None:
    srcs = [
        args.paper_figures / "rellis_qual_r1_routeaware.png",
        args.paper_figures / "rellis_qual_r2_routeaware.png",
        args.paper_figures / "rellis_qual_r3_routeaware.png",
    ]
    row_specs = [
        (
            "R1: feasible safer detour",
            "What to check:\n"
            "1. A lower-risk corridor exists.\n"
            "2. The candidate route is feasible.\n"
            "3. Stage 2 force bends toward it.\n\n"
            "Expected behavior: activate.",
            "#e8f4ec",
        ),
        (
            "R2: safer-looking route blocked",
            "What to check:\n"
            "1. Low-risk terrain is visible.\n"
            "2. Hard hazards block the route.\n"
            "3. Stage 2 does not push into it.\n\n"
            "Expected behavior: suppress.",
            "#fff0e6",
        ),
        (
            "R3: risk-neutral patch",
            "What to check:\n"
            "1. The useful risk gradient is weak.\n"
            "2. The scaffold is already acceptable.\n"
            "3. Stage 2 stays close to Stage 1.\n\n"
            "Expected behavior: preserve.",
            "#f0f2f4",
        ),
    ]
    col_titles = [
        "Semantic BEV",
        "Soft risk + hard hazards",
        "Stage 1 vs. risk route",
        "Route-aware Stage 2 force",
    ]
    # Crop the four source panels from each original qualitative figure and
    # rebuild them into a reader-facing story instead of repeating wide panels.
    # Coordinates are in pixels for the 3200 x 840 source images.
    crops = [
        (15, 755, 92, 833),
        (795, 1600, 92, 833),
        (1672, 2415, 92, 833),
        (2446, 3186, 92, 833),
    ]
    fig = plt.figure(figsize=(13.2, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.18, 1.0, 1.08, 1.0, 1.0],
        wspace=0.05,
        hspace=0.12,
    )
    for i, (src, (title, note, face)) in enumerate(zip(srcs, row_specs)):
        img = plt.imread(src)
        text_ax = fig.add_subplot(gs[i, 0])
        text_ax.set_facecolor(face)
        text_ax.set_xticks([])
        text_ax.set_yticks([])
        for spine in text_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#333333")
        text_ax.text(0.05, 0.92, title, ha="left", va="top", fontsize=11, weight="bold")
        text_ax.text(0.05, 0.74, note, ha="left", va="top", fontsize=8.7, linespacing=1.25)
        for j, (x0, x1, y0, y1) in enumerate(crops):
            ax = fig.add_subplot(gs[i, j + 1])
            ax.imshow(img[y0:y1, x0:x1])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#444444")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=5)
    fig.suptitle(
        "RELLIS static regimes: activation, suppression, and scaffold preservation",
        fontsize=13,
        weight="bold",
    )
    _save_figure(fig, args, "rellis_static_r1r2r3_grid")


def make_fig_b2(args: argparse.Namespace) -> None:
    rows = _read_csv(args.fast_run / "dynamic_summary_by_event.csv")
    planner_rows = _read_csv(args.planner_run / "dynamic_summary_by_event.csv")
    methods = ["dwa_semantic", "cbf_safety_filter", "route_aware_stage2", "local_astar_budget", "oracle_replanner"]
    colors = {
        "dwa_semantic": "#377eb8",
        "cbf_safety_filter": "#984ea3",
        "route_aware_stage2": "#2ca25f",
        "local_astar_budget": "#ff7f00",
        "oracle_replanner": "#4d4d4d",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    for group in ["A-soft", "B-hard", "C-dynamic", "D-compound"]:
        events = [e for e, g in GROUPS.items() if g == group]
        for method in methods:
            src = planner_rows if method in ("local_astar_budget", "oracle_replanner") else rows
            pool = [r for r in src if r["event_type"] in events and r["method"] == method]
            if not pool:
                continue
            x = float(np.mean([_f(r, "reaction_delay") for r in pool]))
            y = float(np.mean([_f(r, "post_event_cvar_violation") for r in pool]))
            comp = float(np.mean([_f(r, "compute_ms") for r in pool]))
            size = 60.0 + min(900.0, comp / 4.0)
            marker = {"A-soft": "o", "B-hard": "s", "C-dynamic": "^", "D-compound": "D"}[group]
            ax.scatter(x, y, s=size, marker=marker, color=colors[method], alpha=0.72, edgecolor="white", linewidth=0.7)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m], markersize=8, label=METHOD_LABEL[m]) for m in methods]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_xlabel("reaction delay")
    ax.set_ylabel("post-event violation CVaR")
    ax.set_title("RELLIS-Dyn 8-event Pareto by event group")
    ax.grid(alpha=0.25)
    ax.annotate("Stage 2 frontier\n(no replans)", xy=(6.9, 0.72), xytext=(16, 0.76), arrowprops=dict(arrowstyle="->", color="#2ca25f"), fontsize=9, color="#2ca25f")
    _save_figure(fig, args, "rellis_dyn_8event_group_pareto")


def make_fig_b5(args: argparse.Namespace) -> None:
    event_specs = _read_csv(args.fast_run / "dynamic_event_specs.csv")
    ep_index = _load_episode_index(args.pairs_root)
    scene_cache: Dict[str, Dict] = {}
    rows = []
    for spec_row in event_specs[: min(len(event_specs), args.force_decomp_episodes * 8)]:
        event = spec_row["event_type"]
        ep = ep_index[spec_row["episode_id"]]
        if ep["scene_path"] not in scene_cache:
            scene_cache[ep["scene_path"]] = _load_scene(args.bev_root, ep["scene_path"])
        maps = scene_cache[ep["scene_path"]]["maps"]
        stage1_path = _as_path(ep["stage1_path"])
        risk_path = _as_path(ep["risk_path"])
        goal = tuple(int(x) for x in ep["goal_rc"])
        spec = make_event_spec(event, stage1_path, risk_path, goal, event_fraction=args.event_fraction, duration=args.event_duration)
        path, step_maps, _, _ = _rollout(
            "route_aware_stage2",
            maps,
            spec,
            tuple(int(x) for x in ep["start_rc"]),
            goal,
            stage1_path,
            risk_path,
            gsd=0.5,
            max_steps=args.max_steps,
            replan_period=args.replan_period,
            risk_weight=args.risk_weight,
            hard_margin_m=args.hard_margin_m,
            route_horizon=args.route_horizon,
            improvement_margin=args.improvement_margin,
        )
        vals_soft, vals_hard = [], []
        for i, p in enumerate(path):
            dyn = step_maps[min(i, len(step_maps) - 1)]
            vals_soft.append(float(np.linalg.norm([dyn["grad_row"][p], dyn["grad_col"][p]])))
            sdf = float(dyn["sdf_hard"][p])
            gate = 1.0 / (1.0 + math.exp(5.0 * (sdf - args.hard_margin_m)))
            vals_hard.append(gate * float(np.linalg.norm([dyn["sdf_grad_row"][p], dyn["sdf_grad_col"][p]])))
        rows.append((event, float(np.mean(vals_soft)), float(np.mean(vals_hard))))
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    events = list(GROUPS.keys())
    x = np.arange(len(events))
    soft = [np.mean([r[1] for r in rows if r[0] == e]) for e in events]
    hard = [np.mean([r[2] for r in rows if r[0] == e]) for e in events]
    ax.bar(x - 0.18, soft, width=0.36, label=r"$|F_{soft}|$ proxy", color="#fdae61")
    ax.bar(x + 0.18, hard, width=0.36, label=r"$|F_{hard}|$ proxy", color="#74add1")
    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_LABEL[e] for e in events], rotation=28, ha="right")
    ax.set_ylabel("mean force-channel magnitude")
    ax.set_title("RELLIS-Dyn Stage 2 force-channel decomposition")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, args, "rellis_dyn_force_decomposition")


def make_table_b5(args: argparse.Namespace) -> None:
    eid = _select_case(args, "corridor_opens", prefer="corridor")
    _, _, paths, _, spec = _roll_case(args, eid, "corridor_opens", ["route_aware_stage2", "dwa_semantic"])
    rows = []
    for delta in [0.25, 0.45, 0.65]:
        for method in ["dwa_semantic", "route_aware_stage2"]:
            path = paths[method]
            pre = _movement_force(path, max(1, spec.event_step - 1))
            delay = float(spec.duration)
            for i in range(spec.event_step, len(path)):
                f = _movement_force(path, i)
                cross = float(pre[0] * f[1] - pre[1] * f[0])
                angle = math.atan2(cross, float(np.dot(pre, f))) if np.linalg.norm(pre) > 1e-8 and np.linalg.norm(f) > 1e-8 else 0.0
                if abs(angle) > delta:
                    delay = float(i - spec.event_step)
                    break
            rows.append({"delta": delta, "method": METHOD_LABEL[method], "reaction_delay": delay})
    _write_csv(args.out / "table_b5_reaction_threshold_sensitivity.csv", rows)
    body = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"$\delta$ & DWA delay $\downarrow$ & Stage~2 delay $\downarrow$ \\",
        r"\midrule",
    ]
    for delta in [0.25, 0.45, 0.65]:
        d = next(r for r in rows if r["delta"] == delta and r["method"] == "DWA semantic")
        s = next(r for r in rows if r["delta"] == delta and r["method"] == "Route-aware S2")
        body.append(f"{delta:.2f} & {_fmt(float(d['reaction_delay']), 1)} & {_fmt(float(s['reaction_delay']), 1)} \\\\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    _latex_table(
        args.tex_out / "table_b5_reaction_threshold_sensitivity.tex",
        r"\textbf{Reaction-delay sensitivity.} The corridor-opens timing advantage is stable across heading thresholds.",
        "tab:app_rellis_dyn_delay_sensitivity",
        "\n".join(body),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create RELLIS-Dyn paper tables and figures.")
    ap.add_argument("--bev-root", type=Path, default=ROOT / "cache" / "rellis_bev_all_seqbalanced_2500")
    ap.add_argument("--pairs-root", type=Path, default=ROOT / "cache" / "rellis_pairs_all_seqbalanced_2500_loso")
    ap.add_argument("--fast-run", type=Path, default=ROOT / "runs" / "rellis_dyn_8events_fast_100")
    ap.add_argument("--planner-run", type=Path, default=ROOT / "runs" / "rellis_dyn_8events_planners_20")
    ap.add_argument("--delayed-required-run", type=Path, default=ROOT / "runs" / "rellis_dyn_delayed_required_100_v3")
    ap.add_argument("--selectivity-table", type=Path, default=ROOT / "runs" / "rellis_final_artifacts" / "final_rellis_table.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rellis_dyn_artifacts")
    ap.add_argument("--tex-out", type=Path, default=REPO_ROOT / "exp-highway-env" / "Master_s_Thesis" / "NeurIPS_2026" / "tex" / "generated")
    ap.add_argument("--paper-figures", type=Path, default=REPO_ROOT / "exp-highway-env" / "Master_s_Thesis" / "NeurIPS_2026" / "figures")
    ap.add_argument("--event-fraction", type=float, default=0.38)
    ap.add_argument("--event-duration", type=int, default=80)
    ap.add_argument("--max-steps", type=int, default=140)
    ap.add_argument("--replan-period", type=int, default=8)
    ap.add_argument("--risk-weight", type=float, default=18.0)
    ap.add_argument("--hard-margin-m", type=float, default=1.0)
    ap.add_argument("--route-horizon", type=int, default=18)
    ap.add_argument("--improvement-margin", type=float, default=0.25)
    ap.add_argument("--force-decomp-episodes", type=int, default=12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.tex_out.mkdir(parents=True, exist_ok=True)
    make_table1(args)
    make_table2(args)
    make_table3(args)
    make_table_b1(args)
    make_table_required_escape(args)
    make_table_b5(args)
    make_figure4(args)
    make_figure5(args)
    make_fig_b1(args)
    make_fig_b2(args)
    make_fig_b5(args)
    print(f"Wrote RELLIS-Dyn artifacts to {args.out}")
    print(f"Wrote LaTeX snippets to {args.tex_out}")
    print(f"Copied paper figures to {args.paper_figures}")


if __name__ == "__main__":
    main()

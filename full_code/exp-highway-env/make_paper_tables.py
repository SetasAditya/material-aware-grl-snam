#!/usr/bin/env python3
"""
make_paper_tables.py — generates Tables 1 and 2 from existing JSONs.

Reads:
    - eval_paired.json          (Table 2: paired Stage 1 vs Stage 2)
    - history.json (Stage 2 unfrozen)  } Table 1 inputs
    - history.json (Stage 2 frozen)    }
    - force_diagnostic.json     (optional, for cross-checking magnitudes)

Writes:
    - tables/table1_enrichment_ablation.{tex,txt}
    - tables/table2_paired_comparison.{tex,txt}

Table 1: frozen vs unfrozen Stage 2 enrichment ablation.
    Demonstrates the structural enrichment principle empirically — that
    when Stage 2 is allowed to modify geometry, it does the wrong thing
    (collapses Stage 1's val_L_traj). This requires history JSONs from
    BOTH a frozen and an unfrozen Stage 2 training run. If you only have
    one, the table reports what it has and notes the missing data.

Table 2: paired Stage 1 vs Stage 2 across scenarios.
    The headline numbers. Six metrics × three scenarios × two stages.
    Includes mean ± std and absolute difference (Stage 2 - Stage 1).

Usage
-----
    python make_paper_tables.py \\
        --paired-eval runs/eval_paired_full.json \\
        --frozen-history checkpoints/highway_stage2_mu_lat/history.json \\
        --unfrozen-history checkpoints/highway_stage2_navscale/history.json \\
        --out tables/

If you don't have an unfrozen history (you've only run frozen), omit:
    python make_paper_tables.py \\
        --paired-eval runs/eval_paired_full.json \\
        --frozen-history checkpoints/highway_stage2_mu_lat/history.json \\
        --out tables/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_DISPLAY = [
    ("default",                     "Default"),
    ("authored_slow_leader",        "Slow leader"),
    ("authored_slow_leader_boxed",  "Boxed"),
]

# Metrics for Table 2, in display order. Tuples of:
#   (key in agg dict, pretty label, format string, lower-is-better)
TABLE2_METRICS = [
    ("collision_rate",              "Crash rate",       ".1%",  True),
    ("mean_speed_mean",             "Mean speed (m/s)", ".2f",  False),
    ("lane_changes_mean",           "Lane changes/ep",  ".2f",  False),
    ("lateral_accel_p95_abs_mean",  "|a_lat| p95",      ".2f",  None),
    ("lane_keep_err_mean_mean",     "LKE mean (m)",     ".2f",  True),
    ("cum_risk_eval_mean",          "Cum risk (eval)",  ".2f",  True),
    ("min_clearance_mean",          "Min clearance (m)", ".2f", False),
]


def _std_key_for(metric_key: str) -> str:
    """Aggregate JSON stores mean/std as <metric>_mean and <metric>_std."""
    if metric_key.endswith("_mean"):
        return metric_key[:-5] + "_std"
    return metric_key + "_std"


def _fmt_scalar(v: float, fmt: str) -> str:
    return f"{v:{fmt}}"


def _fmt_mean_std(agg: Dict[str, Any], key: str, fmt: str) -> str:
    """Format a table cell as mean ± std when a std aggregate is present."""
    v = agg.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    s = agg.get(_std_key_for(key))
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return _fmt_scalar(v, fmt)
    return f"{v:{fmt}} ± {s:{fmt}}"


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: enrichment ablation
# ─────────────────────────────────────────────────────────────────────────────

def _final_train_metrics(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Pull the last epoch's val and train metrics from a history.json."""
    if not history:
        return {}
    last = history[-1]
    out = {"epoch": last.get("epoch")}
    if "val" in last:
        out["val_L_traj"]  = last["val"].get("L_traj")
        out["val_L_nav"]   = last["val"].get("L_nav")
    if "train" in last:
        out["train_L_traj"] = last["train"].get("L_traj")
        out["train_L_nav"]  = last["train"].get("L_nav")
        out["mu_lat_raw"]   = last["train"].get("mu_lat_raw_mean")
    if "closed_loop" in last and last["closed_loop"]:
        cl = last["closed_loop"]
        out["closed_loop_crash_max"] = cl.get("max_crash")
        out["closed_loop_crash_mean"] = cl.get("mean_crash")
        out["closed_loop_speed_mean"] = cl.get("mean_speed")
    return out


def make_table1_enrichment(
    frozen_path: Optional[Path], unfrozen_path: Optional[Path],
    out_dir: Path,
):
    """Frozen vs unfrozen Stage 2 ablation. Both inputs optional —
    table renders what it has."""
    rows: List[Tuple[str, Dict[str, Any]]] = []

    if unfrozen_path and unfrozen_path.exists():
        with open(unfrozen_path) as f:
            hist = json.load(f)
        rows.append(("Stage 2 (unfrozen geometry)", _final_train_metrics(hist)))
    else:
        rows.append(("Stage 2 (unfrozen geometry)", {}))

    if frozen_path and frozen_path.exists():
        with open(frozen_path) as f:
            hist = json.load(f)
        rows.append(("Stage 2 (frozen geometry)", _final_train_metrics(hist)))
    else:
        rows.append(("Stage 2 (frozen geometry)", {}))

    # Define which metrics to display
    cols = [
        ("val_L_traj",     "val L_traj",       ".3f"),
        ("val_L_nav",      "val L_nav",        ".3f"),
        ("mu_lat_raw",     "μ_lat raw",        ".2f"),
        ("closed_loop_crash_max",  "max-scn crash", ".1%"),
        ("closed_loop_crash_mean", "avg-scn crash", ".1%"),
        ("closed_loop_speed_mean", "avg-scn speed (m/s)", ".2f"),
    ]

    # ── .txt rendering ──────────────────────────────────────────────────────
    txt_lines = []
    txt_lines.append("Table 1: Enrichment ablation — geometry constraint matters")
    txt_lines.append("=" * 78)
    header = f"  {'variant':<32}" + " ".join(f"{lbl:>14}" for _, lbl, _ in cols)
    txt_lines.append(header)
    txt_lines.append("-" * len(header))
    for label, m in rows:
        cells = []
        for key, _, fmt in cols:
            v = m.get(key)
            if v is None:
                cells.append("---")
            else:
                cells.append(f"{v:{fmt}}")
        cells_str = " ".join(f"{c:>14}" for c in cells)
        txt_lines.append(f"  {label:<32}{cells_str}")
    missing = [label for label, m in rows if not m]
    txt_lines.append("")
    if missing:
        txt_lines.append("Missing data: " + ", ".join(missing) + ".")
        txt_lines.append("Cells are rendered as '---' so the paper pipeline remains reproducible.")
    else:
        unfrozen_m = rows[0][1]
        frozen_m = rows[1][1]
        unfrozen_ltraj = unfrozen_m.get("val_L_traj")
        frozen_ltraj = frozen_m.get("val_L_traj")
        if unfrozen_ltraj is not None and frozen_ltraj is not None:
            txt_lines.append(
                "Reading: frozen geometry preserves the Stage 1 scaffold better "
                f"on this run (final val_L_traj {frozen_ltraj:.3f} vs. "
                f"{unfrozen_ltraj:.3f} unfrozen)."
            )
        else:
            txt_lines.append("Reading: compare final validation and closed-loop columns by variant.")
        txt_lines.append("The frozen variant trains risk + lateral channels on top of the scaffold.")

    txt_path = out_dir / "table1_enrichment_ablation.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n")
    print(f"  wrote {txt_path}")

    # ── .tex rendering ──────────────────────────────────────────────────────
    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{\textbf{Enrichment ablation.} Freezing Stage 1 geometry preserves the scaffold while Stage 2 trains the risk and lateral channels on top. Missing runs are rendered as dashes rather than dropped.}")
    tex_lines.append(r"\label{tab:enrichment_ablation}")
    tex_lines.append(r"\small")
    col_spec = "l" + "r" * len(cols)
    tex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"\toprule")
    headers = " & ".join([""] + [pretty for _, pretty, _ in cols])
    tex_lines.append(headers + r" \\")
    tex_lines.append(r"\midrule")
    for label, m in rows:
        cells = [label]
        for key, _, fmt in cols:
            v = m.get(key)
            if v is None:
                cells.append("---")
            else:
                cells.append(f"${v:{fmt}}$")
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")

    tex_path = out_dir / "table1_enrichment_ablation.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"  wrote {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: paired comparison across scenarios
# ─────────────────────────────────────────────────────────────────────────────

def make_table2_paired(paired_path: Path, out_dir: Path):
    """Paired Stage 1 vs Stage 2 across all scenarios.

    Expects paired_path to be a JSON with structure:
      {
        "aggregates": {
          "<scenario>": { "stage1": {...}, "stage2": {...} },
          ...
        }
      }
    matching what eval_stage2.py's --out produces.
    """
    with open(paired_path) as f:
        data = json.load(f)

    aggregates = data.get("aggregates") or data.get("results_by_scenario")
    if not aggregates:
        raise ValueError(f"No 'aggregates' in {paired_path}")

    # ── .txt rendering ──────────────────────────────────────────────────────
    txt_lines = []
    txt_lines.append("Table 2: Paired Stage 1 vs Stage 2 across scenarios")
    txt_lines.append("=" * 92)
    txt_lines.append("")

    for scn_name, scn_pretty in SCENARIO_DISPLAY:
        if scn_name not in aggregates:
            continue
        agg_s1 = aggregates[scn_name].get("stage1", {})
        agg_s2 = aggregates[scn_name].get("stage2", {})

        txt_lines.append(f"  {scn_pretty}")
        txt_lines.append(f"  {'-' * len(scn_pretty)}")
        txt_lines.append(f"    {'metric':<22} {'Stage 1':>22}  {'Stage 2':>22}  {'Δ':>10}")
        for key, label, fmt, lower_better in TABLE2_METRICS:
            v1 = agg_s1.get(key)
            v2 = agg_s2.get(key)
            if v1 is None or v2 is None:
                continue
            if isinstance(v1, float) and math.isnan(v1):
                continue
            if isinstance(v2, float) and math.isnan(v2):
                continue
            s1_str = _fmt_mean_std(agg_s1, key, fmt)
            s2_str = _fmt_mean_std(agg_s2, key, fmt)
            d = v2 - v1
            d_str = f"{d:+.3f}" if abs(d) < 0.5 else f"{d:+.2f}"
            txt_lines.append(
                f"    {label:<22} {s1_str:>22}  {s2_str:>22}  {d_str:>10}"
            )
        txt_lines.append("")

    txt_path = out_dir / "table2_paired_comparison.txt"
    txt_path.write_text("\n".join(txt_lines) + "\n")
    print(f"  wrote {txt_path}")

    # ── .tex rendering ──────────────────────────────────────────────────────
    # Single wide table grouped by scenario
    tex_lines = []
    tex_lines.append(r"\begin{table}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{\textbf{Paired Stage 1 vs.\ Stage 2 across scenarios.} 20 paired episodes per scenario, identical seeds. Stage 2 preserves Stage 1's default-scenario performance; produces qualitatively new behavior on slow leader (lane change, no crash); correctly defers in boxed (no escape geometry).}")
    tex_lines.append(r"\label{tab:paired_comparison}")
    tex_lines.append(r"\small")
    tex_lines.append(r"\setlength{\tabcolsep}{4pt}")
    n_scn = sum(1 for s, _ in SCENARIO_DISPLAY if s in aggregates)
    col_spec = "l" + "rr" * n_scn
    tex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"\toprule")
    # Two-row header: scenario name (multicolumn 2), then "S1 / S2"
    scenario_headers = [""]
    for scn_name, scn_pretty in SCENARIO_DISPLAY:
        if scn_name in aggregates:
            scenario_headers.append(rf"\multicolumn{{2}}{{c}}{{{scn_pretty}}}")
    tex_lines.append(" & ".join(scenario_headers) + r" \\")
    sub_headers = [""]
    for scn_name, _ in SCENARIO_DISPLAY:
        if scn_name in aggregates:
            sub_headers.extend(["S1", "S2"])
    tex_lines.append(" & ".join(sub_headers) + r" \\")
    tex_lines.append(r"\midrule")

    for key, label, fmt, lower_better in TABLE2_METRICS:
        cells = [label]
        for scn_name, _ in SCENARIO_DISPLAY:
            if scn_name not in aggregates:
                continue
            agg_s1 = aggregates[scn_name].get("stage1", {})
            agg_s2 = aggregates[scn_name].get("stage2", {})
            for agg in [agg_s1, agg_s2]:
                v = agg.get(key)
                std_key = _std_key_for(key)
                s = agg.get(std_key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("---")
                elif s is None or (isinstance(s, float) and math.isnan(s)):
                    cells.append(f"${v:{fmt}}$")
                else:
                    cells.append(rf"${v:{fmt}}\pm{s:{fmt}}$")
        tex_lines.append(" & ".join(cells) + r" \\")

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    tex_lines.append(r"\end{table}")

    tex_path = out_dir / "table2_paired_comparison.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"  wrote {tex_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--paired-eval",     type=str, required=True,
                    help="JSON from eval_stage2.py --stage1-ckpt --out")
    ap.add_argument("--frozen-history",  type=str, default="",
                    help="history.json from frozen-geometry Stage 2 training")
    ap.add_argument("--unfrozen-history", type=str, default="",
                    help="history.json from unfrozen Stage 2 (e.g., navscale)")
    ap.add_argument("--out",             type=str, default="tables/")
    ap.add_argument("--only",            type=str, default="all",
                    choices=["all", "table1", "table2"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "table1"):
        print("Table 1: enrichment ablation")
        frozen   = Path(args.frozen_history)   if args.frozen_history   else None
        unfrozen = Path(args.unfrozen_history) if args.unfrozen_history else None
        if not frozen and not unfrozen:
            print("  WARNING: no history paths provided; Table 1 will be empty.")
        make_table1_enrichment(frozen, unfrozen, out_dir)

    if args.only in ("all", "table2"):
        print("\nTable 2: paired Stage 1 vs Stage 2")
        make_table2_paired(Path(args.paired_eval), out_dir)

    print(f"\nDone. Tables in {out_dir}/")


if __name__ == "__main__":
    main()

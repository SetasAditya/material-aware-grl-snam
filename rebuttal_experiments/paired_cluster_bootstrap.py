#!/usr/bin/env python3
"""Paired episode-cluster bootstrap confidence intervals for rebuttal experiments.

The dynamic mode operates on one-row-per-(event, episode, method) rollout
artifacts.  The static mode operates on RELLIS force samples and treats an
episode as the resampling cluster, preserving all path samples in that episode.

The reported contrast is always ``method_a - method_b``.  Resampling is paired:
an episode cluster is sampled once and the same multiplicity is applied to both
methods and to every event/path sample contained in that cluster.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_SEED = 27_370


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool | None
    description: str


DYNAMIC_DERIVED = {
    "false_pre_activation": MetricSpec(
        "false_pre_activation",
        False,
        "1 when route_deviation_delay is earlier than the event-spec open_delay",
    ),
    "suppression": MetricSpec(
        "suppression",
        True,
        "1 - false_pre_activation",
    ),
    "hard_contact": MetricSpec(
        "hard_contact",
        False,
        "1 when hard_hazard_length_m is positive",
    ),
}

DIRECTION = {
    "false_pre_activation": False,
    "suppression": True,
    "hard_contact": False,
    "success": True,
    "stuck": False,
    "event_window_cvar_violation": False,
    "post_event_cvar_violation": False,
    "full_episode_cvar_violation": False,
    "event_window_cvar_risk": False,
    "post_event_cvar_risk": False,
    "full_episode_cvar_risk": False,
    "risk_exposure": False,
    "event_window_risk_exposure": False,
    "post_event_risk_exposure": False,
    "hard_hazard_length_m": False,
    "stale_exposure": False,
    "reaction_delay": False,
    "route_deviation_delay": False,
    "opportunity_normalized_delay": False,
    "path_length_m": False,
    "path_length_ratio": False,
    "curvature_energy": False,
    "compute_ms": False,
    "replans": False,
    "revisit_count": False,
    "correct_activation": True,
    "false_activation": False,
    "force_norm": None,
    # Raw force magnitudes/alignment have no unconditional preference direction:
    # activation is desirable in R1 but undesirable in R2/R3.
    "force_perp_norm": None,
    "dot_safe": None,
    "force_risk_alignment": None,
    "selectivity_ratio": True,
}


def _comparison(value: str) -> tuple[str, str]:
    for separator in (":", "="):
        if separator in value:
            a, b = value.split(separator, 1)
            if a.strip() and b.strip():
                return a.strip(), b.strip()
    raise argparse.ArgumentTypeError(
        f"Invalid comparison {value!r}; expected METHOD_A:METHOD_B."
    )


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_id(series: pd.Series) -> pd.Series:
    # Dynamic episode ids are often parsed as integers, while static ids contain
    # underscores.  A string representation gives stable merge/cluster keys.
    return series.map(lambda x: str(x).strip())


def _finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def _merge_open_delay(rollouts: pd.DataFrame, specs_path: Path | None) -> pd.DataFrame:
    out = rollouts.copy()
    if specs_path is None:
        out["open_delay"] = np.nan
        return out
    specs = _read_csv(specs_path)
    required = {"event_type", "episode_id", "open_delay"}
    missing = required - set(specs.columns)
    if missing:
        raise ValueError(f"{specs_path} lacks required columns: {sorted(missing)}")
    specs = specs.copy()
    specs["event_type"] = _as_id(specs["event_type"])
    specs["episode_id"] = _as_id(specs["episode_id"])
    join_cols = ["event_type", "episode_id"]
    if specs.duplicated(join_cols).any():
        examples = specs.loc[specs.duplicated(join_cols, keep=False), join_cols].head()
        raise ValueError(f"Event specs are not unique on {join_cols}:\n{examples}")
    delay = specs[join_cols + ["open_delay"]].copy()
    delay["open_delay"] = _finite_numeric(delay["open_delay"])
    out = out.merge(delay, on=join_cols, how="left", validate="many_to_one")
    return out


def prepare_dynamic(rollouts_path: Path, specs_path: Path | None) -> pd.DataFrame:
    frame = _read_csv(rollouts_path)
    required = {"method", "event_type", "episode_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{rollouts_path} lacks required columns: {sorted(missing)}")
    frame = frame.copy()
    for col in required:
        frame[col] = _as_id(frame[col])
    identity = ["method", "event_type", "episode_id"]
    if frame.duplicated(identity).any():
        examples = frame.loc[frame.duplicated(identity, keep=False), identity].head()
        raise ValueError(f"Dynamic rollout rows are not unique on {identity}:\n{examples}")
    frame = _merge_open_delay(frame, specs_path)
    if "route_deviation_delay" in frame:
        delay = _finite_numeric(frame["route_deviation_delay"])
        delayed_event = frame["event_type"].str.contains("delayed", case=False, na=False)
        has_open = _finite_numeric(frame["open_delay"]).notna()
        eligible = delayed_event & has_open & delay.notna()
        false_pre = pd.Series(np.nan, index=frame.index, dtype=float)
        false_pre.loc[eligible] = (
            delay.loc[eligible] < _finite_numeric(frame.loc[eligible, "open_delay"])
        ).astype(float)
        frame["false_pre_activation"] = false_pre
        frame["suppression"] = 1.0 - false_pre
    else:
        frame["false_pre_activation"] = np.nan
        frame["suppression"] = np.nan
    if "hard_hazard_length_m" in frame:
        hard = _finite_numeric(frame["hard_hazard_length_m"])
        frame["hard_contact"] = np.where(hard.notna(), (hard > 0.0).astype(float), np.nan)
    else:
        frame["hard_contact"] = np.nan
    return frame


def prepare_static(force_samples_path: Path, force_eps: float) -> pd.DataFrame:
    frame = _read_csv(force_samples_path)
    required = {
        "episode_id",
        "regime",
        "path_index",
        "has_safe_alt",
        "dot_safe",
        "force_perp_norm",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{force_samples_path} lacks required columns: {sorted(missing)}")
    frame = frame.copy()
    if "force_source" not in frame:
        # Preserve the evaluator's historical name rather than inventing a
        # route-aware or gate-off interpretation.
        frame["force_source"] = "analytic_fixed_lambda"
    frame["method"] = _as_id(frame["force_source"])
    frame["episode_id"] = _as_id(frame["episode_id"])
    frame["regime"] = _as_id(frame["regime"])
    frame["path_index"] = _as_id(frame["path_index"])
    identity = ["method", "episode_id", "regime", "path_index"]
    if frame.duplicated(identity).any():
        examples = frame.loc[frame.duplicated(identity, keep=False), identity].head()
        raise ValueError(f"Static force rows are not unique on {identity}:\n{examples}")

    alt = _finite_numeric(frame["has_safe_alt"]) > 0.5
    car_pool = frame["regime"].eq("R1") & alt
    far_pool = frame["regime"].isin(["R2", "R3"])
    dot = _finite_numeric(frame["dot_safe"])
    perp = _finite_numeric(frame["force_perp_norm"])
    frame["correct_activation"] = np.where(
        car_pool & dot.notna(), (dot > force_eps).astype(float), np.nan
    )
    frame["false_activation"] = np.where(
        far_pool & perp.notna(), (perp > force_eps).astype(float), np.nan
    )
    for col in ("force_norm", "force_perp_norm", "dot_safe", "force_risk_alignment"):
        if col in frame:
            frame[col] = _finite_numeric(frame[col])
    return frame


def _scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "aggregate" or scope == "static":
        return frame
    if not scope.startswith("event:"):
        raise ValueError(f"Unknown scope {scope}")
    event = scope.split(":", 1)[1]
    return frame.loc[frame["event_type"].eq(event)]


def _pair_keys(mode: str) -> list[str]:
    if mode == "dynamic":
        return ["event_type", "episode_id"]
    return ["episode_id", "regime", "path_index"]


def validate_pairing(
    frame: pd.DataFrame,
    *,
    mode: str,
    scope: str,
    method_a: str,
    method_b: str,
) -> dict[str, object]:
    scoped = _scope_frame(frame, scope)
    keys = _pair_keys(mode)
    a_keys = scoped.loc[scoped["method"].eq(method_a), keys].drop_duplicates()
    b_keys = scoped.loc[scoped["method"].eq(method_b), keys].drop_duplicates()
    joined = a_keys.merge(b_keys, on=keys, how="outer", indicator=True)
    a_only = int((joined["_merge"] == "left_only").sum())
    b_only = int((joined["_merge"] == "right_only").sum())
    paired = int((joined["_merge"] == "both").sum())
    if mode == "dynamic" and paired:
        cluster_counts = (
            joined.loc[joined["_merge"].eq("both")]
            .groupby("episode_id", sort=False)
            .size()
        )
        min_per_cluster = int(cluster_counts.min())
        max_per_cluster = int(cluster_counts.max())
    else:
        min_per_cluster = max_per_cluster = 1
    return {
        "mode": mode,
        "scope": scope,
        "method_a": method_a,
        "method_b": method_b,
        "paired_observations": paired,
        "method_a_only": a_only,
        "method_b_only": b_only,
        "complete_pairing": a_only == 0 and b_only == 0 and paired > 0,
        "min_observations_per_cluster": min_per_cluster,
        "max_observations_per_cluster": max_per_cluster,
    }


def _metric_pair_table(
    frame: pd.DataFrame,
    *,
    mode: str,
    scope: str,
    method_a: str,
    method_b: str,
    metric: str,
) -> pd.DataFrame:
    scoped = _scope_frame(frame, scope)
    if metric not in scoped:
        raise ValueError(f"Metric {metric!r} is not present; columns={list(scoped.columns)}")
    keys = _pair_keys(mode)
    subset = scoped.loc[
        scoped["method"].isin([method_a, method_b]), keys + ["method", metric]
    ].copy()
    subset[metric] = _finite_numeric(subset[metric])
    wide = subset.pivot(index=keys, columns="method", values=metric)
    if method_a not in wide or method_b not in wide:
        return pd.DataFrame(columns=keys + [method_a, method_b])
    wide = wide[[method_a, method_b]].dropna().reset_index()
    return wide


def _cluster_sufficient_statistics(
    paired: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if paired.empty:
        return tuple(np.asarray([], dtype=float) for _ in range(5))  # type: ignore[return-value]
    grouped = paired.groupby("episode_id", sort=True)
    cluster_ids = np.asarray(list(grouped.groups), dtype=object)
    a_sum = grouped[method_a].sum().reindex(cluster_ids).to_numpy(dtype=float)
    b_sum = grouped[method_b].sum().reindex(cluster_ids).to_numpy(dtype=float)
    # The metric table is complete-case paired, so counts are equal.  Keep both
    # explicitly to make the weighted statistic auditable.
    a_count = grouped[method_a].count().reindex(cluster_ids).to_numpy(dtype=float)
    b_count = grouped[method_b].count().reindex(cluster_ids).to_numpy(dtype=float)
    return cluster_ids, a_sum, a_count, b_sum, b_count


def _bootstrap_means(
    a_sum: np.ndarray,
    a_count: np.ndarray,
    b_sum: np.ndarray,
    b_count: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk_size: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = len(a_sum)
    boot_a = np.empty(n_bootstrap, dtype=float)
    boot_b = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        boot_a[start:stop] = a_sum[indices].sum(axis=1) / a_count[indices].sum(axis=1)
        boot_b[start:stop] = b_sum[indices].sum(axis=1) / b_count[indices].sum(axis=1)
    return boot_a, boot_b


def _bootstrap_static_ratio(
    frame: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    """Bootstrap R1/R2 mean-perpendicular-force selectivity ratios."""
    subset = frame.loc[
        frame["method"].isin([method_a, method_b])
        & frame["regime"].isin(["R1", "R2"]),
        ["method", "episode_id", "regime", "path_index", "force_perp_norm"],
    ].copy()
    subset["force_perp_norm"] = _finite_numeric(subset["force_perp_norm"])
    wide = subset.pivot(
        index=["episode_id", "regime", "path_index"],
        columns="method",
        values="force_perp_norm",
    )
    if method_a not in wide or method_b not in wide:
        return None
    paired = wide[[method_a, method_b]].dropna().reset_index()
    if paired.empty:
        return None
    clusters = sorted(paired["episode_id"].unique())
    cluster_index = {cluster: i for i, cluster in enumerate(clusters)}
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    for method in (method_a, method_b):
        for regime in ("R1", "R2"):
            sums = np.zeros(len(clusters), dtype=float)
            counts = np.zeros(len(clusters), dtype=float)
            pool = paired.loc[paired["regime"].eq(regime), ["episode_id", method]]
            grouped = pool.groupby("episode_id")[method].agg(["sum", "count"])
            for episode_id, row in grouped.iterrows():
                i = cluster_index[str(episode_id)]
                sums[i] = float(row["sum"])
                counts[i] = float(row["count"])
            arrays[(method, regime, "sum")] = sums
            arrays[(method, regime, "count")] = counts

    def ratio(method: str, indices: np.ndarray | None = None) -> np.ndarray | float:
        def total(regime: str, kind: str) -> np.ndarray | float:
            values = arrays[(method, regime, kind)]
            return values.sum() if indices is None else values[indices].sum(axis=1)

        r1 = total("R1", "sum") / total("R1", "count")
        r2 = total("R2", "sum") / total("R2", "count")
        return r1 / np.maximum(r2, 1e-12)

    point_a = float(ratio(method_a))
    point_b = float(ratio(method_b))
    boot_a = np.empty(n_bootstrap, dtype=float)
    boot_b = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 2_000):
        stop = min(start + 2_000, n_bootstrap)
        indices = rng.integers(0, len(clusters), size=(stop - start, len(clusters)))
        boot_a[start:stop] = ratio(method_a, indices)  # type: ignore[assignment]
        boot_b[start:stop] = ratio(method_b, indices)  # type: ignore[assignment]
    return _result_row(
        mode="static",
        scope="static",
        method_a=method_a,
        method_b=method_b,
        metric="selectivity_ratio",
        point_a=point_a,
        point_b=point_b,
        boot_a=boot_a,
        boot_b=boot_b,
        n_clusters=len(clusters),
        n_observations=len(paired),
    )


def _result_row(
    *,
    mode: str,
    scope: str,
    method_a: str,
    method_b: str,
    metric: str,
    point_a: float,
    point_b: float,
    boot_a: np.ndarray,
    boot_b: np.ndarray,
    n_clusters: int,
    n_observations: int,
) -> dict[str, object]:
    diff = boot_a - boot_b
    low, high = np.quantile(diff, [0.025, 0.975])
    direction = DIRECTION.get(metric)
    if direction is True:
        probability_better = float(np.mean(diff > 0.0))
    elif direction is False:
        probability_better = float(np.mean(diff < 0.0))
    else:
        probability_better = float("nan")
    return {
        "mode": mode,
        "scope": scope,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "higher_is_better": direction,
        "n_clusters": n_clusters,
        "n_paired_observations": n_observations,
        "method_a_mean": point_a,
        "method_b_mean": point_b,
        "difference_a_minus_b": point_a - point_b,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_probability_a_better": probability_better,
    }


def analyze(
    frame: pd.DataFrame,
    *,
    mode: str,
    comparisons: Sequence[tuple[str, str]],
    metrics: Sequence[str],
    scopes: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, object]] = []
    validations: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    available = set(frame["method"])
    for method_a, method_b in comparisons:
        absent = {method_a, method_b} - available
        if absent:
            raise ValueError(f"Methods absent from input: {sorted(absent)}; available={sorted(available)}")
        for scope in scopes:
            validation = validate_pairing(
                frame,
                mode=mode,
                scope=scope,
                method_a=method_a,
                method_b=method_b,
            )
            validations.append(validation)
            if not validation["complete_pairing"]:
                raise ValueError(
                    "Incomplete episode pairing: "
                    f"{method_a} vs {method_b}, {scope}: "
                    f"A-only={validation['method_a_only']}, "
                    f"B-only={validation['method_b_only']}"
                )
            for metric in metrics:
                if mode == "static" and metric == "selectivity_ratio":
                    ratio_row = _bootstrap_static_ratio(
                        frame,
                        method_a=method_a,
                        method_b=method_b,
                        n_bootstrap=n_bootstrap,
                        rng=rng,
                    )
                    if ratio_row is not None:
                        results.append(ratio_row)
                    continue
                paired = _metric_pair_table(
                    frame,
                    mode=mode,
                    scope=scope,
                    method_a=method_a,
                    method_b=method_b,
                    metric=metric,
                )
                cluster_ids, a_sum, a_count, b_sum, b_count = _cluster_sufficient_statistics(
                    paired,
                    method_a=method_a,
                    method_b=method_b,
                )
                if len(cluster_ids) == 0:
                    continue
                boot_a, boot_b = _bootstrap_means(
                    a_sum,
                    a_count,
                    b_sum,
                    b_count,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                )
                results.append(
                    _result_row(
                        mode=mode,
                        scope=scope,
                        method_a=method_a,
                        method_b=method_b,
                        metric=metric,
                        point_a=float(a_sum.sum() / a_count.sum()),
                        point_b=float(b_sum.sum() / b_count.sum()),
                        boot_a=boot_a,
                        boot_b=boot_b,
                        n_clusters=len(cluster_ids),
                        n_observations=len(paired),
                    )
                )
    return pd.DataFrame(results), pd.DataFrame(validations)


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{float(value):.3f}"


def write_outputs(
    out_dir: Path,
    *,
    results: pd.DataFrame,
    validations: pd.DataFrame,
    config: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "paired_bootstrap_results.csv", index=False)
    validations.to_csv(out_dir / "pairing_validation.csv", index=False)
    payload = {
        "config": {key: _json_value(value) for key, value in config.items()},
        "pairing_validation": _records(validations),
        "results": _records(results),
    }
    (out_dir / "paired_bootstrap_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        f"# {config['analysis_name']}",
        "",
        (
            f"Paired episode-cluster bootstrap with {int(config['n_bootstrap']):,} "
            f"replicates (seed {config['seed']}). Differences are method A minus method B."
        ),
        "",
        "| Scope | Method A | Method B | Metric | A | B | Difference (95% CI) | P(A better) | N clusters |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.to_dict(orient="records"):
        lines.append(
            f"| {row['scope']} | `{row['method_a']}` | `{row['method_b']}` | "
            f"{row['metric']} | {_fmt(row['method_a_mean'])} | {_fmt(row['method_b_mean'])} | "
            f"{_fmt(row['difference_a_minus_b'])} "
            f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] | "
            f"{_fmt(row['bootstrap_probability_a_better'])} | {int(row['n_clusters'])} |"
        )
    lines.extend(
        [
            "",
            "Pairing validation:",
            "",
            "| Scope | Method A | Method B | Paired observations | A-only | B-only | Complete |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in validations.to_dict(orient="records"):
        lines.append(
            f"| {row['scope']} | `{row['method_a']}` | `{row['method_b']}` | "
            f"{int(row['paired_observations'])} | {int(row['method_a_only'])} | "
            f"{int(row['method_b_only'])} | {bool(row['complete_pairing'])} |"
        )
    if config["mode"] == "static":
        lines.extend(
            [
                "",
                (
                    "Static labels are the exact `force_source` values in the raw artifact. "
                    "`analytic_fixed_lambda`, `s2_model_lambda`, and "
                    "`stage2_directional_head` are not labeled as gate-off or route-aware."
                ),
            ]
        )
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--comparison",
            action="append",
            type=_comparison,
            required=True,
            help="Paired contrast METHOD_A:METHOD_B; repeat for multiple contrasts.",
        )
        subparser.add_argument(
            "--metrics",
            type=_csv_list,
            required=True,
            help="Comma-separated metric names.",
        )
        subparser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        subparser.add_argument("--out-dir", type=Path, required=True)
        subparser.add_argument("--analysis-name", required=True)

    dynamic = subparsers.add_parser("dynamic", help="Analyze dynamic_rollouts.csv")
    common(dynamic)
    dynamic.add_argument("--rollouts", type=Path, required=True)
    dynamic.add_argument("--event-specs", type=Path)
    dynamic.add_argument(
        "--scopes",
        choices=["aggregate", "events", "both"],
        default="aggregate",
        help="Analyze all selected events jointly, each event, or both.",
    )
    dynamic.add_argument(
        "--events",
        type=_csv_list,
        help="Optional comma-separated event filter.",
    )

    static = subparsers.add_parser("static", help="Analyze static force_samples.csv")
    common(static)
    static.add_argument("--force-samples", type=Path, required=True)
    static.add_argument("--force-eps", type=float, default=0.001)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.mode == "dynamic":
        frame = prepare_dynamic(args.rollouts, args.event_specs)
        if args.events:
            unknown = set(args.events) - set(frame["event_type"])
            if unknown:
                parser.error(f"Unknown events {sorted(unknown)}")
            frame = frame.loc[frame["event_type"].isin(args.events)].copy()
        events = sorted(frame["event_type"].unique())
        if args.scopes == "aggregate":
            scopes = ["aggregate"]
        elif args.scopes == "events":
            scopes = [f"event:{event}" for event in events]
        else:
            scopes = ["aggregate"] + [f"event:{event}" for event in events]
        input_path = args.rollouts
        event_specs = args.event_specs
    else:
        frame = prepare_static(args.force_samples, args.force_eps)
        scopes = ["static"]
        input_path = args.force_samples
        event_specs = None

    results, validations = analyze(
        frame,
        mode=args.mode,
        comparisons=args.comparison,
        metrics=args.metrics,
        scopes=scopes,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    if results.empty:
        raise ValueError("No analyzable metric rows were produced")
    config = {
        "analysis_name": args.analysis_name,
        "mode": args.mode,
        "input": input_path.resolve(),
        "event_specs": event_specs.resolve() if event_specs else None,
        "comparisons": [f"{a}:{b}" for a, b in args.comparison],
        "metrics": args.metrics,
        "scopes": scopes,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "force_eps": getattr(args, "force_eps", None),
    }
    write_outputs(args.out_dir, results=results, validations=validations, config=config)
    print(f"Wrote {len(results)} estimates to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

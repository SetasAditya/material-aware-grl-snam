#!/usr/bin/env python3
"""
train_stage1.py — Step 5, component 2.

Stage 1 IDM-imitation training in highway-env.

Loads the IDM dataset produced by data_collect_idm.py, trains a
CoefEnergyNetMaterial model with risk heads frozen at λ=0 (geometry-only),
and saves checkpoints. The Stage 1 checkpoint is the warm start for
Stage 2 risk-aware training.

Mirrors `train_material.py:MaterialTrainer.step_batch` (lines 713–840) but:
  • integrate_surrogate_highway instead of integrate_surrogate_material
  • combine_losses(cfg=HighwayLossWeights.stage1()) instead of inline weighted sum
  • bicycle dynamics, ego-frame patch sampling (handled inside the integrator)
  • L_multi skipped for v1 (geometric robustness; not load-bearing for warm start)

Usage
-----
    python train_stage1.py \\
        --data runs/stage1_data \\
        --out checkpoints/highway_stage1 \\
        --epochs 30 --bs 64 --lr 3e-4

For first-shot validation (small dataset, few epochs):
    python train_stage1.py --data runs/stage1_data_smoke --epochs 2 --bs 8
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
LOCAL_HIGHWAY_ENV = HERE / "HighwayEnv"
if LOCAL_HIGHWAY_ENV.exists():
    sys.path.insert(0, str(LOCAL_HIGHWAY_ENV))
sys.path.insert(0, str(HERE))

# DFC tree (parent of exp-highway-env by default; override via --dfc-root)
DEFAULT_DFC_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_DFC_ROOT))

from bicycle_surrogate import ACCEL_RANGE, STEER_RANGE, force_to_action  # noqa: E402
from surrogate_integrator import (  # noqa: E402
    compute_surrogate_highway_force,
    integrate_surrogate_highway,
)
from episode_costs import HighwayLossWeights, combine_losses  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _first_step_action_target_from_tensors(
    v0: torch.Tensor, v1: torch.Tensor, dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finite-difference IDM target in the same action coordinates as deploy."""
    v0_b = v0.float().view(1, 2)
    v1_b = v1.float().view(1, 2)
    dt_f = float(dt.reshape(-1)[0].item()) if torch.is_tensor(dt) else float(dt)
    F_tgt = (v1_b - v0_b) / max(dt_f, 1e-6)
    speed0 = torch.linalg.norm(v0_b, dim=-1).clamp_min(1e-3)
    heading0 = torch.atan2(v0_b[:, 1], v0_b[:, 0])
    accel, steer = force_to_action(F_tgt, heading0, speed0)
    return (
        accel.squeeze(0).clamp(*ACCEL_RANGE),
        steer.squeeze(0).clamp(*STEER_RANGE),
    )


class Stage1IDMDataset(Dataset):
    """Flattened view over IDM-collected episodes.

    Each item is one (obs_dict, o_tgt, v_tgt) sample. Episodes are loaded
    lazily and cached (LRU, size 16) so we don't blow memory on large
    collections. The full 100-episode Stage 1 dataset is about 1.2GB, so
    caching all train episodes is reasonable on the training host and avoids
    random-shuffle disk thrash.
    """

    def __init__(self, manifest_path: Path, split: str):
        self.split = split
        with open(manifest_path) as f:
            records = json.load(f)
        records = [r for r in records if r.get("split", "train") == split]
        if not records:
            raise RuntimeError(f"No records for split={split!r} in {manifest_path}")

        # Resolve relative episode paths against manifest's directory
        self._root = Path(manifest_path).resolve().parent
        self._records = records

        # Flat (record_idx, sample_idx) index
        self._index: List[Tuple[int, int]] = []
        for ri, rec in enumerate(records):
            for si in range(rec["n_samples"]):
                self._index.append((ri, si))

    def __len__(self) -> int:
        return len(self._index)

    @functools.lru_cache(maxsize=128)
    def _load_episode(self, ri: int) -> Dict[str, Any]:
        rec = self._records[ri]
        return torch.load(self._root / rec["path"], weights_only=False,
                           map_location="cpu")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ri, si = self._index[idx]
        ep = self._load_episode(ri)
        s = ep["samples"][si]
        out = dict(s["obs"])  # 12 keys; tensors already
        out["o_tgt"] = s["o_tgt"]
        out["v_tgt"] = s["v_tgt"]
        if "accel_tgt" in s and "steer_tgt" in s:
            out["o_next"] = s.get("o_next", out["o0"])
            out["v_next"] = s.get("v_next", out["v0"])
            out["accel_tgt"] = s["accel_tgt"].float()
            out["steer_tgt"] = s["steer_tgt"].float()
            out["has_action_tgt"] = s.get(
                "has_action_tgt", torch.tensor(1.0, dtype=torch.float32)
            ).float()
        elif si + 1 < len(ep["samples"]):
            # Backward compatibility for datasets collected before one-step
            # targets were stored: adjacent samples are exactly one env step
            # apart, so next obs gives the IDM velocity at t+dt.
            nxt_obs = ep["samples"][si + 1]["obs"]
            out["o_next"] = nxt_obs["o0"]
            out["v_next"] = nxt_obs["v0"]
            accel_tgt, steer_tgt = _first_step_action_target_from_tensors(
                out["v0"], out["v_next"], out["dt"]
            )
            out["accel_tgt"] = accel_tgt
            out["steer_tgt"] = steer_tgt
            out["has_action_tgt"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            out["o_next"] = out["o0"]
            out["v_next"] = out["v0"]
            out["accel_tgt"] = out["o0"].new_tensor(0.0)
            out["steer_tgt"] = out["o0"].new_tensor(0.0)
            out["has_action_tgt"] = out["o0"].new_tensor(0.0)
        return out


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack each key. All shapes are fixed across samples."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# Forward + loss for one batch
# ─────────────────────────────────────────────────────────────────────────────

def _model_forward(
    model, batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """Build (obs_feats, goal_feats) from batch and call model.

    Mirrors train_material.py:723-737 verbatim.
    """
    B = batch["o0"].shape[0]
    N = batch["C"].shape[1]

    if N > 0:
        obs_feats = torch.cat([
            batch["C"],                                    # (B, N, 2)
            batch["R"].unsqueeze(-1),                      # (B, N, 1)
            batch["W"].unsqueeze(-1),                      # (B, N, 1)
            batch["goal"].unsqueeze(1) - batch["C"],       # (B, N, 2)
        ], dim=-1)
    else:
        obs_feats = batch["o0"].new_zeros(B, 0, 6)

    goal_delta = batch["goal"] - batch["o0"]
    goal_feats = torch.cat([
        goal_delta,
        torch.linalg.norm(goal_delta, dim=-1, keepdim=True),
        batch["o0"].new_ones(B, 1),
    ], dim=-1)

    return model(
        obs_feats=obs_feats,
        obs_mask=batch["mask"],
        goal_feats=goal_feats,
        risk_patch=batch["risk_patch"],
    )


def _straight_through_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    clipped = x.clamp(lo, hi)
    return x + (clipped - x).detach()


def step_batch(
    model, batch: Dict[str, torch.Tensor],
    weights: HighwayLossWeights,
    *, train: bool, optimizer=None, grad_clip: float = 5.0,
    alpha_floor: float = 0.0,
    alpha_floor_ahead_only: bool = True,
    action_weight: float = 1.0,
    steer_action_weight: float = 10.0,
) -> Dict[str, float]:
    """One forward (+ optional backward) over a batch. Returns metrics dict."""
    alphas, beta, gamma, lam_soft, lam_hard, _mu_lat = _model_forward(model, batch)

    if alpha_floor > 0 and alphas.numel() > 0:
        floor_mask = batch["mask"]
        if alpha_floor_ahead_only:
            vhat = F.normalize(batch["v0"], dim=-1, eps=1e-6)
            ahead = ((batch["C"] - batch["o0"].unsqueeze(1)) * vhat.unsqueeze(1)).sum(dim=-1) > 0.0
            floor_mask = floor_mask & ahead
        alphas = alphas + float(alpha_floor) * floor_mask.to(alphas.dtype)

    # Stage 1: zero out material outputs (paranoid — heads are also frozen).
    if weights.stage == 1:
        lam_soft = torch.zeros_like(lam_soft)
        lam_hard = torch.zeros_like(lam_hard)

    # H is per-batch but identical across batch; integrator handles (B,) tensor.
    H = batch["H"].long() if batch["H"].dtype != torch.long else batch["H"]

    oT, vT, min_clear, cum_risk, hard_count, arc_length = integrate_surrogate_highway(
        o0            = batch["o0"],
        v0            = batch["v0"],
        goal          = batch["goal"],
        C             = batch["C"],
        R             = batch["R"],
        mask          = batch["mask"],
        alphas        = alphas,
        beta          = beta,
        gamma         = gamma,
        lam_soft      = lam_soft,
        lam_hard      = lam_hard,
        rollout_patch = batch["rollout_patch"],
        d_hat         = batch["d_hat"],
        dt            = batch["dt"],
        H             = H,
    )

    L, metrics = combine_losses(
        oT=oT, vT=vT,
        min_clear=min_clear, cum_risk=cum_risk,
        hard_count=hard_count, arc_length=arc_length,
        goal=batch["goal"],
        o_tgt=batch["o_tgt"], v_tgt=batch["v_tgt"],
        lam_soft=lam_soft, lam_hard=lam_hard,
        L_multi=None,
        cfg=weights,
    )

    if action_weight > 0 and "accel_tgt" in batch and "steer_tgt" in batch:
        v0 = batch["v0"]
        speed0 = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
        heading0 = torch.atan2(v0[:, 1], v0[:, 0])
        F0, dmin0, risk0, sdf0 = compute_surrogate_highway_force(
            o=batch["o0"],
            heading=heading0,
            speed=speed0,
            o0=batch["o0"],
            heading_0=heading0,
            goal=batch["goal"],
            C=batch["C"],
            R_eff=batch["R"],
            mask=batch["mask"],
            alphas=alphas,
            beta=beta,
            gamma=gamma,
            lam_soft=lam_soft,
            lam_hard=lam_hard,
            rollout_patch=batch["rollout_patch"],
            d_hat=batch["d_hat"],
        )
        accel0, steer0 = force_to_action(F0, heading0, speed0)
        accel0_clip = _straight_through_clip(accel0, *ACCEL_RANGE)
        steer0_clip = _straight_through_clip(steer0, *STEER_RANGE)
        has = batch["has_action_tgt"].float().view(-1)
        denom = has.sum().clamp_min(1.0)
        L_accel = (((accel0_clip - batch["accel_tgt"].view(-1)) ** 2) * has).sum() / denom
        L_steer = (((steer0_clip - batch["steer_tgt"].view(-1)) ** 2) * has).sum() / denom
        L_action = L_accel + float(steer_action_weight) * L_steer
        L = L + float(action_weight) * L_action
        metrics["loss"] = float(L.detach())
        metrics["L_action"] = float(L_action.detach())
        metrics["L_accel"] = float(L_accel.detach())
        metrics["L_steer"] = float(L_steer.detach())
        with torch.no_grad():
            metrics["action_tgt_frac"] = float(has.mean().detach())
            metrics["accel0_mean"] = float(accel0_clip.detach().mean())
            metrics["accel_tgt_mean"] = float(batch["accel_tgt"].detach().float().mean())
            metrics["steer0_mean"] = float(steer0_clip.detach().mean())
            metrics["steer_tgt_mean"] = float(batch["steer_tgt"].detach().float().mean())
            metrics["dmin0_mean"] = float(dmin0.detach().mean())
    else:
        metrics.setdefault("L_action", 0.0)
        metrics.setdefault("L_accel", 0.0)
        metrics.setdefault("L_steer", 0.0)
        metrics.setdefault("action_tgt_frac", 0.0)

    with torch.no_grad():
        valid = batch["mask"]
        if alphas.numel() > 0 and bool(valid.any().item()):
            alpha_valid = alphas[valid].detach()
            metrics["alpha_max"] = float(alpha_valid.max())
            metrics["alpha_mean"] = float(alpha_valid.mean())
        else:
            metrics["alpha_max"] = 0.0
            metrics["alpha_mean"] = 0.0
        metrics["beta_mean"] = float(beta.detach().mean())
        metrics["gamma_mean"] = float(gamma.detach().mean())

    if train:
        assert optimizer is not None
        if not torch.isfinite(L):
            metrics["grad_norm"] = float("nan")
            metrics["skipped_step"] = 1.0
            optimizer.zero_grad(set_to_none=True)
            return metrics

        optimizer.zero_grad(set_to_none=True)
        L.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, grad_clip, error_if_nonfinite=False
        )
        metrics["grad_norm"] = float(grad_norm.detach())
        if torch.isfinite(grad_norm):
            optimizer.step()
            metrics["skipped_step"] = 0.0
        else:
            optimizer.zero_grad(set_to_none=True)
            metrics["skipped_step"] = 1.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 freezing
# ─────────────────────────────────────────────────────────────────────────────

def freeze_risk_heads(model) -> None:
    """Freeze risk_enc + lam_soft_head + lam_hard_head."""
    for p in model.risk_enc.parameters():
        p.requires_grad = False
    for p in model.lam_soft_head.parameters():
        p.requires_grad = False
    for p in model.lam_hard_head.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Stage 1: risk heads frozen. "
          f"Trainable params: {n_train:,} / {n_total:,}")


def disable_transformer_nested_tensors(model) -> None:
    """Avoid PyTorch nested-tensor TransformerEncoder backward segfaults.

    Torch 2.7's prototype nested tensor path can be unstable in backward on
    CPU for this masked TransformerEncoder. Disabling it keeps the same model
    math and uses the mature dense tensor path.
    """
    fuser = getattr(model, "fuser", None)
    changed = False
    for attr in ("enable_nested_tensor", "use_nested_tensor"):
        if hasattr(fuser, attr):
            setattr(fuser, attr, False)
            changed = True
    if changed:
        print("  Transformer nested tensors: disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainCfg:
    data:        str = "runs/stage1_data"
    out:         str = "checkpoints/highway_stage1"
    dfc_root:    str = ""           # blank → autodetect from HERE.parent
    epochs:      int = 30
    bs:          int = 64
    lr:          float = 3e-4
    workers:     int = 0
    grad_clip:   float = 5.0
    save_every:  int = 5
    device:      str = "cuda" if torch.cuda.is_available() else "cpu"
    lam_soft_max: float = 50.0      # highway scale
    lam_hard_max: float = 10.0
    d_hat:      float = 15.0        # highway-scale IPC activation distance (m)
    alpha_floor: float = 0.02       # keep geometry active in closed loop
    alpha_floor_ahead_only: bool = True
    w_action:   float = 1.0         # first-step deploy action imitation
    w_steer_action: float = 10.0    # steering target is radian-scale
    seed:        int = 0


def _to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    non_blocking = device.startswith("cuda")
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _override_d_hat(batch: Dict[str, torch.Tensor], d_hat: float) -> Dict[str, torch.Tensor]:
    """Use a highway-scale IPC activation distance without recollecting data."""
    if d_hat > 0:
        batch["d_hat"] = torch.full_like(batch["d_hat"], float(d_hat))
    return batch


def _avg(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         type=str, default="runs/stage1_data")
    ap.add_argument("--out",          type=str, default="checkpoints/highway_stage1")
    ap.add_argument("--dfc-root",     type=str, default="",
                    help="DFC tree root (default: parent of exp-highway-env).")
    ap.add_argument("--epochs",       type=int, default=30)
    ap.add_argument("--bs",           type=int, default=64)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--workers",      type=int, default=0)
    ap.add_argument("--grad-clip",    type=float, default=5.0)
    ap.add_argument("--save-every",   type=int, default=5)
    ap.add_argument("--device",       type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam-soft-max", type=float, default=50.0)
    ap.add_argument("--lam-hard-max", type=float, default=10.0)
    ap.add_argument("--d-hat",        type=float, default=15.0,
                    help="Override dataset d_hat for IPC barrier activation; "
                         "use <=0 to keep stored dataset values.")
    ap.add_argument("--alpha-floor",  type=float, default=0.02,
                    help="Minimum alpha added on valid obstacles. Prevents "
                         "Stage 1 imitation from turning geometry off.")
    ap.add_argument("--alpha-floor-all-obstacles",
                    dest="alpha_floor_ahead_only", action="store_false",
                    help="Apply alpha floor to rear/side vehicles too. Default "
                         "floors only vehicles ahead of ego.")
    ap.set_defaults(alpha_floor_ahead_only=True)
    ap.add_argument("--w-action",    type=float, default=1.0,
                    help="Weight for one-step deploy action imitation loss.")
    ap.add_argument("--w-steer-action", type=float, default=10.0,
                    help="Multiplier on steering MSE inside action loss.")
    ap.add_argument("--seed",         type=int, default=0)
    args = ap.parse_args()

    cfg = TrainCfg(**{k: getattr(args, k) for k in TrainCfg.__dataclass_fields__})

    torch.manual_seed(cfg.seed)

    # Resolve DFC root and import the model
    if cfg.dfc_root:
        sys.path.insert(0, cfg.dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_cfg.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ── Data ─────────────────────────────────────────────────────────────────
    manifest_path = Path(cfg.data) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run data_collect_idm.py first."
        )
    train_ds = Stage1IDMDataset(manifest_path, split="train")
    try:
        val_ds = Stage1IDMDataset(manifest_path, split="val")
        val_source = "val"
    except RuntimeError as exc:
        if "No records for split='val'" not in str(exc):
            raise
        val_ds = train_ds
        val_source = "train fallback"
        print("WARNING: no val split in manifest; using train split for validation. "
              "This is expected for the 2-episode smoke dataset.")
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True,
        num_workers=cfg.workers, collate_fn=collate, drop_last=True,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.bs, shuffle=False,
        num_workers=cfg.workers, collate_fn=collate, drop_last=False,
        pin_memory=cfg.device.startswith("cuda"),
        persistent_workers=cfg.workers > 0,
    )
    print(f"Validation source: {val_source}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = CoefEnergyNetMaterial(
        lam_soft_max=cfg.lam_soft_max,
        lam_hard_max=cfg.lam_hard_max,
    ).to(cfg.device)
    disable_transformer_nested_tensors(model)
    freeze_risk_heads(model)

    weights = HighwayLossWeights.stage1()
    weights.lam_soft_max = cfg.lam_soft_max
    weights.lam_hard_max = cfg.lam_hard_max

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.lr * 0.1
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    print(f"\nDevice: {cfg.device}  Epochs: {cfg.epochs}  Batch: {cfg.bs}  "
          f"LR: {cfg.lr}")
    print(f"Stage 1 weights: w_traj={weights.w_traj} w_vel={weights.w_vel} "
          f"w_clear={weights.w_clear}\n")
    if cfg.w_action > 0:
        print(f"First-step action loss: w_action={cfg.w_action}  "
              f"w_steer_action={cfg.w_steer_action}\n")
    if cfg.d_hat > 0:
        print(f"IPC d_hat override: {cfg.d_hat:.1f} m "
              "(dataset values are left on disk unchanged)\n")
    if cfg.alpha_floor > 0:
        scope = "vehicles ahead of ego" if cfg.alpha_floor_ahead_only else "all valid obstacles"
        print(f"Alpha floor: {cfg.alpha_floor:.4f} on {scope}\n")

    for epoch in range(cfg.epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_metrics_list: List[Dict[str, float]] = []
        for i, batch in enumerate(train_loader):
            batch = _to_device(batch, cfg.device)
            batch = _override_d_hat(batch, cfg.d_hat)
            m = step_batch(model, batch, weights,
                            train=True, optimizer=optimizer,
                            grad_clip=cfg.grad_clip,
                            alpha_floor=cfg.alpha_floor,
                            alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                            action_weight=cfg.w_action,
                            steer_action_weight=cfg.w_steer_action)
            train_metrics_list.append(m)
            if i % 50 == 0:
                print(f"  ep{epoch:03d} it{i:04d}  "
                      f"L={m['loss']:.4f}  L_traj={m['L_traj']:.4f}  "
                      f"L_vel={m['L_vel']:.4f}  L_clear={m['L_clear']:.4f}  "
                      f"L_act={m['L_action']:.4f}  "
                      f"a0/tgt={m.get('accel0_mean', 0.0):+.2f}/"
                      f"{m.get('accel_tgt_mean', 0.0):+.2f}  "
                      f"a={m['alpha_max']:.3f}")
        train_dt = time.time() - t0
        train_avg = _avg(train_metrics_list)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_metrics_list: List[Dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, cfg.device)
                batch = _override_d_hat(batch, cfg.d_hat)
                m = step_batch(model, batch, weights,
                                train=False, optimizer=None,
                                alpha_floor=cfg.alpha_floor,
                                alpha_floor_ahead_only=cfg.alpha_floor_ahead_only,
                                action_weight=cfg.w_action,
                                steer_action_weight=cfg.w_steer_action)
                val_metrics_list.append(m)
        val_avg = _avg(val_metrics_list)

        # Stage-1 score: H-step imitation plus first-step deploy action match.
        val_imit = val_avg.get("L_traj", float("inf")) + val_avg.get("L_vel", float("inf"))
        val_score = val_imit + cfg.w_action * val_avg.get("L_action", 0.0)

        scheduler.step()

        print(f"\n[epoch {epoch:03d}] {train_dt:.1f}s  lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  train: L={train_avg['loss']:.4f}  L_traj={train_avg['L_traj']:.4f}  "
              f"L_vel={train_avg['L_vel']:.4f}  L_clear={train_avg['L_clear']:.4f}  "
              f"L_act={train_avg['L_action']:.4f}")
        print(f"  val  : L={val_avg['loss']:.4f}  L_traj={val_avg['L_traj']:.4f}  "
              f"L_vel={val_avg['L_vel']:.4f}  imit={val_imit:.4f}  "
              f"L_act={val_avg['L_action']:.4f}  score={val_score:.4f}  "
              f"a0/tgt={val_avg.get('accel0_mean', 0.0):+.2f}/"
              f"{val_avg.get('accel_tgt_mean', 0.0):+.2f}  "
              f"a={val_avg['alpha_max']:.3f}\n")

        history.append({"epoch": epoch, "train": train_avg, "val": val_avg,
                         "lr": scheduler.get_last_lr()[0],
                         "train_dt_s": train_dt})

        # ── Checkpoint ───────────────────────────────────────────────────────
        ck = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": asdict(cfg),
            "weights": asdict(weights),
        }
        torch.save(ck, out_dir / "last.pt")
        if cfg.save_every > 0 and (epoch % cfg.save_every == 0
                                    or epoch == cfg.epochs - 1):
            torch.save(ck, out_dir / f"epoch_{epoch:03d}.pt")
        if val_score < best_val:
            best_val = val_score
            torch.save(ck, out_dir / "best.pt")
            print(f"  ✓ new best val_score={val_score:.4f} → best.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val_score={best_val:.4f}")
    print(f"Checkpoints: {out_dir}/{{last,best}}.pt")


if __name__ == "__main__":
    main()

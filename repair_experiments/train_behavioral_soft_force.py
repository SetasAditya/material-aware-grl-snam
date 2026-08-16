#!/usr/bin/env python3
"""Validation-selected repair of the learned soft-material magnitude.

This trainer is deliberately narrower than ``train_material.py``:

* it starts from an existing ``CoefEnergyNetMaterial`` checkpoint;
* it freezes geometry, the risk encoder, the hard-hazard head, and ``mu_lat``;
* it fine-tunes only ``lam_soft_head``;
* it never opens the sealed test manifests or sequence 00004; and
* it selects the checkpoint and force operating point on sequence 00003.

The repaired controller turns ``-grad(risk)`` into a unit direction, projects
that direction into the feasible primitive cone, and falls back to the
primitive axis when the gradient is uninformative.  Thus ``lam_soft`` is the
executed soft-force magnitude.  The active loss operates on the paired
constant-acceleration displacement

    delta_soft_m =
        0.5 * lam_soft * horizon_seconds**2 * resolution_m_per_cell.

This makes the optimization target a behavioral effect rather than a large
coefficient in isolation.  R1 points with a feasible, risk-improving primitive
must reach ``min_effect_m``; R2/R3 and inactive R1 points are explicitly
suppressed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
FULL_CODE = WORKSPACE / "full_code"
EXP_RELLIS = FULL_CODE / "exp-rellis"
for import_root in (FULL_CODE, EXP_RELLIS):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-repair-soft-force")

from scripts.baselines.dfc.models import _build_goal_feats, _build_obs_feats  # noqa: E402
from scripts.build_dfc2018_stagewise import (  # noqa: E402
    extract_local_geom_obstacles,
    extract_risk_patch,
)
from train_material import CoefEnergyNetMaterial  # noqa: E402
from train_rellis_directional_force import _build_point  # noqa: E402


ALLOWED_TRAIN_SEQUENCES = frozenset({"00000", "00001", "00002"})
ALLOWED_VALIDATION_SEQUENCES = frozenset({"00003"})
SEALED_SEQUENCE = "00004"


@dataclass(frozen=True)
class RepairConfig:
    seed: int = 27370
    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    stride: int = 3
    waypoint_stride: int = 6
    patch_size: int = 32
    horizon_cells: int = 8
    long_horizon_cells: int = 24
    hard_margin_m: float = 1.0
    improvement_margin: float = 0.1
    horizon_seconds: float = 1.0
    resolution_m_per_cell: float = 0.5
    min_effect_m: float = 0.5
    active_weight: float = 3.0
    inactive_weight: float = 1.0
    effect_weight: float = 4.0
    target_weight: float = 1.0
    separation_weight: float = 0.5
    normalized_separation: float = 0.25
    target_far: float = 0.25
    lam_soft_max: float = 5.0
    lam_hard_max: float = 10.0
    mu_lat_max: float = 5.0
    train_risk_encoder: bool = True
    preservation_weight: float = 10.0
    device: str = "cpu"
    threads: int = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _stable_seed(text: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(raw[:4], "little")


@contextmanager
def _numpy_seed(seed: int):
    """Make random obstacle subsampling reproducible without leaking RNG state."""

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _load_manifest(
    path: Path,
    *,
    expected_split: str,
    allowed_sequences: frozenset[str],
) -> Tuple[Dict[str, Any], Path]:
    if "test_" in path.name.lower():
        raise ValueError(f"Refusing to open sealed test manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("split_name") != expected_split:
        raise ValueError(
            f"{path} has split_name={payload.get('split_name')!r}; "
            f"expected {expected_split!r}"
        )
    sequences = frozenset(str(x) for x in payload.get("sequences", []))
    if not sequences or not sequences.issubset(allowed_sequences):
        raise ValueError(
            f"{path} contains disallowed sequences {sorted(sequences)}; "
            f"allowed={sorted(allowed_sequences)}"
        )
    if SEALED_SEQUENCE in sequences:
        raise ValueError("Sealed sequence 00004 must not be loaded during tuning")
    bev_manifest = Path(str(payload["bev_source"]["path"]))
    bev_root = bev_manifest.parent
    return payload, bev_root


def _stratified_limit(
    episodes: Sequence[Mapping[str, Any]],
    maximum: int | None,
) -> List[Dict[str, Any]]:
    """Deterministic round-robin limit over sequence/regime groups."""

    rows = [dict(ep) for ep in episodes]
    if maximum is None or maximum >= len(rows):
        return rows
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for episode in rows:
        key = (str(episode["sequence"]), str(episode["regime"]))
        groups.setdefault(key, []).append(episode)
    for values in groups.values():
        values.sort(key=lambda item: str(item["episode_uid"]))
    selected: List[Dict[str, Any]] = []
    depth = 0
    keys = sorted(groups)
    while len(selected) < maximum:
        changed = False
        for key in keys:
            if depth < len(groups[key]) and len(selected) < maximum:
                selected.append(groups[key][depth])
                changed = True
        if not changed:
            break
        depth += 1
    return selected


def _load_scene(bev_root: Path, scene_path: str) -> Dict[str, Any]:
    return torch.load(
        bev_root / scene_path,
        map_location="cpu",
        weights_only=False,
    )


def _as_path(raw: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return [(int(point[0]), int(point[1])) for point in raw]


def build_decision_specs(
    episodes: Sequence[Mapping[str, Any]],
    bev_root: Path,
    cfg: RepairConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build labels without materializing image patches or model features."""

    specs: List[Dict[str, Any]] = []
    scene_cache: Dict[str, Dict[str, Any]] = {}
    label_counts = {"active": 0, "inactive": 0}
    counts_by_regime: Dict[str, Dict[str, int]] = {}
    started = time.perf_counter()
    for episode_number, episode in enumerate(episodes, start=1):
        scene_path = str(episode["scene_path"])
        if scene_path not in scene_cache:
            scene_cache[scene_path] = _load_scene(bev_root, scene_path)
        maps = scene_cache[scene_path]["maps"]
        path = _as_path(episode["stage1_path"])
        regime = str(episode["regime"])
        uid = str(episode["episode_uid"])
        regime_counts = counts_by_regime.setdefault(
            regime, {"active": 0, "inactive": 0}
        )
        for index in range(0, max(0, len(path) - 1), max(1, cfg.stride)):
            point = _build_point(
                maps,
                path,
                index,
                regime=regime,
                episode_id=uid,
                horizon_cells=cfg.horizon_cells,
                long_horizon_cells=cfg.long_horizon_cells,
                hard_margin_m=cfg.hard_margin_m,
                improvement_margin=cfg.improvement_margin,
                route=None,
            )
            if point is None:
                continue
            active = int(point["label"]) > 0
            label_name = "active" if active else "inactive"
            label_counts[label_name] += 1
            regime_counts[label_name] += 1
            specs.append(
                {
                    "episode_uid": uid,
                    "sequence": str(episode["sequence"]),
                    "regime": regime,
                    "scene_path": scene_path,
                    "path": path,
                    "path_index": index,
                    "active": float(active),
                    "best_idx": int(point["best_idx"]),
                    "scaffold_risk": float(point["scaffold_risk"]),
                    "safe_risk": float(point["safe_risk"]),
                }
            )
        if episode_number % 100 == 0:
            print(
                f"  labelled {episode_number}/{len(episodes)} episodes, "
                f"{len(specs)} decisions",
                flush=True,
            )
    return specs, {
        "episodes": len(episodes),
        "decisions": len(specs),
        "labels": label_counts,
        "labels_by_regime": counts_by_regime,
        "seconds": time.perf_counter() - started,
    }


class DecisionDataset(Dataset):
    """Lazy deterministic construction of model inputs from decision specs."""

    def __init__(
        self,
        specs: Sequence[Mapping[str, Any]],
        bev_root: Path,
        cfg: RepairConfig,
    ):
        self.specs = [dict(spec) for spec in specs]
        self.bev_root = bev_root
        self.cfg = cfg
        self.scene_cache: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.specs)

    def _maps(self, scene_path: str) -> Dict[str, np.ndarray]:
        if scene_path not in self.scene_cache:
            self.scene_cache[scene_path] = _load_scene(
                self.bev_root, scene_path
            )
        return self.scene_cache[scene_path]["maps"]

    def __getitem__(self, item: int) -> Dict[str, Any]:
        spec = self.specs[item]
        maps = self._maps(str(spec["scene_path"]))
        path = spec["path"]
        index = int(spec["path_index"])
        row, col = path[index]
        goal_index = min(index + max(1, self.cfg.waypoint_stride), len(path) - 1)
        goal_row, goal_col = path[goal_index]
        pos_xy = np.asarray([float(col), float(row)], dtype=np.float32)
        goal_xy = np.asarray([float(goal_col), float(goal_row)], dtype=np.float32)
        sample_seed = _stable_seed(str(spec["episode_uid"]) + f":{index}", self.cfg.seed)
        with _numpy_seed(sample_seed):
            centers, radii, widths, _ = extract_local_geom_obstacles(
                maps["geom_occ"],
                (row, col),
                patch_size=64,
                robot_radius=1.5,
                margin_factor=0.5,
            )
        obs_feats = _build_obs_feats(
            pos_xy, goal_xy, centers, radii, widths, "cpu"
        ).squeeze(0)
        goal_feats = _build_goal_feats(pos_xy, goal_xy, "cpu").squeeze(0)
        risk_patch, _ = extract_risk_patch(
            maps, (row, col), self.cfg.patch_size
        )
        grad_norm = math.hypot(
            float(maps["grad_col"][row, col]),
            float(maps["grad_row"][row, col]),
        )
        return {
            "obs_feats": obs_feats,
            "goal_feats": goal_feats,
            "risk_patch": torch.as_tensor(risk_patch, dtype=torch.float32),
            "active": torch.tensor(float(spec["active"]), dtype=torch.float32),
            "grad_norm": torch.tensor(grad_norm, dtype=torch.float32),
            "regime": str(spec["regime"]),
            "episode_uid": str(spec["episode_uid"]),
        }


class EncodedDecisionDataset(Dataset):
    """Fixed upstream representation; only ``lam_soft_head`` is trainable."""

    def __init__(
        self,
        material_features: torch.Tensor,
        active: torch.Tensor,
        grad_norm: torch.Tensor,
        regimes: Sequence[str],
        episode_uids: Sequence[str],
        geometry_context: torch.Tensor | None = None,
        risk_patch: torch.Tensor | None = None,
        teacher_lam_hard: torch.Tensor | None = None,
        teacher_mu_lat: torch.Tensor | None = None,
    ):
        self.material_features = material_features
        self.active = active
        self.grad_norm = grad_norm
        self.regimes = list(regimes)
        self.episode_uids = list(episode_uids)
        self.geometry_context = geometry_context
        self.risk_patch = risk_patch
        self.teacher_lam_hard = teacher_lam_hard
        self.teacher_mu_lat = teacher_mu_lat

    def __len__(self) -> int:
        return int(self.material_features.shape[0])

    def __getitem__(self, item: int) -> Dict[str, Any]:
        output = {
            "material_features": self.material_features[item],
            "active": self.active[item],
            "grad_norm": self.grad_norm[item],
            "regime": self.regimes[item],
            "episode_uid": self.episode_uids[item],
        }
        if self.geometry_context is not None:
            output.update(
                {
                    "geometry_context": self.geometry_context[item],
                    "risk_patch": self.risk_patch[item],
                    "teacher_lam_hard": self.teacher_lam_hard[item],
                    "teacher_mu_lat": self.teacher_mu_lat[item],
                }
            )
        return output


def collate_decisions(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    max_obstacles = max(int(item["obs_feats"].shape[0]) for item in batch)
    padded: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for item in batch:
        obs = item["obs_feats"]
        count = int(obs.shape[0])
        if count < max_obstacles:
            obs = torch.cat(
                [obs, torch.zeros(max_obstacles - count, 6, dtype=obs.dtype)],
                dim=0,
            )
        padded.append(obs)
        mask = torch.zeros(max_obstacles, dtype=torch.bool)
        mask[:count] = True
        masks.append(mask)
    return {
        "obs_feats": torch.stack(padded),
        "obs_mask": torch.stack(masks),
        "goal_feats": torch.stack([item["goal_feats"] for item in batch]),
        "risk_patch": torch.stack([item["risk_patch"] for item in batch]),
        "active": torch.stack([item["active"] for item in batch]),
        "grad_norm": torch.stack([item["grad_norm"] for item in batch]),
        "regime": [str(item["regime"]) for item in batch],
        "episode_uid": [str(item["episode_uid"]) for item in batch],
    }


@torch.no_grad()
def _frozen_material_features(
    model: CoefEnergyNetMaterial,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen model prefix exactly once."""

    obs_feats = batch["obs_feats"].to(device)
    obs_mask = batch["obs_mask"].to(device)
    goal_feats = batch["goal_feats"].to(device)
    risk_patch = batch["risk_patch"].to(device)
    batch_size, obstacle_count = obs_feats.shape[:2]
    goal_token = model.goal_enc(goal_feats).unsqueeze(1)
    if obstacle_count == 0:
        tokens = goal_token
        padding = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=device
        )
    else:
        obstacle_tokens = model.obs_enc(
            obs_feats.reshape(batch_size * obstacle_count, -1)
        ).reshape(batch_size, obstacle_count, -1)
        tokens = torch.cat([goal_token, obstacle_tokens], dim=1)
        padding = torch.cat(
            [
                torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=device
                ),
                ~obs_mask,
            ],
            dim=1,
        )
    context = model.fuser(tokens, src_key_padding_mask=padding)[:, 0]
    risk_context = model.risk_enc(risk_patch)
    return torch.cat([risk_context, context], dim=-1), context


def encode_frozen_dataset(
    model: CoefEnergyNetMaterial,
    loader: DataLoader,
    cfg: RepairConfig,
) -> Tuple[EncodedDecisionDataset, float]:
    """Cache the fixed prefix so head-only epochs are fast and exact."""

    model.eval()
    device = torch.device(cfg.device)
    features: List[torch.Tensor] = []
    active: List[torch.Tensor] = []
    grad_norm: List[torch.Tensor] = []
    regimes: List[str] = []
    episode_uids: List[str] = []
    geometry_contexts: List[torch.Tensor] = []
    risk_patches: List[torch.Tensor] = []
    teacher_lam_hard: List[torch.Tensor] = []
    teacher_mu_lat: List[torch.Tensor] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        material_features, geometry_context = _frozen_material_features(
            model, batch, device
        )
        features.append(material_features.cpu())
        if cfg.train_risk_encoder:
            geometry_contexts.append(geometry_context.cpu())
            risk_patches.append(batch["risk_patch"].to(torch.float16).cpu())
            teacher_lam_hard.append(
                (
                    model.lam_hard_max
                    * torch.sigmoid(
                        model.lam_hard_head(material_features).squeeze(-1)
                    )
                ).cpu()
            )
            teacher_mu_lat.append(
                (
                    model.mu_lat_max
                    * torch.sigmoid(
                        model.mu_lat_head(material_features).squeeze(-1)
                    )
                ).cpu()
            )
        active.append(batch["active"].cpu())
        grad_norm.append(batch["grad_norm"].cpu())
        regimes.extend(batch["regime"])
        episode_uids.extend(batch["episode_uid"])
        if batch_index % 50 == 0:
            print(
                f"  encoded {batch_index}/{len(loader)} frozen batches",
                flush=True,
            )
    dataset = EncodedDecisionDataset(
        torch.cat(features),
        torch.cat(active),
        torch.cat(grad_norm),
        regimes,
        episode_uids,
        geometry_context=(
            torch.cat(geometry_contexts) if geometry_contexts else None
        ),
        risk_patch=torch.cat(risk_patches) if risk_patches else None,
        teacher_lam_hard=(
            torch.cat(teacher_lam_hard) if teacher_lam_hard else None
        ),
        teacher_mu_lat=(
            torch.cat(teacher_mu_lat) if teacher_mu_lat else None
        ),
    )
    return dataset, time.perf_counter() - started


def load_initial_model(
    checkpoint: Path,
    cfg: RepairConfig,
) -> Tuple[CoefEnergyNetMaterial, Dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old_cfg = dict(payload.get("cfg", {}))
    model = CoefEnergyNetMaterial(
        patch_size=int(old_cfg.get("patch_size", cfg.patch_size)),
        lam_soft_max=float(old_cfg.get("lam_soft_max", cfg.lam_soft_max)),
        lam_hard_max=float(old_cfg.get("lam_hard_max", cfg.lam_hard_max)),
        mu_lat_max=float(old_cfg.get("mu_lat_max", cfg.mu_lat_max)),
    )
    state = payload.get("model_state_dict", payload.get("model", payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {key for key in missing if key.startswith("mu_lat_head.")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible initialization: missing={missing}, unexpected={unexpected}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        (model.risk_enc, model.lam_soft_head)
        if cfg.train_risk_encoder
        else (model.lam_soft_head,)
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "missing_initialized_parameters": sorted(missing),
        "unexpected_parameters": sorted(unexpected),
        "trainable_parameter_names": [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    return model, metadata


def _forward_lambda(
    model: CoefEnergyNetMaterial,
    batch: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    return _forward_material_outputs(model, batch, device)[0]


def _forward_material_outputs(
    model: CoefEnergyNetMaterial,
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return lambda_soft, preservation loss, and hard/mu absolute drift."""

    if "geometry_context" in batch:
        risk_patch = batch["risk_patch"].to(
            device=device, dtype=torch.float32
        )
        geometry_context = batch["geometry_context"].to(device)
        risk_context = model.risk_enc(risk_patch)
        features = torch.cat([risk_context, geometry_context], dim=-1)
        lam_soft = model.lam_soft_max * torch.sigmoid(
            model.lam_soft_head(features).squeeze(-1)
        )
        lam_hard = model.lam_hard_max * torch.sigmoid(
            model.lam_hard_head(features).squeeze(-1)
        )
        mu_lat = model.mu_lat_max * torch.sigmoid(
            model.mu_lat_head(features).squeeze(-1)
        )
        teacher_hard = batch["teacher_lam_hard"].to(device)
        teacher_mu = batch["teacher_mu_lat"].to(device)
        hard_error = (
            F.mse_loss(
                lam_hard / float(model.lam_hard_max),
                teacher_hard / float(model.lam_hard_max),
            )
        )
        mu_error = F.mse_loss(
                mu_lat / float(model.mu_lat_max),
                teacher_mu / float(model.mu_lat_max),
            )
        preservation = hard_error + mu_error
        hard_drift = torch.mean(torch.abs(lam_hard - teacher_hard))
        mu_drift = torch.mean(torch.abs(mu_lat - teacher_mu))
        hard_max_drift = torch.max(torch.abs(lam_hard - teacher_hard))
        mu_max_drift = torch.max(torch.abs(mu_lat - teacher_mu))
        return (
            lam_soft,
            preservation,
            hard_drift,
            mu_drift,
            hard_max_drift,
            mu_max_drift,
        )
    if "material_features" in batch:
        features = batch["material_features"].to(device)
        zero = features.sum() * 0.0
        return (
            model.lam_soft_max * torch.sigmoid(
                model.lam_soft_head(features).squeeze(-1)
            ),
            zero,
            zero,
            zero,
            zero,
            zero,
        )
    outputs = model(
        batch["obs_feats"].to(device),
        batch["obs_mask"].to(device),
        batch["goal_feats"].to(device),
        batch["risk_patch"].to(device),
    )
    zero = outputs[3].sum() * 0.0
    return outputs[3], zero, zero, zero, zero, zero


def _balanced_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def behavioral_loss(
    lam_soft: torch.Tensor,
    active: torch.Tensor,
    grad_norm: torch.Tensor,
    cfg: RepairConfig,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Loss on normalized/projected force magnitude and paired displacement."""

    active = active.to(lam_soft.dtype)
    inactive = 1.0 - active
    # The repaired controller normalizes the projected gradient and uses the
    # selected primitive as a fallback direction in flat-risk regions.  Keep
    # grad_norm in the signature because it is logged as a confidence and
    # legacy-attainability diagnostic; it no longer scales the repaired force.
    del grad_norm
    force_magnitude = lam_soft
    effect = (
        0.5
        * force_magnitude
        * float(cfg.horizon_seconds) ** 2
        * float(cfg.resolution_m_per_cell)
    )
    required_force = (
        2.0 * float(cfg.min_effect_m)
        / max(
            float(cfg.horizon_seconds) ** 2
            * float(cfg.resolution_m_per_cell),
            1e-8,
        )
    )
    target_lambda_active = lam_soft.new_full(
        lam_soft.shape,
        min(required_force, float(cfg.lam_soft_max)),
    )
    target_lambda = active * target_lambda_active

    normalized = (lam_soft / float(cfg.lam_soft_max)).clamp(1e-6, 1.0 - 1e-6)
    classification = -(
        cfg.active_weight * active * torch.log(normalized)
        + cfg.inactive_weight * inactive * torch.log1p(-normalized)
    ).mean()
    active_shortfall = _balanced_mean(
        F.relu(float(cfg.min_effect_m) - effect).square(), active
    )
    inactive_effect = _balanced_mean(
        effect.square()
        + 0.1 * (lam_soft / float(cfg.lam_soft_max)).square(),
        inactive,
    )
    target = F.smooth_l1_loss(
        lam_soft / float(cfg.lam_soft_max),
        target_lambda / float(cfg.lam_soft_max),
        reduction="none",
    )
    target = (
        cfg.active_weight * active * target
        + cfg.inactive_weight * inactive * target
    ).mean()

    if bool(active.any()) and bool(inactive.any()):
        separation = F.relu(
            float(cfg.normalized_separation)
            - normalized[active.bool()].mean()
            + normalized[inactive.bool()].mean()
        )
    else:
        separation = lam_soft.sum() * 0.0

    loss = (
        classification
        + cfg.effect_weight * (active_shortfall + inactive_effect)
        + cfg.target_weight * target
        + cfg.separation_weight * separation
    )
    metrics = {
        "loss": float(loss.detach()),
        "classification_loss": float(classification.detach()),
        "active_shortfall_loss": float(active_shortfall.detach()),
        "inactive_effect_loss": float(inactive_effect.detach()),
        "target_loss": float(target.detach()),
        "separation_loss": float(separation.detach()),
        "lambda_mean": float(lam_soft.mean().detach()),
        "force_mean": float(force_magnitude.mean().detach()),
        "effect_mean": float(effect.mean().detach()),
    }
    return loss, metrics, effect


def _aggregate(records: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    rows = list(records)
    if not rows:
        return {}
    keys = rows[0].keys()
    output: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output[key] = (
            float(np.max(values))
            if key.endswith("_max_abs_drift")
            else float(np.mean(values))
        )
    return output


@torch.no_grad()
def collect_predictions(
    model: CoefEnergyNetMaterial,
    loader: DataLoader,
    cfg: RepairConfig,
) -> Dict[str, Any]:
    model.eval()
    device = torch.device(cfg.device)
    records: List[Dict[str, Any]] = []
    losses: List[Dict[str, float]] = []
    for batch in loader:
        active = batch["active"].to(device)
        grad_norm = batch["grad_norm"].to(device)
        (
            lam_soft,
            preservation,
            hard_drift,
            mu_drift,
            hard_max_drift,
            mu_max_drift,
        ) = _forward_material_outputs(model, batch, device)
        _, loss_metrics, effect = behavioral_loss(
            lam_soft, active, grad_norm, cfg
        )
        loss_metrics["preservation_loss"] = float(preservation.detach())
        loss_metrics["lam_hard_mean_abs_drift"] = float(hard_drift.detach())
        loss_metrics["mu_lat_mean_abs_drift"] = float(mu_drift.detach())
        loss_metrics["lam_hard_max_abs_drift"] = float(
            hard_max_drift.detach()
        )
        loss_metrics["mu_lat_max_abs_drift"] = float(mu_max_drift.detach())
        loss_metrics["loss"] += (
            cfg.preservation_weight * loss_metrics["preservation_loss"]
        )
        losses.append(loss_metrics)
        force = lam_soft
        for index in range(lam_soft.shape[0]):
            records.append(
                {
                    "episode_uid": batch["episode_uid"][index],
                    "regime": batch["regime"][index],
                    "active": float(active[index].cpu()),
                    "grad_norm": float(grad_norm[index].cpu()),
                    "lam_soft": float(lam_soft[index].cpu()),
                    "force_magnitude": float(force[index].cpu()),
                    "predicted_effect_m": float(effect[index].cpu()),
                }
            )
    return {"loss": _aggregate(losses), "records": records}


def calibrate_lambda_threshold(
    records: Sequence[Mapping[str, Any]],
    target_far: float,
    *,
    min_threshold: float = 0.0,
    target_r3: float = 0.20,
) -> Tuple[float, Dict[str, float]]:
    """Validation-only magnitude threshold maximizing CAR under frozen FAR."""

    values = np.asarray([float(row["lam_soft"]) for row in records])
    active = np.asarray([float(row["active"]) > 0.5 for row in records])
    false_pool = np.asarray(
        [str(row["regime"]) in ("R2", "R3") for row in records]
    )
    r2_pool = np.asarray([str(row["regime"]) == "R2" for row in records])
    r3_pool = np.asarray([str(row["regime"]) == "R3" for row in records])
    candidates = np.unique(values)
    candidates = np.concatenate(
        [
            [max(float(values.max() + 1e-6), float(min_threshold))],
            candidates[::-1],
            [float(values.min())],
        ]
    )
    best: Tuple[Tuple[float, float, float], float, Dict[str, float]] | None = None
    for threshold in candidates:
        if threshold < min_threshold:
            continue
        predicted = values >= threshold
        car = float((predicted & active).sum() / max(1, active.sum()))
        far = float(
            (predicted & false_pool).sum() / max(1, false_pool.sum())
        )
        if far > target_far + 1e-12:
            continue
        r2_rate = float(
            (predicted & r2_pool).sum() / max(1, r2_pool.sum())
        )
        r3_rate = float(
            (predicted & r3_pool).sum() / max(1, r3_pool.sum())
        )
        if r2_rate > target_far + 1e-12 or r3_rate > target_r3 + 1e-12:
            continue
        accuracy = float(np.mean(predicted == active))
        key = (car, -far, accuracy)
        metrics = {
            "threshold": float(threshold),
            "correct_activation_rate": car,
            "false_activation_rate_R2_R3": far,
            "R2_activation_rate": r2_rate,
            "R3_activation_rate": r3_rate,
            "accuracy": accuracy,
            "active_denominator": int(active.sum()),
            "false_activation_denominator": int(false_pool.sum()),
            "R2_denominator": int(r2_pool.sum()),
            "R3_denominator": int(r3_pool.sum()),
            "minimum_effect_threshold": float(min_threshold),
        }
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise RuntimeError("No lambda threshold candidate satisfies target FAR")
    return best[1], best[2]


def checkpoint_selection_key(
    calibration: Mapping[str, float],
    validation_loss: float,
) -> Tuple[float, float, float]:
    """Preregistered model-selection order: CAR, FAR, then validation loss."""

    return (
        float(calibration["correct_activation_rate"]),
        -float(calibration["false_activation_rate_R2_R3"]),
        -float(validation_loss),
    )


def summarize_predictions(
    records: Sequence[Mapping[str, Any]],
    threshold: float,
    cfg: RepairConfig,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name, subset in (
        ("all", list(records)),
        ("active", [row for row in records if float(row["active"]) > 0.5]),
        ("inactive", [row for row in records if float(row["active"]) <= 0.5]),
        ("R1", [row for row in records if row["regime"] == "R1"]),
        ("R2", [row for row in records if row["regime"] == "R2"]),
        ("R3", [row for row in records if row["regime"] == "R3"]),
    ):
        if not subset:
            output[name] = {"n": 0}
            continue
        lam = np.asarray([float(row["lam_soft"]) for row in subset])
        grad = np.asarray([float(row["grad_norm"]) for row in subset])
        force = np.asarray([float(row["force_magnitude"]) for row in subset])
        effect = np.asarray([float(row["predicted_effect_m"]) for row in subset])
        max_effect = np.full(
            grad.shape,
            0.5
            * float(cfg.lam_soft_max)
            * float(cfg.horizon_seconds) ** 2
            * float(cfg.resolution_m_per_cell),
            dtype=np.float64,
        )
        legacy_max_effect = (
            0.5
            * float(cfg.lam_soft_max)
            * grad
            * float(cfg.horizon_seconds) ** 2
            * float(cfg.resolution_m_per_cell)
        )
        output[name] = {
            "n": len(subset),
            "lambda_mean": float(lam.mean()),
            "lambda_median": float(np.median(lam)),
            "force_mean": float(force.mean()),
            "effect_mean_m": float(effect.mean()),
            "effect_median_m": float(np.median(effect)),
            "activation_rate_at_threshold": float(np.mean(lam >= threshold)),
            "fraction_reaching_min_effect": float(
                np.mean(effect >= cfg.min_effect_m)
            ),
            "grad_norm_mean": float(grad.mean()),
            "grad_norm_median": float(np.median(grad)),
            "near_zero_gradient_fraction": float(np.mean(grad < 1e-4)),
            "max_effect_mean_m": float(max_effect.mean()),
            "effect_target_attainable_fraction": float(
                np.mean(max_effect >= cfg.min_effect_m)
            ),
            "legacy_unnormalized_max_effect_mean_m": float(
                legacy_max_effect.mean()
            ),
            "legacy_unnormalized_target_attainable_fraction": float(
                np.mean(legacy_max_effect >= cfg.min_effect_m)
            ),
        }
    return output


def _write_predictions(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    fieldnames = [
        "episode_uid",
        "regime",
        "active",
        "grad_norm",
        "lam_soft",
        "force_magnitude",
        "predicted_effect_m",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def train(
    model: CoefEnergyNetMaterial,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: RepairConfig,
    output: Path,
    provenance: Mapping[str, Any],
) -> Tuple[CoefEnergyNetMaterial, Dict[str, Any]]:
    device = torch.device(cfg.device)
    model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.epochs),
        eta_min=cfg.learning_rate * 0.1,
    )
    best_loss = float("inf")
    best_selection: Tuple[float, float, float] | None = None
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    curves: List[Dict[str, Any]] = []
    for epoch in range(cfg.epochs):
        model.train()
        batch_metrics: List[Dict[str, float]] = []
        for batch in train_loader:
            active = batch["active"].to(device)
            grad_norm = batch["grad_norm"].to(device)
            (
                lam_soft,
                preservation,
                hard_drift,
                mu_drift,
                hard_max_drift,
                mu_max_drift,
            ) = _forward_material_outputs(model, batch, device)
            loss, metrics, _ = behavioral_loss(
                lam_soft, active, grad_norm, cfg
            )
            loss = loss + cfg.preservation_weight * preservation
            metrics["preservation_loss"] = float(preservation.detach())
            metrics["lam_hard_mean_abs_drift"] = float(hard_drift.detach())
            metrics["mu_lat_mean_abs_drift"] = float(mu_drift.detach())
            metrics["lam_hard_max_abs_drift"] = float(
                hard_max_drift.detach()
            )
            metrics["mu_lat_max_abs_drift"] = float(mu_max_drift.detach())
            metrics["loss"] = float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=5.0
            )
            optimizer.step()
            batch_metrics.append(metrics)
        scheduler.step()
        train_metrics = _aggregate(batch_metrics)
        validation = collect_predictions(model, validation_loader, cfg)
        validation_loss = float(validation["loss"]["loss"])
        required_force = (
            2.0 * cfg.min_effect_m
            / (
                cfg.horizon_seconds ** 2
                * cfg.resolution_m_per_cell
            )
        )
        epoch_threshold, epoch_calibration = calibrate_lambda_threshold(
            validation["records"],
            cfg.target_far,
            min_threshold=required_force,
            target_r3=0.20,
        )
        selection = checkpoint_selection_key(
            epoch_calibration, validation_loss
        )
        curve = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": validation["loss"],
            "validation_lambda_threshold": epoch_threshold,
            "validation_calibration": epoch_calibration,
            "checkpoint_selection_key": list(selection),
        }
        curves.append(curve)
        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['loss']:.5f} "
            f"val={validation_loss:.5f} "
            f"lambda={train_metrics['lambda_mean']:.3f} "
            f"effect={train_metrics['effect_mean']:.3f}m",
            f"CAR={epoch_calibration['correct_activation_rate']:.3f} "
            f"R2={epoch_calibration['R2_activation_rate']:.3f} "
            f"R3={epoch_calibration['R3_activation_rate']:.3f}",
            flush=True,
        )
        epoch_payload = {
            "epoch": epoch,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "cfg": {
                "patch_size": cfg.patch_size,
                "lam_soft_max": cfg.lam_soft_max,
                "lam_hard_max": cfg.lam_hard_max,
                "mu_lat_max": cfg.mu_lat_max,
            },
            "repair_training_config": asdict(cfg),
            "train_metrics": train_metrics,
            "validation_metrics": validation["loss"],
            "provenance": dict(provenance),
        }
        torch.save(epoch_payload, output / f"epoch_{epoch:03d}.pt")
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_loss = validation_loss
            best_epoch = epoch
            best_state = epoch_payload["model_state_dict"]

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    validation = collect_predictions(model, validation_loader, cfg)
    required_force = (
        2.0 * cfg.min_effect_m
        / (
            cfg.horizon_seconds ** 2
            * cfg.resolution_m_per_cell
        )
    )
    threshold, calibration = calibrate_lambda_threshold(
        validation["records"],
        cfg.target_far,
        min_threshold=required_force,
        target_r3=0.20,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_checkpoint_selection_key": list(best_selection),
        "validation_lambda_threshold": threshold,
        "validation_calibration": calibration,
        "validation_summary": summarize_predictions(
            validation["records"], threshold, cfg
        ),
        "curves": curves,
    }
    final_payload = {
        "epoch": best_epoch,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "cfg": {
            "patch_size": cfg.patch_size,
            "lam_soft_max": cfg.lam_soft_max,
            "lam_hard_max": cfg.lam_hard_max,
            "mu_lat_max": cfg.mu_lat_max,
        },
        "repair_training_config": asdict(cfg),
        "repair_calibration": {
            "lambda_active_threshold": threshold,
            "target_far": cfg.target_far,
            "horizon_seconds": cfg.horizon_seconds,
            "resolution_m_per_cell": cfg.resolution_m_per_cell,
            "min_effect_m": cfg.min_effect_m,
            "minimum_effect_lambda": required_force,
            "checkpoint_selection": (
                "minimum behavioral validation loss on RELLIS sequence 00003"
            ),
            "validation_metrics": calibration,
        },
        "provenance": dict(provenance),
        "train_metrics": curves[best_epoch]["train"],
        "validation_metrics": curves[best_epoch]["validation"],
    }
    torch.save(final_payload, output / "best.pt")
    _write_predictions(output / "validation_predictions.csv", validation["records"])
    (output / "curves.json").write_text(json.dumps(curves, indent=2) + "\n")
    with (output / "curves.csv").open("w", newline="") as stream:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "validation_loss",
            "train_lambda_mean",
            "validation_lambda_mean",
            "train_effect_mean_m",
            "validation_effect_mean_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "epoch": curve["epoch"],
                    "learning_rate": curve["learning_rate"],
                    "train_loss": curve["train"]["loss"],
                    "validation_loss": curve["validation"]["loss"],
                    "train_lambda_mean": curve["train"]["lambda_mean"],
                    "validation_lambda_mean": curve["validation"]["lambda_mean"],
                    "train_effect_mean_m": curve["train"]["effect_mean"],
                    "validation_effect_mean_m": curve["validation"]["effect_mean"],
                }
            )
    return model, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=HERE / "splits" / "train_static.json",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=HERE / "splits" / "validation_static.json",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path(
            "/mnt/data/adityas/GRL-SNAM/exp-rellis/checkpoints/"
            "rellis_stage2_decision_mid_ep12/best.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "behavioral_soft_force",
    )
    parser.add_argument("--seed", type=int, default=27370)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--horizon-seconds", type=float, default=1.0)
    parser.add_argument("--resolution-m-per-cell", type=float, default=0.5)
    parser.add_argument("--min-effect-m", type=float, default=0.5)
    parser.add_argument("--target-far", type=float, default=0.25)
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze risk_enc as well; retained for the failed capacity control.",
    )
    parser.add_argument("--preservation-weight", type=float, default=10.0)
    parser.add_argument("--active-weight", type=float, default=3.0)
    parser.add_argument("--effect-weight", type=float, default=4.0)
    parser.add_argument("--separation-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--max-train-episodes",
        type=int,
        default=None,
        help="Stratified development limit; omit for the frozen full train split.",
    )
    parser.add_argument(
        "--max-validation-episodes",
        type=int,
        default=None,
        help="Stratified development limit; omit for the frozen full validation split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RepairConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        stride=args.stride,
        horizon_seconds=args.horizon_seconds,
        resolution_m_per_cell=args.resolution_m_per_cell,
        min_effect_m=args.min_effect_m,
        target_far=args.target_far,
        train_risk_encoder=not args.head_only,
        preservation_weight=args.preservation_weight,
        active_weight=args.active_weight,
        effect_weight=args.effect_weight,
        separation_weight=args.separation_weight,
        device=args.device,
        threads=args.threads,
    )
    seed_everything(cfg.seed, cfg.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    train_manifest, train_bev = _load_manifest(
        args.train_manifest,
        expected_split="train",
        allowed_sequences=ALLOWED_TRAIN_SEQUENCES,
    )
    validation_manifest, validation_bev = _load_manifest(
        args.validation_manifest,
        expected_split="validation",
        allowed_sequences=ALLOWED_VALIDATION_SEQUENCES,
    )
    if train_bev.resolve() != validation_bev.resolve():
        raise ValueError("Train and validation manifests refer to different BEV roots")
    train_episodes = _stratified_limit(
        train_manifest["episodes"], args.max_train_episodes
    )
    validation_episodes = _stratified_limit(
        validation_manifest["episodes"], args.max_validation_episodes
    )
    print(
        f"Building labels from {len(train_episodes)} train and "
        f"{len(validation_episodes)} validation episodes"
    )
    train_specs, train_data_summary = build_decision_specs(
        train_episodes, train_bev, cfg
    )
    validation_specs, validation_data_summary = build_decision_specs(
        validation_episodes, validation_bev, cfg
    )
    if not train_specs or not validation_specs:
        raise RuntimeError("Train and validation decisions must be nonempty")

    raw_train_dataset = DecisionDataset(train_specs, train_bev, cfg)
    raw_validation_dataset = DecisionDataset(validation_specs, validation_bev, cfg)
    raw_train_loader = DataLoader(
        raw_train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_decisions,
    )
    raw_validation_loader = DataLoader(
        raw_validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_decisions,
    )
    model, initialization = load_initial_model(args.init_checkpoint, cfg)
    print("Encoding frozen train representation once ...", flush=True)
    train_dataset, train_encode_seconds = encode_frozen_dataset(
        model, raw_train_loader, cfg
    )
    print("Encoding frozen validation representation once ...", flush=True)
    validation_dataset, validation_encode_seconds = encode_frozen_dataset(
        model, raw_validation_loader, cfg
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    provenance = {
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": _sha256(args.train_manifest),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "validation_manifest_sha256": _sha256(args.validation_manifest),
        "sealed_test_loaded": False,
        "train_sequences": sorted(train_manifest["sequences"]),
        "validation_sequences": sorted(validation_manifest["sequences"]),
        "bev_root": str(train_bev.resolve()),
        "initialization": initialization,
        "train_data": train_data_summary,
        "validation_data": validation_data_summary,
        "frozen_feature_cache": {
            "train_seconds": train_encode_seconds,
            "validation_seconds": validation_encode_seconds,
            "feature_dim": int(train_dataset.material_features.shape[1]),
            "policy": (
                "Cached frozen geometry context and teacher outputs once; "
                "retained raw risk patches because risk_enc remains trainable."
                if cfg.train_risk_encoder
                else
                "Cached the full fixed material representation once because "
                "every upstream parameter is frozen."
            ),
        },
        "development_limits": {
            "max_train_episodes": args.max_train_episodes,
            "max_validation_episodes": args.max_validation_episodes,
        },
    }
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "repair_training_config": asdict(cfg),
                "provenance": provenance,
            },
            indent=2,
        )
        + "\n"
    )

    # Log initialization behavior before changing the head.
    initial_validation = collect_predictions(model, validation_loader, cfg)
    required_force = (
        2.0 * cfg.min_effect_m
        / (
            cfg.horizon_seconds ** 2
            * cfg.resolution_m_per_cell
        )
    )
    initial_threshold, initial_calibration = calibrate_lambda_threshold(
        initial_validation["records"],
        cfg.target_far,
        min_threshold=required_force,
        target_r3=0.20,
    )
    initial_summary = {
        "lambda_threshold": initial_threshold,
        "calibration": initial_calibration,
        "summary": summarize_predictions(
            initial_validation["records"], initial_threshold, cfg
        ),
    }
    (args.output / "initial_validation.json").write_text(
        json.dumps(initial_summary, indent=2) + "\n"
    )

    _, training_summary = train(
        model,
        train_loader,
        validation_loader,
        cfg,
        args.output,
        provenance,
    )
    report = {
        "status": "completed",
        "method": (
            "risk-encoder + lambda-head behavioral force-effect fine-tuning; "
            "geometry and hard/mu heads frozen with teacher distillation"
            if cfg.train_risk_encoder
            else
            "lambda-head-only behavioral force-effect fine-tuning; geometry, "
            "risk encoder, hard-hazard head, and mu_lat frozen"
        ),
        "initial_validation": initial_summary,
        "trained": training_summary,
        "checkpoint": str((args.output / "best.pt").resolve()),
        "test_data_used": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

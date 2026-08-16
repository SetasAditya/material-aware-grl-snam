from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import HighwayBaseline, normalize_action

from bicycle_surrogate import ACCEL_RANGE, STEER_RANGE, force_to_action  # noqa: E402
from eval_stage1 import _apply_alpha_floor, disable_transformer_nested_tensors  # noqa: E402
from surrogate_integrator import compute_surrogate_highway_force  # noqa: E402


def _to_batch(obs_np: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in obs_np.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                t = torch.from_numpy(v.copy()).bool()
            elif v.dtype in (np.int32, np.int64):
                t = torch.from_numpy(v.copy()).long()
            else:
                t = torch.from_numpy(v.astype(np.float32, copy=False))
        elif isinstance(v, (np.floating, float)):
            t = torch.tensor(float(v), dtype=torch.float32)
        elif isinstance(v, (np.integer, int)):
            t = torch.tensor(int(v), dtype=torch.long)
        else:
            t = torch.as_tensor(v)
        out[k] = t.unsqueeze(0).to(device)
    return out


def _model_coeffs(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    B = batch["o0"].shape[0]
    N = batch["C"].shape[1]
    if N > 0:
        obs_feats = torch.cat(
            [
                batch["C"],
                batch["R"].unsqueeze(-1),
                batch["W"].unsqueeze(-1),
                batch["goal"].unsqueeze(1) - batch["C"],
            ],
            dim=-1,
        )
    else:
        obs_feats = batch["o0"].new_zeros(B, 0, 6)
    goal_delta = batch["goal"] - batch["o0"]
    goal_feats = torch.cat(
        [
            goal_delta,
            torch.linalg.norm(goal_delta, dim=-1, keepdim=True),
            batch["o0"].new_ones(B, 1),
        ],
        dim=-1,
    )
    return model(
        obs_feats=obs_feats,
        obs_mask=batch["mask"],
        goal_feats=goal_feats,
        risk_patch=batch["risk_patch"],
    )


def _load_stage_model(ckpt_path: Path, device: str, dfc_root: str = ""):
    if dfc_root:
        sys.path.insert(0, dfc_root)
    from train_material import CoefEnergyNetMaterial  # noqa: E402

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_train = ck.get("cfg", {})
    model = CoefEnergyNetMaterial(
        lam_soft_max=float(cfg_train.get("lam_soft_max", 50.0)),
        lam_hard_max=float(cfg_train.get("lam_hard_max", 10.0)),
    ).to(device)
    disable_transformer_nested_tensors(model)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model, ck


class StageModelBaseline(HighwayBaseline):
    def __init__(
        self,
        *,
        name: str,
        ckpt_path: str,
        stage: int,
        device: str = "cpu",
        dfc_root: str = "",
        d_hat_override: float = 0.0,
        alpha_floor_override: float = -1.0,
        alpha_floor_ahead_only_override: Optional[bool] = None,
        disable_mu_lat: bool = False,
        ttc_gain_override: Optional[float] = None,
        ttc_threshold_s_override: Optional[float] = None,
        ttc_softness_s_override: Optional[float] = None,
        ttc_min_closing_speed_override: Optional[float] = None,
        ttc_lane_halfwidth_override: Optional[float] = None,
        ttc_boxed_risk_thresh_override: Optional[float] = None,
        ttc_boxed_gate_sharpness_override: Optional[float] = None,
    ):
        self.name = name
        self.stage = int(stage)
        self.device = device
        self.ckpt_path = str(ckpt_path)
        self.model, self.ck = _load_stage_model(Path(ckpt_path), device, dfc_root)
        cfg = self.ck.get("cfg", {})
        self.runtime = {
            "d_hat": float(d_hat_override) if d_hat_override > 0 else float(cfg.get("d_hat", 0.0)),
            "alpha_floor": (
                float(alpha_floor_override)
                if alpha_floor_override >= 0
                else float(cfg.get("alpha_floor", 0.0))
            ),
            "alpha_floor_ahead_only": (
                bool(alpha_floor_ahead_only_override)
                if alpha_floor_ahead_only_override is not None
                else bool(cfg.get("alpha_floor_ahead_only", False))
            ),
            "disable_mu_lat": bool(disable_mu_lat),
            "ttc_gain": float(ttc_gain_override) if ttc_gain_override is not None else float(cfg.get("ttc_gain", 0.0)),
            "ttc_threshold_s": float(ttc_threshold_s_override) if ttc_threshold_s_override is not None else float(cfg.get("ttc_threshold_s", 3.0)),
            "ttc_softness_s": float(ttc_softness_s_override) if ttc_softness_s_override is not None else float(cfg.get("ttc_softness_s", 0.5)),
            "ttc_min_closing_speed": float(ttc_min_closing_speed_override) if ttc_min_closing_speed_override is not None else float(cfg.get("ttc_min_closing_speed", 0.5)),
            "ttc_lane_halfwidth": float(ttc_lane_halfwidth_override) if ttc_lane_halfwidth_override is not None else float(cfg.get("ttc_lane_halfwidth", 2.0)),
            "ttc_boxed_risk_thresh": float(ttc_boxed_risk_thresh_override) if ttc_boxed_risk_thresh_override is not None else float(cfg.get("ttc_boxed_risk_thresh", 0.25)),
            "ttc_boxed_gate_sharpness": float(ttc_boxed_gate_sharpness_override) if ttc_boxed_gate_sharpness_override is not None else float(cfg.get("ttc_boxed_gate_sharpness", 20.0)),
        }

    @torch.no_grad()
    def act(self, env: Any, observer: Any) -> np.ndarray:
        obs_np = observer.build(env)
        batch = _to_batch(obs_np, self.device)
        if self.runtime["d_hat"] > 0:
            batch["d_hat"] = torch.full_like(batch["d_hat"], float(self.runtime["d_hat"]))

        alphas, beta, gamma, lam_soft, lam_hard, mu_lat = _model_coeffs(self.model, batch)
        alphas = _apply_alpha_floor(
            batch,
            alphas,
            self.runtime["alpha_floor"],
            ahead_only=self.runtime["alpha_floor_ahead_only"],
        )
        if self.stage == 1:
            lam_soft = torch.zeros_like(lam_soft)
            lam_hard = torch.zeros_like(lam_hard)
            mu_lat = None
        elif self.runtime["disable_mu_lat"]:
            mu_lat = None

        v0 = batch["v0"]
        speed_0 = torch.linalg.norm(v0, dim=-1).clamp_min(1e-3)
        heading_0 = torch.atan2(v0[:, 1], v0[:, 0])
        F_tot, _, _, _ = compute_surrogate_highway_force(
            o=batch["o0"],
            heading=heading_0,
            speed=speed_0,
            o0=batch["o0"],
            heading_0=heading_0,
            goal=batch["goal"],
            C=batch["C"],
            V_neighbors=batch.get("V_neighbors"),
            R_eff=batch["R"],
            mask=batch["mask"],
            alphas=alphas,
            beta=beta,
            gamma=gamma,
            lam_soft=lam_soft,
            lam_hard=lam_hard,
            mu_lat=mu_lat,
            rollout_patch=batch["rollout_patch"],
            d_hat=batch["d_hat"],
            ttc_gain=self.runtime["ttc_gain"],
            ttc_threshold_s=self.runtime["ttc_threshold_s"],
            ttc_softness_s=self.runtime["ttc_softness_s"],
            ttc_min_closing_speed=self.runtime["ttc_min_closing_speed"],
            ttc_lane_halfwidth=self.runtime["ttc_lane_halfwidth"],
            ttc_boxed_risk_thresh=self.runtime["ttc_boxed_risk_thresh"],
            ttc_boxed_gate_sharpness=self.runtime["ttc_boxed_gate_sharpness"],
        )
        accel, steer = force_to_action(F_tot, heading_0, speed_0)
        accel_phys = float(accel.clamp(*ACCEL_RANGE).item())
        steer_phys = float(steer.clamp(*STEER_RANGE).item())
        return normalize_action(accel_phys, steer_phys)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "stage": self.stage, "ckpt_path": self.ckpt_path}


class SB3PPOBaseline(HighwayBaseline):
    name = "ppo"

    @staticmethod
    def observation_config() -> Dict[str, Any]:
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "normalize": True,
                "absolute": False,
                "order": "sorted",
            }
        }

    def __init__(self, ckpt_path: str):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError(
                "stable_baselines3 is required for the PPO baseline. "
                "Install it and provide --ppo-ckpt."
            ) from exc
        self.ckpt_path = str(ckpt_path)
        self.model = PPO.load(self.ckpt_path)

    def env_config_overrides(self) -> Dict[str, Any]:
        return self.observation_config()

    def act(self, env: Any, observer: Any) -> np.ndarray:
        obs = env.unwrapped.observation_type.observe()
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(2)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "ckpt_path": self.ckpt_path}


class SB3SACBaseline(SB3PPOBaseline):
    name = "sac"

    def __init__(self, ckpt_path: str):
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise RuntimeError(
                "stable_baselines3 is required for the SAC baseline. "
                "Install it and provide --sac-ckpt."
            ) from exc
        self.ckpt_path = str(ckpt_path)
        self.model = SAC.load(self.ckpt_path)


class SB3GenericPolicyBaseline(SB3PPOBaseline):
    def __init__(self, *, name: str, ckpt_path: str, algo: str):
        self.name = name
        self.ckpt_path = str(ckpt_path)
        algo_name = str(algo).lower()
        try:
            if algo_name == "ppo":
                from stable_baselines3 import PPO

                self.model = PPO.load(self.ckpt_path)
            elif algo_name == "sac":
                from stable_baselines3 import SAC

                self.model = SAC.load(self.ckpt_path)
            else:
                raise RuntimeError(f"Unsupported SB3 algorithm '{algo}' for {name}")
        except ImportError as exc:
            raise RuntimeError(
                f"stable_baselines3 is required for the {name} baseline. "
                f"Install it and provide a compatible checkpoint."
            ) from exc


class ExportedContinuousActor(nn.Module):
    """Small actor used for exported safe-RL baselines when no external runtime is available."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _flatten_obs(obs: Any) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1).astype(np.float32, copy=False)


class GenericContinuousPolicyBaseline(HighwayBaseline):
    """Loader for TorchScript or exported PyTorch continuous-control actors."""

    def __init__(self, *, name: str, ckpt_path: str):
        self.name = name
        self.ckpt_path = str(ckpt_path)
        self._mode = ""
        self._predictor = None

        path = Path(ckpt_path)
        suffix = path.suffix.lower()
        if suffix in {".zip"}:
            raise RuntimeError(
                f"{name} expects either a TorchScript/exported `.pt` actor or a dedicated runtime wrapper. "
                f"Received unsupported checkpoint format: {ckpt_path}"
            )

        try:
            mod = torch.jit.load(self.ckpt_path, map_location="cpu")
            mod.eval()
            self._mode = "torchscript"
            self._predictor = mod
            return
        except Exception:
            pass

        obj = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            obj.eval()
            self._mode = "module"
            self._predictor = obj
            return
        if isinstance(obj, dict) and "model_state" in obj and "obs_dim" in obj:
            actor = ExportedContinuousActor(
                obs_dim=int(obj["obs_dim"]),
                hidden_dim=int(obj.get("hidden_dim", 256)),
            )
            actor.load_state_dict(obj["model_state"], strict=True)
            actor.eval()
            self._mode = "exported_state"
            self._predictor = actor
            return
        if hasattr(obj, "predict"):
            self._mode = "predict_api"
            self._predictor = obj
            return

        raise RuntimeError(
            f"Could not load {name} checkpoint '{ckpt_path}'. "
            "Supported formats: TorchScript `.pt`, saved `nn.Module`, "
            "dict with `model_state` + `obs_dim`, or object with `.predict()`."
        )

    def env_config_overrides(self) -> Dict[str, Any]:
        return SB3PPOBaseline.observation_config()

    def act(self, env: Any, observer: Any) -> np.ndarray:
        obs = env.unwrapped.observation_type.observe()
        flat = _flatten_obs(obs)
        if self._mode == "predict_api":
            action = self._predictor.predict(obs)
            if isinstance(action, tuple):
                action = action[0]
        else:
            with torch.no_grad():
                inp = torch.from_numpy(flat).unsqueeze(0)
                action = self._predictor(inp)
                if isinstance(action, (tuple, list)):
                    action = action[0]
                action = action.squeeze(0).detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(2)
        return np.clip(action, -1.0, 1.0)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "ckpt_path": self.ckpt_path, "loader": self._mode}

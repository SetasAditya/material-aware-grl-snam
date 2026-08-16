from __future__ import annotations

from typing import Any, Dict

from .learned import (
    GenericContinuousPolicyBaseline,
    SB3GenericPolicyBaseline,
    SB3PPOBaseline,
    SB3SACBaseline,
    StageModelBaseline,
)
from .mpc import ChanceConstrainedMPCBaseline, RiskAwareMPCBaseline
from .rule_based import (
    ConstantVelocityBaseline,
    IDMBaseline,
    IDMFollowOnlyBaseline,
    IDMMOBILBaseline,
    MOBILIDMBaseline,
    SafeStopBaseline,
)
from .safety import CBFQPFilterBaseline


BASELINE_NAMES = (
    "constant_velocity",
    "idm",
    "mobil_idm",
    "ppo",
    "sac",
    "ppo_lagrangian",
    "sac_lagrangian",
    "cpo",
    "risk_aware_mpc",
    "chance_constrained_mpc",
    "cbf_qp_filter",
    "s1_model",
    "s2_model",
    # Backward-compatible aliases
    "safe_stop",
    "idm_follow_only",
    "idm_mobil",
)


def create_baseline(name: str, **kwargs):
    if name == "constant_velocity":
        return ConstantVelocityBaseline(**{k: v for k, v in kwargs.items() if k in {"target_speed"}})
    if name == "safe_stop":
        allowed = {"cruise_speed", "stop_gap_m", "caution_gap_m", "min_follow_speed"}
        return SafeStopBaseline(**{k: v for k, v in kwargs.items() if k in allowed})
    if name == "idm":
        return IDMBaseline()
    if name == "mobil_idm":
        return MOBILIDMBaseline()
    if name == "idm_follow_only":
        return IDMFollowOnlyBaseline()
    if name == "idm_mobil":
        return IDMMOBILBaseline()
    if name == "ppo":
        ckpt = kwargs.get("ppo_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("ppo baseline requires ppo_ckpt")
        return SB3PPOBaseline(ckpt)
    if name == "sac":
        ckpt = kwargs.get("sac_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("sac baseline requires sac_ckpt")
        return SB3SACBaseline(ckpt)
    if name == "ppo_lagrangian":
        ckpt = kwargs.get("ppo_lagrangian_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("ppo_lagrangian baseline requires ppo_lagrangian_ckpt")
        if str(ckpt).lower().endswith(".zip"):
            return SB3GenericPolicyBaseline(name="ppo_lagrangian", ckpt_path=ckpt, algo="ppo")
        return GenericContinuousPolicyBaseline(name="ppo_lagrangian", ckpt_path=ckpt)
    if name == "sac_lagrangian":
        ckpt = kwargs.get("sac_lagrangian_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("sac_lagrangian baseline requires sac_lagrangian_ckpt")
        if str(ckpt).lower().endswith(".zip"):
            return SB3GenericPolicyBaseline(name="sac_lagrangian", ckpt_path=ckpt, algo="sac")
        return GenericContinuousPolicyBaseline(name="sac_lagrangian", ckpt_path=ckpt)
    if name == "cpo":
        ckpt = kwargs.get("cpo_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("cpo baseline requires cpo_ckpt")
        if str(ckpt).lower().endswith(".zip"):
            return SB3GenericPolicyBaseline(name="cpo", ckpt_path=ckpt, algo="ppo")
        return GenericContinuousPolicyBaseline(name="cpo", ckpt_path=ckpt)
    if name == "risk_aware_mpc":
        allowed = {"horizon", "dt", "target_speed", "safe_clearance_m", "ttc_safe_s", "lane_change_penalty"}
        return RiskAwareMPCBaseline(**{k: v for k, v in kwargs.items() if k in allowed})
    if name == "chance_constrained_mpc":
        allowed = {
            "horizon",
            "dt",
            "target_speed",
            "safe_clearance_m",
            "ttc_safe_s",
            "lane_change_penalty",
            "chance_margin_m",
            "tail_risk_weight",
        }
        return ChanceConstrainedMPCBaseline(**{k: v for k, v in kwargs.items() if k in allowed})
    if name == "cbf_qp_filter":
        allowed = {
            "nominal_speed",
            "safe_time_headway_s",
            "min_gap_m",
            "gamma_long",
            "gamma_lat",
            "dt",
        }
        return CBFQPFilterBaseline(**{k: v for k, v in kwargs.items() if k in allowed})
    if name == "s1_model":
        ckpt = kwargs.get("s1_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("s1_model baseline requires s1_ckpt")
        return StageModelBaseline(
            name="s1_model",
            ckpt_path=ckpt,
            stage=1,
            device=kwargs.get("device", "cpu"),
            dfc_root=kwargs.get("dfc_root", ""),
            d_hat_override=kwargs.get("d_hat_override", 0.0),
            alpha_floor_override=kwargs.get("alpha_floor_override", -1.0),
            alpha_floor_ahead_only_override=kwargs.get("alpha_floor_ahead_only_override"),
        )
    if name == "s2_model":
        ckpt = kwargs.get("s2_ckpt") or kwargs.get("ckpt_path")
        if not ckpt:
            raise ValueError("s2_model baseline requires s2_ckpt")
        return StageModelBaseline(
            name="s2_model",
            ckpt_path=ckpt,
            stage=2,
            device=kwargs.get("device", "cpu"),
            dfc_root=kwargs.get("dfc_root", ""),
            d_hat_override=kwargs.get("d_hat_override", 0.0),
            alpha_floor_override=kwargs.get("alpha_floor_override", -1.0),
            alpha_floor_ahead_only_override=kwargs.get("alpha_floor_ahead_only_override"),
            disable_mu_lat=bool(kwargs.get("disable_mu_lat", False)),
            ttc_gain_override=kwargs.get("ttc_gain_override"),
            ttc_threshold_s_override=kwargs.get("ttc_threshold_s_override"),
            ttc_softness_s_override=kwargs.get("ttc_softness_s_override"),
            ttc_min_closing_speed_override=kwargs.get("ttc_min_closing_speed_override"),
            ttc_lane_halfwidth_override=kwargs.get("ttc_lane_halfwidth_override"),
            ttc_boxed_risk_thresh_override=kwargs.get("ttc_boxed_risk_thresh_override"),
            ttc_boxed_gate_sharpness_override=kwargs.get("ttc_boxed_gate_sharpness_override"),
        )
    raise ValueError(f"Unknown baseline '{name}'. Available: {', '.join(BASELINE_NAMES)}")

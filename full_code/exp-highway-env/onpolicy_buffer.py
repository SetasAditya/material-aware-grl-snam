#!/usr/bin/env python3
"""
onpolicy_buffer.py — Step 6, component 1.

FIFO sample buffer for on-policy trajectories collected by running the
current Stage 2 model in highway-env. Used by train_stage2.py alongside
the static IDM imitation dataset.

The buffer holds *timestep* samples, not trajectories. Each sample carries:
    • The 12-key observation dict (same as IDM samples) — input to the model
    • The action the model took at deployment (accel, steer)
    • Per-episode outcome metadata (collided, min_clearance, length)
    • A stream tag (`is_onpolicy=1`) for downstream loss masking
    • `has_imit_target=0` and zero-placeholder imitation targets so the
      collate function is identical to the IDM dataloader's

Stream identity is explicit at every interface:
    OnPolicyBuffer.stats()  reports counts and crash rate for buffer contents
    to_torch_dict()         emits is_onpolicy=1 and has_imit_target=0
    The trainer multiplies imitation losses by has_imit_target

Typical usage
-------------
    buf = OnPolicyBuffer(capacity=5000)
    # ... collect_onpolicy.py runs episodes, calls:
    buf.append_episode(samples_for_one_episode)
    # ... train_stage2.py at start of epoch:
    snap = buf.snapshot_dataset()
    loader = DataLoader(snap, batch_size=64, shuffle=True, collate_fn=collate)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


# Stream id constants. Kept as integers so they round-trip through tensor
# batches cleanly.
STREAM_IDM      = 0
STREAM_ONPOLICY = 1


# ─────────────────────────────────────────────────────────────────────────────
# Sample record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OnPolicySample:
    """One timestep from a closed-loop on-policy rollout in highway-env.

    All tensor fields are CPU torch.Tensors. Conversion to GPU happens at the
    DataLoader → trainer boundary, not in the buffer.
    """
    # ── Required at construction time ────────────────────────────────────────
    obs: Dict[str, torch.Tensor]            # 12-key obs dict, same as IDM
    action: torch.Tensor                    # (2,) accel, steer the model output
    o_next: torch.Tensor                    # (2,) ego pos one env-step later
    v_next: torch.Tensor                    # (2,) ego vel one env-step later
    step_in_episode: int                    # 0-indexed timestep within episode

    # ── Stamped by buffer.append_episode (not by collector) ──────────────────
    episode_id: int = -1                    # buffer-local; -1 until appended

    # ── Episode-level outcome (set by collector, replicated to every sample) ─
    deploy_collided: bool = False
    deploy_min_clearance: float = float("inf")
    deploy_episode_length: int = 0

    # ── Stream tag — always STREAM_ONPOLICY for buffer entries ───────────────
    stream: int = STREAM_ONPOLICY

    def to_torch_dict(self) -> Dict[str, torch.Tensor]:
        """Flat dict consumable by the same collate function used for IDM.

        Imitation-target keys are zero placeholders gated by has_imit_target=0.
        The trainer's loss combiner multiplies imitation losses by
        has_imit_target so on-policy samples contribute zero to L_traj/L_vel
        /L_lane/L_act regardless of placeholder values.
        """
        out: Dict[str, torch.Tensor] = dict(self.obs)        # 12 obs keys

        # Action and next-step state (used by trainer for action loss
        # *only* on IDM samples, gated by has_action_tgt; on-policy passes
        # has_action_tgt=0 so this is unused).
        out["action_taken"] = self.action.float()
        out["o_next"]       = self.o_next.float()
        out["v_next"]       = self.v_next.float()

        # Imitation target placeholders — values don't matter, mask zeros them
        out["o_tgt"]      = torch.zeros(2, dtype=torch.float32)
        out["v_tgt"]      = torch.zeros(2, dtype=torch.float32)
        out["y_lane_tgt"] = torch.zeros((), dtype=torch.float32)
        out["accel_tgt"]  = torch.zeros((), dtype=torch.float32)
        out["steer_tgt"]  = torch.zeros((), dtype=torch.float32)

        # Stream tag and gates. The trainer reads these to mask losses
        # per sample. All scalars-as-tensors so they stack cleanly.
        out["is_onpolicy"]      = torch.tensor(float(self.stream == STREAM_ONPOLICY))
        out["has_imit_target"]  = torch.tensor(0.0)         # OFF for on-policy
        out["has_action_tgt"]   = torch.tensor(0.0)         # OFF for on-policy

        # Episode metadata (useful for per-episode analysis at training time)
        out["episode_id"]            = torch.tensor(self.episode_id, dtype=torch.long)
        out["step_in_episode"]       = torch.tensor(self.step_in_episode, dtype=torch.long)
        out["deploy_collided"]       = torch.tensor(float(self.deploy_collided))
        out["deploy_min_clearance"]  = torch.tensor(float(self.deploy_min_clearance))

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot dataset — frozen view used by DataLoader during one epoch
# ─────────────────────────────────────────────────────────────────────────────

class OnPolicySnapshot(Dataset):
    """Frozen list-snapshot of buffer contents at one moment.

    Why this exists: OnPolicyBuffer mutates as new episodes are collected.
    DataLoader workers cannot tolerate the underlying container changing
    during iteration. Solution: at the start of each training epoch, take
    a snapshot via `buf.snapshot_dataset()`, hand that to DataLoader, and
    let the buffer keep mutating in the background.

    The snapshot copies references to the underlying OnPolicySample objects,
    not deep copies — they're not mutated during training so this is safe.
    """

    def __init__(self, samples: List[OnPolicySample]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._samples[idx].to_torch_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Buffer
# ─────────────────────────────────────────────────────────────────────────────

class OnPolicyBuffer:
    """FIFO buffer of on-policy timestep samples.

    Capacity-bounded. When append would exceed capacity, oldest samples are
    evicted automatically (collections.deque maxlen semantics).

    Threading: not thread-safe. Single producer (collection), single consumer
    (training); both are sequential in the planned trainer.

    Stats: separate "buffer-current" stats (only over samples still resident
    after eviction) and "lifetime" stats (running counters of everything that
    was ever added). The trainer should log both — buffer-current to track
    distribution drift, lifetime to track total work done.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._samples: "deque[OnPolicySample]" = deque(maxlen=capacity)

        # Episode id assignment. The buffer owns this so samples added via
        # different append_episode calls always get distinct ids even if
        # collector code resets its own counter.
        self._next_episode_id: int = 0

        # Lifetime counters (NOT reset on eviction)
        self._lifetime_episodes: int = 0
        self._lifetime_samples: int = 0
        self._lifetime_collisions: int = 0

    # ── Core API ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterable[OnPolicySample]:
        return iter(self._samples)

    def append_episode(self, samples: List[OnPolicySample]) -> int:
        """Append a complete episode's samples. Returns assigned episode_id.

        All samples are assumed to come from one episode and share identical
        deploy_collided / deploy_min_clearance / deploy_episode_length values.
        We don't check this — the collector is responsible.
        """
        if not samples:
            return -1
        ep_id = self._next_episode_id
        self._next_episode_id += 1

        for s in samples:
            s.episode_id = ep_id

        self._samples.extend(samples)            # FIFO eviction handled by deque

        self._lifetime_episodes  += 1
        self._lifetime_samples   += len(samples)
        if samples[0].deploy_collided:
            self._lifetime_collisions += 1

        return ep_id

    def sample(self, n: int, *, rng: Optional[random.Random] = None
                ) -> List[OnPolicySample]:
        """Draw `n` samples uniformly without replacement.

        If buffer has fewer than `n` samples, returns all of them. Caller
        should check `len(buffer)` before calling for batched training.
        """
        if not self._samples:
            return []
        rng = rng if rng is not None else random
        n = min(n, len(self._samples))
        # deque doesn't support O(1) random access; convert to list. For
        # buffers in the 5k-sample range this is ~us-cheap.
        return rng.sample(list(self._samples), n)

    def snapshot_dataset(self) -> OnPolicySnapshot:
        """Frozen Dataset view of current buffer contents for DataLoader use."""
        return OnPolicySnapshot(list(self._samples))

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Per-stream counts and crash rate. Always non-throwing."""
        n_samples = len(self._samples)
        if n_samples == 0:
            return {
                "size":                  0,
                "capacity":              self.capacity,
                "n_episodes":            0,
                "n_crashed_episodes":    0,
                "collision_rate":        0.0,
                "mean_min_clearance":    0.0,
                "mean_episode_length":   0.0,
                "lifetime_episodes":     self._lifetime_episodes,
                "lifetime_samples":      self._lifetime_samples,
                "lifetime_collisions":   self._lifetime_collisions,
            }

        # One representative sample per resident episode for episode-level stats
        seen_eps: Dict[int, OnPolicySample] = {}
        for s in self._samples:
            if s.episode_id not in seen_eps:
                seen_eps[s.episode_id] = s

        eps = list(seen_eps.values())
        n_eps = len(eps)
        n_crashed = sum(1 for e in eps if e.deploy_collided)
        # Skip inf values in the mean — those are "no neighbours" samples.
        clears = [e.deploy_min_clearance for e in eps
                  if e.deploy_min_clearance != float("inf")]
        lengths = [e.deploy_episode_length for e in eps]

        return {
            "size":                  n_samples,
            "capacity":              self.capacity,
            "n_episodes":            n_eps,
            "n_crashed_episodes":    n_crashed,
            "collision_rate":        n_crashed / n_eps,
            "mean_min_clearance":    float(sum(clears) / len(clears)) if clears else 0.0,
            "mean_episode_length":   float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "lifetime_episodes":     self._lifetime_episodes,
            "lifetime_samples":      self._lifetime_samples,
            "lifetime_collisions":   self._lifetime_collisions,
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> None:
        """Serialize to disk. Use this between training runs / for resumption."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version":               1,
            "capacity":              self.capacity,
            "samples":               list(self._samples),
            "next_episode_id":       self._next_episode_id,
            "lifetime_episodes":     self._lifetime_episodes,
            "lifetime_samples":      self._lifetime_samples,
            "lifetime_collisions":   self._lifetime_collisions,
        }, path)

    @classmethod
    def load(cls, path) -> "OnPolicyBuffer":
        d = torch.load(Path(path), weights_only=False)
        if d.get("version") != 1:
            raise ValueError(f"Unsupported buffer version: {d.get('version')}")
        b = cls(capacity=d["capacity"])
        b._samples.extend(d["samples"])
        b._next_episode_id     = d["next_episode_id"]
        b._lifetime_episodes   = d["lifetime_episodes"]
        b._lifetime_samples    = d["lifetime_samples"]
        b._lifetime_collisions = d["lifetime_collisions"]
        return b

    # ── Useful when juggling a buffer in a Jupyter session etc. ──────────────

    def __repr__(self) -> str:
        s = self.stats()
        return (f"OnPolicyBuffer(size={s['size']}/{self.capacity}, "
                 f"n_episodes={s['n_episodes']}, "
                 f"collision_rate={s['collision_rate']:.1%}, "
                 f"lifetime_episodes={s['lifetime_episodes']})")

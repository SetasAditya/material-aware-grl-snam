#!/usr/bin/env python3
"""
test_onpolicy_buffer.py — T14-T18.

Tests the FIFO buffer in isolation. No env, no model, no training. Validates:
    T14: append + len works correctly, episode_id stamped from buffer counter
    T15: capacity eviction is FIFO (oldest sample first)
    T16: stats() reports per-stream counts and crash rate accurately
    T17: save/load roundtrip preserves everything
    T18: snapshot_dataset works with a torch DataLoader, batches stack cleanly

All tests run in <1 second on CPU.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from onpolicy_buffer import (  # noqa: E402
    OnPolicyBuffer, OnPolicySample, OnPolicySnapshot,
    STREAM_IDM, STREAM_ONPOLICY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_obs(n_max: int = 15, hp: int = 8, wp: int = 4) -> dict:
    """Minimal 12-key obs dict matching env_wrapper output. Tiny patch sizes
    keep tests fast; the buffer doesn't care about actual shapes."""
    return {
        "o0":            torch.zeros(2),
        "v0":            torch.zeros(2),
        "goal":          torch.tensor([100.0, 0.0]),
        "C":             torch.zeros(n_max, 2),
        "R":             torch.zeros(n_max),
        "W":             torch.zeros(n_max),
        "mask":          torch.zeros(n_max, dtype=torch.bool),
        "risk_patch":    torch.zeros(2, hp, wp),
        "rollout_patch": torch.zeros(6, hp, wp),
        "d_hat":         torch.tensor(10.0),
        "dt":            torch.tensor(0.1),
        "H":             torch.tensor(20, dtype=torch.long),
    }


def _fake_episode(n_steps: int, *, collided: bool = False,
                    min_clear: float = 5.0) -> List[OnPolicySample]:
    return [
        OnPolicySample(
            obs                   = _fake_obs(),
            action                = torch.tensor([0.1, 0.0]),
            o_next                = torch.tensor([float(t) * 2.5, 0.0]),
            v_next                = torch.tensor([25.0, 0.0]),
            step_in_episode       = t,
            deploy_collided       = collided,
            deploy_min_clearance  = min_clear,
            deploy_episode_length = n_steps,
        )
        for t in range(n_steps)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# T14: append + len + episode_id stamping
# ─────────────────────────────────────────────────────────────────────────────

def test_t14_append_and_episode_ids():
    print("T14: append + episode_id stamping ...", end=" ")
    buf = OnPolicyBuffer(capacity=100)
    assert len(buf) == 0

    ep1 = _fake_episode(5)
    # Pre-append: episode_id should be the default sentinel
    assert all(s.episode_id == -1 for s in ep1)

    ep_id_1 = buf.append_episode(ep1)
    assert ep_id_1 == 0
    assert len(buf) == 5
    assert all(s.episode_id == 0 for s in ep1), \
        "All samples in one episode should share episode_id"

    # Second episode gets next id
    ep2 = _fake_episode(3)
    ep_id_2 = buf.append_episode(ep2)
    assert ep_id_2 == 1
    assert len(buf) == 8
    assert all(s.episode_id == 1 for s in ep2)

    # Empty episode is no-op, returns -1
    assert buf.append_episode([]) == -1
    assert len(buf) == 8

    # Stream tag is set correctly by default
    assert all(s.stream == STREAM_ONPOLICY for s in buf)

    print("✓")


# ─────────────────────────────────────────────────────────────────────────────
# T15: capacity eviction is FIFO
# ─────────────────────────────────────────────────────────────────────────────

def test_t15_fifo_eviction():
    print("T15: FIFO eviction at capacity ...", end=" ")
    buf = OnPolicyBuffer(capacity=10)

    # Fill to capacity
    buf.append_episode(_fake_episode(10))
    assert len(buf) == 10

    samples_before = list(buf)
    assert all(s.episode_id == 0 for s in samples_before)
    assert [s.step_in_episode for s in samples_before] == list(range(10))

    # Add 5 more — should evict the first 5 of episode 0
    buf.append_episode(_fake_episode(5))
    assert len(buf) == 10  # capacity-bounded

    samples_after = list(buf)
    # First 5 should be remaining episode-0 (steps 5..9)
    assert [s.episode_id for s in samples_after[:5]] == [0] * 5
    assert [s.step_in_episode for s in samples_after[:5]] == [5, 6, 7, 8, 9]
    # Last 5 should be episode 1
    assert [s.episode_id for s in samples_after[5:]] == [1] * 5
    assert [s.step_in_episode for s in samples_after[5:]] == [0, 1, 2, 3, 4]

    # Lifetime counters track everything ever added, not just resident
    s = buf.stats()
    assert s["lifetime_samples"] == 15, f"got {s['lifetime_samples']}"
    assert s["lifetime_episodes"] == 2

    # Capacity validation
    try:
        OnPolicyBuffer(capacity=0)
        assert False, "capacity=0 should raise"
    except ValueError:
        pass

    print("✓")


# ─────────────────────────────────────────────────────────────────────────────
# T16: stats() accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_t16_stats():
    print("T16: stats() reports correct counts ...", end=" ")
    buf = OnPolicyBuffer(capacity=1000)

    # Empty buffer is well-defined
    s = buf.stats()
    assert s["size"] == 0
    assert s["n_episodes"] == 0
    assert s["collision_rate"] == 0.0
    assert s["lifetime_episodes"] == 0

    # 4 episodes: 3 clean, 1 crashed
    buf.append_episode(_fake_episode(10, collided=False, min_clear=8.0))
    buf.append_episode(_fake_episode(10, collided=False, min_clear=6.0))
    buf.append_episode(_fake_episode(8,  collided=True,  min_clear=1.5))
    buf.append_episode(_fake_episode(10, collided=False, min_clear=10.0))

    s = buf.stats()
    assert s["size"] == 38
    assert s["n_episodes"] == 4
    assert s["n_crashed_episodes"] == 1
    assert abs(s["collision_rate"] - 0.25) < 1e-6
    # Mean min clearance: (8 + 6 + 1.5 + 10) / 4 = 6.375
    assert abs(s["mean_min_clearance"] - 6.375) < 1e-6
    # Mean episode length: (10 + 10 + 8 + 10) / 4 = 9.5
    assert abs(s["mean_episode_length"] - 9.5) < 1e-6
    assert s["lifetime_collisions"] == 1
    assert s["lifetime_episodes"] == 4

    # After eviction, buffer-current vs lifetime should diverge.
    small = OnPolicyBuffer(capacity=15)
    small.append_episode(_fake_episode(10, collided=True))   # evicted partly
    small.append_episode(_fake_episode(10, collided=False))
    s = small.stats()
    # Buffer holds 15 samples: last 5 of crashed ep + all 10 of clean ep
    assert s["size"] == 15
    assert s["n_episodes"] == 2  # both still represented
    assert s["n_crashed_episodes"] == 1
    # Lifetime tracks everything added
    assert s["lifetime_samples"] == 20
    assert s["lifetime_collisions"] == 1

    print("✓")


# ─────────────────────────────────────────────────────────────────────────────
# T17: save/load roundtrip
# ─────────────────────────────────────────────────────────────────────────────

def test_t17_save_load_roundtrip():
    print("T17: save/load roundtrip ...", end=" ")
    buf = OnPolicyBuffer(capacity=50)
    buf.append_episode(_fake_episode(10, collided=False, min_clear=7.5))
    buf.append_episode(_fake_episode(8,  collided=True,  min_clear=1.0))

    s_before = buf.stats()

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "buf.pt"
        buf.save(path)
        assert path.exists()

        loaded = OnPolicyBuffer.load(path)
        s_after = loaded.stats()

    # All numeric stats preserved
    for k in ("size", "capacity", "n_episodes", "n_crashed_episodes",
                "lifetime_episodes", "lifetime_samples", "lifetime_collisions"):
        assert s_before[k] == s_after[k], f"{k} mismatch: {s_before[k]} vs {s_after[k]}"

    # Episode-id counter is preserved (so next episode gets a fresh id)
    assert loaded._next_episode_id == 2

    # Sample-level data is preserved
    orig_samples = list(buf)
    new_samples = list(loaded)
    assert len(orig_samples) == len(new_samples)
    for a, b in zip(orig_samples, new_samples):
        assert a.episode_id == b.episode_id
        assert a.step_in_episode == b.step_in_episode
        assert a.deploy_collided == b.deploy_collided
        assert torch.equal(a.action, b.action)
        # Spot-check obs dict
        assert torch.equal(a.obs["o0"], b.obs["o0"])
        assert torch.equal(a.obs["risk_patch"], b.obs["risk_patch"])

    print("✓")


# ─────────────────────────────────────────────────────────────────────────────
# T18: snapshot_dataset + DataLoader integration
# ─────────────────────────────────────────────────────────────────────────────

def _collate(batch):
    """Minimal collate matching what train_stage2 will use: stack each key."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def test_t18_snapshot_dataloader():
    print("T18: snapshot_dataset + DataLoader ...", end=" ")
    buf = OnPolicyBuffer(capacity=100)
    buf.append_episode(_fake_episode(20))
    buf.append_episode(_fake_episode(20))

    snap = buf.snapshot_dataset()
    assert isinstance(snap, OnPolicySnapshot)
    assert len(snap) == 40

    # DataLoader cycles through the snapshot
    loader = DataLoader(snap, batch_size=8, shuffle=True,
                         collate_fn=_collate, num_workers=0)
    batches = list(loader)
    assert len(batches) == 5  # 40 / 8

    # Each batch has the expected keys and shapes
    b = batches[0]
    expected_obs_keys = {"o0", "v0", "goal", "C", "R", "W", "mask",
                          "risk_patch", "rollout_patch", "d_hat", "dt", "H"}
    expected_imit_keys = {"o_tgt", "v_tgt", "y_lane_tgt",
                           "accel_tgt", "steer_tgt"}
    expected_meta_keys = {"is_onpolicy", "has_imit_target", "has_action_tgt",
                           "episode_id", "step_in_episode",
                           "deploy_collided", "deploy_min_clearance",
                           "action_taken", "o_next", "v_next"}
    assert set(b.keys()) >= expected_obs_keys | expected_imit_keys | expected_meta_keys

    # Shapes
    assert b["o0"].shape == (8, 2)
    assert b["risk_patch"].shape == (8, 2, 8, 4)        # batch dim added
    assert b["mask"].shape == (8, 15)
    assert b["mask"].dtype == torch.bool

    # Stream tags: all on-policy
    assert b["is_onpolicy"].shape == (8,)
    assert (b["is_onpolicy"] == 1.0).all()
    assert (b["has_imit_target"] == 0.0).all(), \
        "On-policy samples must have has_imit_target=0 to gate imit losses"
    assert (b["has_action_tgt"] == 0.0).all()

    # Buffer can keep mutating after snapshot — snapshot is frozen
    buf.append_episode(_fake_episode(10))
    assert len(buf) == 50         # buffer grew
    assert len(snap) == 40        # snapshot unchanged
    assert sum(b["o0"].shape[0] for b in batches) == 40

    print("✓")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    test_t14_append_and_episode_ids()
    test_t15_fifo_eviction()
    test_t16_stats()
    test_t17_save_load_roundtrip()
    test_t18_snapshot_dataloader()
    print("\nAll buffer tests (T14-T18) passed.")


if __name__ == "__main__":
    main()
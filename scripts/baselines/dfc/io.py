from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    split: str
    scene_id: str
    path: Path
    start_rc: tuple[int, int]
    goal_rc: tuple[int, int]
    success: bool


def _resolve_entry_path(root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    candidate = root / raw_path
    if candidate.exists():
        return candidate
    candidate = root / "episodes" / f"ep_{Path(raw_path).stem}" / "episode.pt"
    return candidate


def load_manifest_entries(
    root: Path,
    *,
    split: Optional[str] = None,
    max_episodes: Optional[int] = None,
) -> List[EpisodeSpec]:
    manifest = json.loads((root / "manifest.json").read_text())
    entries: List[EpisodeSpec] = []
    for raw in manifest:
        if split is not None and raw.get("split") != split:
            continue
        entry = EpisodeSpec(
            episode_id=str(raw["episode_id"]),
            split=str(raw["split"]),
            scene_id=str(raw["scene_id"]),
            path=_resolve_entry_path(root, str(raw["path"])),
            start_rc=tuple(int(x) for x in raw["start_rc"]),
            goal_rc=tuple(int(x) for x in raw["goal_rc"]),
            success=bool(raw.get("success", True)),
        )
        entries.append(entry)
        if max_episodes is not None and len(entries) >= max_episodes:
            break
    return entries


def load_episode(entry: EpisodeSpec) -> Dict[str, Any]:
    return torch.load(entry.path, map_location="cpu", weights_only=False)


def load_scene(root: Path, scene_id: str) -> Dict[str, Any]:
    scene_path = root / f"scene_{scene_id}.pt"
    return torch.load(scene_path, map_location="cpu", weights_only=False)

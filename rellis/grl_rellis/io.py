from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class RellisFrame:
    sequence: str
    frame_id: str
    scan_path: Path
    label_path: Path
    split: str


def iter_split_frames(
    data_root: str | Path,
    split_file: str | Path,
    *,
    split: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> Iterator[RellisFrame]:
    data_root = Path(data_root)
    split_file = Path(split_file)
    inferred_split = split or split_file.stem.replace("pt_", "")
    count = 0
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        scan_rel, label_rel = line.split()[:2]
        scan_path = data_root / scan_rel
        label_path = data_root / label_rel
        seq = Path(scan_rel).parts[0]
        frame_id = Path(scan_rel).stem
        yield RellisFrame(
            sequence=seq,
            frame_id=frame_id,
            scan_path=scan_path,
            label_path=label_path,
            split=inferred_split,
        )
        count += 1
        if max_frames is not None and count >= max_frames:
            break


def load_frame(frame: RellisFrame) -> tuple[np.ndarray, np.ndarray]:
    points = np.fromfile(frame.scan_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Scan is not Nx4 float32: {frame.scan_path}")
    points = points.reshape(-1, 4)
    labels = np.fromfile(frame.label_path, dtype=np.uint32) & 0xFFFF
    if points.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Point/label mismatch for {frame.sequence}/{frame.frame_id}: "
            f"{points.shape[0]} points vs {labels.shape[0]} labels"
        )
    return points, labels.astype(np.uint16, copy=False)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Set

import yaml


RAW_LABELS: Dict[int, str] = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}


@dataclass(frozen=True)
class RellisOntology:
    name: str
    description: str
    rho_by_id: Dict[int, float]
    hard_ids: Set[int]
    soft_ids: Set[int]
    low_ids: Set[int]
    unknown_is_obstacle: bool
    void_risk: float

    def class_name(self, label_id: int) -> str:
        return RAW_LABELS.get(int(label_id), f"unknown_{int(label_id)}")


def _invert_names(values: Mapping[str, float]) -> Dict[int, float]:
    name_to_id = {name: idx for idx, name in RAW_LABELS.items()}
    out: Dict[int, float] = {}
    for name, risk in values.items():
        if name not in name_to_id:
            raise KeyError(f"Unknown RELLIS class in ontology: {name}")
        out[name_to_id[name]] = float(risk)
    return out


def load_ontology(path: str | Path, mapping: str = "main") -> RellisOntology:
    data = yaml.safe_load(Path(path).read_text())
    if mapping not in data:
        raise KeyError(f"Ontology mapping '{mapping}' not found in {path}")
    raw = data[mapping]
    void_risk = float(raw.get("void_risk", 0.55))
    rho_by_id = {idx: void_risk for idx in RAW_LABELS}

    low = _invert_names(raw.get("low_risk", {}))
    med = _invert_names(raw.get("medium_risk", {}))
    high = _invert_names(raw.get("high_soft_risk", {}))
    hard = _invert_names(raw.get("hard_hazard", {}))
    for group in (low, med, high, hard):
        rho_by_id.update(group)

    return RellisOntology(
        name=str(mapping),
        description=str(raw.get("description", "")),
        rho_by_id=rho_by_id,
        hard_ids=set(hard),
        soft_ids=set(med) | set(high),
        low_ids=set(low),
        unknown_is_obstacle=bool(raw.get("unknown_is_obstacle", True)),
        void_risk=void_risk,
    )

"""Core 3D data structures for LogicSplat."""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Object3D:
    """A discovered 3D object from point cloud clustering."""
    uid: int
    centroid: np.ndarray   # [x, y, z]
    bbox_min: np.ndarray   # [x, y, z]
    bbox_max: np.ndarray   # [x, y, z]
    color: np.ndarray      # mean RGB [0-255]
    point_count: int
    label: str = "object"

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def volume(self) -> float:
        s = self.size
        return float(s[0] * s[1] * s[2])

    @property
    def bottom_z(self) -> float:
        return float(self.bbox_min[2])

    @property
    def top_z(self) -> float:
        return float(self.bbox_max[2])

    def __repr__(self) -> str:
        x, y, z = self.centroid
        return f"<Obj {self.uid} '{self.label}' @ [{x:.2f},{y:.2f},{z:.2f}] pts={self.point_count}>"


@dataclass
class Relation:
    """A directed spatial/physical relation between two objects."""
    subject_id: int
    relation: str
    object_id: int
    confidence: float = 1.0

    def to_text(self, objects: List[Object3D]) -> str:
        subj = next((o for o in objects if o.uid == self.subject_id), None)
        obj  = next((o for o in objects if o.uid == self.object_id),  None)
        s = subj.label if subj else f"Object_{self.subject_id}"
        o = obj.label  if obj  else f"Object_{self.object_id}"
        templates = {
            "on_top_of":  f"{s} is resting on top of {o}.",
            "inside":     f"{s} is contained inside {o}.",
            "occludes":   f"{s} is blocking {o} from view.",
            "adjacent_to":f"{s} is adjacent to {o}.",
            "supported_by":f"{s} is supported by {o}.",
        }
        return templates.get(self.relation, f"{s} {self.relation} {o}.")

    def __repr__(self) -> str:
        return f"Object_{self.subject_id} --[{self.relation}]--> Object_{self.object_id}"


@dataclass
class SceneGraph:
    """Full scene representation with objects and relations."""
    scene_id: str
    objects: List[Object3D] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def get_object(self, uid: int) -> Optional[Object3D]:
        return next((o for o in self.objects if o.uid == uid), None)

    def __repr__(self) -> str:
        return f"<SceneGraph '{self.scene_id}': {len(self.objects)} objects, {len(self.relations)} relations>"

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
        return (f"<Obj {self.uid} '{self.label}' "
                f"@ [{x:.2f},{y:.2f},{z:.2f}] pts={self.point_count}>")


@dataclass
class RelationEdge:
    """
    A directed spatial/physical relation between two objects.

    Named RelationEdge to avoid collision with the Relation enum
    in src.relations.schema.
    """
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
            "on_top_of":   f"The {s} is resting on top of the {o}.",
            "under":       f"The {s} is underneath the {o}.",
            "inside":      f"The {s} is inside the {o}.",
            "adjacent_to": f"The {s} is adjacent to the {o}.",
            "left_of":     f"The {s} is to the left of the {o}.",
            "right_of":    f"The {s} is to the right of the {o}.",
            "in_front_of": f"The {s} is in front of the {o}.",
            "behind":      f"The {s} is behind the {o}.",
            "higher_than": f"The {s} is higher than the {o}.",
            "lower_than":  f"The {s} is lower than the {o}.",
            "attached_to": f"The {s} is attached to the {o}.",
            "hanging_from":f"The {s} is hanging from the {o}.",
        }
        return templates.get(self.relation, f"The {s} {self.relation} the {o}.")

    def __repr__(self) -> str:
        return (f"Object_{self.subject_id} "
                f"--[{self.relation} ({self.confidence:.2f})]--> "
                f"Object_{self.object_id}")


# Keep backward-compatible alias
Relation = RelationEdge


@dataclass
class SceneGraph:
    """Full scene representation with objects and relations."""
    scene_id: str
    objects: List[Object3D] = field(default_factory=list)
    relations: List[RelationEdge] = field(default_factory=list)

    def get_object(self, uid: int) -> Optional[Object3D]:
        return next((o for o in self.objects if o.uid == uid), None)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (f"SceneGraph '{self.scene_id}': "
                f"{len(self.objects)} objects, {len(self.relations)} relations")

    def __repr__(self) -> str:
        return f"<{self.summary()}>"

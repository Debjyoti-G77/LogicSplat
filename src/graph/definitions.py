"""3D Object and Scene Graph definitions."""
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Object3D:
    """Represents a 3D object in the scene."""
    uid: int
    label: str
    centroid: np.ndarray  # [x, y, z]
    bbox: np.ndarray  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    
    def __repr__(self) -> str:
        """Pretty print the object."""
        x, y, z = self.centroid
        return f"<Object {self.uid}: {self.label.capitalize()} at [{x:.2f}, {y:.2f}, {z:.2f}]>"


@dataclass
class SceneGraph:
    """Holds a collection of 3D objects in the scene."""
    objects: List[Object3D]
    
    def __repr__(self) -> str:
        """Pretty print the scene graph."""
        return f"<SceneGraph with {len(self.objects)} objects>"

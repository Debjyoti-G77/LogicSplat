"""Physics-based reasoning rules for 3D scene understanding."""
import numpy as np
from typing import List
from src.graph.definitions import Object3D, SceneGraph


def is_supported(obj_top: Object3D, obj_bottom: Object3D, 
                 vertical_threshold: float = 0.05,
                 horizontal_threshold: float = 0.5) -> bool:
    """
    Check if obj_top is supported by obj_bottom.
    
    Args:
        obj_top: The object on top
        obj_bottom: The potential supporting object below
        vertical_threshold: Maximum vertical gap for contact (default 0.05 units)
        horizontal_threshold: Maximum horizontal distance for overlap
        
    Returns:
        True if obj_top is supported by obj_bottom
    """
    # Check vertical contact: bottom of top object near top of bottom object
    top_min_z = obj_top.bbox[0][2]  # min_z of top object
    bottom_max_z = obj_bottom.bbox[1][2]  # max_z of bottom object
    
    vertical_contact = abs(top_min_z - bottom_max_z) <= vertical_threshold
    
    if not vertical_contact:
        return False
    
    # Check horizontal overlap: centroids close in XY plane
    top_xy = obj_top.centroid[:2]
    bottom_xy = obj_bottom.centroid[:2]
    horizontal_distance = np.linalg.norm(top_xy - bottom_xy)
    
    horizontal_overlap = horizontal_distance <= horizontal_threshold
    
    return horizontal_overlap


def find_unstable_objects(scene_graph: SceneGraph) -> List[Object3D]:
    """
    Find all objects that have no support below them.
    
    Args:
        scene_graph: The scene containing all objects
        
    Returns:
        List of unstable objects (excluding floor objects)
    """
    unstable = []
    
    for obj in scene_graph.objects:
        # Skip floor objects (objects at or near z=0)
        if obj.bbox[0][2] <= 0.1:  # min_z near ground
            continue
        
        # Check if this object is supported by any other object
        has_support = False
        for potential_support in scene_graph.objects:
            if obj.uid == potential_support.uid:
                continue
            
            if is_supported(obj, potential_support):
                has_support = True
                break
        
        if not has_support:
            unstable.append(obj)
    
    return unstable

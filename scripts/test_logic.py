"""Test script to verify the physics reasoning logic."""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.graph.definitions import Object3D, SceneGraph
from src.logic.rules import find_unstable_objects


def main():
    """Create a mock scene and test the stability logic."""
    print("=" * 60)
    print("Logic-Splat Physics Engine Test")
    print("=" * 60)
    
    # Create a table at ground level (z=0)
    table = Object3D(
        uid=1,
        label="table",
        centroid=np.array([0.0, 0.0, 0.5]),
        bbox=np.array([
            [-1.0, -1.0, 0.0],  # min corner
            [1.0, 1.0, 1.0]     # max corner
        ])
    )
    
    # Create a cup sitting on the table (z=1)
    cup = Object3D(
        uid=2,
        label="cup",
        centroid=np.array([0.0, 0.0, 1.1]),
        bbox=np.array([
            [-0.05, -0.05, 1.0],  # min corner (bottom at z=1, touching table top)
            [0.05, 0.05, 1.2]     # max corner
        ])
    )
    
    # Create a floating ball in mid-air (z=5)
    floating_ball = Object3D(
        uid=3,
        label="ball",
        centroid=np.array([0.0, 0.0, 5.0]),
        bbox=np.array([
            [-0.1, -0.1, 4.9],  # min corner
            [0.1, 0.1, 5.1]     # max corner
        ])
    )
    
    # Create the scene
    scene = SceneGraph(objects=[table, cup, floating_ball])
    
    print(f"\nScene created: {scene}")
    print("\nObjects in scene:")
    for obj in scene.objects:
        print(f"  {obj}")
    
    # Find unstable objects
    print("\n" + "-" * 60)
    print("Running stability analysis...")
    print("-" * 60)
    
    unstable = find_unstable_objects(scene)
    
    if unstable:
        print(f"\n⚠️  Found {len(unstable)} unstable object(s):")
        for obj in unstable:
            print(f"  - {obj}")
    else:
        print("\n✓ All objects are stable!")
    
    # Verify expected result
    print("\n" + "=" * 60)
    if len(unstable) == 1 and unstable[0].label == "ball":
        print("✓ TEST PASSED: Floating ball correctly identified as unstable!")
    else:
        print("✗ TEST FAILED: Expected only the floating ball to be unstable.")
    print("=" * 60)


if __name__ == "__main__":
    main()

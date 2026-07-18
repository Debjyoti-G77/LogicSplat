import os
for i in range(6, 14):
    s = f"scene_{i:02d}"
    ply = os.path.exists(f"data/processed/{s}/splat.ply")
    gt = os.path.exists(f"data/processed/{s}/ground_truth_relations.json")
    print(f"{s}: ply={ply}, gt={gt}")

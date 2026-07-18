"""Check edge feature dimensions in cache vs trained model."""
import sys
sys.path.insert(0, ".")
import torch, os

cache_dir = "D:/logicsplat_data/scannet_cache"
files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")][:5]
print("Cache edge dims:")
for f in files:
    g = torch.load(os.path.join(cache_dir, f), weights_only=False)
    print(f"  {f}: edge_attr={g['edge_attr'].shape}, x={g['x'].shape}")

print()
for model_file in os.listdir("models"):
    if not model_file.endswith(".pt"):
        continue
    sd = torch.load(f"models/{model_file}", map_location="cpu", weights_only=True)
    node_dim = edge_dim = "?"
    for k, v in sd.items():
        if "node_encoder.0.weight" in k:
            node_dim = v.shape[1]
        if "edge_classifier.0.weight" in k:
            edge_dim = v.shape[1] - 2 * 128
    print(f"{model_file}: node={node_dim} edge={edge_dim}")

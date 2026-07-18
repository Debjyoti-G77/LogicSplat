"""Check label distribution across all cached ScanNet graphs."""
import sys
sys.path.insert(0, ".")
import torch, os
from collections import Counter
from src.relations.schema import RELATION_NAMES

cache_dir = "D:/logicsplat_data/scannet_cache"
files = [f for f in os.listdir(cache_dir) if f.endswith(".pt")]
print(f"Total cached graphs: {len(files)}")

# Sample one to check edge_attr dims
sample = torch.load(os.path.join(cache_dir, files[0]), weights_only=False)
print(f"Node feat dim: {sample['x'].shape[1]}")
print(f"Edge feat dim: {sample['edge_attr'].shape[1]}")

counts = Counter()
for i, f in enumerate(files):
    g = torch.load(os.path.join(cache_dir, f), weights_only=False)
    counts.update(g["edge_label"].tolist())
    if i % 500 == 0:
        print(f"  {i}/{len(files)}...")

total = sum(counts.values())
print(f"\nLabel distribution ({total} total edges across {len(files)} graphs):")
for idx in sorted(counts):
    pct = 100 * counts[idx] / total
    bar = "#" * int(40 * counts[idx] / max(counts.values()))
    print(f"  {RELATION_NAMES[idx]:20s} {counts[idx]:7d}  {pct:5.1f}%  {bar}")

print(f"\nMost common: {RELATION_NAMES[counts.most_common(1)[0][0]]}")
print(f"Least common: {RELATION_NAMES[counts.most_common()[-1][0]]}")
print(f"Imbalance ratio: {counts.most_common(1)[0][1] / counts.most_common()[-1][1]:.1f}x")

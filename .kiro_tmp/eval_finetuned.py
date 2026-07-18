"""
Evaluate the fine-tuned model using the proper evaluate_scenes.py pipeline.
Temporarily swaps the model path in gaussian_inference, runs eval, restores.
"""
import sys
sys.path.insert(0, ".")

import src.inference.gaussian_inference as gi

# Swap to fine-tuned model
FINETUNED_MODEL = "models/relation_gnn_v7_finetuned_tabletop.pt"
FINETUNED_THRESH = "models/relation_gnn_v7_finetuned_tabletop_thresholds.json"

print(f"Original model: {gi.MODEL_PATH}")
print(f"Swapping to:    {FINETUNED_MODEL}")
print(f"Thresholds:     {FINETUNED_THRESH}")

gi.MODEL_PATH = FINETUNED_MODEL
gi.THRESHOLDS_PATH = FINETUNED_THRESH

# Now run evaluate_scenes in hybrid mode
sys.path.insert(0, "scripts")
from evaluate_scenes import evaluate_scene, aggregate_results, print_results_table

scenes = [f"scene_{i:02d}" for i in range(6, 14)]
results = []
for s in scenes:
    r = evaluate_scene(s, confidence_threshold=0.25, labeler="none", mode="hybrid")
    if r is not None:
        results.append(r)

if results:
    agg = aggregate_results(results)
    print_results_table(results, agg)
else:
    print("No results!")

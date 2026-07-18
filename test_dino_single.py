"""Quick test: run Grounding DINO on a single frame and print detections."""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")

from groundingdino.util.inference import load_model, predict
from PIL import Image
import torchvision.transforms as T
import torch

print("Loading model...")
model = load_model(
    "models/GroundingDINO_SwinT_OGC.py",
    "models/groundingdino_swint_ogc.pth",
    device="cpu",
)
print("Model loaded.")

# Load a frame
img_path = "D:/logicsplat_data/processed/scene_01/ns_data/images/frame_00167.png"
image_pil = Image.open(img_path).convert("RGB")
orig_w, orig_h = image_pil.size
print(f"Image: {orig_w}x{orig_h}")

transform = T.Compose([
    T.Resize((800, 1333)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image_tensor = transform(image_pil)

prompt = "router . box . coke can . comb . perfume bottle . hair dryer"
print(f"Prompt: {prompt}")
print("Running inference...")

boxes, logits, phrases = predict(
    model=model,
    image=image_tensor,
    caption=prompt,
    box_threshold=0.25,
    text_threshold=0.20,
    device="cpu",
)

print(f"\nDetections ({len(phrases)}):")
for box, logit, phrase in zip(boxes, logits, phrases):
    cx, cy, bw, bh = box.tolist()
    x1 = (cx - bw/2) * orig_w
    y1 = (cy - bh/2) * orig_h
    x2 = (cx + bw/2) * orig_w
    y2 = (cy + bh/2) * orig_h
    print(f"  '{phrase}' conf={logit:.2f} bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

if not phrases:
    print("  (no detections above threshold)")

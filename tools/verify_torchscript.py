"""Quick check to ensure the exported TorchScript model is readable."""
from pathlib import Path

import torch

MODEL = Path(__file__).resolve().parents[1] / "app" / "src" / "main" / "assets" / "passt_model.pt"

if not MODEL.exists():
    raise SystemExit(f"TorchScript file not found: {MODEL}")

model = torch.jit.load(str(MODEL))
model.eval()
with torch.no_grad():
    dummy = torch.zeros(1, 32000 * 10)
    logits = model(dummy)
print("logits shape:", tuple(logits.shape))

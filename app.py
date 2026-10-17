"""
Gradio web demo for AlexNet.

Run:
    python app.py

Then open the local URL shown in the terminal.
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from alexnet import AlexNet

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # CIFAR-10 classes for the demo
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# -------------------------------------------------
# Model
# -------------------------------------------------
model = AlexNet(num_classes=NUM_CLASSES)
model.to(DEVICE)
model.eval()

CKPT_PATH = Path("checkpoints/alexnet_cifar10_best.pth")
if CKPT_PATH.exists():
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    STATUS = f"Loaded trained checkpoint: {CKPT_PATH}"
else:
    STATUS = "No trained checkpoint found — using randomly initialized weights (demo only)."

print(STATUS)
print(f"Device: {DEVICE}")

# -------------------------------------------------
# Preprocessing (same as training)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])


def predict(image: Image.Image | None):
    """Run inference and return top-5 predictions."""
    if image is None:
        return {}

    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=min(5, NUM_CLASSES))
    results = {
        CLASS_NAMES[idx]: float(prob)
        for prob, idx in zip(topk.values, topk.indices)
    }
    return results


# -------------------------------------------------
# Gradio Interface
# -------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="AlexNet — Image Classification Demo",
    description=(
        "Classic **AlexNet** (Krizhevsky et al., 2012) implemented in PyTorch.\n\n"
        f"**Status:** {STATUS}\n\n"
        "This demo uses the CIFAR-10 class set. "
        "Upload any image to see the model’s top-5 predictions."
    ),
    article=(
        "Built by [Stanley Samwel](https://github.com/Stanley03-arch) • "
        "[Source code](https://github.com/Stanley03-arch/alexnet-pytorch)"
    ),
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()

"""
Gradio demo for AlexNet.

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
# Model setup
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # CIFAR-10 by default for the demo
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = AlexNet(num_classes=NUM_CLASSES)
model.to(DEVICE)
model.eval()

# Optional: load a trained checkpoint if it exists
CKPT_PATH = Path("checkpoints/alexnet_cifar10_best.pth")
if CKPT_PATH.exists():
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {CKPT_PATH}")
else:
    print("No checkpoint found — using randomly initialized weights (for demo only).")


# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def predict(image: Image.Image):
    if image is None:
        return None

    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    # Return top-5 predictions as a dictionary (Gradio Label)
    top5 = torch.topk(probs, k=min(5, NUM_CLASSES))
    result = {CLASS_NAMES[idx]: float(prob) for prob, idx in zip(top5.values, top5.indices)}
    return result


# -------------------------------------------------
# Gradio interface
# -------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="AlexNet Demo",
    description=(
        "Classic AlexNet (Krizhevsky et al., 2012) implemented in PyTorch.\n\n"
        "This demo uses the CIFAR-10 class set. "
        "Upload any image to see the model's top predictions."
    ),
    examples=None,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()

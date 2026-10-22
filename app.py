"""
Powerful Gradio web demo for AlexNet.

Features:
- Image classification (top-5)
- Grad-CAM visual explanation
- Model info & architecture summary
- LRN toggle awareness

Run:
    python app.py
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from alexnet import AlexNet, count_parameters, count_flops, summary
from alexnet.gradcam import GradCAM, overlay_cam_on_image

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  # CIFAR-10
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
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        STATUS = (
            f"✅ Loaded trained checkpoint (epoch {state.get('epoch', '?')}, "
            f"best acc {state.get('best_acc', 0):.2f}%)"
        )
    else:
        model.load_state_dict(state)
        STATUS = f"✅ Loaded trained checkpoint: {CKPT_PATH}"
else:
    STATUS = "⚠️ No trained checkpoint found — using randomly initialized weights (demo only)."

print(STATUS)
print(f"Device: {DEVICE}")

# Grad-CAM
cam_extractor = GradCAM(model)

# -------------------------------------------------
# Preprocessing
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
    """Classification + Grad-CAM."""
    if image is None:
        return {}, None, "Please upload an image."

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

    # Grad-CAM for the top prediction
    top_class = topk.indices[0].item()
    heatmap = cam_extractor(x, target_class=top_class)
    cam_image = overlay_cam_on_image(img.resize((224, 224)), heatmap)

    explanation = (
        f"**Top prediction:** {CLASS_NAMES[top_class]} "
        f"({results[CLASS_NAMES[top_class]]*100:.1f}%)\n\n"
        f"The heatmap highlights regions the model focused on for this decision."
    )

    return results, cam_image, explanation


def get_model_info():
    """Return model statistics."""
    params = count_parameters(model)
    try:
        flops = count_flops(model) / 1e9
        flops_str = f"{flops:.2f} GFLOPs"
    except Exception:
        flops_str = "N/A"

    info = f"""
### Model Information

- **Architecture**: AlexNet (paper-faithful)
- **Channels**: 96 → 256 → 384 → 384 → 256
- **Parameters**: {params:,}
- **FLOPs (224×224)**: {flops_str}
- **Device**: {DEVICE}
- **Status**: {STATUS}
- **Classes**: CIFAR-10 ({NUM_CLASSES} classes)

### Why this implementation is special

Most public AlexNets (including torchvision) use a modified 64→192 channel layout.  
This one stays true to the **2012 paper** and supports optional **Local Response Normalization**.
"""
    return info


# -------------------------------------------------
# Gradio Blocks UI (much more powerful)
# -------------------------------------------------
with gr.Blocks(
    title="AlexNet — Powerful Demo",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 1100px !important}
    """
) as demo:
    gr.Markdown(
        f"""
        # 🧠 AlexNet — Paper-Faithful & Powerful Demo
        Classic **AlexNet** (Krizhevsky et al., 2012) with modern extras:  
        Grad-CAM explanations • Transfer-learning ready • FLOPs counting

        **Status:** {STATUS}
        """
    )

    with gr.Tabs():
        with gr.Tab("🔍 Classify + Explain"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload an image", height=320)
                    run_btn = gr.Button("Run Prediction + Grad-CAM", variant="primary")
                with gr.Column(scale=1):
                    output_label = gr.Label(num_top_classes=5, label="Top Predictions")
                    cam_output = gr.Image(type="pil", label="Grad-CAM Explanation", height=320)
            explanation_md = gr.Markdown()
            run_btn.click(
                fn=predict,
                inputs=input_image,
                outputs=[output_label, cam_output, explanation_md],
            )

        with gr.Tab("📊 Model Info"):
            info_md = gr.Markdown(get_model_info())
            gr.Markdown(
                """
                ### Quick tips
                - Train your own weights: `python train_cifar10.py --epochs 20`
                - Use paper-style LRN: `AlexNet(use_lrn=True)`
                - Extract features for transfer learning: `model.extract_features(x)`
                """
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ### About this project

                This is a **clean, modern, paper-faithful** re-implementation of AlexNet.

                - Original paper channels (96 → 256 → 384)
                - Optional Local Response Normalization
                - Full transfer-learning helpers
                - FLOPs / parameter counting
                - TorchScript & ONNX export
                - Grad-CAM visual explanations (this demo)

                **Repository**: [Stanley03-arch/alexnet-pytorch](https://github.com/Stanley03-arch/alexnet-pytorch)

                Built with ❤️ by [Stanley Samwel](https://github.com/Stanley03-arch)
                """
            )

if __name__ == "__main__":
    demo.launch()

# Training & Visualization Walkthrough

This short guide shows how to train AlexNet on CIFAR-10 and plot the results.

## 1. Install dependencies

```bash
pip install -e ".[train,demo]"
# or
pip install -r requirements.txt
```

## 2. Train the model

```bash
python train_cifar10.py --epochs 15 --batch-size 128 --lr 0.01
```

The best checkpoint will be saved to:

```
checkpoints/alexnet_cifar10_best.pth
```

Training history is saved to:

```
checkpoints/training_history.pt
```

## 3. Visualize the curves

```bash
python visualize_history.py
```

This generates a plot at `checkpoints/training_curves.png` showing:

- Training & Test Loss
- Training & Test Accuracy

## 4. Run inference on an image

```bash
python inference.py \
  --image path/to/your/image.jpg \
  --checkpoint checkpoints/alexnet_cifar10_best.pth \
  --num-classes 10
```

## 5. Launch the Gradio demo

```bash
python app.py
```

Then open the local URL shown in the terminal.

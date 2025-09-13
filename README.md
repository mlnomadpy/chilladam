# ChillAdam

A modular deep learning library featuring a custom ChillAdam optimizer and ResNet implementations from scratch.

## Features

- **ChillAdam Optimizer**: Custom optimizer with adaptive learning rates based on parameter norms
- **ResNet from Scratch**: Full implementations of ResNet-18 and ResNet-50 architectures
- **Modular Design**: Clean, production-ready code structure
- **Tiny ImageNet Support**: Optimized for Tiny ImageNet dataset training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mlnomadpy/chilladam.git
cd chilladam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train with ResNet-18 (default):
```bash
python main.py
```

Train with ResNet-50:
```bash
python main.py --model resnet50
```

### Advanced Options

```bash
python main.py --model resnet18 \
               --epochs 20 \
               --batch-size 64 \
               --min-lr 1e-5 \
               --max-lr 1.0 \
               --weight-decay 1e-4
```

### Command Line Arguments

- `--model`: Choose between `resnet18` or `resnet50` (default: resnet18)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--device`: Training device, `cuda` or `cpu` (auto-detected)
- `--min-lr`: Minimum learning rate for ChillAdam (default: 1e-5)
- `--max-lr`: Maximum learning rate for ChillAdam (default: 1.0)
- `--weight-decay`: Weight decay for regularization (default: 0)
- `--image-size`: Input image size (default: 64 for Tiny ImageNet)

## Architecture

```
chilladam/
├── chilladam/
│   ├── optimizers/          # ChillAdam optimizer implementation
│   ├── models/              # ResNet architectures from scratch
│   ├── data/                # Data loading utilities
│   ├── training/            # Training and validation logic
│   └── config.py            # Configuration management
├── main.py                  # Main training script
├── requirements.txt         # Dependencies
└── README.md
```

## ChillAdam Optimizer

The ChillAdam optimizer adapts learning rates based on parameter norms, providing:
- Automatic learning rate scaling
- Gradient normalization
- Momentum-based updates
- Configurable learning rate bounds

## ResNet Implementation

Full implementations from scratch:
- **ResNet-18**: 18-layer network with BasicBlock
- **ResNet-50**: 50-layer network with Bottleneck blocks
- Adaptive architecture for different input sizes
- Optimized for Tiny ImageNet (64x64) and ImageNet (224x224)

## Example Output

```
==================================================
CONFIGURATION
==================================================
Model: resnet18
Device: cuda
Epochs: 10
Batch Size: 64
Image Size: 64
Number of Classes: 200
ChillAdam Min LR: 1e-05
ChillAdam Max LR: 1.0
Weight Decay: 0
==================================================

Loading Tiny ImageNet dataset...
Creating ResNet-18 model from scratch...
Setting up ChillAdam optimizer...
Initializing trainer with device: cuda

Starting training for 10 epochs...
Epoch 1/10
Training: 100%|██████████| 1563/1563 [02:30<00:00, 10.38it/s]
Validation: 100%|██████████| 157/157 [00:15<00:00, 10.12it/s]
Training Loss: 4.2451
Validation Loss: 3.8923
Validation Accuracy: 12.45%
...
```

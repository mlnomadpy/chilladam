# ChillAdam

A modular deep learning library featuring a custom ChillAdam optimizer and ResNet implementations from scratch with efficient dataset streaming and support for multiple PyTorch optimizers.

## Features

- **Multiple Optimizer Support**: ChillAdam custom optimizer plus 7 PyTorch built-in optimizers (Adam, AdamW, SGD, RMSprop, Adamax, NAdam, RAdam)
- **ChillAdam Optimizer**: Custom optimizer with adaptive learning rates based on parameter norms
- **ResNet from Scratch**: Full implementations of ResNet-18 and ResNet-50 architectures
- **Modular Design**: Clean, production-ready code structure
- **Streaming Dataset Support**: Efficient streaming of multiple datasets from Hugging Face without local downloads
- **Multiple Dataset Support**: Support for Tiny ImageNet, ImageNet-1k, Food-101, and STL-10 from Hugging Face

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

Train with ResNet-18 and ChillAdam optimizer (default):
```bash
python main.py
```

Train with different optimizers:
```bash
# Adam optimizer
python main.py --optimizer adam --lr 0.001

# SGD with momentum
python main.py --optimizer sgd --lr 0.01 --momentum 0.9

# AdamW with weight decay
python main.py --optimizer adamw --lr 0.002 --weight-decay 0.01
```

Train with ResNet-50 on ImageNet-1k:
```bash
python main.py --model resnet50 --dataset imagenet-1k --optimizer adam --lr 0.001
```

### Dataset Options

Train on different datasets with various optimizers:
```bash
# Tiny ImageNet (200 classes, 64x64 images) with ChillAdam
python main.py --dataset tiny-imagenet --optimizer chilladam

# ImageNet-1k (1000 classes, 224x224 images) with Adam
python main.py --dataset imagenet-1k --model resnet50 --optimizer adam --lr 0.001

# Food-101 (101 classes, 224x224 images) with SGD
python main.py --dataset food101 --batch-size 32 --optimizer sgd --lr 0.01 --momentum 0.9

# STL-10 (10 classes, 96x96 images) with AdamW
python main.py --dataset stl10 --epochs 50 --optimizer adamw --lr 0.002 --weight-decay 0.01
```

### Advanced Options

```bash
python main.py --model resnet18 \
               --dataset food101 \
               --optimizer adamw \
               --lr 0.002 \
               --weight-decay 0.01 \
               --epochs 20 \
               --batch-size 64 \
               --min-lr 1e-5 \
               --max-lr 1.0 \
               --weight-decay 1e-4 \
               --image-size 256 \
               --shuffle-buffer-size 20000
```

### Command Line Arguments

#### Model and Training Arguments
- `--model`: Choose between `resnet18` or `resnet50` (default: resnet18)
- `--dataset`: Choose dataset: `tiny-imagenet`, `imagenet-1k`, `food101`, `stl10` (default: tiny-imagenet)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--device`: Training device, `cuda` or `cpu` (auto-detected)
- `--image-size`: Input image size (auto-detected based on dataset if not specified)

#### Optimizer Arguments
- `--optimizer`: Choose optimizer: `chilladam`, `adam`, `adamw`, `sgd`, `rmsprop`, `adamax`, `nadam`, `radam` (default: chilladam)

#### Standard Optimizer Parameters (Adam, AdamW, SGD, RMSprop, etc.)
- `--lr`: Learning rate for standard optimizers (default: 1e-3)
- `--momentum`: Momentum for SGD and RMSprop (default: 0.9)
- `--alpha`: Alpha parameter for RMSprop (default: 0.99)
- `--weight-decay`: Weight decay for regularization (default: 0)
- `--image-size`: Input image size (auto-detected based on dataset if not specified)
- `--shuffle-buffer-size`: Buffer size for shuffling streaming datasets (default: 10000)

#### ChillAdam Specific Parameters (only used when `--optimizer chilladam`)
- `--min-lr`: Minimum learning rate for ChillAdam (default: 1e-5)
- `--max-lr`: Maximum learning rate for ChillAdam (default: 1.0)

## Supported Optimizers

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| **ChillAdam** | Custom adaptive optimizer with parameter norm-based learning rates | `min_lr`, `max_lr`, `eps`, `betas`, `weight_decay` |
| **Adam** | Adaptive moment estimation | `lr`, `betas`, `eps`, `weight_decay` |
| **AdamW** | Adam with decoupled weight decay | `lr`, `betas`, `eps`, `weight_decay` |
| **SGD** | Stochastic Gradient Descent | `lr`, `momentum`, `weight_decay` |
| **RMSprop** | Root Mean Square Propagation | `lr`, `alpha`, `eps`, `weight_decay`, `momentum` |
| **Adamax** | Adam with infinity norm | `lr`, `betas`, `eps`, `weight_decay` |
| **NAdam** | Adam with Nesterov momentum | `lr`, `betas`, `eps`, `weight_decay` |
| **RAdam** | Rectified Adam | `lr`, `betas`, `eps`, `weight_decay` |
=======
## Supported Datasets

| Dataset | Classes | Default Image Size | Hugging Face ID |
|---------|---------|-------------------|-----------------|
| Tiny ImageNet | 200 | 64x64 | zh-plus/tiny-imagenet |
| ImageNet-1k | 1000 | 224x224 | imagenet-1k |
| Food-101 | 101 | 224x224 | food101 |
| STL-10 | 10 | 96x96 | stl10 |

## Architecture

```
chilladam/
├── chilladam/
│   ├── optimizers/          # ChillAdam and PyTorch optimizer support
│   ├── models/              # ResNet architectures from scratch
│   ├── data/                # Data loading utilities with multi-dataset support
│   ├── training/            # Training and validation logic
│   └── config.py            # Configuration management
├── main.py                  # Main training script
├── requirements.txt         # Dependencies
└── README.md
```

## Optimizers

### ChillAdam Optimizer

The ChillAdam optimizer adapts learning rates based on parameter norms, providing:
- Automatic learning rate scaling
- Gradient normalization
- Momentum-based updates
- Configurable learning rate bounds

### PyTorch Built-in Optimizers

All standard PyTorch optimizers are supported with their native parameters:
- **Adam/AdamW**: Adaptive moment estimation with optional weight decay
- **SGD**: Stochastic Gradient Descent with optional momentum
- **RMSprop**: Root Mean Square Propagation
- **Adamax, NAdam, RAdam**: Advanced Adam variants

## ResNet Implementation

Full implementations from scratch:
- **ResNet-18**: 18-layer network with BasicBlock
- **ResNet-50**: 50-layer network with Bottleneck blocks
- Adaptive architecture for different input sizes
- Optimized for various image sizes (64x64 to 224x224)

## Example Output

```
==================================================
CONFIGURATION
==================================================
Dataset: food101
Model: resnet18
Device: cuda
Epochs: 10
Batch Size: 64
Image Size: 224
Number of Classes: 101
ChillAdam Min LR: 1e-05
ChillAdam Max LR: 1.0
Weight Decay: 0
==================================================

Loading food101 dataset...
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

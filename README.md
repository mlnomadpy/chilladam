# ChillAdam

A modular deep learning library featuring a custom ChillAdam optimizer and ResNet implementations from scratch with efficient dataset streaming and support for multiple PyTorch optimizers.

## Features

- **Multiple Optimizer Support**: ChillAdam custom optimizer plus 7 PyTorch built-in optimizers (Adam, AdamW, SGD, RMSprop, Adamax, NAdam, RAdam)
- **ChillAdam Optimizer**: Custom optimizer with adaptive learning rates based on parameter norms
- **ResNet from Scratch**: Full implementations of ResNet-18 and ResNet-50 architectures
- **SE-ResNet Models**: ResNet with Squeeze-and-Excitation blocks for improved feature representation
- **YAT-ResNet Models**: Yet Another Transformation-based ResNet variants with configurable SE blocks or LayerNorm normalization
- **Advanced YAT Features**: YAT models support alpha scaling, DropConnect regularization, and configurable dropout rates
- **NMN Integration**: Utilizes the Neural Masked Networks (nmn) package for YAT transformations and advanced convolutions
- **Modular Design**: Clean, production-ready code structure
- **Weights & Biases Integration**: Full support for experiment tracking, model monitoring, and gradient/parameter logging
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

### Key Dependencies
- **PyTorch >= 1.9.0**: Deep learning framework
- **torchvision >= 0.10.0**: Computer vision utilities
- **datasets >= 2.0.0**: Hugging Face datasets for streaming
- **wandb >= 0.15.0**: Weights & Biases for experiment tracking
- **nmn**: Neural Masked Networks package for YAT transformations
- **tqdm, pillow**: Progress bars and image processing

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

# ChillSGD optimizer (SGD with gradient normalization and adaptive learning rate)
python main.py --optimizer chillsgd --min-lr 1e-5 --max-lr 0.1

# SGD with momentum
python main.py --optimizer sgd --lr 0.01 --momentum 0.9

# AdamW with weight decay
python main.py --optimizer adamw --lr 0.002 --weight-decay 0.01
```

Train with ResNet-50 on ImageNet-1k:
```bash
python main.py --model resnet50 --dataset imagenet-1k --optimizer adam --lr 0.001
```

### Advanced Model Options

Train with different ResNet architectures:
```bash
# SE-ResNet with Squeeze-and-Excitation blocks
python main.py --model standard_se_resnet18 --dataset tiny-imagenet

# YAT-ResNet with SE layers
python main.py --model yat_resnet18 --dataset tiny-imagenet

# YAT-ResNet without SE layers (with LayerNorm after skip connections)
python main.py --model yat_resnet18_no_se --dataset tiny-imagenet

# Larger YAT-ResNet variant
python main.py --model yat_resnet34_no_se --dataset food101 --optimizer adam --lr 0.001
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

### Weights & Biases Integration

ChillAdam provides comprehensive Weights & Biases support for experiment tracking and model monitoring:

```bash
# Enable basic wandb logging
python main.py --use-wandb --wandb-project my-chilladam-experiments

# Enable wandb.watch() for gradient and parameter tracking
python main.py --use-wandb --wandb-watch --wandb-watch-log-freq 50

# Full wandb integration with custom run name
python main.py --model yat_resnet18 \
               --dataset food101 \
               --optimizer chilladam \
               --use-wandb \
               --wandb-project food-classification \
               --wandb-run-name yat-resnet18-chilladam \
               --wandb-watch \
               --wandb-watch-log-freq 100
```

#### Wandb Features:
- **Experiment Tracking**: Automatically logs training/validation loss, accuracy, and hyperparameters
- **Model Monitoring**: Track gradients, weights, and model topology with `wandb.watch()`
- **Custom Projects**: Organize experiments with custom project names
- **Run Names**: Set custom run names for easy identification
- **Configurable Logging**: Adjust gradient/parameter logging frequency

### Command Line Arguments

#### Model and Training Arguments
- `--model`: Choose model architecture:
  - **Basic ResNet**: `resnet18`, `resnet50`
  - **SE-ResNet**: `standard_se_resnet18`, `standard_se_resnet34`
  - **YAT-ResNet (with SE)**: `yat_resnet18`, `yat_resnet34`
  - **YAT-ResNet (no SE, with LayerNorm)**: `yat_resnet18_no_se`, `yat_resnet34_no_se`
  
  Default: `resnet18`
- `--dataset`: Choose dataset: `tiny-imagenet`, `imagenet-1k`, `food101`, `stl10` (default: tiny-imagenet)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--device`: Training device, `cuda` or `cpu` (auto-detected)
- `--image-size`: Input image size (auto-detected based on dataset if not specified)

#### Optimizer Arguments
- `--optimizer`: Choose optimizer: `chilladam`, `chillsgd`, `adam`, `adamw`, `sgd`, `rmsprop`, `adamax`, `nadam`, `radam` (default: chilladam)

#### Standard Optimizer Parameters (Adam, AdamW, SGD, RMSprop, etc.)
- `--lr`: Learning rate for standard optimizers (default: 1e-3)
- `--momentum`: Momentum for SGD and RMSprop (default: 0.9)
- `--alpha`: Alpha parameter for RMSprop (default: 0.99)
- `--weight-decay`: Weight decay for regularization (default: 0)
- `--shuffle-buffer-size`: Buffer size for shuffling streaming datasets (default: 10000)

#### ChillAdam & ChillSGD Specific Parameters (used when `--optimizer chilladam` or `--optimizer chillsgd`)
- `--min-lr`: Minimum learning rate for ChillAdam and ChillSGD (default: 1e-5)
- `--max-lr`: Maximum learning rate for ChillAdam and ChillSGD (default: 1.0)

#### Weights & Biases Arguments
- `--use-wandb`: Enable Weights & Biases logging (default: disabled)
- `--wandb-project`: Wandb project name (default: "chilladam-training")
- `--wandb-run-name`: Custom run name for easy identification (auto-generated if not specified)
- `--wandb-watch`: Enable wandb.watch() to log model gradients and parameters (default: disabled)
- `--wandb-watch-log-freq`: Log frequency for wandb.watch() in steps (default: 100)

## Supported Optimizers

| Optimizer | Description | Key Parameters |
|-----------|-------------|----------------|
| **ChillAdam** | Custom adaptive optimizer with parameter norm-based learning rates | `min_lr`, `max_lr`, `eps`, `betas`, `weight_decay` |
| **ChillSGD** | Custom SGD with gradient normalization and adaptive learning rates (no momentum) | `min_lr`, `max_lr`, `eps`, `weight_decay` |
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

### ChillSGD Optimizer

The ChillSGD optimizer combines SGD with the "chill" mechanism from ChillAdam:
- **Gradient normalization**: Divides gradients by their L2 norm for stable training
- **Adaptive learning rate**: Uses inverse of parameter norm instead of fixed learning rate
- **No momentum**: Pure SGD without momentum for simplicity
- **Configurable bounds**: Min/max learning rate constraints for stability

### PyTorch Built-in Optimizers

All standard PyTorch optimizers are supported with their native parameters:
- **Adam/AdamW**: Adaptive moment estimation with optional weight decay
- **SGD**: Stochastic Gradient Descent with optional momentum
- **RMSprop**: Root Mean Square Propagation
- **Adamax, NAdam, RAdam**: Advanced Adam variants

## Model Architectures

### ResNet Implementation

Full implementations from scratch with multiple variants:

#### Basic ResNet Models
- **ResNet-18**: 18-layer network with BasicBlock (2 layers per block)
- **ResNet-50**: 50-layer network with Bottleneck blocks (3 layers per block)
- Adaptive architecture for different input sizes
- Optimized for various image sizes (64x64 to 224x224)

#### SE-ResNet Models (Squeeze-and-Excitation)
- **SE-ResNet-18**: ResNet-18 with SE blocks for channel-wise attention
- **SE-ResNet-34**: ResNet-34 with SE blocks for improved feature representation
- SE blocks adaptively recalibrate channel-wise feature responses
- Reduction ratio of 16 for efficient computation

#### YAT-ResNet Models (Yet Another Transformation)
ChillAdam includes advanced YAT-ResNet models powered by the Neural Masked Networks (nmn) package:

**YAT-ResNet with SE blocks:**
- **yat_resnet18**: YAT-based ResNet-18 with SE attention mechanism
- **yat_resnet34**: YAT-based ResNet-34 with SE attention mechanism

**YAT-ResNet without SE (LayerNorm variant):**
- **yat_resnet18_no_se**: YAT-based ResNet-18 with LayerNorm after skip connections
- **yat_resnet34_no_se**: YAT-based ResNet-34 with LayerNorm after skip connections

#### YAT Model Features
- **Alpha Scaling**: Configurable with `use_alpha` parameter for adaptive feature scaling
- **DropConnect**: Optional regularization with `use_dropconnect` and configurable `drop_rate`
- **YatConv2d**: Advanced convolution layers from the nmn package
- **YatNMN**: Neural masked network transformations for enhanced feature learning

### Model Comparison

| Model Family | Parameters (ImageNet) | Special Features |
|-------------|----------------------|------------------|
| ResNet-18 | ~11.2M | Basic residual connections |
| ResNet-50 | ~23.5M | Deeper network with bottlenecks |
| SE-ResNet-18 | ~11.3M | Channel attention with SE blocks |
| YAT-ResNet-18 | ~11.3M | Advanced transformations + SE |
| YAT-ResNet-18 (no SE) | ~11.2M | YAT transformations + LayerNorm |

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

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torchvision import models
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

class ChillAdam(Optimizer):
    def __init__(self, params, min_lr=1e-5, max_lr=1.0, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if not 0.0 <= max_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(min_lr=min_lr, max_lr=max_lr, eps=eps, betas=betas, weight_decay=weight_decay)
        super(ChillAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            min_lr, max_lr, eps, betas, weight_decay = group['min_lr'], group['max_lr'], group['eps'], group['betas'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamChill does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                grad_norm = grad.norm(p=2).clamp(min=eps)
                grad_normalized = grad / grad_norm

                if weight_decay != 0:
                    grad_normalized = grad_normalized.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_normalized, grad_normalized, value=1 - beta2)

                param_norm = p.norm(p=2).clamp(min=eps)
                lr = 1.0 / param_norm
                lr = lr.clamp(min=min_lr, max=max_lr)
                self.state[p]["lr"] = lr.item()

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
# --- Configuration ---
NUM_CLASSES = 200
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Loading and Preprocessing ---
print("Loading Tiny ImageNet dataset...")
try:
    dataset = load_dataset("zh-plus/tiny-imagenet")
    train_dataset = dataset["train"]
    val_dataset = dataset["valid"]
except Exception as e:
    print(f"Error loading dataset: {e}. Trying an alternative...")
    try:
        dataset = load_dataset("Maysee/tiny-imagenet")
        train_dataset = dataset["train"]
        val_dataset = dataset["valid"]
    except Exception as e:
        print(f"Failed to load dataset from both sources. Please ensure you have internet access and the Hugging Face `datasets` library is installed correctly.")
        exit()

# Define image transformations for training and validation
image_size = 64
train_transforms = Compose([
    Resize((image_size, image_size)),
    RandomCrop(image_size, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use set_transform to apply the transformations and return tensors
# This is a cleaner way to handle the transformation and prevents the TypeError.
train_dataset.set_transform(lambda examples: {
    'pixel_values': [train_transforms(image.convert("RGB")) for image in examples['image']],
    'label': examples['label']
})
val_dataset.set_transform(lambda examples: {
    'pixel_values': [val_transforms(image.convert("RGB")) for image in examples['image']],
    'label': examples['label']
})

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- The rest of the script (model, optimizer, training loop) remains the same ---
# --- Model Definition ---
print("Initializing ResNet model...")
model = models.resnet18(weights=None) # We will train from scratch

# Adapt the first conv layer for 64x64 images
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()

# Replace the final fully connected layer for Tiny ImageNet
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.to(DEVICE)

# --- Training Setup ---
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamChill(model.parameters(), min_lr=1e-5, max_lr=1.0)

# --- Training and Validation Loop ---
# --- Training and Validation Loop ---
print("Starting training loop...")
for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Training)"):
        # This is the corrected section.
        if isinstance(batch["pixel_values"], list):
            inputs = torch.stack(batch["pixel_values"]).to(DEVICE)
        else:
            inputs = batch["pixel_values"].to(DEVICE)

        labels = torch.tensor(batch["label"]).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    avg_train_loss = train_loss / len(train_dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)"):
            # This is the corrected section.
            if isinstance(batch["pixel_values"], list):
                inputs = torch.stack(batch["pixel_values"]).to(DEVICE)
            else:
                inputs = batch["pixel_values"].to(DEVICE)

            labels = torch.tensor(batch["label"]).to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    accuracy = 100 * correct_predictions / total_predictions

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Training Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"  Validation Accuracy: {accuracy:.2f}%")

print("Training finished.")

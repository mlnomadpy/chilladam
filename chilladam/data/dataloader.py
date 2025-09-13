"""
Data loading utilities for Tiny ImageNet dataset.
"""

from torch.utils.data import DataLoader
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Resize,
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
)
from datasets import load_dataset


def get_tiny_imagenet_loaders(batch_size=64, image_size=64):
    """
    Get DataLoaders for Tiny ImageNet dataset.
    
    Arguments:
        batch_size: batch size for training and validation
        image_size: size to resize images to
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    print("Loading Tiny ImageNet dataset...")
    
    # Try to load the dataset from different sources
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
            raise e

    # Define image transformations for training and validation
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

    # Apply transformations to the datasets
    train_dataset.set_transform(lambda examples: {
        'pixel_values': [train_transforms(image.convert("RGB")) for image in examples['image']],
        'label': examples['label']
    })
    val_dataset.set_transform(lambda examples: {
        'pixel_values': [val_transforms(image.convert("RGB")) for image in examples['image']],
        'label': examples['label']
    })

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
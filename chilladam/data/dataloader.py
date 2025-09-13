"""
Data loading utilities for multiple datasets from Hugging Face.
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


# Dataset registry with Hugging Face dataset names and configurations
DATASET_REGISTRY = {
    "tiny-imagenet": {
        "hf_names": ["zh-plus/tiny-imagenet", "Maysee/tiny-imagenet"],
        "num_classes": 200,
        "default_image_size": 64,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "splits": {"train": "train", "val": "valid"}
    },
    "imagenet-1k": {
        "hf_names": ["imagenet-1k", "ILSVRC/imagenet-1k"],
        "num_classes": 1000,
        "default_image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "splits": {"train": "train", "val": "validation"}
    },
    "food101": {
        "hf_names": ["food101"],
        "num_classes": 101,
        "default_image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "splits": {"train": "train", "val": "validation"}
    },
    "stl10": {
        "hf_names": ["stl10"],
        "num_classes": 10,
        "default_image_size": 96,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "splits": {"train": "train", "val": "test"}
    }
}


def load_dataset_with_fallbacks(dataset_name):
    """
    Load dataset from Hugging Face with multiple fallback options.
    
    Arguments:
        dataset_name: Name of dataset in DATASET_REGISTRY
        
    Returns:
        Loaded dataset object
        
    Raises:
        Exception if none of the dataset sources work
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_REGISTRY.keys())}")
    
    dataset_config = DATASET_REGISTRY[dataset_name]
    hf_names = dataset_config["hf_names"]
    
    last_error = None
    for hf_name in hf_names:
        try:
            print(f"Trying to load dataset from '{hf_name}'...")
            dataset = load_dataset(hf_name)
            print(f"Successfully loaded dataset from '{hf_name}'")
            return dataset
        except Exception as e:
            print(f"Failed to load from '{hf_name}': {e}")
            last_error = e
            continue
    
    # If we get here, all sources failed
    raise Exception(f"Failed to load dataset '{dataset_name}' from any source. Last error: {last_error}")


def get_data_loaders(dataset_name="tiny-imagenet", batch_size=64, image_size=None):
    """
    Get DataLoaders for specified dataset.
    
    Arguments:
        dataset_name: Name of dataset to load
        batch_size: batch size for training and validation
        image_size: size to resize images to (uses dataset default if None)
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_REGISTRY.keys())}")
    
    dataset_config = DATASET_REGISTRY[dataset_name]
    
    # Use provided image_size or dataset default
    if image_size is None:
        image_size = dataset_config["default_image_size"]
    
    # Load the dataset
    dataset = load_dataset_with_fallbacks(dataset_name)
    
    # Get the appropriate splits
    splits = dataset_config["splits"]
    train_dataset = dataset[splits["train"]]
    val_dataset = dataset[splits["val"]]
    
    # Get normalization parameters
    mean = dataset_config["mean"]
    std = dataset_config["std"]
    
    # Define image transformations for training and validation
    train_transforms = Compose([
        Resize((image_size, image_size)),
        RandomCrop(image_size, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])

    val_transforms = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=mean, std=std),
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


# Keep backward compatibility
def get_tiny_imagenet_loaders(batch_size=64, image_size=64):
    """
    Get DataLoaders for Tiny ImageNet dataset.
    This function is kept for backward compatibility.
    
    Arguments:
        batch_size: batch size for training and validation
        image_size: size to resize images to
        
    Returns:
        tuple: (train_dataloader, val_dataloader)
    """
    return get_data_loaders("tiny-imagenet", batch_size, image_size)
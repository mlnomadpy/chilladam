"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Trainer class for model training and validation.
    
    Arguments:
        model: PyTorch model to train
        optimizer: optimizer for training
        device: device to train on ('cuda' or 'cpu')
        criterion: loss function (default: CrossEntropyLoss)
    """
    
    def __init__(self, model, optimizer, device='cuda', criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Move model to device
        self.model.to(self.device)
    
    def train_epoch(self, train_dataloader):
        """
        Train the model for one epoch.
        
        Arguments:
            train_dataloader: DataLoader for training data
            
        Returns:
            float: average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            # Handle different batch formats
            if isinstance(batch["pixel_values"], list):
                inputs = torch.stack(batch["pixel_values"]).to(self.device)
            else:
                inputs = batch["pixel_values"].to(self.device)
            
            labels = torch.tensor(batch["label"]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
        
        return total_loss / num_samples
    
    def validate_epoch(self, val_dataloader):
        """
        Validate the model for one epoch.
        
        Arguments:
            val_dataloader: DataLoader for validation data
            
        Returns:
            tuple: (average validation loss, accuracy percentage)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Handle different batch formats
                if isinstance(batch["pixel_values"], list):
                    inputs = torch.stack(batch["pixel_values"]).to(self.device)
                else:
                    inputs = batch["pixel_values"].to(self.device)
                
                labels = torch.tensor(batch["label"]).to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_dataloader.dataset)
        accuracy = 100.0 * correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_dataloader, val_dataloader, num_epochs):
        """
        Train the model for multiple epochs.
        
        Arguments:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            num_epochs: number of epochs to train
        """
        print("Starting training...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation phase
            val_loss, accuracy = self.validate_epoch(val_dataloader)
            
            # Print results
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {accuracy:.2f}%")
        
        print("\nTraining finished!")
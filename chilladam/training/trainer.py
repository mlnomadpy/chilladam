"""
Training utilities and trainer class.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

# Import L1RegularizedLoss - try relative import first, fall back to absolute
try:
    from ..loss import L1RegularizedLoss
except ImportError:
    from chilladam.loss import L1RegularizedLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """
    Trainer class for model training and validation.
    
    Arguments:
        model: PyTorch model to train
        optimizer: optimizer for training
        device: device to train on ('cuda' or 'cpu')
        criterion: loss function (default: CrossEntropyLoss)
        l1_lambda: L1 regularization strength (default: 0, no regularization)
        use_wandb: whether to use Weights & Biases logging
        wandb_config: configuration dict for wandb
        wandb_watch: whether to enable wandb.watch() for model logging
        wandb_watch_log_freq: log frequency for wandb.watch()
    """
    
    def __init__(self, model, optimizer, device='cuda', criterion=None, l1_lambda=0, use_wandb=False, wandb_config=None, wandb_watch=False, wandb_watch_log_freq=100):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.l1_lambda = l1_lambda
        
        # Set up the loss function with L1 regularization if needed
        base_criterion = criterion or nn.CrossEntropyLoss()
        if l1_lambda > 0:
            self.criterion = L1RegularizedLoss(base_criterion, l1_lambda=l1_lambda)
            self.use_l1_regularization = True
        else:
            self.criterion = base_criterion
            self.use_l1_regularization = False
            
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_watch = wandb_watch
        self.wandb_watch_log_freq = wandb_watch_log_freq
        self.step_count = 0
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize wandb if requested
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available, disabling wandb logging")
                self.use_wandb = False
                self.wandb_watch = False
            else:
                project = wandb_config.get('project', 'chilladam-training') if wandb_config else 'chilladam-training'
                name = wandb_config.get('run_name') if wandb_config else None
                
                wandb.init(
                    project=project,
                    name=name,
                    config=wandb_config if wandb_config else {}
                )
                
                # Set up wandb.watch() if requested
                if self.wandb_watch:
                    wandb.watch(self.model, log_freq=self.wandb_watch_log_freq)
                    print(f"Wandb watch enabled with log frequency: {self.wandb_watch_log_freq}")
                
                # Log model info
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                wandb.log({
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                }, step=0)
    
    def train_epoch(self, train_dataloader, epoch=None):
        """
        Train the model for one epoch.
        
        Arguments:
            train_dataloader: DataLoader for training data
            epoch: current epoch number (for logging)
            
        Returns:
            float: average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Handle different batch formats
            if isinstance(batch["pixel_values"], list):
                inputs = torch.stack(batch["pixel_values"]).to(self.device)
            else:
                inputs = batch["pixel_values"].to(self.device)
            
            labels = torch.tensor(batch["label"]).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss (with L1 regularization if enabled)
            if self.use_l1_regularization:
                # Calculate separate loss components for logging
                base_loss, l1_loss, total_loss_tensor = self.criterion.compute_separate_losses(outputs, labels, self.model)
                loss = total_loss_tensor
            else:
                base_loss = self.criterion(outputs, labels)
                l1_loss = None
                loss = base_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
            
            # Log per-step metrics to wandb
            if self.use_wandb:
                log_dict = {
                    "train/step_loss": loss.item(),
                    "train/step_cross_entropy_loss": base_loss.item(),
                    "train/step": self.step_count,
                }
                
                # Log L1 loss if regularization is enabled
                if self.use_l1_regularization and l1_loss is not None:
                    log_dict["train/step_l1_loss"] = l1_loss.item()
                
                if epoch is not None:
                    log_dict["train/epoch"] = epoch
                
                # Log optimizer statistics if available (ChillAdam specific)
                if hasattr(self.optimizer, 'state') and len(self.optimizer.state) > 0:
                    # Get learning rate from first parameter
                    first_param = next(iter(self.optimizer.param_groups[0]['params']))
                    if first_param in self.optimizer.state and 'lr' in self.optimizer.state[first_param]:
                        log_dict["optimizer/learning_rate"] = self.optimizer.state[first_param]['lr']
                
                wandb.log(log_dict, step=self.step_count)
            
            self.step_count += 1
        
        return total_loss / num_samples
    
    def validate_epoch(self, val_dataloader, epoch=None):
        """
        Validate the model for one epoch.
        
        Arguments:
            val_dataloader: DataLoader for validation data
            epoch: current epoch number (for logging)
            
        Returns:
            tuple: (average validation loss, top1 accuracy percentage, top5 accuracy percentage)
        """
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
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
                
                # Calculate loss (with L1 regularization if enabled)
                if self.use_l1_regularization:
                    # Calculate separate loss components for logging
                    base_loss, l1_loss, total_loss_tensor = self.criterion.compute_separate_losses(outputs, labels, self.model)
                    loss = total_loss_tensor
                else:
                    base_loss = self.criterion(outputs, labels)
                    l1_loss = None
                    loss = base_loss
                
                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)
                
                # Calculate top-1 and top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 += top5_pred.eq(labels.view(1, -1).expand_as(top5_pred)).sum().item()
                
                _, top1_pred = torch.max(outputs.data, 1)
                correct_top1 += (top1_pred == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / total_predictions
        top1_accuracy = 100.0 * correct_top1 / total_predictions
        top5_accuracy = 100.0 * correct_top5 / total_predictions
        
        return avg_loss, top1_accuracy, top5_accuracy
    
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
            train_loss = self.train_epoch(train_dataloader, epoch + 1)
            
            # Validation phase
            val_loss, top1_accuracy, top5_accuracy = self.validate_epoch(val_dataloader, epoch + 1)
            
            # Print results
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Top-1 Accuracy: {top1_accuracy:.2f}%")
            print(f"Validation Top-5 Accuracy: {top5_accuracy:.2f}%")
            
            # Log epoch metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "val/top1_accuracy": top1_accuracy,
                    "val/top5_accuracy": top5_accuracy,
                    "epoch": epoch + 1
                }, step=self.step_count)
        
        # Log final test accuracy
        if self.use_wandb:
            wandb.log({
                "test/final_top1_accuracy": top1_accuracy,
                "test/final_top5_accuracy": top5_accuracy
            }, step=self.step_count)
        
        print(f"\nTraining finished!")
        print(f"Final Test Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Final Test Top-5 Accuracy: {top5_accuracy:.2f}%")
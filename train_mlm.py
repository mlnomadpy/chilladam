#!/usr/bin/env python3
"""
Masked Language Model (MLM) and Causal Language Model (CLM) Training Script

A comprehensive training script for language models with streaming dataset support.
Supports both MLM (BERT-style) and CLM (GPT-style) training with configurable optimizers.

Usage:
    # MLM pretraining with streaming
    python train_mlm.py --task_type mlm --mode pretrain --streaming --dataset wikitext --max_samples 10000
    
    # CLM pretraining with streaming  
    python train_mlm.py --task_type clm --mode pretrain --streaming --dataset openwebtext --max_samples 10000
    
    # Evaluation mode
    python train_mlm.py --task_type mlm --mode evaluate --model_path ./saved_model
    
    # Serving mode
    python train_mlm.py --task_type mlm --mode serve --model_path ./saved_model
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
import wandb
from tqdm import tqdm
import math
import json

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chilladam'))

from chilladam.optimizers import create_optimizer
from chilladam.schedulers import create_scheduler


class TextDatasetHandler:
    """Handles text dataset loading with streaming support."""
    
    # Text dataset registry with streaming-compatible datasets
    TEXT_DATASET_REGISTRY = {
        "wikitext": {
            "hf_names": ["wikitext", "wikitext-103-v1", "wikitext-2-v1"],
            "config": "wikitext-103-v1",
            "text_column": "text",
            "splits": {"train": "train", "validation": "validation", "test": "test"}
        },
        "openwebtext": {
            "hf_names": ["openwebtext"],
            "config": None,
            "text_column": "text", 
            "splits": {"train": "train"}
        },
        "c4": {
            "hf_names": ["c4", "allenai/c4"],
            "config": "en",
            "text_column": "text",
            "splits": {"train": "train", "validation": "validation"}
        },
        "bookcorpus": {
            "hf_names": ["bookcorpus"],
            "config": None,
            "text_column": "text",
            "splits": {"train": "train"}
        }
    }
    
    def __init__(self, dataset_name, streaming=True, max_samples=None):
        self.dataset_name = dataset_name
        self.streaming = streaming
        self.max_samples = max_samples
        
        if dataset_name not in self.TEXT_DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(self.TEXT_DATASET_REGISTRY.keys())}")
        
        self.dataset_config = self.TEXT_DATASET_REGISTRY[dataset_name]
    
    def load_dataset_with_fallbacks(self):
        """Load dataset with fallback options and streaming support."""
        hf_names = self.dataset_config["hf_names"]
        config = self.dataset_config["config"]
        
        last_error = None
        for hf_name in hf_names:
            try:
                print(f"Trying to load dataset from '{hf_name}' (streaming={self.streaming})...")
                if config:
                    dataset = load_dataset(hf_name, config, streaming=self.streaming)
                else:
                    dataset = load_dataset(hf_name, streaming=self.streaming)
                print(f"Successfully loaded dataset from '{hf_name}'")
                return dataset
            except Exception as e:
                print(f"Failed to load from '{hf_name}': {e}")
                last_error = e
                continue
        
        raise Exception(f"Failed to load dataset '{self.dataset_name}' from any source. Last error: {last_error}")
    
    def get_datasets(self):
        """Get train/validation datasets with optional sample limiting."""
        dataset = self.load_dataset_with_fallbacks()
        splits = self.dataset_config["splits"]
        
        # Get train dataset
        train_dataset = dataset[splits["train"]]
        
        # Get validation dataset if available
        val_dataset = None
        if "validation" in splits:
            val_dataset = dataset[splits["validation"]]
        elif "test" in splits:
            val_dataset = dataset[splits["test"]]
        
        # Apply sample limiting if specified
        if self.max_samples:
            if self.streaming:
                train_dataset = train_dataset.take(self.max_samples)
                if val_dataset:
                    val_dataset = val_dataset.take(min(self.max_samples // 10, 1000))
            else:
                train_dataset = train_dataset.select(range(min(self.max_samples, len(train_dataset))))
                if val_dataset:
                    val_size = min(self.max_samples // 10, 1000, len(val_dataset))
                    val_dataset = val_dataset.select(range(val_size))
        
        return train_dataset, val_dataset, self.dataset_config["text_column"]


class MLMDataCollator:
    """Data collator for masked language modeling."""
    
    def __init__(self, tokenizer, mlm_probability=0.15, max_length=512):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
    
    def __call__(self, examples):
        # Tokenize the texts
        texts = [example["text"] for example in examples if example["text"].strip()]
        if not texts:
            # Fallback for empty batch
            texts = [""]
        
        # Tokenize and pad
        batch = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # Create masked inputs for MLM
        inputs = batch["input_ids"].clone()
        labels = batch["input_ids"].clone()
        
        # Create mask for tokens to mask (excluding special tokens)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with [MASK] token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return {
            "input_ids": inputs,
            "attention_mask": batch["attention_mask"],
            "labels": labels
        }


class CLMDataCollator:
    """Data collator for causal language modeling."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        # Tokenize the texts
        texts = [example["text"] for example in examples if example["text"].strip()]
        if not texts:
            # Fallback for empty batch
            texts = [""]
        
        # Tokenize and pad
        batch = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # For CLM, labels are the same as input_ids, shifted by one
        batch["labels"] = batch["input_ids"].clone()
        
        return batch


class LanguageModelTrainer:
    """Trainer for language models with ChillAdam optimizer support."""
    
    def __init__(self, model, optimizer, scheduler, device, task_type, use_wandb=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.task_type = task_type
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.watch(self.model)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log to wandb
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": batch_idx + epoch * len(dataloader)
                })
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return avg_loss, perplexity


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MLM/CLM models with streaming dataset support")
    
    # Mode and task type
    parser.add_argument("--mode", choices=["pretrain", "evaluate", "serve", "all"], default="pretrain",
                       help="Training mode")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to pretrained model")
    parser.add_argument("--task_type", choices=["clm", "mlm"], required=True,
                       help="Task type: CLM (causal) or MLM (masked)")
    
    # Model architecture arguments
    parser.add_argument("--vocab_size", type=int, default=30522,
                       help="Vocabulary size")
    parser.add_argument("--embedding_size", type=int, default=768,
                       help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=768,
                       help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=12,
                       help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12,
                       help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072,
                       help="Intermediate layer size")
    parser.add_argument("--num_experts", type=int, default=8,
                       help="Number of experts for MoE models")
    parser.add_argument("--top_k_experts", type=int, default=2,
                       help="Top-k experts to use")
    parser.add_argument("--max_position_embeddings", type=int, default=512,
                       help="Maximum position embeddings")
    parser.add_argument("--aux_loss_coefficient", type=float, default=0.01,
                       help="Auxiliary loss coefficient for MoE")
    
    # Dataset arguments
    parser.add_argument("--dataset", choices=["wikitext", "openwebtext", "c4", "bookcorpus"], 
                       default="wikitext", help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default=None,
                       help="Dataset configuration")
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split to use")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                       help="Tokenizer to use")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of the text column")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save model every N steps")
    
    # ChillAdam optimizer arguments
    parser.add_argument("--min_lr", type=float, default=1e-5,
                       help="Minimum learning rate for ChillAdam")
    parser.add_argument("--max_lr", type=float, default=5e-4,
                       help="Maximum learning rate for ChillAdam")
    parser.add_argument("--eps", type=float, default=1e-8,
                       help="Epsilon for Adam optimizers")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Beta1 for Adam optimizers")
    parser.add_argument("--beta2", type=float, default=0.999,
                       help="Beta2 for Adam optimizers")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Evaluation arguments
    parser.add_argument("--eval_tasks", choices=["basic", "comprehensive"], default="basic",
                       help="Evaluation tasks to run")
    parser.add_argument("--custom_tasks", nargs="+", default=None,
                       help="Custom evaluation tasks")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="chilladam-nlp",
                       help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Wandb run name")
    
    # Hub arguments
    parser.add_argument("--push_to_hub", type=str, default=None,
                       help="Hub repository to push to")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="Hugging Face Hub token")
    parser.add_argument("--hub_private", action="store_true",
                       help="Make Hub repository private")
    parser.add_argument("--hub_commit_message", type=str, default="Add model",
                       help="Commit message for Hub")
    
    # Interactive/serving arguments
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--input_text", type=str, default=None,
                       help="Input text for processing")
    parser.add_argument("--gen_max_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p (nucleus) sampling")
    
    return parser.parse_args()


def create_model(args):
    """Create the language model based on task type and arguments."""
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        if args.task_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(args.model_path)
        else:  # clm
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
    else:
        print(f"Creating new {args.task_type.upper()} model from scratch")
        # For this example, we'll use pretrained models as base
        if args.task_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        else:  # clm
            model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    return model


def main():
    """Main training function."""
    args = parse_arguments()
    
    print(f"Starting {args.task_type.upper()} training in {args.mode} mode")
    print(f"Using streaming: {args.streaming}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )
    
    if args.mode in ["pretrain", "all"]:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load datasets
        dataset_handler = TextDatasetHandler(
            dataset_name=args.dataset,
            streaming=args.streaming,
            max_samples=args.max_samples
        )
        train_dataset, val_dataset, text_column = dataset_handler.get_datasets()
        
        # Create data collators
        if args.task_type == "mlm":
            data_collator = MLMDataCollator(tokenizer, max_length=args.max_length)
        else:  # clm
            data_collator = CLMDataCollator(tokenizer, max_length=args.max_length)
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=not args.streaming  # Don't shuffle streaming datasets
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                collate_fn=data_collator
            )
        
        # Create model
        model = create_model(args)
        
        # Create optimizer
        optimizer = create_optimizer(
            optimizer_name="chilladam",  # Use ChillAdam by default
            model_parameters=model.parameters(),
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )
        
        # Create scheduler
        scheduler = None
        if args.num_epochs > 1:
            total_steps = len(train_dataloader) * args.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=total_steps // 10,
                num_training_steps=total_steps
            )
        
        # Create trainer
        trainer = LanguageModelTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            task_type=args.task_type,
            use_wandb=args.use_wandb
        )
        
        # Training loop
        print(f"Starting training for {args.num_epochs} epochs...")
        for epoch in range(args.num_epochs):
            # Train
            train_loss = trainer.train_epoch(train_dataloader, epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Evaluate
            if val_dataloader:
                val_loss, perplexity = trainer.evaluate(val_dataloader)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Perplexity = {perplexity:.2f}")
                
                if args.use_wandb:
                    wandb.log({
                        "val_loss": val_loss,
                        "perplexity": perplexity,
                        "epoch": epoch
                    })
        
        # Save model
        output_dir = f"./saved_model_{args.task_type}"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
        # Push to hub if requested
        if args.push_to_hub:
            print(f"Pushing model to {args.push_to_hub}")
            model.push_to_hub(args.push_to_hub, token=args.hub_token, private=args.hub_private)
            tokenizer.push_to_hub(args.push_to_hub, token=args.hub_token, private=args.hub_private)
    
    if args.mode in ["evaluate", "all"]:
        print("Evaluation mode - placeholder implementation")
        # Implement evaluation logic here
    
    if args.mode in ["serve", "all"]:
        print("Serving mode - placeholder implementation")
        # Implement serving logic here
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
"""
Trainer module for GBERT-Lite
Handles training, validation, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path


class Trainer:
    """Trainer class for GBERT-Lite and baseline models"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Mixed precision
        self.use_amp = config['training']['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Scheduler
        self.scheduler = None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_f1 = 0.0
        self.best_model_state = None
    
    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler with warmup"""
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        warmup_steps = self.config['training']['warmup_steps']
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps - warmup_steps
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = self.criterion(
                        outputs['falsity_prob'].squeeze(),
                        batch['labels'].float()
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(batch)
                loss = self.criterion(
                    outputs['falsity_prob'].squeeze(),
                    batch['labels'].float()
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                
                loss = self.criterion(
                    outputs['falsity_prob'].squeeze(),
                    batch['labels'].float()
                )
                
                total_loss += loss.item()
                
                probs = outputs['falsity_prob'].squeeze().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                labels = batch['labels'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)
        
        avg_loss = total_loss / len(val_loader)
        
        # Compute metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds),
            'recall': recall_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds),
            'roc_auc': roc_auc_score(all_labels, all_probs)
        }
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int) -> Dict[str, List[float]]:
        """Full training loop"""
        
        # Setup scheduler
        num_training_steps = len(train_loader) * num_epochs
        self.setup_scheduler(num_training_steps)
        
        print(f"Training for {num_epochs} epochs...")
        print(f"Total training steps: {num_training_steps}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model based on F1
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best model! F1: {self.best_val_f1:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_f1': self.best_val_f1
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        print(f"Checkpoint loaded from {path}")
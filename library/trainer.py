import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import logging
import os
import json
import time

from .loss_functions import FocalLoss, QuantileLoss, HuberLoss, DirectionalLoss
from .utils import MetricsCalculator

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.metrics = MetricsCalculator()
        
        # Classes: 0: Bear, 1: Neutral, 2: Bull
        # Weights from config or default (Bear biased)
        bear_weight = config.get('bear_weight', 3.0)
        self.class_weights = torch.tensor([bear_weight, 1.0, 1.0]).to(device)
        
        self.cls_loss_fn = FocalLoss(alpha=self.class_weights, gamma=config.get('focal_gamma', 2.0))
        
        # Regression Loss: MSE or Quantile
        if config.get('loss_type') == 'quantile':
            quantiles = config.get('quantiles', [0.1, 0.5, 0.9])
            self.reg_loss_fn = QuantileLoss(quantiles)
            logger.info(f"Using Quantile Loss with q={quantiles}")
        elif config.get('loss_type') == 'huber':
            delta = config.get('huber_delta', 1.0)
            self.reg_loss_fn = HuberLoss(delta)
            logger.info(f"Using Huber Loss (delta={delta})")
        elif config.get('loss_type') == 'directional':
            lambda_dir = config.get('directional_lambda', 0.5)
            self.reg_loss_fn = DirectionalLoss(lambda_dir)
            logger.info(f"Using Directional Loss (lambda={lambda_dir})")
        else:
            self.reg_loss_fn = nn.MSELoss()
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        weight_cls = self.config.get('cls_weight', 0.8)
        
        for batch in loader:
            b = [t.to(self.device) for t in batch]
            # Batch: [X_dyn, X_stat, X_time, Y_1_reg, Y_1_cls, Y_2_reg, Y_2_cls]
            self.optimizer.zero_grad()
            
            r1, c1, r2, c2 = self.model(b[0], b[1], b[2])
            
            loss_c_1 = self.cls_loss_fn(c1, b[4])
            # If MSE/Huber, ensure target shape matches pred (B, 1)
            # Pred r1 is (B, 1). Target b[3] is (B).
            # Quantile: r1 (B, Q), target b[3] (B) -> QuantileLoss handles it.
            # Directional: r1 (B, 1), target b[3] (B)
            
            target_1 = b[3]
            target_2 = b[5]
            
            if not isinstance(self.reg_loss_fn, QuantileLoss):
                 # For MSE/Huber/Directional, reshape target to (B, 1) to match r1
                 if target_1.dim() == 1: target_1 = target_1.view(-1, 1)
                 if target_2.dim() == 1: target_2 = target_2.view(-1, 1)
            
            loss_r_1 = self.reg_loss_fn(r1, target_1)
            loss_r_2 = self.reg_loss_fn(r2, target_2)
            
            loss_c_2 = self.cls_loss_fn(c2, b[6])
            
            total_cls = loss_c_1 + loss_c_2
            total_reg = loss_r_1 + loss_r_2
            
            loss = (weight_cls * total_cls) + ((1 - weight_cls) * total_reg)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        weight_cls = self.config.get('cls_weight', 0.8)
        
        p1_cls, t1_cls = [], []
        p2_cls, t2_cls = [], []
        
        with torch.no_grad():
            for batch in loader:
                b = [t.to(self.device) for t in batch]
                r1, c1, r2, c2 = self.model(b[0], b[1], b[2])
                
                # Compute Loss
                # Target handling (same as train_epoch)
                target_1, target_2 = b[3], b[5]
                if not isinstance(self.reg_loss_fn, QuantileLoss):
                     if target_1.dim() == 1: target_1 = target_1.view(-1, 1)
                     if target_2.dim() == 1: target_2 = target_2.view(-1, 1)

                loss_c_1 = self.cls_loss_fn(c1, b[4])
                loss_c_2 = self.cls_loss_fn(c2, b[6])
                loss_r_1 = self.reg_loss_fn(r1, target_1)
                loss_r_2 = self.reg_loss_fn(r2, target_2)
                
                loss = (weight_cls * (loss_c_1 + loss_c_2)) + ((1 - weight_cls) * (loss_r_1 + loss_r_2))
                total_loss += loss.item()
                
                p1_cls.extend(torch.argmax(c1, dim=1).cpu().numpy())
                t1_cls.extend(b[4].cpu().numpy())
                
                p2_cls.extend(torch.argmax(c2, dim=1).cpu().numpy())
                t2_cls.extend(b[6].cpu().numpy())

        # Metrics
        val_loss = total_loss / len(loader)
        f1_5_1 = self.metrics.get_f1_5_score(t1_cls, p1_cls)
        f1_5_2 = self.metrics.get_f1_5_score(t2_cls, p2_cls)
        
        return val_loss, f1_5_1, f1_5_2

    def run_training(self, train_loader, val_loader, epochs=50, save_path="best_model.pth", config=None, dyn_cols=None, stat_cols=None):
        best_score = -float('inf')
        patience = self.config.get('patience', 15)
        early_stop = 0
        
        logger.info(f"Starting Training (max {epochs} epochs)...")
        
        for epoch in range(epochs):
            t0 = time.time()
            loss = self.train_epoch(train_loader)
            val_loss, val_f1_1, val_f1_2 = self.validate(val_loader)
            
            # Weighted Score (F1 bias)
            score = (0.7 * val_f1_1) + (0.3 * val_f1_2)
            
            # Use score for scheduler? Or val_loss?
            # Scheduler usually minimizes loss or maximizes score.
            # Convert score to loss-like for scheduler if mode='min'
            # But earlier code used scheduler mode='max'.
            self.scheduler.step(score)
            
            logger.info(f"Ep {epoch+1}: Train Loss {loss:.4f} | Val Loss {val_loss:.4f} | F1: {val_f1_1:.3f}/{val_f1_2:.3f} | Score {score:.4f}")
             
            if score > best_score:
                best_score = score
                early_stop = 0
                self.save_checkpoint(save_path, config=config, dyn_cols=dyn_cols, stat_cols=stat_cols)
                logger.info(f"  --> New Best! Saved to {save_path}")
            else:
                early_stop += 1
                
            if early_stop >= patience:
                logger.info("Early Stopping Triggered.")
                break
                
        return best_score
        
    def save_checkpoint(self, path, config=None, dyn_cols=None, stat_cols=None):
        """Save model checkpoint with metadata.
        
        Creates a dedicated folder structure:
        weights/model_name/
            ├── model.pth           (model weights)
            ├── config.json         (training config)
            └── model_info.json     (architecture metadata)
        """
        # Skip saving if path is devnull or None (used during Optuna trials)
        if path is None or path == os.devnull or 'nul' in str(path).lower():
            return None
            
        # Extract model name from path
        if path.endswith('.pth'):
            model_name = os.path.basename(path)[:-4]
        else:
            model_name = os.path.basename(path)
            
        # Create model directory
        model_dir = os.path.join(os.path.dirname(path) or 'weights', model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model weights
        weights_path = os.path.join(model_dir, 'model.pth')
        torch.save(self.model.state_dict(), weights_path)
        logger.info(f"Saved model weights to {weights_path}")
        
        # Save config if provided
        if config:
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved config to {config_path}")
        
        # Save model metadata
        metadata = {
            "model_name": model_name,
            "architecture": config.get('model_type', 'unknown') if config else 'unknown',
            "input_features": {
                "dynamic_features": len(dyn_cols) if dyn_cols else config.get('input_dim', 'unknown'),
                "static_features": len(stat_cols) if stat_cols else config.get('static_dim', 'unknown'),
                "dynamic_feature_names": list(dyn_cols) if dyn_cols else [],
                "static_feature_names": list(stat_cols) if stat_cols else []
            },
            "outputs": {
                "regression_heads": 2,
                "classification_heads": 2,
                "regression_targets": ["target_1", "target_2"],
                "classification_classes": ["Bear (0)", "Neutral (1)", "Bull (2)"],
                "classification_thresholds": {"bear": -0.02, "neutral_min": -0.02, "neutral_max": 0.02, "bull": 0.02}
            },
            "hyperparameters": {
                "hidden_size": config.get('hidden_size', 'unknown') if config else 'unknown',
                "lstm_layers": config.get('lstm_layers', 'unknown') if config else 'unknown',
                "trans_layers": config.get('trans_layers', 'unknown') if config else 'unknown',
                "dropout": config.get('dropout', 'unknown') if config else 'unknown',
                "learning_rate": config.get('lr', 'unknown') if config else 'unknown',
                "batch_size": config.get('batch_size', 'unknown') if config else 'unknown'
            },
            "training_info": {
                "epochs": config.get('epochs', 'unknown') if config else 'unknown',
                "loss_type": config.get('loss_type', 'mse') if config else 'mse',
                "cls_weight": config.get('cls_weight', 0.8) if config else 0.8
            }
        }
        
        metadata_path = os.path.join(model_dir, 'model_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved model metadata to {metadata_path}")
        
        return model_dir


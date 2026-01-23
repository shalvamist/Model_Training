import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import logging
import os
import time

from .loss_functions import FocalLoss
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
            loss_r_1 = self.reg_loss_fn(r1.squeeze(), b[3])
            
            loss_c_2 = self.cls_loss_fn(c2, b[6])
            loss_r_2 = self.reg_loss_fn(r2.squeeze(), b[5])
            
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
        p1_cls, t1_cls = [], []
        p2_cls, t2_cls = [], []
        
        with torch.no_grad():
            for batch in loader:
                b = [t.to(self.device) for t in batch]
                r1, c1, r2, c2 = self.model(b[0], b[1], b[2])
                
                p1_cls.extend(torch.argmax(c1, dim=1).cpu().numpy())
                t1_cls.extend(b[4].cpu().numpy())
                
                p2_cls.extend(torch.argmax(c2, dim=1).cpu().numpy())
                t2_cls.extend(b[6].cpu().numpy())

        # Metrics
        f1_5_1 = self.metrics.get_f1_5_score(t1_cls, p1_cls)
        f1_5_2 = self.metrics.get_f1_5_score(t2_cls, p2_cls)
        
        return f1_5_1, f1_5_2

    def run_training(self, train_loader, val_loader, epochs=50):
        best_score = 0.0
        patience = self.config.get('patience', 15)
        early_stop = 0
        
        logger.info("Starting Training...")
        
        for epoch in range(epochs):
            t0 = time.time()
            loss = self.train_epoch(train_loader)
            val_score_1, val_score_2 = self.validate(val_loader)
            
            # Weighted Score
            score = (0.7 * val_score_1) + (0.3 * val_score_2)
            
            self.scheduler.step(score)
            
            logger.info(f"Ep {epoch+1}: Loss {loss:.4f} | Val F1.5 (1): {val_score_1:.4f} | Score: {score:.4f} | Time: {time.time()-t0:.1f}s")
             
            if score > best_score:
                best_score = score
                early_stop = 0
                self.save_checkpoint("best_model_checkpoint.pth")
            else:
                early_stop += 1
                
            if early_stop >= patience:
                logger.info("Early Stopping Triggered.")
                break
                
        return best_score
        
    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)


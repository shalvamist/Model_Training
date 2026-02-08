import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os

# Add local path
sys.path.append(os.getcwd())

from library.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockModel(nn.Module):
    def __init__(self, quantiles=3):
        super().__init__()
        # Output 3 quantiles for 2 reg heads
        self.head_1_reg = nn.Linear(10, quantiles)
        self.head_1_cls = nn.Linear(10, 3)
        self.head_2_reg = nn.Linear(10, quantiles)
        self.head_2_cls = nn.Linear(10, 3)
        
    def forward(self, x, stat, time):
        # x is (B, 10)
        r1 = self.head_1_reg(x)
        c1 = self.head_1_cls(x)
        r2 = self.head_2_reg(x)
        c2 = self.head_2_cls(x)
        return r1, c1, r2, c2

def main():
    logger.info("Testing Quantile Training Loop...")
    
    # 1. Config
    config = {
        "loss_type": "quantile",
        "quantiles": [0.1, 0.5, 0.9],
        "lr": 1e-3
    }
    
    # 2. Data
    B = 8
    x_dyn = torch.randn(B, 10)
    x_stat = torch.randn(B, 5)
    x_time = torch.zeros(B, 2, dtype=torch.long)
    y_reg = torch.randn(B)
    y_cls = torch.randint(0, 3, (B,))
    
    dataset = TensorDataset(x_dyn, x_stat, x_time, y_reg, y_cls, y_reg, y_cls)
    loader = DataLoader(dataset, batch_size=4)
    
    # 3. Model & Trainer
    model = MockModel(quantiles=3)
    trainer = ModelTrainer(model, config)
    
    # 4. Train Step
    try:
        loss = trainer.train_epoch(loader)
        logger.info(f"Train Epoch Success. Loss: {loss:.4f}")
        assert loss > 0
    except Exception as e:
        logger.error(f"Train Epoch Failed: {e}")
        raise e
        
    logger.info("SUCCESS: Quantile Training Verified.")

if __name__ == "__main__":
    main()

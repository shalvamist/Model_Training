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
    def __init__(self):
        super().__init__()
        # Output 1 reg head, 3 cls head
        self.head_1_reg = nn.Linear(10, 1)
        self.head_1_cls = nn.Linear(10, 3)
        self.head_2_reg = nn.Linear(10, 1)
        self.head_2_cls = nn.Linear(10, 3)
        
    def forward(self, x, stat, time):
        r1 = self.head_1_reg(x)
        c1 = self.head_1_cls(x)
        r2 = self.head_2_reg(x)
        c2 = self.head_2_cls(x)
        return r1, c1, r2, c2

def test_loss(config, name):
    logger.info(f"Testing {name}...")
    
    # 2. Data
    B = 4
    x_dyn = torch.randn(B, 10)
    x_stat = torch.randn(B, 5)
    x_time = torch.zeros(B, 2, dtype=torch.long)
    y_reg = torch.randn(B)
    y_cls = torch.randint(0, 3, (B,))
    
    dataset = TensorDataset(x_dyn, x_stat, x_time, y_reg, y_cls, y_reg, y_cls)
    loader = DataLoader(dataset, batch_size=4)
    
    # 3. Model & Trainer
    model = MockModel()
    trainer = ModelTrainer(model, config)
    
    # 4. Train Step
    try:
        loss = trainer.train_epoch(loader)
        logger.info(f"{name} Success. Loss: {loss:.4f}")
        assert loss > 0
    except Exception as e:
        logger.error(f"{name} Failed: {e}")
        raise e

def main():
    # Test Huber
    test_loss({"loss_type": "huber", "huber_delta": 0.5}, "Huber Loss")
    
    # Test Directional
    test_loss({"loss_type": "directional", "directional_lambda": 0.5}, "Directional Loss")
    
    logger.info("SUCCESS: All Advanced Losses Verified.")

if __name__ == "__main__":
    main()

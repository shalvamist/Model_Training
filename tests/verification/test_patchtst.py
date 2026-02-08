import torch
import logging
import sys
import os

# Add local path
sys.path.append(os.getcwd())

from library.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing PatchTST Architecture...")
    
    # 1. Config
    config = {
        "model_type": "patchtst",
        "hidden_size": 32,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.1,
        "trans_layers": 1,
        "input_dim": 10, # 10 dynamic features
        "static_cols": ["a", "b"] # placeholder
    }
    
    # 2. Instantiate
    try:
        model = ModelFactory.create_model(config)
        logger.info("Model Instantiated Successfully.")
    except Exception as e:
        logger.error(f"Model Creation Failed: {e}")
        return

    # 3. Dummy Input
    B, L, D = 4, 64, 10
    x_dyn = torch.randn(B, L, D)
    x_stat = torch.randn(B, 10) # 2 stat + 8 placeholders? Factory logic for static dim might differ
    # Network expects (B, static_dim)
    # The factory logic: static_dim = len(static_cols) + 8 = 10
    
    x_time = torch.zeros(B, 2, dtype=torch.long) # dow, moy
    
    # 4. Forward
    try:
        reg1, cls1, reg2, cls2 = model(x_dyn, x_stat, x_time)
        logger.info("Forward Pass Successful.")
        logger.info(f"Reg1 Shape: {reg1.shape} (Expected {B}, 1)")
        logger.info(f"Cls1 Shape: {cls1.shape} (Expected {B}, 3)")
        
        assert reg1.shape == (B, 1)
        assert cls1.shape == (B, 3)
        
    except Exception as e:
        logger.error(f"Forward Pass Failed: {e}")
        raise e
        
    logger.info("SUCCESS: PatchTST Verified.")

if __name__ == "__main__":
    main()

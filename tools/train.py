import argparse
import json
import logging
import sys
import torch
import os
from Model_trading_training.library.factory import ModelFactory
from Model_trading_training.library.trainer import ModelTrainer

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a Trading Model")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--output", type=str, default="trained_model.pth", help="Output path for weights")
    args = parser.parse_args()
    
    logger.info(f"Loading Config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Override config with args if needed
    config['epochs'] = args.epochs
    
    # 1. Get Processor
    logger.info(f"Initializing Processor: {config.get('processor_version', 'v11')}")
    processor = ModelFactory.get_processor(config.get('processor_version', 'v11'), config)
    
    # 2. Fetch Data
    df = processor.fetch_data()
    if df.empty:
        logger.error("Failed to fetch data.")
        return
        
    logger.info(f"Data Fetched: {len(df)} rows")
    
    # 3. Process & Create Sequences
    df_proc, dyn_cols, stat_cols = processor.process(df)
    
    # Update Config with potentially dynamic columns if not present
    if 'input_dim' not in config:
        config['input_dim'] = len(dyn_cols)
    if 'static_dim' not in config:
        config['static_dim'] = len(stat_cols)
    
    data = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    # Pack into Loaders (Simplified for tool)
    from torch.utils.data import DataLoader, TensorDataset
    X_d, X_s, X_t, Y_1, Y_2, dates = data
    
    # Simple Train/Val Split (Last 20% Val)
    split_idx = int(len(X_d) * 0.8)
    
    def to_tensor(arr): return torch.tensor(arr, dtype=torch.float32)
    def to_long(arr): return torch.tensor(arr, dtype=torch.long)
    
    # Targets: Reg (Float), Cls (Long)
    # create_sequences returned raw reg targets. We need to classify them.
    # Simple Logic: > 2% Bull, < -2% Bear
    Y_1_c = np.ones_like(Y_1, dtype=int)
    Y_1_c[Y_1 > 0.02] = 2; Y_1_c[Y_1 < -0.02] = 0
    
    Y_2_c = np.ones_like(Y_2, dtype=int)
    Y_2_c[Y_2 > 0.04] = 2; Y_2_c[Y_2 < -0.04] = 0
    
    train_ds = TensorDataset(
        to_tensor(X_d[:split_idx]), to_tensor(X_s[:split_idx]), to_long(X_t[:split_idx]),
        to_tensor(Y_1[:split_idx]), to_long(Y_1_c[:split_idx]),
        to_tensor(Y_2[:split_idx]), to_long(Y_2_c[:split_idx])
    )
    
    val_ds = TensorDataset(
        to_tensor(X_d[split_idx:]), to_tensor(X_s[split_idx:]), to_long(X_t[split_idx:]),
        to_tensor(Y_1[split_idx:]), to_long(Y_1_c[split_idx:]),
        to_tensor(Y_2[split_idx:]), to_long(Y_2_c[split_idx:])
    )
    
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 64), shuffle=False)
    
    # 4. Create Model
    logger.info("Initializing Model...")
    model = ModelFactory.create_model(config)
    
    # 5. Train
    trainer = ModelTrainer(model, config, device=args.device)
    best_score = trainer.run_training(train_loader, val_loader, epochs=args.epochs)
    
    logger.info(f"Training Complete. Best Score: {best_score:.4f}")
    trainer.save_checkpoint(args.output)
    logger.info(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()

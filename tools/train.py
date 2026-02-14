import argparse
import json
import logging
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add local path (root) to sys.path
# This ensures "library" can be imported whether run from root or tools/
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from library.processors import GenericProcessor
from library.factory import ModelFactory
from library.trainer import ModelTrainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a Trading Model from Config")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, help="Override batch_size from config")
    parser.add_argument("--lr", type=float, help="Override learning rate from config")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Override Config with CLI Args
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['lr'] = args.lr
    if args.force_cpu: config['use_gpu'] = False
        
    # Set Device
    device = "cuda" if torch.cuda.is_available() and config.get("use_gpu", False) else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Data Processing
    logger.info("Initializing Processor...")
    processor_type = config.get("processor", "generic")
    processor = ModelFactory.get_processor(processor_type, config)
    df = processor.fetch_data()
    
    if df.empty:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)
        
    df_proc, dyn_cols, stat_cols = processor.process(df)
    
    # Create Sequences
    logger.info("Creating Sequences...")
    sequences = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    # Handle variable return length (V19=6, V20=8)
    if len(sequences) == 8:
        X_dyn, X_stat, X_time, Y_1, Y_2, dates, Y_1_cls, Y_2_cls = sequences
        logger.info("Loaded Pre-Calculated Classification Targets (V20 Triple Barrier)")
    else:
        X_dyn, X_stat, X_time, Y_1, Y_2, dates = sequences
        
        # Derive classification targets from regression targets (V19 Logic)
        cls_threshold = config.get('cls_threshold', 0.02)
        logger.info(f"Deriving Classification Targets (Threshold: ±{cls_threshold*100:.1f}%)")
        
        def classify_return(returns, thresh):
            labels = np.zeros(len(returns), dtype=np.int64)
            labels[returns > thresh] = 2   # Bull
            labels[returns < -thresh] = 0  # Bear  
            labels[(returns >= -thresh) & (returns <= thresh)] = 1  # Neutral
            return labels
        
        Y_1_cls = classify_return(Y_1, cls_threshold)
        Y_2_cls = classify_return(Y_2, cls_threshold)
    
    if len(X_dyn) == 0:
        logger.error("No sequences created. Check data length vs seq_length.")
        sys.exit(1)
        
    # 3-Way Chronological Split: Train / Val / Test
    train_years = config.get('train_years', 13)
    val_years = config.get('val_years', 2)
    test_years = config.get('test_years', 2)
    total_years = train_years + val_years + test_years
    
    n_samples = len(X_dyn)
    train_frac = train_years / total_years
    val_frac = val_years / total_years
    
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    logger.info(f"Data Split: Train={train_years}y ({train_end} samples), Val={val_years}y ({val_end-train_end} samples), Test={test_years}y ({n_samples-val_end} samples)")
    
    train_data = TensorDataset(
        torch.tensor(X_dyn[:train_end], dtype=torch.float32),
        torch.tensor(X_stat[:train_end], dtype=torch.float32),
        torch.tensor(X_time[:train_end], dtype=torch.long),
        torch.tensor(Y_1[:train_end], dtype=torch.float32), # Target 1 (Reg)
        torch.tensor(Y_1_cls[:train_end], dtype=torch.long), # Target 1 (Class)
        torch.tensor(Y_2[:train_end], dtype=torch.float32), # Target 2 (Reg)
        torch.tensor(Y_2_cls[:train_end], dtype=torch.long)  # Target 2 (Class)
    )
    
    val_data = TensorDataset(
        torch.tensor(X_dyn[train_end:val_end], dtype=torch.float32),
        torch.tensor(X_stat[train_end:val_end], dtype=torch.float32),
        torch.tensor(X_time[train_end:val_end], dtype=torch.long),
        torch.tensor(Y_1[train_end:val_end], dtype=torch.float32),
        torch.tensor(Y_1_cls[train_end:val_end], dtype=torch.long),
        torch.tensor(Y_2[train_end:val_end], dtype=torch.float32),
        torch.tensor(Y_2_cls[train_end:val_end], dtype=torch.long)
    )
    
    train_loader = DataLoader(train_data, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.get("batch_size", 32))
    
    # 2. Model Creation
    logger.info("Building Model...")
    # Inject input dimensions into config for factory
    config['input_dim'] = len(dyn_cols)
    config['static_dim'] = len(stat_cols)
    
    model = ModelFactory.create_model(config).to(device)
    
    # 3. Training
    logger.info("Starting Training...")
    trainer = ModelTrainer(model, config, device=device)
    
    # Weights Dir
    os.makedirs("weights", exist_ok=True)
    # Determine model name from config file path if possible, else use experiment_name
    if args.config:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        experiment_name = config_name
    else:
        experiment_name = config.get('experiment_name', 'model')
        
    save_path = os.path.join("weights", f"{experiment_name}.pth")
    
    epochs = config.get("epochs", 10)
    best_score = trainer.run_training(
        train_loader, val_loader, 
        epochs=epochs, 
        save_path=save_path,
        config=config,
        dyn_cols=dyn_cols,
        stat_cols=stat_cols
    )
            
    logger.info(f"Training Completed. Best Score: {best_score:.4f}")

if __name__ == "__main__":
    main()

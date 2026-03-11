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

from library.processors import GenericProcessor, prepare_training_data
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
    parser.add_argument("--seed", type=int, help="Deterministic seed for PyTorch and Numpy")
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
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set deterministic seed to: {args.seed}")
    
    # 1. Data Processing
    logger.info("Initializing Processor...")
    processor_type = config.get("processor", "generic")
    processor = ModelFactory.get_processor(processor_type, config)
    
    data_pkg = prepare_training_data(processor, config, batch_size_override=args.batch_size)
    if data_pkg is None:
        sys.exit(1)
        
    train_loader, val_loader, dyn_cols, stat_cols = data_pkg
    
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
        
    if args.seed is not None:
        experiment_name = f"{experiment_name}_seed_{args.seed}"
        
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

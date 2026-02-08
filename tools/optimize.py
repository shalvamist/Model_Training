import argparse
import json
import logging
import os
import sys
import torch
import numpy as np
import optuna
from torch.utils.data import DataLoader, TensorDataset

# Add local path (root) to sys.path
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

def objective(trial, config, device, train_loader_full, val_loader_full, epochs=10):
    # 1. Suggest Hyperparams
    # Check for "optuna" section in config, else use defaults
    opt_config = config.get("optuna", {})
    
    # LR
    if "lr" in opt_config:
        lr_min, lr_max = opt_config["lr"]
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        
    # Batch Size
    if "batch_size" in opt_config:
        batch_size = trial.suggest_categorical("batch_size", opt_config["batch_size"])
    else:
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
    # Hidden Size
    if "hidden_size" in opt_config:
        hidden_size = trial.suggest_categorical("hidden_size", opt_config["hidden_size"])
    else:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        
    # Dropout
    if "dropout" in opt_config:
        d_min, d_max = opt_config["dropout"]
        dropout = trial.suggest_float("dropout", d_min, d_max)
    else:
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Update Config
    trial_config = config.copy()
    trial_config["lr"] = lr
    trial_config["batch_size"] = batch_size
    trial_config["hidden_size"] = hidden_size
    trial_config["dropout"] = dropout
    
    # Tune layers for hybrid/v17 models
    if "lstm_layers" in opt_config:
        lstm_vals = opt_config["lstm_layers"]
        if len(lstm_vals) == 2:
            trial_config["lstm_layers"] = trial.suggest_int("lstm_layers", lstm_vals[0], lstm_vals[1])
        else:
            trial_config["lstm_layers"] = trial.suggest_categorical("lstm_layers", lstm_vals)
    
    if "trans_layers" in opt_config:
        trans_vals = opt_config["trans_layers"]
        if len(trans_vals) == 2:
            trial_config["trans_layers"] = trial.suggest_int("trans_layers", trans_vals[0], trans_vals[1])
        else:
            trial_config["trans_layers"] = trial.suggest_categorical("trans_layers", trans_vals)
        
    if "n_experts" in opt_config:
        expert_vals = opt_config["n_experts"]
        if len(expert_vals) == 2:
            trial_config["n_experts"] = trial.suggest_int("n_experts", expert_vals[0], expert_vals[1])
        else:
            trial_config["n_experts"] = trial.suggest_categorical("n_experts", expert_vals)
        
    # PatchTST specific
    if "patch_len" in opt_config:
        p_choices = opt_config["patch_len"]
        trial_config["patch_len"] = trial.suggest_categorical("patch_len", p_choices)
        
    if "stride" in opt_config:
        s_choices = opt_config["stride"]
        trial_config["stride"] = trial.suggest_categorical("stride", s_choices)

    # Legacy fallback (if specific layers not in optuna but in config)
    if "layers" in opt_config:
        l_min, l_max = opt_config["layers"]
        n_layers = trial.suggest_int("n_layers", l_min, l_max)
        trial_config["lstm_layers"] = n_layers
        trial_config["trans_layers"] = n_layers

    # 2. Build Model
    model = ModelFactory.create_model(trial_config).to(device)
    
    # 3. Train (Short duration for optimization)
    # Recreate loaders with new batch_size
    train_dataset = train_loader_full.dataset
    val_dataset = val_loader_full.dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    trainer = ModelTrainer(model, trial_config, device=device)
    
    # Run training
    score = trainer.run_training(train_loader, val_loader, epochs=epochs, save_path=os.devnull)
    
    return score

def main():
    parser = argparse.ArgumentParser(description="Optimize a Trading Model with Optuna")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--top_k", type=int, default=1, help="Save top K best configurations")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Data Prep (Once)
    logger.info("Initializing Processor...")
    processor_type = config.get("processor", "generic")
    processor = ModelFactory.get_processor(processor_type, config)
    df = processor.fetch_data()
    df_proc, dyn_cols, stat_cols = processor.process(df)
    
    # Inject dims
    config['input_dim'] = len(dyn_cols)
    config['static_dim'] = len(stat_cols)
    
    X_dyn, X_stat, X_time, Y_1, Y_2, dates = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    if len(X_dyn) == 0:
        logger.error("No sequences.")
        sys.exit(1)
        
    # 3-Way Chronological Split: Train / Val / Test
    # Get split parameters from config (in years)
    train_years = config.get('train_years', 13)
    val_years = config.get('val_years', 2)
    test_years = config.get('test_years', 2)
    total_years = train_years + val_years + test_years
    
    # Calculate split indices based on proportions
    n_samples = len(X_dyn)
    train_frac = train_years / total_years
    val_frac = val_years / total_years
    
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    logger.info(f"Data Split: Train={train_years}y ({train_end} samples), Val={val_years}y ({val_end-train_end} samples), Test={test_years}y ({n_samples-val_end} samples)")
    
    # Tensors
    t_X_dyn = torch.tensor(X_dyn, dtype=torch.float32)
    t_X_stat = torch.tensor(X_stat, dtype=torch.float32)
    t_X_time = torch.tensor(X_time, dtype=torch.long)
    t_Y_1 = torch.tensor(Y_1, dtype=torch.float32)
    t_Y_2 = torch.tensor(Y_2, dtype=torch.float32)
    
    
    # Derive classification targets from regression targets
    # Bear (0): target < -2%, Neutral (1): -2% to +2%, Bull (2): > +2%
    def classify_return(returns):
        """Convert continuous returns to 3-class labels"""
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[returns > 0.02] = 2   # Bull
        labels[returns < -0.02] = 0  # Bear  
        labels[(returns >= -0.02) & (returns <= 0.02)] = 1  # Neutral
        return labels
    
    
    Y_1_cls = classify_return(Y_1)
    Y_2_cls = classify_return(Y_2)
    
    # Create Full Datasets (Train + Val for Optuna, Test is held out)
    train_data = TensorDataset(
        t_X_dyn[:train_end], t_X_stat[:train_end], t_X_time[:train_end],
        t_Y_1[:train_end], torch.tensor(Y_1_cls[:train_end], dtype=torch.long), 
        t_Y_2[:train_end], torch.tensor(Y_2_cls[:train_end], dtype=torch.long)
    )
    
    val_data = TensorDataset(
        t_X_dyn[train_end:val_end], t_X_stat[train_end:val_end], t_X_time[train_end:val_end],
        t_Y_1[train_end:val_end], torch.tensor(Y_1_cls[train_end:val_end], dtype=torch.long), 
        t_Y_2[train_end:val_end], torch.tensor(Y_2_cls[train_end:val_end], dtype=torch.long)
    )
    
    # We pass Dummy Loaders to objective, which extracts Dataset
    # This avoids passing raw tensors
    train_loader_dummy = DataLoader(train_data, batch_size=32)
    val_loader_dummy = DataLoader(val_data, batch_size=32)
    
    # 2. Optimization
    study_name = args.study_name or config.get("experiment_name", "optimization")
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    logger.info(f"Starting Optuna Optimization ({args.trials} trials)...")
    
    study.optimize(lambda trial: objective(trial, config, device, train_loader_dummy, val_loader_dummy, epochs=args.epochs), n_trials=args.trials)
    
    logger.info("Optimization Complete.")
    logger.info(f"Best Trial: {study.best_trial.params}")
    logger.info(f"Best Score: {study.best_value}")
    
    # 3. Save Top K Params
    top_k = args.top_k if hasattr(args, 'top_k') else 1
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_trials = sorted_trials[:top_k]
    
    logger.info(f"Saving top {len(top_trials)} configurations...")
    
    for rank, trial in enumerate(top_trials, start=1):
        # Create config with trial params
        trial_config = config.copy()
        trial_config.update(trial.params)
        
        # Save with rank suffix (strip existing _optimized to avoid double naming)
        base_path = config_path.replace("_optimized.json", ".json")  # Remove existing _optimized
        
        if top_k == 1:
            save_config_path = base_path.replace(".json", "_optimized.json")
        else:
            save_config_path = base_path.replace(".json", f"_optimized_rank{rank}.json")
            
        with open(save_config_path, 'w') as f:
            json.dump(trial_config, f, indent=4)
        logger.info(f"  Rank {rank}: Score={trial.value:.4f} -> {save_config_path}")
    
if __name__ == "__main__":
    main()

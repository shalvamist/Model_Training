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


def suggest_param(trial, name, spec):
    """Universal parameter suggestion from structured search_space spec.
    
    Supports:
        {"type": "int", "low": 4, "high": 10}
        {"type": "float", "low": 0.1, "high": 0.35}
        {"type": "loguniform", "low": 1e-5, "high": 5e-4}
        {"type": "categorical", "choices": [64, 128, 256]}
    """
    ptype = spec.get("type", "float")
    
    if ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    elif ptype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif ptype == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif ptype == "float":
        return trial.suggest_float(name, spec["low"], spec["high"])
    else:
        raise ValueError(f"Unknown search_space type: {ptype} for param: {name}")



def objective(trial, config, device, df_cache, epochs=10):
    """Optuna objective supporting both V18 (legacy) and V19 (structured) search spaces."""
    
    opt_config = config.get("optuna", {})
    search_space = opt_config.get("search_space", {})
    
    trial_config = config.copy()
    
    if search_space:
        # V19 Structured Search Space
        for param_name, spec in search_space.items():
            trial_config[param_name] = suggest_param(trial, param_name, spec)
    else:
        # Legacy V18 Search Space (flat arrays)
        if "lr" in opt_config:
            lr_min, lr_max = opt_config["lr"]
            trial_config["lr"] = trial.suggest_float("lr", lr_min, lr_max, log=True)
        else:
            trial_config["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            
        if "batch_size" in opt_config:
            trial_config["batch_size"] = trial.suggest_categorical("batch_size", opt_config["batch_size"])
        else:
            trial_config["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
            
        if "hidden_size" in opt_config:
            trial_config["hidden_size"] = trial.suggest_categorical("hidden_size", opt_config["hidden_size"])
        else:
            trial_config["hidden_size"] = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
            
        if "dropout" in opt_config:
            d_min, d_max = opt_config["dropout"]
            trial_config["dropout"] = trial.suggest_float("dropout", d_min, d_max)
        else:
            trial_config["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
        
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
                
        if "n_heads" in opt_config:
            head_vals = opt_config["n_heads"]
            if len(head_vals) == 2:
                # E.g. [4, 8] - we can just pick categorical if we define an array of choices in the json
                # But typically heads jump in powers of 2. We will treat it as categorical if it has >2 elements,
                # or just categorical anyway.
                pass # handled below securely 
            trial_config["n_heads"] = trial.suggest_categorical("n_heads", head_vals)
            
        if "n_experts" in opt_config:
            expert_vals = opt_config["n_experts"]
            if len(expert_vals) == 2:
                trial_config["n_experts"] = trial.suggest_int("n_experts", expert_vals[0], expert_vals[1])
            else:
                trial_config["n_experts"] = trial.suggest_categorical("n_experts", expert_vals)
            
        if "patch_len" in opt_config:
            trial_config["patch_len"] = trial.suggest_categorical("patch_len", opt_config["patch_len"])
            
        if "stride" in opt_config:
            trial_config["stride"] = trial.suggest_categorical("stride", opt_config["stride"])

        # --- V23 Specific Hyperparameters ---
        if "tail_penalty_factor" in opt_config:
            tp_min, tp_max = opt_config["tail_penalty_factor"]
            trial_config["tail_penalty_factor"] = trial.suggest_float("tail_penalty_factor", tp_min, tp_max)
        else:
            # Default V23 search space if not explicitly provided
            trial_config["tail_penalty_factor"] = trial.suggest_float("tail_penalty_factor", 2.0, 20.0)
            
        if "tail_threshold" in opt_config:
            tt_min, tt_max = opt_config["tail_threshold"]
            trial_config["tail_threshold"] = trial.suggest_float("tail_threshold", tt_min, tt_max)
        else:
            trial_config["tail_threshold"] = trial.suggest_float("tail_threshold", -0.05, -0.01)

        if "layers" in opt_config:
            l_min, l_max = opt_config["layers"]
            n_layers = trial.suggest_int("n_layers", l_min, l_max)
            trial_config["lstm_layers"] = n_layers
            trial_config["trans_layers"] = n_layers

    # Always rely on prepare_training_data for dataset construction
    # We pass df_cache directly so no fetching from yfinance happens inside the optuna loop
    # We use trial_config entirely so parameters like batch_size, seq_length, cls_threshold take effect
    processor = ModelFactory.get_processor(trial_config.get("processor", "generic"), trial_config)
    data_pkg = prepare_training_data(processor, trial_config, df=df_cache, batch_size_override=trial_config.get('batch_size'), return_raw=True)
    
    if data_pkg is None:
        raise optuna.exceptions.TrialPruned("Data preparation failed for these parameters.")
        
    train_loader = data_pkg['train_loader']
    val_loader = data_pkg['val_loader']
    trial_config['input_dim'] = len(data_pkg['dyn_cols'])
    trial_config['static_dim'] = len(data_pkg['stat_cols'])
    
    # Build Model
    model = ModelFactory.create_model(trial_config).to(device)
    
    # Train
    trainer = ModelTrainer(model, trial_config, device=device)
    score = trainer.run_training(train_loader, val_loader, epochs=epochs, save_path=os.devnull)
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Optimize a Trading Model with Optuna")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per trial (defaults to config)")
    parser.add_argument("--top_k", type=int, default=3, help="Save top K best configurations")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna")
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
    
    # Fetch data ONCE for caching
    logger.info("Initializing Processor and Caching Data...")
    processor_type = config.get("processor", "generic")
    processor = ModelFactory.get_processor(processor_type, config)
    df_cache = processor.fetch_data()
    if df_cache is None or df_cache.empty:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)
    
    # Read Optuna settings from config
    opt_config = config.get("optuna", {})
    n_trials = opt_config.get("n_trials", args.trials)
    
    # 2. Optimization
    study_name = args.study_name or config.get("experiment_name", "optimization")
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    logger.info(f"Starting Optuna Optimization ({n_trials} trials, {args.epochs} epochs per trial, {args.n_jobs} parallel jobs)...")
    
    study.optimize(
        lambda trial: objective(
            trial, config, device, df_cache, epochs=args.epochs or config.get("epochs", 10)
        ), 
        n_trials=n_trials,
        n_jobs=args.n_jobs
    )
    
    logger.info("Optimization Complete.")
    logger.info(f"Best Trial: {study.best_trial.params}")
    logger.info(f"Best Score: {study.best_value}")
    
    # 3. Save Top K Params
    top_k = args.top_k
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    top_trials = sorted_trials[:top_k]
    
    logger.info(f"Saving top {len(top_trials)} configurations...")
    
    for rank, trial in enumerate(top_trials, start=1):
        trial_config = config.copy()
        trial_config.update(trial.params)
        
        base_path = config_path.replace("_optimized.json", ".json")
        
        if top_k == 1:
            save_config_path = base_path.replace(".json", "_optimized.json")
        else:
            save_config_path = base_path.replace(".json", f"_optimized_rank{rank}.json")
            
        with open(save_config_path, 'w') as f:
            json.dump(trial_config, f, indent=4)
        logger.info(f"  Rank {rank}: Score={trial.value:.4f} -> {save_config_path}")
    
if __name__ == "__main__":
    main()

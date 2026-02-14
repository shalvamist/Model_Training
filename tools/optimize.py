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



def objective(trial, config, device, X_dyn, X_stat, X_time, Y_1, Y_2, 
              train_end, val_end, Y_1_cls_pre=None, Y_2_cls_pre=None, epochs=10):
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

        if "layers" in opt_config:
            l_min, l_max = opt_config["layers"]
            n_layers = trial.suggest_int("n_layers", l_min, l_max)
            trial_config["lstm_layers"] = n_layers
            trial_config["trans_layers"] = n_layers

    # V19: Re-create sequences if seq_length was tuned
    trial_seq_len = trial_config.get('seq_length', config.get('seq_length', 60))
    base_seq_len = config.get('seq_length', 60)
    
    if trial_seq_len != base_seq_len:
        # Need to re-slice sequences with new seq_length
        trial_config_for_proc = config.copy()
        trial_config_for_proc['seq_length'] = trial_seq_len
        processor = ModelFactory.get_processor(trial_config_for_proc.get("processor", "generic"), trial_config_for_proc)
        df = processor.fetch_data()
        df_proc, dyn_cols, stat_cols = processor.process(df)
        
        # Handle variable return
        seq_data = processor.create_sequences(df_proc, dyn_cols, stat_cols)
        if len(seq_data) == 8:
             # Correct V20 Order: (X_dyn, X_stat, X_time, Y_1, Y_2, dates, Y1_cls, Y2_cls)
             t_X_dyn, t_X_stat, t_X_time, t_Y_1, t_Y_2, dates, t_Y1_cls, t_Y2_cls = seq_data
        else:
             t_X_dyn, t_X_stat, t_X_time, t_Y_1, t_Y_2, dates = seq_data
             t_Y1_cls, t_Y2_cls = None, None
        
        trial_config['input_dim'] = len(dyn_cols)
        trial_config['static_dim'] = len(stat_cols)
        
        n = len(t_X_dyn)
        train_years = config.get('train_years', 13)
        val_years = config.get('val_years', 2)
        test_years = config.get('test_years', 2)
        total_years = train_years + val_years + test_years
        t_train_end = int(n * (train_years / total_years))
        t_val_end = int(n * ((train_years + val_years) / total_years))
    else:
        t_X_dyn, t_X_stat, t_X_time, t_Y_1, t_Y_2 = X_dyn, X_stat, X_time, Y_1, Y_2
        t_train_end, t_val_end = train_end, val_end
        t_Y1_cls, t_Y2_cls = Y_1_cls_pre, Y_2_cls_pre
    
    # V19: Classification with configurable threshold if not pre-calc
    cls_threshold = trial_config.get('cls_threshold', 0.02)
    
    def classify_return(returns, thresh):
        labels = np.zeros(len(returns), dtype=np.int64)
        labels[returns > thresh] = 2
        labels[returns < -thresh] = 0
        labels[(returns >= -thresh) & (returns <= thresh)] = 1
        return labels
    
    Y_1_np = t_Y_1 if isinstance(t_Y_1, np.ndarray) else t_Y_1.numpy()
    Y_2_np = t_Y_2 if isinstance(t_Y_2, np.ndarray) else t_Y_2.numpy()
    
    if t_Y1_cls is not None:
        Y_1_cls = t_Y1_cls if isinstance(t_Y1_cls, np.ndarray) else t_Y1_cls
    else:
        Y_1_cls = classify_return(Y_1_np, cls_threshold)
        
    if t_Y2_cls is not None:
        Y_2_cls = t_Y2_cls if isinstance(t_Y2_cls, np.ndarray) else t_Y2_cls
    else:
        Y_2_cls = classify_return(Y_2_np, cls_threshold)
    
    batch_size = trial_config.get('batch_size', 32)
    
    train_data = TensorDataset(
        torch.tensor(t_X_dyn[:t_train_end], dtype=torch.float32) if isinstance(t_X_dyn, np.ndarray) else t_X_dyn[:t_train_end],
        torch.tensor(t_X_stat[:t_train_end], dtype=torch.float32) if isinstance(t_X_stat, np.ndarray) else t_X_stat[:t_train_end],
        torch.tensor(t_X_time[:t_train_end], dtype=torch.long) if isinstance(t_X_time, np.ndarray) else t_X_time[:t_train_end],
        torch.tensor(Y_1_np[:t_train_end], dtype=torch.float32),
        torch.tensor(Y_1_cls[:t_train_end], dtype=torch.long),
        torch.tensor(Y_2_np[:t_train_end], dtype=torch.float32),
        torch.tensor(Y_2_cls[:t_train_end], dtype=torch.long)
    )
    
    val_data = TensorDataset(
        torch.tensor(t_X_dyn[t_train_end:t_val_end], dtype=torch.float32) if isinstance(t_X_dyn, np.ndarray) else t_X_dyn[t_train_end:t_val_end],
        torch.tensor(t_X_stat[t_train_end:t_val_end], dtype=torch.float32) if isinstance(t_X_stat, np.ndarray) else t_X_stat[t_train_end:t_val_end],
        torch.tensor(t_X_time[t_train_end:t_val_end], dtype=torch.long) if isinstance(t_X_time, np.ndarray) else t_X_time[t_train_end:t_val_end],
        torch.tensor(Y_1_np[t_train_end:t_val_end], dtype=torch.float32),
        torch.tensor(Y_1_cls[t_train_end:t_val_end], dtype=torch.long),
        torch.tensor(Y_2_np[t_train_end:t_val_end], dtype=torch.float32),
        torch.tensor(Y_2_cls[t_train_end:t_val_end], dtype=torch.long)
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
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
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument("--top_k", type=int, default=3, help="Save top K best configurations")
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
    
    # Handle variable return length (V19=6, V20=8)
    seq_data = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    if len(seq_data) == 8:
        X_dyn, X_stat, X_time, Y_1, Y_2, dates, Y_1_cls_pre, Y_2_cls_pre = seq_data
    else:
        X_dyn, X_stat, X_time, Y_1, Y_2, dates = seq_data
        Y_1_cls_pre, Y_2_cls_pre = None, None
    
    if len(X_dyn) == 0:
        logger.error("No sequences.")
        sys.exit(1)
        
    # 3-Way Chronological Split
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
    
    # Read Optuna settings from config
    opt_config = config.get("optuna", {})
    n_trials = opt_config.get("n_trials", args.trials)
    
    # 2. Optimization
    study_name = args.study_name or config.get("experiment_name", "optimization")
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    logger.info(f"Starting Optuna Optimization ({n_trials} trials, {args.epochs} epochs per trial)...")
    
    study.optimize(
        lambda trial: objective(
            trial, config, device, X_dyn, X_stat, X_time, Y_1, Y_2, 
            train_end, val_end, Y_1_cls_pre, Y_2_cls_pre, epochs=args.epochs
        ), 
        n_trials=n_trials
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

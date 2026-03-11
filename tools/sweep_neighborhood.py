import os
import sys
import json
import logging
import argparse
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

NEIGHBORHOOD_BOUNDS = {
    # Typical hyper-local stabilization parameters for V23 Transformers
    "dropout_bounds": [0.05, 0.25],          # Narrower dropout for fine-tuning
    "lr_bounds": [1e-5, 5e-5],              # Very low learning rate
    "tail_penalty_bounds": [2.0, 6.0]       # Sharpe tuning
}

def create_neighborhood_sweep(base_config_path, output_dir, trials=20):
    if not os.path.exists(base_config_path):
        logger.error(f"Base config not found: {base_config_path}")
        return None

    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Inject tight neighborhood bounds into the optuna config
    if "optuna" not in config:
        config["optuna"] = {}
        
    config["optuna"]["dropout_bounds"] = NEIGHBORHOOD_BOUNDS["dropout_bounds"]
    config["optuna"]["lr_bounds"] = NEIGHBORHOOD_BOUNDS["lr_bounds"]
    config["optuna"]["tail_penalty_bounds"] = NEIGHBORHOOD_BOUNDS["tail_penalty_bounds"]
    
    # Do NOT sweep architecture capacity or anything else (Lock them)
    # Ensure they are removed from sweeps so they stay fixed
    keys_to_remove = ["hidden_size_bounds", "lstm_layers_bounds", "trans_layers_bounds", "batch_size_bounds"]
    for k in keys_to_remove:
         if k in config["optuna"]:
              del config["optuna"][k]
              
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(base_config_path).split('.')[0]
    sweep_name = f"{basename}_neighborhood_sweep"
    out_config = os.path.join(output_dir, f"{sweep_name}.json")
    
    with open(out_config, 'w') as f:
         json.dump(config, f, indent=4)
         
    return out_config

def main():
    parser = argparse.ArgumentParser(description="Hyper-Local Optuna Stabilizer")
    parser.add_argument("--config", type=str, required=True, help="Base JSON config of the successful Seed")
    parser.add_argument("--trials", type=int, default=20, help="Number of tight Optuna trials")
    parser.add_argument("--outdir", type=str, default="configs/sweeps", help="Where to save the sweep config")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    sweep_cfg = create_neighborhood_sweep(args.config, args.outdir, args.trials)
    
    if sweep_cfg:
        logger.info(f"Neighborhood Config Generated: {sweep_cfg}")
        logger.info("Launching Optimizer...")
        
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        optimize_script = os.path.join(script_dir, "tools", "optimize.py")
        
        cmd = [
            sys.executable, optimize_script,
            "--config", sweep_cfg,
            "--n_trials", str(args.trials),
            "--device", args.device
        ]
        
        try:
             subprocess.run(cmd, check=True)
             logger.info("[+] Neighborhood Optimization Completed.")
        except subprocess.CalledProcessError as e:
             logger.error(f"[X] Neighborhood Optimization Failed: {e}")

if __name__ == "__main__":
    main()

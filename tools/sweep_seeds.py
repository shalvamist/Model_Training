import os
import sys
import json
import logging
import argparse
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def sweep_seeds(base_config_path, num_seeds, output_dir, device="cuda"):
    if not os.path.exists(base_config_path):
        logger.error(f"Base config not found: {base_config_path}")
        return

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # 1. Enforce Titan Locks
    logger.info("Locking capacity to Titan Scale (hidden=1024, bs=32)")
    base_config["hidden_size"] = 1024
    base_config["batch_size"] = 32
    
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(base_config_path).split('.')[0]
    
    # 2. Sweep
    for seed in range(40, 40 + num_seeds):  # Standard deterministic sequence
        logger.info(f"--- Launching Sweep Trial with Seed {seed} ---")
        
        trial_config = base_config.copy()
        trial_config["random_seed"] = seed
        
        trial_name = f"{basename}_seed_{seed}"
        trial_config_path = os.path.join(output_dir, f"{trial_name}.json")
        
        with open(trial_config_path, 'w') as f:
            json.dump(trial_config, f, indent=4)
            
        script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        train_script = os.path.join(script_dir, "tools", "train.py")
        
        cmd = [
            sys.executable, train_script,
            "--config", trial_config_path,
            "--save", os.path.join(script_dir, "weights", trial_name, "model.pth"),
            "--device", device,
            "--seed", str(seed)
        ]
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            # We use a synchronous run here for simplicity, but in a real massive sweep 
            # this would dispatch to SLURM or background processes.
            subprocess.run(cmd, check=True)
            logger.info(f"[+] Trial {seed} Completed.")
        except subprocess.CalledProcessError as e:
            logger.error(f"[X] Trial {seed} Failed: {e}")
            
    logger.info("--- [DONE] Seed Sweep Completed ---")

def main():
    parser = argparse.ArgumentParser(description="Deterministic Seed Sweeper")
    parser.add_argument("--config", type=str, required=True, help="Base JSON config path")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to sweep")
    parser.add_argument("--outdir", type=str, default="configs/sweeps", help="Where to save generated trial configs")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    sweep_seeds(args.config, args.seeds, args.outdir, args.device)
    
if __name__ == "__main__":
    main()

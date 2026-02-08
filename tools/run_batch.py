import argparse
import os
import subprocess
import sys
import glob
import time

def main():
    parser = argparse.ArgumentParser(description="Batch Train Models from Config Directory")
    parser.add_argument("--config_dir", type=str, default="configs/strategies", help="Directory containing JSON configs")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for config files")
    parser.add_argument("--epochs", type=int, help="Override epochs for all models")
    parser.add_argument("--batch_size", type=int, help="Override batch_size for all models")
    parser.add_argument("--lr", type=float, help="Override learning rate for all models")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization instead of training")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials for optimization")
    parser.add_argument("--top_k", type=int, default=3, help="Save and train top K best models after optimization")
    parser.add_argument("--final_epochs", type=int, help="Epochs for final training of top K models (if --optimize is set)")
    args = parser.parse_args()
    
    config_dir = args.config_dir
    if not os.path.exists(config_dir):
        print(f"Error: Directory {config_dir} not found.")
        sys.exit(1)
        
    search_pattern = os.path.join(config_dir, args.pattern)
    config_files = glob.glob(search_pattern)
    
    if not config_files:
        print(f"No config files found matching {search_pattern}")
        sys.exit(1)
        
    print(f"Found {len(config_files)} configs to train:")
    for f in config_files:
        print(f" - {f}")
        
    results = {}
    optimized_configs = []  # Track optimized configs for final training
    
    start_time_all = time.time()
    
    # Path to tools for subprocesses
    # Assuming run from root, tools are in "tools/"
    # Check if we are in "tools/" or root
    if os.path.exists("train.py"):
        # We are in tools/
        prefix = ""
    elif os.path.exists("tools/train.py"):
        # We are in root
        prefix = "tools/"
    else:
        # Fallback
        prefix = "tools/"
    
    train_script = os.path.join(prefix, "train.py")
    optimize_script = os.path.join(prefix, "optimize.py")

    for i, config_file in enumerate(config_files):
        print(f"\n[{i+1}/{len(config_files)}] Starting Training: {config_file}")
        print("="*60)
        
        start_time = time.time()
        
        if args.optimize:
            print(f"   -> Mode: OPTIMIZE (Trials: {args.trials}, Top K: {args.top_k})")
            cmd = [sys.executable, optimize_script, "--config", config_file, "--trials", str(args.trials), "--top_k", str(args.top_k)]
            if args.epochs: cmd.extend(["--epochs", str(args.epochs)])
            if args.force_cpu: cmd.append("--force_cpu")
        else:
            # Run train.py as a subprocess
            cmd = [sys.executable, train_script, "--config", config_file]
            
            # Append overrides
            if args.epochs: cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size: cmd.extend(["--batch_size", str(args.batch_size)])
            if args.lr: cmd.extend(["--lr", str(args.lr)])
            if args.force_cpu: cmd.append("--force_cpu")
        
        try:
            # Stream output to console
            process = subprocess.run(cmd, check=True)
            elapsed = time.time() - start_time
            print(f"SUCCESS: {config_file} (Time: {elapsed:.2f}s)")
            results[config_file] = "SUCCESS"
            
            # If optimizing, track optimized configs for final training
            if args.optimize:
                base_name = config_file.replace(".json", "")
                
                # Check for single optimized file (top_k=1 case)
                single_opt_path = f"{base_name}_optimized.json"
                if os.path.exists(single_opt_path):
                    optimized_configs.append(single_opt_path)
                
                # Check for ranked files
                for rank in range(1, args.top_k + 1):
                    optimized_path = f"{base_name}_optimized_rank{rank}.json"
                    if os.path.exists(optimized_path):
                        optimized_configs.append(optimized_path)
            
        except subprocess.CalledProcessError:
            print(f"FAILURE: {config_file}")
            results[config_file] = "FAILED"
    
    # Final training phase: Train all top K models if optimization was run
    if args.optimize and args.final_epochs and optimized_configs:
        print("\n" + "="*60)
        print(f"FINAL TRAINING PHASE: Training {len(optimized_configs)} optimized models with {args.final_epochs} epochs")
        print("="*60)
        
        for i, opt_config in enumerate(optimized_configs):
            print(f"\n[{i+1}/{len(optimized_configs)}] Final Training: {opt_config}")
            print("="*60)
            
            start_time = time.time()
            cmd = [sys.executable, train_script, "--config", opt_config, "--epochs", str(args.final_epochs)]
            if args.force_cpu: cmd.append("--force_cpu")
            
            try:
                process = subprocess.run(cmd, check=True)
                elapsed = time.time() - start_time
                print(f"SUCCESS: {opt_config} (Time: {elapsed:.2f}s)")
                results[opt_config] = "SUCCESS (FINAL)"
            except subprocess.CalledProcessError:
                print(f"FAILURE: {opt_config}")
                results[opt_config] = "FAILED (FINAL)"
            
    print("\n" + "="*60)
    print("BATCH TRAINING SUMMARY")
    print("="*60)
    
    success_count = sum(1 for status in results.values() if "SUCCESS" in status)
    total_elapsed = time.time() - start_time_all
    
    for f, status in results.items():
        print(f"{status}: {os.path.basename(f)}")
        
    print(f"\nTotal Success: {success_count}/{len(results)}")
    print(f"Total Time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()

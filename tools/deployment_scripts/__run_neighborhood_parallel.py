import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def main():
    base_config_path = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy\v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04.json"
    
    if not os.path.exists(base_config_path):
        print(f"Error: Could not find base config at {base_config_path}")
        return

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # We know the massive 1024 capacity works best with a lower dropout (0.25). 
    # Now we will test variations on the actual model depth (Transformer and LSTM layers).
    # Base structure is: trans: 3, lstm: 4
    variations = [
        {"name": "v23_alpha_gen_arch_sweep_trans_2", "updates": {"dropout": 0.25, "trans_layers": 2}},
        {"name": "v23_alpha_gen_arch_sweep_trans_4", "updates": {"dropout": 0.25, "trans_layers": 4}},
        {"name": "v23_alpha_gen_arch_sweep_lstm_3", "updates": {"dropout": 0.25, "lstm_layers": 3}},
        {"name": "v23_alpha_gen_arch_sweep_lstm_5", "updates": {"dropout": 0.25, "lstm_layers": 5}}
    ]

    configs_to_run = []
    config_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\configs"

    print("Generating dropout-focused neighborhood configurations...")
    for var in variations:
        new_config = base_config.copy()
        new_config.update(var["updates"])
        
        new_path = os.path.join(config_dir, var["name"] + ".json")
        with open(new_path, 'w') as f:
            json.dump(new_config, f, indent=4)
        configs_to_run.append(new_path)
        print(f" -> Created: {var['name']}.json")

    # Generate exactly 25 seeds for each variant (4 variants * 25 seeds = 100 training runs)
    random.seed(77777)
    seeds = random.sample(range(10000, 99999), 25)

    commands = []
    for config in configs_to_run:
        for seed in seeds:
            cmd = [sys.executable, "tools/train.py", "--config", config, "--seed", str(seed)]
            commands.append((cmd, config, seed))

    print(f"\nTotal runs scheduled: {len(commands)} (4 configs * 25 seeds)")
    print(f"Starting parallel execution with 4 concurrent tasks...\n")

    def run_training(cmd_info):
        cmd, config, seed = cmd_info
        config_name = os.path.basename(config).replace('.json', '')
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"SUCCESS: {config_name} initialized with seed {seed}"
        except subprocess.CalledProcessError:
            return f"FAILED: {config_name} with seed {seed}"

    # Execute 4 instances in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_training, cmd): cmd for cmd in commands}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{i}/{len(commands)}] {result}")
            
    print("\nAll 100 dropout-sweep runs have completed!")

if __name__ == "__main__":
    main()

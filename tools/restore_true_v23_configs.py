import json
import os

configs = [
    "configs/v23_alpha_generation.json",
    "configs/v23_bear_headhunter.json",
    "configs/v23_fortress_master.json",
    "configs/v23_sector_alpha.json"
]

for filepath in configs:
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}, not found.")
        continue
        
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Reverting to true Deep V23 baselines
    data["seq_length"] = 60
    data["epochs"] = 150
    data["early_stopping_patience"] = 20
    
    # Remove hallucinated n_experts
    data.pop("n_experts", None)
    
    if "optuna" in data:
        data["optuna"]["hidden_size"] = [128, 256, 512, 1024]
        data["optuna"]["dropout"] = [0.1, 0.4]
        data["optuna"]["lstm_layers"] = [2, 6]
        data["optuna"]["trans_layers"] = [1, 6]  # CRITICAL FIX
        data["optuna"]["batch_size"] = [32, 64, 128]
        data["optuna"]["lr"] = [1e-05, 0.005]
        data["optuna"].pop("n_experts", None)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully restored Deep V23 configuration for {filepath}")

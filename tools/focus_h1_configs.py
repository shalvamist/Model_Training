import os
import json

configs = [
    "configs/v23_alpha_generation.json",
    "configs/v23_bear_headhunter.json",
    "configs/v23_fortress_master.json",
    "configs/v23_sector_alpha.json"
]

for cfg_path in configs:
    # Use absolute path relative to Model_trading_training
    full_path = os.path.join(r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training", cfg_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Inject the new aggressive H1 weighting paradigm
        data["h1_weight"] = 0.9
        data["h2_weight"] = 0.1
        
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully tuned {cfg_path} for 7-Day Optimization (h1_weight=0.9)")
    else:
        print(f"ERROR: Could not find {full_path}")

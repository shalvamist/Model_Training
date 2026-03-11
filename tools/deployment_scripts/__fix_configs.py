import json
import os

base_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\configs"
deploy_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy"

# 1. Base files and their respective trans_layers count based on the original run
base_files = {
    "v23_alpha_generation.json": 3,
    "v23_bear_headhunter.json": 2,
    "v23_fortress_master.json": 3,
    "v23_sector_alpha.json": 4
}

for b_file, t_layers in base_files.items():
    p = os.path.join(base_dir, b_file)
    with open(p, 'r') as f:
        data = json.load(f)
    
    # Add trans layers to root
    data["trans_layers"] = t_layers
    
    # Add horizon weights to optuna to carry over to future optimizations
    if "optuna" in data:
        data["optuna"]["h1_weight"] = [0.7]
        data["optuna"]["h2_weight"] = [0.3]
        
    with open(p, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated Base Config: {b_file}")

# 2. Deploy files
deploy_files = [
    "v23_alpha_generation_optimized_rank1.json",
    "v23_bear_headhunter_optimized_rank1.json",
    "v23_fortress_master_optimized_rank1.json",
    "v23_sector_alpha_optimized_rank1.json",
    "v23_sector_alpha_optimized_rank4.json"
]

for d_file in deploy_files:
    p = os.path.join(deploy_dir, d_file)
    if not os.path.exists(p):
        print(f"WARNING: Deploy file not found -> {p}")
        continue
    with open(p, 'r') as f:
        data = json.load(f)
    
    # Inject missing horizon weights
    data["h1_weight"] = 0.7
    data["h2_weight"] = 0.3
        
    with open(p, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated Deploy Config: {d_file}")

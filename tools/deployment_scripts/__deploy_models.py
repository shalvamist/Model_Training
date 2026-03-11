import os
import shutil

base_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training"
weights_dir = os.path.join(base_dir, "weights")
deploy_dir = os.path.join(weights_dir, "V23_Deploy")

os.makedirs(deploy_dir, exist_ok=True)

models_to_move = [
    ("v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04_seed_84667", "PROTECTED_H1_MONSTER_118SHARPE_2026_03_07", "v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04.json"),
    ("v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04_seed_51807", "PROTECTED_H1_WINRATE_104SHARPE_2026_03_07", "v23_alpha_generation_rank5_PROTECTED_CAPACITY_BRUTE_58PERCENT_2026_03_04.json")
]

for old_name, new_suffix, config_name in models_to_move:
    # Build new name
    new_base = "v23_alpha_generation_rank5_" + old_name.split("_seed_")[-1] 
    new_name = f"{new_base}_{new_suffix}"
    
    # 1. Copy Config
    old_config = os.path.join(deploy_dir, config_name)
    if not os.path.exists(old_config):
        old_config = os.path.join(base_dir, "configs", config_name)
        
    new_config = os.path.join(deploy_dir, f"{new_name}.json")
    if os.path.exists(old_config):
        shutil.copy(old_config, new_config)
        print(f"Copied config: {old_config} -> {new_config}")
    else:
        print(f"MISSING CONFIG: {old_config}")
        
    # 2. Copy Weights Dir
    old_weight_dir = os.path.join(weights_dir, old_name)
    new_weight_dir = os.path.join(deploy_dir, new_name)
    if os.path.exists(old_weight_dir):
        if os.path.exists(new_weight_dir):
            shutil.rmtree(new_weight_dir)
        shutil.copytree(old_weight_dir, new_weight_dir)
        print(f"Copied weight dir: {old_weight_dir} -> {new_weight_dir}")
    else:
        print(f"MISSING WEIGHT DIR: {old_weight_dir}")

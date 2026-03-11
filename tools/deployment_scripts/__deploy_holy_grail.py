import os
import shutil

base_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training"
weights_dir = os.path.join(base_dir, "weights")
config_dir = os.path.join(base_dir, "configs")
deploy_dir = os.path.join(weights_dir, "V23_Deploy")

os.makedirs(deploy_dir, exist_ok=True)

models_to_move = [
    ("v23_alpha_gen_arch_sweep_lstm_3_seed_68302", "PROTECTED_HOLY_GRAIL_87PERCENT_2026_03_07", "v23_alpha_gen_arch_sweep_lstm_3.json"),
    ("v23_alpha_gen_arch_sweep_lstm_3_seed_69731", "PROTECTED_HOLY_GRAIL_83PERCENT_2026_03_07", "v23_alpha_gen_arch_sweep_lstm_3.json")
]

for old_name, new_suffix, config_name in models_to_move:
    # Build new name
    new_base = "v23_alpha_gen_lstm3_" + old_name.split("_seed_")[-1] 
    new_name = f"{new_base}_{new_suffix}"
    
    # 1. Copy Config
    old_config = os.path.join(config_dir, config_name)
    if not os.path.exists(old_config):
        print(f"MISSING CONFIG: {old_config}")
        continue
        
    new_config = os.path.join(deploy_dir, f"{new_name}.json")
    shutil.copy(old_config, new_config)
    print(f"Copied config: {old_config} -> {new_config}")
        
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

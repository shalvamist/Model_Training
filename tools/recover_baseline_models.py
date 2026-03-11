import os
import subprocess
import json
import shutil

root = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training"
deploy_dir = os.path.join(root, "weights", "V23_Deploy")
weights_dir = os.path.join(root, "weights")

models = [
    "v23_alpha_generation_optimized_rank1",
    "v23_bear_headhunter_optimized_rank1",
    "v23_fortress_master_optimized_rank1",
    "v23_sector_alpha_optimized_rank1"
]

for model in models:
    json_path = os.path.join(deploy_dir, f"{model}.json")
    if not os.path.exists(json_path):
        print(f"Could not find {json_path}")
        continue
        
    print(f"Found preserved configuration for {model}.")
    with open(json_path, 'r') as f:
        config = json.load(f)
        
    config['h1_weight'] = 0.7
    config['h2_weight'] = 0.3
    
    # We will train these into an exact-recreation folder
    model_folder = os.path.join(weights_dir, f"{model}_exact_recreation")
    os.makedirs(model_folder, exist_ok=True)
    
    # train.py uses the config filename as the experiment name, so we name it exactly the target name
    temp_config_name = f"{model}_exact.json"
    new_config_path = os.path.join(model_folder, temp_config_name)
    
    with open(new_config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Training exactly on optimal parameters for {model}...")
    cmd = [
        r"C:\projects\QQQ_Reg_Models_v2\venv\Scripts\python.exe",
        os.path.join(root, "tools", "train.py"),
        "--config", new_config_path
    ]
    subprocess.run(cmd, check=True)
    
    # Move the spawned weights/model_exact.pth -> weights/model_exact_recreation/model.pth
    spawned_pth = os.path.join(weights_dir, f"{model}_exact.pth")
    if os.path.exists(spawned_pth):
        shutil.move(spawned_pth, os.path.join(model_folder, "model.pth"))
        
    # We must also rename the config file to `config.json` so evaluate_v23.py can parse the folder
    final_config_path = os.path.join(model_folder, "config.json")
    if os.path.exists(new_config_path):
        os.rename(new_config_path, final_config_path)

print("\nExact Parametric Retraining Complete.")

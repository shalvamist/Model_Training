import os
import shutil
import datetime

deploy_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy"
weights_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights"

models = {
    "v23_bear_headhunter_optimized_rank4": "v23_bear_headhunter_rank4_PROTECTED_CRASH_CATCHER",
    "v23_bear_headhunter_optimized_rank2": "v23_bear_headhunter_rank2_PROTECTED_HIGH_ACCURACY"
}

date_str = datetime.datetime.now().strftime('%Y_%m_%d')

for src_name, protected_name in models.items():
    src_path = os.path.join(weights_dir, src_name)
    dest_path = os.path.join(deploy_dir, f"{protected_name}_{date_str}")
    
    if os.path.exists(src_path):
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(src_path, dest_path)
        print(f"Copied {src_name} to -> {os.path.basename(dest_path)}")
        
        # also copy the primary config file into the deploy folder so evaluate_all can pick it up
        dest_config_path = os.path.join(deploy_dir, f"{protected_name}_{date_str}.json")
        src_config_file = os.path.join(src_path, "config.json")
        
        if os.path.exists(src_config_file):
            shutil.copy2(src_config_file, dest_config_path)
            print(f"Copied root config -> {os.path.basename(dest_config_path)}")
            
    else:
        print(f"Warning: {src_path} not found.")

print("Finished deploying Bear Headhunter models.")

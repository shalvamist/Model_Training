import os
import shutil
import datetime

deploy_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy"
weights_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights"

models_to_move = [
    "v23_alpha_generation_optimized_rank1",
    "v23_bear_headhunter_optimized_rank1",
    "v23_fortress_master_optimized_rank1",
    "v23_sector_alpha_optimized_rank1"
]

date_suffix = datetime.datetime.now().strftime("%Y_%m_%d")

for model in models_to_move:
    # Source is the exact recreation folder we just made
    src_dir = os.path.join(weights_dir, f"{model}_exact_recreation")
    # Dest is inside V23_Deploy with the date appended
    dest_dir = os.path.join(deploy_dir, f"{model}_{date_suffix}")
    
    if os.path.exists(src_dir):
        print(f"Moving {src_dir} \n -> {dest_dir}")
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
    else:
        print(f"Warning: Could not find {src_dir}")

print("Successfully deployed models to V23_Deploy.")

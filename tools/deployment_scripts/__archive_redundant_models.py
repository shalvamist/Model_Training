import os
import shutil

deploy_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy"
archive_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Archive"

os.makedirs(archive_dir, exist_ok=True)

# The 6 Apex models must NEVER be touched
apex_models = [
    "v23_alpha_gen_lstm3_68302_PROTECTED_HOLY_GRAIL_87PERCENT_2026_03_07",
    "v23_alpha_generation_rank5_84667_PROTECTED_H1_MONSTER_118SHARPE_2026_03_07",
    "v23_sector_alpha_rank4_seed_25613_PROTECTED_SEED_SWEEP_135PERCENT_2026_03_06",
    "v23_sector_alpha_rank5_seed_88778_PROTECTED_ROTATIONAL_SWEEP_55PERCENT_2026_03_06",
    "v23_fortress_master_rank5_PROTECTED_UNIFIED_LONG_2026_03_04",
    "v23_bear_headhunter_rank6_PROTECTED_CRASH_CATCHER_PRIMARY_2026_03_04"
]

models_archived = 0

# Get list of items in deploy directory
items = os.listdir(deploy_dir)

# First pass: collect bases
all_bases = set()
for item in items:
    if item.endswith(".json"):
        all_bases.add(item[:-5]) # remove .json
    elif os.path.isdir(os.path.join(deploy_dir, item)):
        all_bases.add(item)
        
for base in all_bases:
    if base not in apex_models and "summary" not in base:
        # Move both dir and json
        dir_path = os.path.join(deploy_dir, base)
        json_path = os.path.join(deploy_dir, f"{base}.json")
        
        if os.path.exists(dir_path):
            shutil.move(dir_path, os.path.join(archive_dir, base))
            models_archived += 1
            print(f"Archived dir: {base}")
            
        if os.path.exists(json_path):
            shutil.move(json_path, os.path.join(archive_dir, f"{base}.json"))

print(f"\nArchived {models_archived} redundant model instances to `V23_Archive`.")
print("The V23_Deploy directory is now perfectly clean.")

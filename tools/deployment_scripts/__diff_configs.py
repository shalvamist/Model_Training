import json

path_old = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\V23_Deploy\v23_sector_alpha_optimized_rank1.json"
path_new = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training\configs\v23_sector_alpha.json"

with open(path_old) as f:
    old = json.load(f)
with open(path_new) as f:
    new = json.load(f)

print("=== DIFFERENCES: OLD DEPLOY vs NEW BASE (SECTOR ALPHA) ===")

for k in old:
    if k not in new:
        print(f"Key '{k}' is in OLD but not NEW")
    elif old[k] != new[k]:
        print(f"Key '{k}' differs:\nOLD: {old[k]}\nNEW: {new[k]}\n")

for k in new:
    if k not in old:
         print(f"Key '{k}' is in NEW but not OLD")

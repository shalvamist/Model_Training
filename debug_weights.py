import torch
import os
import sys

weights_path = r"c:\projects\QQQ_Reg_Models_v2\Model_trading_training\weights\v22_real_economy_pilot_optimized_rank1\model.pth"

if not os.path.exists(weights_path):
    print(f"File not found: {weights_path}")
    sys.exit(1)

try:
    state_dict = torch.load(weights_path, map_location='cpu')
    print("Keys found:", len(state_dict.keys()))
    
    if 'revin.affine_weight' in state_dict:
        print(f"revin.affine_weight shape: {state_dict['revin.affine_weight'].shape}")
    else:
        print("revin.affine_weight NOT found in state_dict")
        
    if 'input_proj.weight' in state_dict:
        print(f"input_proj.weight shape: {state_dict['input_proj.weight'].shape}")
        
except Exception as e:
    print(f"Error loading weights: {e}")

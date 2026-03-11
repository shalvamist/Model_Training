import os
import json

configs = [
    "configs/v23_alpha_generation.json",
    "configs/v23_bear_headhunter.json",
    "configs/v23_fortress_master.json",
    "configs/v23_sector_alpha.json"
]

base_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training"

for cfg_path in configs:
    full_path = os.path.join(base_dir, cfg_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            data = json.load(f)

        # 1. Truncate Sequence Memory
        data["seq_length"] = 30
        
        # 2. Restore Dual-Horizon Loss Weights to Baseline
        data["h1_weight"] = 0.7
        data["h2_weight"] = 0.3
        
        # 3. Modify Feature Pipeline
        pipeline = data.get("feature_pipeline", [])
        
        # A. Add 1 and 2 to log_return lags
        for step in pipeline:
            if step.get("step") == "log_return" and step.get("output") == "log_ret":
                lags = step.get("lags", [])
                for lag in [1, 2]:
                    if lag not in lags:
                        lags.append(lag)
                lags.sort()
                step["lags"] = lags
                break
                
        # B. Inject Micro-Feature Steps if not present
        new_steps = [
            {
                "step": "technical_indicator",
                "indicator": "rsi",
                "period": 3,
                "output": "rsi_3"
            },
            {
                "step": "bollinger_position",
                "window": 5,
                "output": "bb_pos_5"
            },
            {
                "step": "close_vs_ma",
                "window": 3,
                "output": "close_vs_ma_3"
            },
            {
                "step": "normalize",
                "columns": ["close_vs_ma_3"],
                "window": 30,
                "suffix": "_norm"
            }
        ]
        
        # Check if they exist to avoid duplicates
        existing_steps = [s.get("output", "") for s in pipeline]
        if "rsi_3" not in existing_steps:
            pipeline.extend(new_steps)
            data["feature_pipeline"] = pipeline
            
        # 4. Add to Dynamic Columns
        dyn_cols = data.get("dynamic_columns", [])
        new_dyn = ["log_ret_1", "log_ret_2", "rsi_3", "bb_pos_5", "close_vs_ma_3_norm"]
        for c in new_dyn:
            if c not in dyn_cols:
                dyn_cols.append(c)
        data["dynamic_columns"] = dyn_cols
        
        # 5. Update Input Dim
        data["input_dim"] = len(dyn_cols)
        
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Successfully injected Micro-Features into {cfg_path}. Input dim: {data['input_dim']}, Seq_len: 30")
    else:
        print(f"ERROR: Could not find {full_path}")

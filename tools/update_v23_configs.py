import json

def update_config(filepath, strategy):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # 1. Global Updates
    data["seq_length"] = 120
    data["epochs"] = 300
    data["early_stopping_patience"] = 40
    
    # Global Optuna Bounds
    data["optuna"]["hidden_size"] = [256, 512, 1024, 2048]
    data["optuna"]["dropout"] = [0.2, 0.6]
    data["optuna"]["tail_penalty_factor"] = [2.0, 15.0]
    data["optuna"]["tail_threshold"] = [-0.05, -0.01]
    
    # 2. Strategy Specific Inputs
    if strategy == "sector_alpha":
        sectors = ["xlk", "xly", "xlp", "xlv", "xlf"]
        for s in sectors:
            data["feature_pipeline"].append({
                "step": "log_return",
                "input": s.lower(),
                "output": f"{s.lower()}_ret"
            })
            
        data["feature_pipeline"].append({
            "step": "rolling_correlation",
            "input1": "xly_ret",
            "input2": "xlp_ret",
            "window": 20,
            "output": "corr_xly_xlp_20"
        })
        data["feature_pipeline"].append({
            "step": "rolling_correlation",
            "input1": "xlk_ret",
            "input2": "xlv_ret",
            "window": 20,
            "output": "corr_xlk_xlv_20"
        })

        norm_step = next(s for s in data["feature_pipeline"] if s["step"] == "normalize")
        norm_step["columns"].extend([f"{s}_ret" for s in sectors] + ["corr_xly_xlp_20", "corr_xlk_xlv_20"])
        data["dynamic_columns"].extend([f"{s}_ret_norm" for s in sectors] + ["corr_xly_xlp_20_norm", "corr_xlk_xlv_20_norm"])
        data["input_dim"] = len(data["dynamic_columns"])

    elif strategy == "bear_headhunter":
        bears = ["sqqq", "shy", "vix"]
        for b in ["sqqq", "shy"]:
            data["feature_pipeline"].append({
                "step": "log_return",
                "input": b.lower(),
                "output": f"{b.lower()}_ret"
            })
            
        data["feature_pipeline"].append({
            "step": "rolling_correlation",
            "input1": "log_ret",
            "input2": "sqqq_ret",
            "window": 20,
            "output": "corr_qqq_sqqq_20"
        })
        
        norm_step = next(s for s in data["feature_pipeline"] if s["step"] == "normalize")
        norm_step["columns"].extend([f"{b}_ret" for b in ["sqqq", "shy"]] + ["corr_qqq_sqqq_20"])
        data["dynamic_columns"].extend([f"{b}_ret_norm" for b in ["sqqq", "shy"]] + ["corr_qqq_sqqq_20_norm"])
        data["input_dim"] = len(data["dynamic_columns"])

    else:
        # Just update input dim just in case
        data["input_dim"] = len(data["dynamic_columns"])

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Updated {filepath} (Input Dim: {data['input_dim']})")

if __name__ == "__main__":
    import os
    os.chdir(r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training")
    
    update_config("configs/v23_sector_alpha.json", "sector_alpha")
    update_config("configs/v23_bear_headhunter.json", "bear_headhunter")
    update_config("configs/v23_fortress_master.json", "fortress_master")
    update_config("configs/v23_alpha_generation.json", "alpha_generation")
    
    # Also fix the previous tail_threshold on alpha_generation and sector_alpha
    # and bear_headhunter out of the optuna block. It needs to be present
    for conf in ["configs/v23_sector_alpha.json", "configs/v23_bear_headhunter.json", "configs/v23_alpha_generation.json"]:
        with open(conf, 'r') as f:
            c = json.load(f)
            c["tail_threshold"] = -0.05
            c["tail_penalty_factor"] = 5.0
        with open(conf, 'w') as f:
            json.dump(c, f, indent=4)

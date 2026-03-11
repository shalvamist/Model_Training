import json
import os

def fix_pipeline(filepath, strategy):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # We will reconstruct the feature_pipeline completely so that `normalize` is guaranteed to be last
    base_pipeline = [
        {"step": "log_return", "input": "underlying_close", "output": "log_ret", "lags": [3, 5, 10, 20]},
        {"step": "technical_indicator", "indicator": "atr", "period": 5, "output": "atr_5"},
        {"step": "technical_indicator", "indicator": "atr", "period": 20, "output": "atr_20"},
        {"step": "arithmetic", "formula": "atr_5 / (atr_20 + 1e-6)", "output": "atr_ratio"},
        {"step": "rolling_correlation", "input1": "log_ret", "input2": "tlt", "window": 20, "output": "corr_qqq_tlt_20"},
        {"step": "fourier", "input": "log_ret", "window": 60, "output": "log_ret_entropy_60"},
        {"step": "wavelet", "input": "log_ret", "widths": [2, 5, 10], "prefix": "log_ret_w"},
        {"step": "greeks", "col_S": "underlying_close", "col_K": "strike", "col_r": "us_treasury_10y", "col_sigma": "vix", "T_years": 0.08, "rate_pct": True}
    ]
    
    norm_columns = [
        "us_treasury_10y", "vix", "atr_ratio", "corr_qqq_tlt_20", "log_ret_entropy_60",
        "log_ret_w_2", "log_ret_w_5", "log_ret_w_10", "delta", "gamma"
    ]
    
    dynamic_cols = [
        "log_ret", "log_ret_3", "log_ret_5", "log_ret_10", "log_ret_20", "atr_ratio_norm",
        "corr_qqq_tlt_20_norm", "log_ret_entropy_60_norm", "log_ret_w_2_norm", "log_ret_w_5_norm",
        "log_ret_w_10_norm", "us_treasury_10y_norm", "vix_norm", "delta_norm", "gamma_norm",
        "rsi_abs", "vix_abs", "vix_slope_5", "vix_accel_5", "crash_delta", "crash_gamma"
    ]
    
    if strategy == "sector_alpha":
        sectors = ["xlk", "xly", "xlp", "xlv", "xlf"]
        for s in sectors:
            base_pipeline.append({"step": "log_return", "input": s, "output": f"{s}_ret"})
            norm_columns.append(f"{s}_ret")
            dynamic_cols.append(f"{s}_ret_norm")
            
        base_pipeline.append({"step": "rolling_correlation", "input1": "xly_ret", "input2": "xlp_ret", "window": 20, "output": "corr_xly_xlp_20"})
        base_pipeline.append({"step": "rolling_correlation", "input1": "xlk_ret", "input2": "xlv_ret", "window": 20, "output": "corr_xlk_xlv_20"})
        norm_columns.extend(["corr_xly_xlp_20", "corr_xlk_xlv_20"])
        dynamic_cols.extend(["corr_xly_xlp_20_norm", "corr_xlk_xlv_20_norm"])
        
    elif strategy == "bear_headhunter":
        bears = ["sqqq", "shy"]
        for b in bears:
            base_pipeline.append({"step": "log_return", "input": b, "output": f"{b}_ret"})
            norm_columns.append(f"{b}_ret")
            dynamic_cols.append(f"{b}_ret_norm")
            
        base_pipeline.append({"step": "rolling_correlation", "input1": "log_ret", "input2": "sqqq_ret", "window": 20, "output": "corr_qqq_sqqq_20"})
        norm_columns.append("corr_qqq_sqqq_20")
        dynamic_cols.append("corr_qqq_sqqq_20_norm")

    # Add the normalize step LAST!
    base_pipeline.append({
        "step": "normalize",
        "columns": norm_columns,
        "window": 60,
        "suffix": "_norm"
    })
    
    data["feature_pipeline"] = base_pipeline
    data["dynamic_columns"] = dynamic_cols
    data["input_dim"] = len(dynamic_cols)
    
    # Ensure correct optuna settings
    data["seq_length"] = 120
    data["epochs"] = 300
    data["early_stopping_patience"] = 40
    data["optuna"]["hidden_size"] = [256, 512, 1024, 2048]
    data["optuna"]["dropout"] = [0.2, 0.6]
    data["optuna"]["tail_penalty_factor"] = [2.0, 15.0]
    data["optuna"]["tail_threshold"] = [-0.05, -0.01]
    
    data["tail_threshold"] = -0.05
    data["tail_penalty_factor"] = 5.0
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Fixed {filepath} - New Input Dim: {data['input_dim']}")

if __name__ == "__main__":
    os.chdir(r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training")
    fix_pipeline("configs/v23_sector_alpha.json", "sector_alpha")
    fix_pipeline("configs/v23_bear_headhunter.json", "bear_headhunter")
    fix_pipeline("configs/v23_fortress_master.json", "fortress_master")
    fix_pipeline("configs/v23_alpha_generation.json", "alpha_generation")

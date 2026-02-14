import argparse
import json
import logging
import sys
import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report

# Add root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from library.factory import ModelFactory
from library.utils import MetricsCalculator

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Trading Model")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weights file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV predictions")
    parser.add_argument("--report_file", type=str, default="evaluation_report.txt", help="Output Report File")
    parser.add_argument("--metrics_json", type=str, default="evaluation_metrics.json", help="Output Metrics JSON")
    args = parser.parse_args()
    
    logger.info(f"Loading Config: {args.config}")
    
    # 1. Load Model
    model = ModelFactory.load_model_from_weights(args.config, args.weights, device=args.device)
    
    # 2. Load Data Config
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # 3. Get Processed Data
    logger.info("Preparing Data...")
    # Prefer 'processor' key (new), fallback to 'processor_version' (old), default to 'generic'
    proc_type = config.get('processor', config.get('processor_version', 'generic'))
    processor = ModelFactory.get_processor(proc_type, config)
    df = processor.fetch_data()
    df_proc, dyn_cols, stat_cols = processor.process(df)
    data = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    # Handle variable return length (V19=6, V20=8)
    if len(data) == 8:
        X_d, X_s, X_t, Y_1, Y_2, dates, Y_1_cls_pre, Y_2_cls_pre = data
        logger.info("Using Pre-Calculated Classification Targets (V20 Triple Barrier)")
    else:
        X_d, X_s, X_t, Y_1, Y_2, dates = data
        Y_1_cls_pre = None
        Y_2_cls_pre = None
    
    # Use Last 20% for Evaluation (Test Set)
    split_idx = int(len(X_d) * 0.8)
    
    X_d_test = torch.tensor(X_d[split_idx:], dtype=torch.float32).to(args.device)
    X_s_test = torch.tensor(X_s[split_idx:], dtype=torch.float32).to(args.device)
    X_t_test = torch.tensor(X_t[split_idx:], dtype=torch.long).to(args.device)
    
    # 4. Inference
    logger.info("Running Inference...")
    with torch.no_grad():
        outputs = model(X_d_test, X_s_test, X_t_test)
        
    # Handle V19 6-output vs legacy 4-output
    if len(outputs) == 6:
        r1, c1, r2, c2, d1, d2 = outputs
        # V19: Also compute directional head accuracy
        dir_preds = torch.sigmoid(d1).cpu().numpy().flatten()
    else:
        r1, c1, r2, c2 = outputs
        dir_preds = None
        
    p1_reg = r1.cpu().numpy()
    p2_reg = r2.cpu().numpy()
    
    # Handle Quantile Output (Select Median 0.5)
    if p1_reg.ndim > 1 and p1_reg.shape[1] > 1:
        # Assuming quantiles are sorted or we use the middle one
        # For [0.1, 0.5, 0.9], median is index 1
        # Check config for quantiles
        qs = config.get('quantiles', [0.1, 0.5, 0.9])
        if 0.5 in qs:
            idx = qs.index(0.5)
            p1_reg = p1_reg[:, idx]
        else:
            # Fallback: take middle
            p1_reg = p1_reg[:, p1_reg.shape[1] // 2]
    else:
        p1_reg = p1_reg.flatten()
        
    if p2_reg.ndim > 1 and p2_reg.shape[1] > 1:
        qs = config.get('quantiles', [0.1, 0.5, 0.9])
        if 0.5 in qs:
            idx = qs.index(0.5)
            p2_reg = p2_reg[:, idx]
        else:
            p2_reg = p2_reg[:, p2_reg.shape[1] // 2]
    else:
        p2_reg = p2_reg.flatten()

    p1_cls = torch.argmax(c1, dim=1).cpu().numpy()
    p2_cls = torch.argmax(c2, dim=1).cpu().numpy()
    
    # Targets for Report
    t1_raw = Y_1[split_idx:]
    t2_raw = Y_2[split_idx:]
    
    if Y_1_cls_pre is not None:
        # Use V20 Pre-calculated labels (Triple Barrier)
        t1_cls = Y_1_cls_pre[split_idx:]
        t2_cls = Y_2_cls_pre[split_idx:]
    else:
        # Derive from V19 Regression Targets
        cls_threshold = config.get('cls_threshold', 0.02)
        
        t1_cls = np.ones_like(t1_raw, dtype=int)
        t1_cls[t1_raw > cls_threshold] = 2; t1_cls[t1_raw < -cls_threshold] = 0
        
        t2_cls = np.ones_like(t2_raw, dtype=int)
        h2_threshold = cls_threshold * 3  # Wider threshold for longer horizon
        t2_cls[t2_raw > h2_threshold] = 2; t2_cls[t2_raw < -h2_threshold] = 0
    
    # 5. Metrics Calculation
    
    # Extract Daily Returns for Simulation (Fix for Overlapping Returns Bug)
    # df_proc has 'log_ret'. We need to slice it to match X_d_test.
    # X_d starts at seq_len. split_idx is relative to X_d.
    # So test set indices in df_proc are: seq_len + split_idx : end
    seq_len = config.get('seq_length', 60)
    
    # Ensure log_ret exists
    if 'log_ret' in df_proc.columns:
        daily_rets = df_proc['log_ret'].values
    else:
        # Fallback calculation
        daily_rets = np.log(df_proc['underlying_close'] / df_proc['underlying_close'].shift(1)).fillna(0).values
        
    start_idx = seq_len + split_idx
    # Ensure we don't go out of bounds (though sequences len check handles this)
    if start_idx < len(daily_rets):
        test_daily_rets = daily_rets[start_idx : start_idx + len(X_d_test)]
    else:
        test_daily_rets = np.zeros(len(X_d_test))
        
    # H1 Metrics
    metrics_h1_cls = MetricsCalculator.get_classification_metrics(t1_cls, p1_cls)
    metrics_h1_reg = MetricsCalculator.get_regression_metrics(t1_raw, p1_reg)
    metrics_h1_dir = MetricsCalculator.get_directional_accuracy(t1_raw, p1_reg)
    # Use DAILY returns for simulation, but steered by H1 predictions
    sim_h1 = MetricsCalculator.get_simple_trading_simulation(test_daily_rets, p1_cls)
    
    # H2 Metrics
    metrics_h2_cls = MetricsCalculator.get_classification_metrics(t2_cls, p2_cls)
    metrics_h2_reg = MetricsCalculator.get_regression_metrics(t2_raw, p2_reg)
    metrics_h2_dir = MetricsCalculator.get_directional_accuracy(t2_raw, p2_reg)
    # Use DAILY returns for simulation, but steered by H2 predictions
    sim_h2 = MetricsCalculator.get_simple_trading_simulation(test_daily_rets, p2_cls)
    
    # Optional V19 Directional Head Stats
    dir_head_acc = None
    if dir_preds is not None:
        # Binary target for directional head (+1 if return > 0, 0 otherwise)
        dir_targets = (t1_raw > 0).astype(float)
        dir_head_preds = (dir_preds > 0.5).astype(float)
        dir_head_acc = np.mean(dir_targets == dir_head_preds)
    
    # 6. Reporting
    report = []
    report.append("==================================================")
    report.append(f"EVALUATION REPORT: {os.path.basename(args.config)}")
    report.append("==================================================")
    report.append(f"Model ID: {config.get('experiment_name', 'N/A')}")
    report.append(f"Test Samples: {len(X_d_test)}")
    report.append("")
    
    def format_metrics(h_name, cls_m, reg_m, dir_acc, sim_m, d_head_acc=None):
        s = []
        s.append(f"--- {h_name} Horizon ---")
        s.append(f"[Classification] Accuracy: {cls_m.get('Accuracy', 'N/A')}")
        s.append(f"[Classification] F1 (Macro): {cls_m['F1']:.4f}")
        s.append(f"[Classification] MCC: {cls_m['MCC']:.4f}")
        s.append(f"[Regression]     MAE: {reg_m['MAE']:.5f} | RMSE: {reg_m['RMSE']:.5f} | R2: {reg_m['R2']:.4f}")
        s.append(f"[Directional]    Reg Direction Acc: {dir_acc*100:.2f}%")
        if d_head_acc is not None:
            s.append(f"[Directional]    Head Direction Acc: {d_head_acc*100:.2f}%")
        s.append(f"[Trading Sim]    Total Return: {sim_m['Total_Strategy_Return']*100:.2f}% (Market: {sim_m['Total_Market_Return']*100:.2f}%)")
        s.append(f"[Trading Sim]    Win Rate: {sim_m['Win_Rate']*100:.2f}% | Sharpe: {sim_m['Sharpe_Ratio']:.2f}")
        return "\n".join(s)

    report.append(format_metrics("Horizon 1 (Short)", metrics_h1_cls, metrics_h1_reg, metrics_h1_dir, sim_h1, dir_head_acc))
    report.append("")
    report.append(format_metrics("Horizon 2 (Medium)", metrics_h2_cls, metrics_h2_reg, metrics_h2_dir, sim_h2))
    report.append("")
    
    report.append("=== Detailed Classification Report (H1) ===")
    report.append(classification_report(t1_cls, p1_cls, target_names=['Bear', 'Neutral', 'Bull']))
    
    report.append("=== Detailed Classification Report (H2) ===")
    report.append(classification_report(t2_cls, p2_cls, target_names=['Bear', 'Neutral', 'Bull']))
    
    report_text = "\n".join(report)
    
    # Print to Console (Brief)
    print("\n" + report_text)
    
    # Save Report
    with open(args.report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"Detailed report saved to {args.report_file}")
    
    # Save JSON
    all_metrics = {
        "h1": {
            **metrics_h1_cls, 
            **metrics_h1_reg, 
            "directional_acc": metrics_h1_dir, 
            "dir_head_acc": dir_head_acc,
            "trading_sim": sim_h1
        },
        "h2": {**metrics_h2_cls, **metrics_h2_reg, "directional_acc": metrics_h2_dir, "trading_sim": sim_h2}
    }
    # Convert numpy types to native
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(args.metrics_json, 'w') as f:
        json.dump(all_metrics, f, indent=4, default=convert)
    logger.info(f"Metrics JSON saved to {args.metrics_json}")

    # 7. Save Predictions CSV
    df_res = pd.DataFrame({
        'Date': dates[split_idx:],
        'Act_H1_Reg': t1_raw, 'Pred_H1_Reg': p1_reg, 'Pred_H1_Cls': p1_cls,
        'Act_H2_Reg': t2_raw, 'Pred_H2_Reg': p2_reg, 'Pred_H2_Cls': p2_cls
    })
    df_res.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()


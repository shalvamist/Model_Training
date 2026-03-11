import os
import sys
import json
import logging
import subprocess
import argparse
import pandas as pd

# Add root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from library.utils import find_config_for_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(root_dir, "weights")
CONFIGS_DIR = os.path.join(root_dir, "configs")
EVALUATE_SCRIPT = os.path.join(root_dir, "tools", "evaluate.py")

def get_nested(d, *keys, default="N/A"):
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val

def analyze_model_persona(row, config):
    h1_acc = row.get('H1_Acc', 0)
    h2_return = row.get('H2_Ret', 0)
    market_return = row.get('H2_Market_Return', 0)
    
    alpha = h2_return - market_return
    
    if alpha > 0.05:
        classification = "**Alpha Generator** (Beats Market)"
    elif abs(alpha) < 0.05:
        classification = "**Market Tracker** (Matches Market)"
    else:
        classification = "**Underperformer** (Lags Market)"
        
    strengths, weaknesses = [], []
    if h1_acc > 0.58 and h2_return < market_return:
        weaknesses.append("High Accuracy Trap: Predicts 'Neutral' too often, missing trends.")
    if h2_return > market_return:
        strengths.append(f"Strong Trend Capturing (Alpha: {alpha*100:.1f}%)")
    
    bear_recall = row.get('H1_Recall_Bear', 0)
    if bear_recall > 0.3:
        strengths.append("High Bear Sensitivity (Good for hedging)")
    elif bear_recall < 0.1:
        weaknesses.append("Blind to Bear Markets")

    return classification, strengths, weaknesses

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Batch of Models")
    parser.add_argument("--filter", type=str, default="all", help="all, rank_candidates, or specific model name substring")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of models")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--generate_report", action="store_true", help="Generate detailed Markdown catalogue")
    args = parser.parse_args()
    
    # 1. Gather Models based on filter
    target_models = []
    if os.path.exists(MODELS_DIR):
        for root, dirs, files in os.walk(MODELS_DIR):
            if "model.pth" in files:
                dir_name = os.path.basename(root)
                if args.filter == "all":
                    target_models.append(root)
                elif args.filter == "rank_candidates" and "rank" in root.lower():
                    target_models.append(root)
                elif args.filter != "all" and args.filter != "rank_candidates" and args.filter in root:
                    target_models.append(root)

    logger.info(f"Found {len(target_models)} models matching filter '{args.filter}'")
    
    # 2. Evaluate
    results = []
    for count, model_dir in enumerate(target_models):
        if args.limit > 0 and count >= args.limit:
            break
            
        model_name = os.path.basename(model_dir)
        if "rank_candidates" in model_dir:
            model_name = f"rank_candidate_{model_name}"
            
        weights_file = os.path.join(model_dir, "model.pth")
        config_file = find_config_for_model(model_name, model_dir, CONFIGS_DIR)
        
        if not config_file:
            logger.warning(f"Skipping {model_name}: No config found.")
            continue
            
        logger.info(f"Evaluating {model_name}...")
        metrics_file = os.path.join(model_dir, "evaluation_metrics.json")
        report_file = os.path.join(model_dir, "evaluation_report.txt")
        csv_file = os.path.join(model_dir, "predictions.csv")
        
        cmd = [
            sys.executable, EVALUATE_SCRIPT,
            "--config", config_file, "--weights", weights_file,
            "--device", args.device, "--metrics_json", metrics_file,
            "--report_file", report_file, "--output_csv", csv_file
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            h1 = metrics.get('h1', {})
            h2 = metrics.get('h2', {})
            
            row = {
                "Model": model_name,
                "Model_Dir": model_dir,
                "Config_File": config_file,
                
                "H1_Acc": h1.get("Accuracy", 0),
                "H1_MCC": h1.get("MCC", 0),
                "H1_DirAcc": h1.get("directional_acc", 0),
                "H1_Recall_Bear": get_nested(h1, "detailed_cls_report", "Bear", "recall"),
                "H1_Ret": get_nested(h1, "trading_sim", "Total_Strategy_Return", default=0),
                "H1_Market_Return": get_nested(h1, "trading_sim", "Total_Market_Return", default=0),
                "H1_Sharpe": get_nested(h1, "trading_sim", "Sharpe_Ratio", default=0),
                
                "H2_Acc": h2.get("Accuracy", 0),
                "H2_MCC": h2.get("MCC", 0),
                "H2_DirAcc": h2.get("directional_acc", 0),
                "H2_Ret": get_nested(h2, "trading_sim", "Total_Strategy_Return", default=0),
                "H2_Market_Return": get_nested(h2, "trading_sim", "Total_Market_Return", default=0),
                "H2_Sharpe": get_nested(h2, "trading_sim", "Sharpe_Ratio", default=0),
            }
            results.append(row)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed {model_name}: {e.stderr}")
            
    # 3. Summarize and Rank
    if results:
        df = pd.DataFrame(results)
        df['Alpha'] = df['H2_Ret'] - df['H2_Market_Return']
        df = df.sort_values(by="H1_Sharpe", ascending=False)
        
        out_csv = os.path.join(MODELS_DIR, f"evaluation_summary_{args.filter}.csv")
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved summary CSV to {out_csv}")
        
        # 4. Generate Markdown Report
        if args.generate_report:
            md = [f"# Model Evaluation Report (Filter: {args.filter})", "\n## Summary Table", df.to_markdown(index=False), "\n## Detailed Model Cards"]
            
            for _, row in df.iterrows():
                with open(row["Config_File"], 'r') as f:
                    cfg = json.load(f)
                    
                persona, strengths, weaknesses = analyze_model_persona(row, cfg)
                md.append(f"### 📦 {row['Model']}")
                md.append(f"**Persona**: {persona}")
                md.append(f"- **H1 Sharpe**: {row['H1_Sharpe']:.2f} | **Return**: {row['H1_Ret']*100:.1f}%")
                md.append(f"- **H2 Sharpe**: {row['H2_Sharpe']:.2f} | **Return**: {row['H2_Ret']*100:.1f}%")
                md.append(f"- **Strengths**: {', '.join(strengths) if strengths else 'None'}")
                md.append(f"- **Weaknesses**: {', '.join(weaknesses) if weaknesses else 'None'}")
                md.append("---")
                
            report_out = os.path.join(root_dir, f"model_catalogue_{args.filter}.md")
            with open(report_out, "w", encoding='utf-8') as f:
                f.write("\n".join(md))
            logger.info(f"Detailed Markdown report saved to {report_out}")
    else:
        logger.warning("No models successfully evaluated.")

if __name__ == "__main__":
    main()

import os
import sys
import json
import logging
import subprocess
import glob
import pandas as pd
import argparse

# Add root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(root_dir, "weights")
CONFIGS_DIR = os.path.join(root_dir, "configs")
EVALUATE_SCRIPT = os.path.join(root_dir, "tools", "evaluate.py")

def find_config_for_model(model_name, model_dir):
    """
    Heuristic to find the correct config file for a model.
    1. Check for config.json inside the model dir
    2. Check configs/zoo/<model_name>.json
    3. Check configs/templates/<model_name>.json
    """
    # 1. Config in model dir
    local_config = os.path.join(model_dir, "config.json")
    if os.path.exists(local_config):
        return local_config
    
    # 2. Config in zoo
    zoo_config = os.path.join(CONFIGS_DIR, "zoo", f"{model_name}.json")
    if os.path.exists(zoo_config):
        return zoo_config

    # 3. Recursive search in configs dir
    for root, dirs, files in os.walk(CONFIGS_DIR):
        if f"{model_name}.json" in files:
            return os.path.join(root, f"{model_name}.json")
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate All Models")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of models to evaluate (for testing)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    model_dirs = []
    for root, dirs, files in os.walk(MODELS_DIR):
        if "model.pth" in files:
            model_dirs.append(root)
    
    results = []
    
    count = 0
    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        # Use relative path for clarity if needed, but basename is usually enough for the report
        # If there are duplicate names, we might want to include parent dir.
        # Let's check if it's a "winner" from rank_candidates
        if "rank_candidates" in model_dir:
            model_name = f"rank_candidate_{model_name}"
        if args.limit > 0 and count >= args.limit:
            break
            

        weights_file = os.path.join(model_dir, "model.pth")
        
        if not os.path.exists(weights_file):
            logger.warning(f"Skipping {model_name}: No model.pth found")
            continue
            
        config_file = find_config_for_model(model_name, model_dir)
        
        if not config_file:
            logger.warning(f"Skipping {model_name}: No matching config found")
            continue
            
        logger.info(f"Evaluating {model_name} with config {config_file}")
        
        # Define output files
        metrics_file = os.path.join(model_dir, "evaluation_metrics.json")
        report_file = os.path.join(model_dir, "evaluation_report.txt")
        csv_file = os.path.join(model_dir, "predictions.csv")
        
        # Run evaluate.py
        cmd = [
            sys.executable, EVALUATE_SCRIPT,
            "--config", config_file,
            "--weights", weights_file,
            "--device", args.device,
            "--metrics_json", metrics_file,
            "--report_file", report_file,
            "--output_csv", csv_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Parse results
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Extract Key Metrics for Summary
            h1 = metrics.get('h1', {})
            h2 = metrics.get('h2', {})
            
             # Helper to safely get nested keys
            def get_nested(d, *keys, default="N/A"):
                val = d
                for k in keys:
                    if isinstance(val, dict):
                        val = val.get(k, default)
                    else:
                        return default
                return val

            row = {
                "Model": model_name,
                "H1_Accuracy": h1.get("Accuracy", 0),
                "H1_Directional": h1.get("directional_acc", 0),
                "H1_RMSE": h1.get("RMSE", 0),
                "H1_Recall_Bear": get_nested(h1, "detailed_cls_report", "Bear", "recall"),
                "H1_Recall_Bull": get_nested(h1, "detailed_cls_report", "Bull", "recall"),
                "H1_Precision_Bear": get_nested(h1, "detailed_cls_report", "Bear", "precision"),
                "H1_Precision_Bull": get_nested(h1, "detailed_cls_report", "Bull", "precision"),
                "H1_Return": get_nested(h1, "trading_sim", "Total_Strategy_Return"),
                "H1_Market_Return": get_nested(h1, "trading_sim", "Total_Market_Return"),
                
                "H2_Accuracy": h2.get("Accuracy", 0),
                "H2_Directional": h2.get("directional_acc", 0),
                "H2_RMSE": h2.get("RMSE", 0),
                "H2_Recall_Bear": get_nested(h2, "detailed_cls_report", "Bear", "recall"),
                "H2_Recall_Bull": get_nested(h2, "detailed_cls_report", "Bull", "recall"),
                "H2_Return": get_nested(h2, "trading_sim", "Total_Strategy_Return"),
                "H2_Market_Return": get_nested(h2, "trading_sim", "Total_Market_Return"),
            }
            results.append(row)
            count += 1
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save Aggregate Results
    if results:
        df_results = pd.DataFrame(results)
        df_results.sort_values(by="H1_Accuracy", ascending=False, inplace=True)
        
        summary_csv = "model_metrics_summary.csv"
        df_results.to_csv(summary_csv, index=False)
        logger.info(f"Summary CSV saved to {summary_csv}")
        
        # Generate Markdown Report
        md_report = []
        md_report.append("# Comprehensive Model Evaluation Report")
        md_report.append(f"Models Evaluated: {len(results)}")
        md_report.append("")
        md_report.append("## Top Models by H1 Accuracy")
        md_report.append(df_results.head(5).to_markdown(index=False))
        md_report.append("")
        
        md_report.append("## Top Models by H1 Directional Accuracy")
        md_report.append(df_results.sort_values(by="H1_Directional", ascending=False).head(5).to_markdown(index=False))
        md_report.append("")
        
        md_report.append("## Top Models by H1 Bear Recall (Risk Management)")
        md_report.append(df_results.sort_values(by="H1_Recall_Bear", ascending=False).head(5).to_markdown(index=False))
        md_report.append("")
        
        md_report.append("## Full Results")
        md_report.append(df_results.to_markdown(index=False))
        
        with open("comprehensive_report.md", "w") as f:
            f.write("\n".join(md_report))
        logger.info("Comprehensive report saved to comprehensive_report.md")
        
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()

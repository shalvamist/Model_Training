import os
import shutil
import datetime
import argparse
import json
import logging
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(root_dir, "weights")
    configs_dir = os.path.join(root_dir, "configs")
    
    parser = argparse.ArgumentParser(description="Unified Model Deployment Tool")
    parser.add_argument("--models", nargs="*", help="List of model names to deploy (space separated). Assumes they are in weights/<name>")
    parser.add_argument("--env", type=str, default="V23_Deploy", help="Target deployment folder inside weights/")
    args = parser.parse_args()
    
    if not args.models:
        logger.error("No models provided to deploy. Use --models <model1> <model2>")
        sys.exit(1)
        
    deploy_dir = os.path.join(weights_dir, args.env)
    os.makedirs(deploy_dir, exist_ok=True)
    
    logger.info(f"Deploying models to {deploy_dir}")
    
    summary_lines = [
        f"# Model Deployment Package: {args.env}",
        "",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Included Models",
        ""
    ]
    
    # Check if we have an evaluation summary CSV to pull performance metrics from
    summary_df = None
    summary_csv = os.path.join(weights_dir, "evaluation_summary_all.csv")
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
        summary_df.set_index("Model", inplace=True)
    
    for target in args.models:
        src_weights = os.path.join(weights_dir, target)
        if not os.path.exists(src_weights):
            logger.error(f"  [!] Weights directory not found for {target}: {src_weights}")
            continue
            
        # Read Metadata to get timestamp for intelligent versioning
        metadata_path = os.path.join(src_weights, "metadata.json")
        version_suffix = ""
        if os.path.exists(metadata_path):
            try:
                 with open(metadata_path, 'r') as f:
                     meta = json.load(f)
                 timestamp = meta.get("timestamp", "")
                 if timestamp:
                     # Convert ISO 8601 to a safe folder suffix
                     clean_ts = timestamp.replace(":", "_").replace("-", "").replace("T", "_").replace("Z", "")
                     version_suffix = f"_{clean_ts}"
            except Exception as e:
                 logger.warning(f"  [~] Failed to read metadata.json for {target}: {e}")
                 version_suffix = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
             version_suffix = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
             
        dest_name = f"{target}{version_suffix}"
        dst_weights = os.path.join(deploy_dir, dest_name)
        
        # 1. Copy Model Weights Directory
        if os.path.exists(dst_weights):
            shutil.rmtree(dst_weights)
        shutil.copytree(src_weights, dst_weights)
        logger.info(f"  [+] Copied weights: {target} -> {dest_name}")
        
        # 2. Add to Summary
        summary_lines.append(f"### {dest_name}")
        if summary_df is not None and target in summary_df.index:
            row = summary_df.loc[target]
            h1_dir_acc = row.get("H1_DirAcc", 0) * 100
            h2_dir_acc = row.get("H2_DirAcc", 0) * 100
            h1_ret = row.get("H1_Ret", 0) * 100
            h2_ret = row.get("H2_Ret", 0) * 100
            h1_sharpe = row.get("H1_Sharpe", 0)
            h2_sharpe = row.get("H2_Sharpe", 0)
            
            summary_lines.extend([
                f"- **H1 (Short-Term)**: Return {h1_ret:.2f}% | Sharpe {h1_sharpe:.2f} | DirAcc {h1_dir_acc:.2f}%",
                f"- **H2 (Medium-Term)**: Return {h2_ret:.2f}% | Sharpe {h2_sharpe:.2f} | DirAcc {h2_dir_acc:.2f}%"
            ])
        else:
             summary_lines.append("- Performance metrics not found in summary cache.")
             
        summary_lines.append("")
        
    # Write Summary File
    summary_path = os.path.join(deploy_dir, "deployment_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
        
    logger.info(f"Deployment complete. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

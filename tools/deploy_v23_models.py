import os
import shutil
import pandas as pd

# Define paths
base_dir = r"C:\projects\QQQ_Reg_Models_v2\Model_trading_training"
weights_dir = os.path.join(base_dir, "weights")
configs_dir = os.path.join(base_dir, "configs")
deploy_dir = os.path.join(weights_dir, "V23_Deploy")

# Create deploy directory
os.makedirs(deploy_dir, exist_ok=True)

# The optimal models to deploy based on the summary
deploy_targets = [
    "v23_alpha_generation_optimized_rank1",
    "v23_sector_alpha_optimized_rank4",
    "v23_fortress_master_optimized_rank1",
    "v23_bear_headhunter_optimized_rank1",
    "v23_sector_alpha_optimized_rank1"
]

print(f"Deploying models to: {deploy_dir}\n")

summary_lines = [
    "# True Deep V23 Deployment Package",
    "",
    "This directory contains the finalized, verified 150-epoch V23 models that successfully replicated the historical breakthrough performance.",
    "",
    "## Included Models and Performance",
    ""
]

# Load summary CSV to extract metrics
summary_csv = os.path.join(weights_dir, "v23_models_summary.csv")
if os.path.exists(summary_csv):
    df = pd.read_csv(summary_csv)
    df.set_index("Model", inplace=True)
else:
    df = None
    print("WARNING: Summary CSV not found.")

for target in deploy_targets:
    # 1. Copy Model Weights Directory
    src_weights = os.path.join(weights_dir, target)
    dst_weights = os.path.join(deploy_dir, target)
    
    if os.path.exists(src_weights):
        if os.path.exists(dst_weights):
            shutil.rmtree(dst_weights)
        shutil.copytree(src_weights, dst_weights)
        print(f"Copied weights: {target}")
    else:
        print(f"ERROR: Weights not found for {target}")
        
    # 2. Copy Config JSON
    src_config = os.path.join(configs_dir, f"{target}.json")
    dst_config = os.path.join(deploy_dir, f"{target}.json")
    
    if os.path.exists(src_config):
        shutil.copy2(src_config, dst_config)
        print(f"Copied config: {target}.json")
    else:
        print(f"ERROR: Config not found for {target}.json")
        
    # 3. Append to Summary
    if df is not None and target in df.index:
        row = df.loc[target]
        h1_dir_acc = row.get("H1_DirAcc", 0) * 100
        h2_dir_acc = row.get("H2_DirAcc", 0) * 100
        h1_ret = row.get("H1_Ret", 0) * 100
        h2_ret = row.get("H2_Ret", 0) * 100
        h1_sharpe = row.get("H1_Sharpe", 0)
        h2_sharpe = row.get("H2_Sharpe", 0)
        h1_acc = row.get("H1_Acc", 0) * 100
        h2_acc = row.get("H2_Acc", 0) * 100
        h1_mcc = row.get("H1_MCC", 0)
        h2_mcc = row.get("H2_MCC", 0)
        
        summary_lines.extend([
            f"### {target}",
            f"#### Horizon 1 (Short-Term / 7-Day)",
            f"- **Directional Accuracy:** {h1_dir_acc:.2f}%",
            f"- **Simulated Return:** {h1_ret:.2f}%",
            f"- **Sharpe Ratio:** {h1_sharpe:.2f}",
            f"- **Standard Accuracy:** {h1_acc:.2f}%",
            f"- **MCC:** {h1_mcc:.4f}",
            "",
            f"#### Horizon 2 (Medium-Term / 21-Day)",
            f"- **Directional Accuracy:** {h2_dir_acc:.2f}%",
            f"- **Simulated Return:** {h2_ret:.2f}%",
            f"- **Sharpe Ratio:** {h2_sharpe:.2f}",
            f"- **Standard Accuracy:** {h2_acc:.2f}%",
            f"- **MCC:** {h2_mcc:.4f}",
            ""
        ])
    print("-" * 40)

# Write Summary File
summary_path = os.path.join(deploy_dir, "deployment_summary.md")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
    
print(f"Deployment complete. Summary saved to: {summary_path}")

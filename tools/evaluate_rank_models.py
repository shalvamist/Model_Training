import os
import sys
import json
import pandas as pd
import subprocess

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(root_dir, "weights")
    
    # Find all subdirectories with "rank" in the name
    rank_models = []
    if os.path.exists(weights_dir):
        for d in os.listdir(weights_dir):
            path = os.path.join(weights_dir, d)
            if os.path.isdir(path) and "rank" in d.lower():
                if os.path.exists(os.path.join(path, "config.json")) and os.path.exists(os.path.join(path, "model.pth")):
                    rank_models.append(d)
    
    print(f"Found {len(rank_models)} rank models: {rank_models}")
    
    results = []
    
    for model_name in rank_models:
        print(f"Evaluating {model_name}...")
        model_dir = os.path.join(weights_dir, model_name)
        config_path = os.path.join(model_dir, "config.json")
        weights_path = os.path.join(model_dir, "model.pth")
        metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
        report_path = os.path.join(model_dir, "evaluation_report.txt")
        csv_path = os.path.join(model_dir, "predictions.csv")
        
        cmd = [
            sys.executable,
            os.path.join(root_dir, "tools", "evaluate.py"),
            "--config", config_path,
            "--weights", weights_path,
            "--metrics_json", metrics_path,
            "--report_file", report_path,
            "--output_csv", csv_path,
            "--device", "cpu"
        ]
        
        try:
            # Run evaluation
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Read metrics
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Flatten metrics for CSV
                row = {"Model": model_name}
                
                # H1
                h1 = metrics.get("h1", {})
                row["H1_Acc"] = h1.get("Accuracy", 0)
                row["H1_MCC"] = h1.get("MCC", 0)
                row["H1_DirAcc"] = h1.get("directional_acc", 0)
                
                trading_h1 = h1.get("trading_sim", {})
                if trading_h1:
                     row["H1_Sharpe"] = trading_h1.get("Sharpe_Ratio", 0)
                     row["H1_Ret"] = trading_h1.get("Total_Strategy_Return", 0)
                else:
                     row["H1_Sharpe"] = 0
                     row["H1_Ret"] = 0

                # H2
                h2 = metrics.get("h2", {})
                row["H2_Acc"] = h2.get("Accuracy", 0)
                row["H2_MCC"] = h2.get("MCC", 0)
                row["H2_DirAcc"] = h2.get("directional_acc", 0)
                
                trading_h2 = h2.get("trading_sim", {})
                if trading_h2:
                    row["H2_Sharpe"] = trading_h2.get("Sharpe_Ratio", 0)
                    row["H2_Ret"] = trading_h2.get("Total_Strategy_Return", 0)
                else:
                    row["H2_Sharpe"] = 0
                    row["H2_Ret"] = 0
                
                results.append(row)
            else:
                print(f"Error: No metrics found for {model_name}")

        except subprocess.CalledProcessError as e:
            print(f"Failed to evaluate {model_name}: {e.stderr}")
            
    # Save Summary
    if results:
        df = pd.DataFrame(results)
        # Sort by H1 Sharpe
        df = df.sort_values(by="H1_Sharpe", ascending=False)
        output_path = os.path.join(weights_dir, "rank_models_summary.csv")
        df.to_csv(output_path, index=False)
        print("\n\n" + "="*50)
        print(f"Summary saved to {output_path}")
        print("="*50)
        print(df.to_string(index=False))
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

import argparse
import json
import logging
import sys
import torch
import pandas as pd
import os
from sklearn.metrics import classification_report
from Model_trading_training.library.factory import ModelFactory

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Trading Model")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth weights file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV predictions")
    args = parser.parse_args()
    
    logger.info(f"Loading Config: {args.config}")
    
    # 1. Load Model
    model = ModelFactory.load_model_from_weights(args.config, args.weights, device=args.device)
    
    # 2. Load Data Config
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # 3. Get Processed Data
    logger.info("Preparing Data...")
    processor = ModelFactory.get_processor(config.get('processor_version', 'v11'), config)
    df = processor.fetch_data()
    df_proc, dyn_cols, stat_cols = processor.process(df)
    data = processor.create_sequences(df_proc, dyn_cols, stat_cols)
    
    X_d, X_s, X_t, Y_1, Y_2, dates = data
    
    # Use Last 20% for Evaluation (Test Set)
    split_idx = int(len(X_d) * 0.8)
    
    X_d_test = torch.tensor(X_d[split_idx:], dtype=torch.float32).to(args.device)
    X_s_test = torch.tensor(X_s[split_idx:], dtype=torch.float32).to(args.device)
    X_t_test = torch.tensor(X_t[split_idx:], dtype=torch.long).to(args.device)
    
    # 4. Inference
    logger.info("Running Inference...")
    with torch.no_grad():
        r1, c1, r2, c2 = model(X_d_test, X_s_test, X_t_test)
        
    p1_cls = torch.argmax(c1, dim=1).cpu().numpy()
    p2_cls = torch.argmax(c2, dim=1).cpu().numpy()
    
    # Targets for Report
    # Logic: > 2% Bull, < -2% Bear
    t1_raw = Y_1[split_idx:]
    t1_cls = np.ones_like(t1_raw, dtype=int)
    t1_cls[t1_raw > 0.02] = 2; t1_cls[t1_raw < -0.02] = 0
    
    t2_raw = Y_2[split_idx:]
    t2_cls = np.ones_like(t2_raw, dtype=int)
    t2_cls[t2_raw > 0.04] = 2; t2_cls[t2_raw < -0.04] = 0
    
    # 5. Reports
    print("\n=== Horizon 1 (Short Term) Report ===")
    print(classification_report(t1_cls, p1_cls, target_names=['Bear', 'Neutral', 'Bull']))
    
    print("\n=== Horizon 2 (Medium Term) Report ===")
    print(classification_report(t2_cls, p2_cls, target_names=['Bear', 'Neutral', 'Bull']))
    
    # 6. Save
    df_res = pd.DataFrame({
        'Date': dates[split_idx:],
        'Act_H1': t1_raw, 'Pred_H1': p1_cls,
        'Act_H2': t2_raw, 'Pred_H2': p2_cls
    })
    df_res.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()

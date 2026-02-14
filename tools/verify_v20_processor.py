
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from library.processors import ProcessorV20
from library.factory import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyV20")

def main():
    config = {
        "seq_length": 60,
        "train_years": 2,
        "test_years": 0.5
    }
    
    logger.info("Initializing ProcessorV20...")
    processor = ProcessorV20(config)
    
    # 1. Fetch
    logger.info("Fetching Data (QQQ)...")
    df = processor.fetch_data("QQQ")
    if df.empty:
        logger.error("Fetch failed.")
        return
        
    logger.info(f"Data Fetched: {len(df)} rows.")
    
    # 2. Process
    logger.info("Processing Data...")
    df_proc, dyn, stat = processor.process(df)
    
    # Check Columns
    logging.info(f"Dynamic Columns: {len(dyn)}")
    if 'rsi_abs' in dyn:
        logger.info("[PASS] rsi_abs found in dynamic columns.")
    else:
        logger.error("[FAIL] rsi_abs NOT found.")
        
    if 'vix_abs' in dyn:
        logger.info("[PASS] vix_abs found.")
    else:
        logger.warning("[WARN] vix_abs not found (maybe missing symbol?)")
        
    # Check Targets
    if 'target_1_cls' in df_proc.columns:
        counts = df_proc['target_1_cls'].value_counts().sort_index()
        logger.info(f"Target 1 Class Distribution (H1):\n{counts}")
        if len(counts) > 1:
            logger.info("[PASS] Target 1 Classification Labels Generated.")
        else:
            logger.warning("[WARN] Target 1 Distribution is trivial.")
    else:
        logger.error("[FAIL] target_1_cls column missing.")
        
    # 3. Sequences
    logger.info("Creating Sequences...")
    seqs = processor.create_sequences(df_proc, dyn, stat)
    
    logger.info(f"Sequence Tuple Length: {len(seqs)}")
    if len(seqs) == 8:
        logger.info("[PASS] 8-Tuple Returned (V20 standard).")
        X_d, X_s, X_t, Y1, Y2, dates, Y1c, Y2c = seqs
        logger.info(f"Y1_cls shape: {Y1c.shape}")
        
        # Check alignment
        logger.info(f"Dates shape: {dates.shape}")
        logger.info(f"X_d shape: {X_d.shape}")
        
        if len(Y1c) == len(X_d):
            logger.info("[PASS] Label length matches sequence length.")
        else:
            logger.error(f"[FAIL] Length mismatch: Labels {len(Y1c)} != Seqs {len(X_d)}")
            
        # Inspect a few values
        logger.info(f"First 5 labels: {Y1c[:5]}")
        
    else:
        logger.error(f"[FAIL] Expected 8 items, got {len(seqs)}")

if __name__ == "__main__":
    main()

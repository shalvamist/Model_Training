import pandas as pd
import numpy as np
import logging
import sys
import os
import json

# Add local path
sys.path.append(os.getcwd())

from library.processors import GenericProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_fetch_data(self, ticker="QQQ"):
    # Create dummy data with all needed columns
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame(index=dates)
    df['underlying_close'] = 100 + np.random.randn(100).cumsum()
    df['underlying_open'] = df['underlying_close'] + np.random.randn(100)
    df['underlying_high'] = df['underlying_close'] + 5
    df['underlying_low'] = df['underlying_close'] - 5
    df['volume'] = 1000000
    
    # Macro
    df['vix'] = 20 + np.random.randn(100)
    df['us_treasury_10y'] = 4.0
    
    # Extra Tickers for V15
    if hasattr(self, 'additional_tickers'):
        for tkr in self.additional_tickers:
            df[tkr.lower()] = 100 + np.random.randn(100).cumsum()
            
    # Options (mocked by base processor usually, but we need them for greeks step)
    df['iv'] = 0.2
    
    return df

# Monkey patch fetch_data to avoid network calls
GenericProcessor.fetch_data = mock_fetch_data

def test_config(config_path):
    logger.info(f"Testing Config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    processor = GenericProcessor(config)
    df = processor.fetch_data()
    df_proc, dyn_cols, stat_cols = processor.process(df)
    
    logger.info(f"Dynamic Cols: {len(dyn_cols)}")
    
    # Validation
    if 'v15' in config_path:
        if 'ratio_consumer_norm' not in df_proc.columns:
            raise ValueError("V15 Failed: Missing ratio_consumer_norm")
        logger.info("V15 Sector Ratios Verified.")
        
    if 'v13' in config_path:
        if 'delta' not in df_proc.columns:
             raise ValueError("V13 Failed: Missing Delta")
        logger.info("V13 Greeks Verified.")

def main():
    configs = [
        "configs/zoo/v11_replicated.json",
        "configs/zoo/v13_replicated.json",
        "configs/zoo/v15_replicated.json"
    ]
    
    for c in configs:
        try:
            test_config(c)
        except Exception as e:
            logger.error(f"Failed {c}: {e}")
            raise e
            
    logger.info("SUCCESS: All Ported Configs Verified.")

if __name__ == "__main__":
    main()

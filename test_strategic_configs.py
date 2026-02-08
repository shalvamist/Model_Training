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
    dates = pd.date_range("2023-01-01", periods=300) # Need > 256 for Deep Cycle
    df = pd.DataFrame(index=dates)
    df['underlying_close'] = 100 + np.random.randn(300).cumsum()
    df['underlying_open'] = df['underlying_close'] + np.random.randn(300)
    df['underlying_high'] = df['underlying_close'] + 5
    df['underlying_low'] = df['underlying_close'] - 5
    df['volume'] = 1000000.0
    
    # Macro
    df['vix'] = 20 + np.random.randn(300)
    df['us_treasury_10y'] = 4.0
    
    # Mock Extra Tickers
    # We just create columns for every possible ticker code lowercased
    possible_tickers = ["XLY", "XLP", "XLK", "XLU", "SPY", "USO", "XLI", "^OVX", "ITA", "CPER", "^TNX", "GLD"]
    
    # GenericProcessor fetch_extras logic:
    # It calls get_col_func(tkr, 'Close').
    # Our mock_fetch_data replaces the whole method. 
    # But GenericProcessor calls `self.fetch_data()` which returns a DF.
    # INSTEAD of mocking fetch_data entirely, we should rely on the fact that `fetch_data` 
    # calls `yf.download`. We should mock `yf.download`.
    # OR, we can just monkeypatch `fetch_data` to return a pre-built compatible DF.
    
    # Let's populate the extras manually in the returned DF
    if hasattr(self, 'additional_tickers'):
        for tkr in self.additional_tickers:
            col_name = tkr.lower()
            df[col_name] = 100 + np.random.randn(300).cumsum()
            
    # Options (mocked)
    df['iv'] = 0.2
    
    return df

# Monkey patch
GenericProcessor.fetch_data = mock_fetch_data

def test_config(config_path):
    logger.info(f"Testing Config: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    processor = GenericProcessor(config)
    try:
        # Mocking data fetch
        df = processor.fetch_data()
        
        # Determine cols
        # process() calls fetch_data internally? No, user calls fetch_data then process.
        # Check BaseProcessor code:
        # def fetch_data... returns df
        # def process(df)... returns results
        
        df_proc, dyn_cols, stat_cols = processor.process(df)
        
        if df_proc is None or df_proc.empty:
            raise ValueError("Processed DF is empty")

        logger.info(f"Dynamic Cols: {len(dyn_cols)}")
        
        # Strategy Specific Checks
        if 'sector' in config_path:
            if 'ratio_risk_norm' not in df_proc.columns: raise ValueError("Missing ratio_risk")
        if 'real_economy' in config_path:
            if 'ratio_hard_soft_norm' not in df_proc.columns: raise ValueError("Missing ratio_hard_soft")
        if 'industrial' in config_path:
            if 'vol_spike_norm' not in df_proc.columns: raise ValueError("Missing vol_spike")
        if 'deep_cycle' in config_path:
            if 'cycle_phase_256_norm' not in df_proc.columns: raise ValueError("Missing cycle_phase")
            
        logger.info(f"Success: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed {config_path}: {e}")
        # raise e # Don't raise to allow others to run

def main():
    configs = [
        "configs/strategies/sector_alpha_rotator.json",
        "configs/strategies/real_economy_rotator.json",
        "configs/strategies/industrial_sniper.json",
        "configs/strategies/deep_cycle_arbitrage.json"
    ]
    
    for c in configs:
        test_config(c)
            
    logger.info("Verification Complete.")

if __name__ == "__main__":
    main()

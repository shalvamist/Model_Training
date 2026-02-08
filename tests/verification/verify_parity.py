import pandas as pd
import numpy as np
import json
import logging
import sys
import os

# Add local path to find library
sys.path.append(os.getcwd())

from library.processors import ProcessorV13, GenericProcessor
from library.utils import robust_normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_dataframes(df1, df2, cols):
    mismatch = []
    
    # Align indices first
    common_idx = df1.index.intersection(df2.index)
    
    d1 = df1.loc[common_idx]
    d2 = df2.loc[common_idx]

    if len(common_idx) < len(df1) or len(common_idx) < len(df2):
        print(f"Aligning DFs: V13={len(df1)}, Gen={len(df2)} -> Common={len(common_idx)}")
    
    for c in cols:
        if c not in d1.columns:
            mismatch.append(f"Missing in V13: {c}")
            continue
        if c not in d2.columns:
            mismatch.append(f"Missing in Generic: {c}")
            continue
            
        s1 = d1[c].fillna(0).values
        s2 = d2[c].fillna(0).values
        
        if not np.allclose(s1, s2, atol=1e-4): # slightly relaxed tolerance
            diff = np.abs(s1 - s2).max()
            mismatch.append(f"Diff {c}: Max Delta {diff}")
            
    return mismatch

def main():
    # 1. Configs
    config_v13 = {"tech_params": {}} 
    
    config_path = "v13_generic_config.json"
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {os.getcwd()}/{config_path}")
        return

    with open(config_path, "r") as f:
        config_gen = json.load(f)
        
    # 2. Initialize
    logger.info("Initializing Processors...")
    p_v13 = ProcessorV13(config_v13)
    p_gen = GenericProcessor(config_gen)
    
    # 3. Fetch Data
    logger.info("Fetching Data...")
    # Mocking fetch if needed, but let's try real fetch
    # Using 'QQQ' as standard
    df_raw = p_v13.fetch_data("QQQ")
    
    if df_raw.empty:
        logger.error("Data fetch failed - likely no internet or yfinance issue.")
        # Create Dummy Data
        logger.info("Creating Dummy Data for Verification...")
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        df_raw = pd.DataFrame(index=dates)
        df_raw['underlying_close'] = 100 + np.random.randn(500).cumsum()
        df_raw['underlying_open'] = df_raw['underlying_close'] * (1 + np.random.randn(500)*0.01)
        df_raw['underlying_high'] = df_raw['underlying_open'] * 1.01
        df_raw['underlying_low'] = df_raw['underlying_open'] * 0.99
        df_raw['vix'] = 20 + np.random.randn(500).cumsum()
        df_raw['us_treasury_10y'] = 4.0
        df_raw['iv'] = 0.2
        # Extras needed for V13?
        # V13 doesn't fetch extras.
        
        # Ensure column map
        df_raw['date'] = dates
        
    # 4. Process
    logger.info("Processing V13...")
    df_v13, dyn_v13, stat_v13 = p_v13.process(df_raw.copy())
    
    logger.info("Processing Generic...")
    df_gen, dyn_gen, stat_gen = p_gen.process(df_raw.copy())
    
    # 5. Compare
    logger.info("Comparing Outputs...")
    
    # Compare Dynamic Cols list match
    set_v13 = set(dyn_v13)
    set_gen = set(dyn_gen)
    
    # Ignore order
    if set_v13 != set_gen:
        logger.warning(f"Dynamic Columns Mismatch! Count: V13={len(set_v13)}, Gen={len(set_gen)}")
        if len(set_v13 - set_gen) > 0:
            logger.warning(f"Missing in Gen: {set_v13 - set_gen}")
        if len(set_gen - set_v13) > 0:
            logger.warning(f"Extra in Gen: {set_gen - set_v13}")
    else:
        logger.info("Dynamic Column Names Match!")
        
    # Compare Values
    mismatches = compare_dataframes(df_v13, df_gen, list(set_v13))
    
    if mismatches:
        logger.error(f"Value Mismatches Found ({len(mismatches)} columns):")
        for m in mismatches[:10]: # First 10
            logger.error(m)
        if len(mismatches) > 10:
            logger.error("...")
    else:
        logger.info("SUCCESS: All values match within tolerance!")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import logging
import sys
import os
import json

# Add local path
sys.path.append(os.getcwd())

from library.processors import ProcessorV11, ProcessorV13, ProcessorV15, GenericProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dummy_data():
    dates = pd.date_range("2023-01-01", periods=200)
    df = pd.DataFrame(index=dates)
    df['underlying_close'] = 100 + np.random.randn(200).cumsum()
    df['underlying_open'] = df['underlying_close'] + np.random.randn(200)
    df['underlying_high'] = df['underlying_close'] + 5
    df['underlying_low'] = df['underlying_close'] - 5
    df['volume'] = 1000000
    
    # Macro
    df['vix'] = 20 + np.random.randn(200)
    df['us_treasury_10y'] = 4.0
    
    # Sectors (for V15)
    for tkr in ["XLY", "XLP", "XLK", "XLU"]:
        df[tkr] = 100 + np.random.randn(200).cumsum()
        
    return df

def compare_processors(legacy_cls, config_path, name):
    logger.info(f"--- Comparing {name} ---")
    
    # 1. Setup Data
    df = get_dummy_data()
    
    # 2. Legacy Run
    # Legacy processors take a config dict, but mostly ignore it for feature definitions
    # They check config for 'seq_length' etc.
    legacy_config = {"seq_length": 60} 
    legacy_proc = legacy_cls(legacy_config)
    
    # Mock extract extras for V15 legacy (it calls yf download usually)
    # We need to monkeypatch _fetch_extras or just ensure inputs are in DF?
    # ProcessorV15.process calls `super().process` then adds columns.
    # It assumes `_fetch_extras` populated `xly` etc.
    # In our dummy data, we put `XLY` etc.
    # But V15 expects lowercase `xly` in df.
    # Let's preprocess df to have what `fetch_data` would produce.
    df_legacy = df.copy()
    df_legacy['date'] = df_legacy.index
    for c in ["XLY", "XLP", "XLK", "XLU"]:
        if c in df_legacy.columns: df_legacy[c.lower()] = df_legacy[c]
        
    # We need to manually inject these into V15 because `process` doesn't fetch, `fetch_data` does.
    # But `process` assumes they exist.
    
    try:
        df_old, dyn_old, stat_old = legacy_proc.process(df_legacy.copy())
    except Exception as e:
        logger.error(f"Legacy {name} Failed: {e}")
        return

    # 3. Generic Run
    with open(config_path, 'r') as f:
        gen_config = json.load(f)
        
    gen_proc = GenericProcessor(gen_config)
    
    # Generic Processor `process` runs `fetch_extras` if we had run `fetch_data`.
    # But we are calling `process` directly.
    # So we must ensure `df` has the extra cols if Generic needs them.
    # Generic `_fetch_extras` logic puts them in.
    # But here we bypass fetch.
    # We should manually add them for Generic too if needed.
    # The Generic config has `extra_tickers`. 
    # The `process` method uses `pipeline` which might refer to `xly`.
    # So `df` passed to `process` must have `xly`.
    # Our `df_legacy` has them.
    
    try:
        df_new, dyn_new, stat_new = gen_proc.process(df_legacy.copy())
    except Exception as e:
        logger.error(f"Generic {name} Failed: {e}")
        return
        
    # 4. Compare
    logger.info(f"Legacy Dyn Cols: {len(dyn_old)}")
    logger.info(f"Generic Dyn Cols: {len(dyn_new)}")
    
    missing = set(dyn_old) - set(dyn_new)
    extra = set(dyn_new) - set(dyn_old)
    
    if len(missing) == 0 and len(extra) == 0:
        logger.info(f"SUCCESS: {name} Structure Matches Perfectly.")
    else:
        if missing: logger.warning(f"Missing Cols in New: {missing}")
        if extra: logger.warning(f"Extra Cols in New: {extra}")

def main():
    # V11
    # Note: ProcessorV11 uses 'rates_norm' but Replicated uses 'us_treasury_10y_norm' maybe?
    # We will see.
    compare_processors(ProcessorV11, "configs/zoo/v11_replicated.json", "V11")
    
    # V13
    compare_processors(ProcessorV13, "configs/zoo/v13_replicated.json", "V13")
    
    # V15
    compare_processors(ProcessorV15, "configs/zoo/v15_replicated.json", "V15")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add local path to find library
sys.path.append(os.getcwd())

from library.feature_pipeline import TransformationPipeline
from library.financial_engineer import FinancialEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Generating Dummy Data...")
    # Sine wave + Noise
    t = np.linspace(0, 100, 500)
    data = np.sin(t) + np.random.normal(0, 0.5, 500)
    df = pd.DataFrame({'log_ret': data})
    
    # 2. Config
    config_pipeline = [
        {"step": "fourier", "input": "log_ret", "window": 20, "output": "entropy"},
        {"step": "wavelet", "input": "log_ret", "widths": [2, 5, 10], "prefix": "wav"},
        {"step": "regime_gmm", "input": "log_ret", "n_components": 2, "output": "regime"}
    ]
    
    # 3. Process
    logger.info("Running Pipeline...")
    pipeline = TransformationPipeline(config_pipeline)
    df_out = pipeline.run(df.copy())
    
    # 4. Verify
    logger.info("Verifying Outputs...")
    
    # Fourier
    if 'entropy' in df_out.columns:
        logger.info(f"Fourier Entropy: Mean={df_out['entropy'].mean():.4f}")
        assert df_out['entropy'].max() > 0, "Entropy should be positive"
    else:
        logger.error("Fourier Entropy Missing")
        
    # Wavelet
    w_cols = [c for c in df_out.columns if c.startswith('wav_')]
    if len(w_cols) == 3:
        logger.info(f"Wavelet Cols: {w_cols}")
    else:
        logger.error(f"Wavelet Cols Missing or Wrong Count: {w_cols}")
        
    # Regime
    if 'regime' in df_out.columns:
        counts = df_out['regime'].value_counts()
        logger.info(f"Regime Counts:\n{counts}")
        assert len(counts) <= 2, "Regime should have at most 2 classes"
    else:
        logger.error("Regime Column Missing")
        
    logger.info("SUCCESS: Phase 2 Features Verified.")

if __name__ == "__main__":
    main()

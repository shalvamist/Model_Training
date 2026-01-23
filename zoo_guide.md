# The Model Zoo Guide (V16 Library)

Welcome to the unified `Model_trading_training` library. This project consolidates our best performing trading models into a single, reproducible framework.

## 🏁 Quick Start

### Training a Model
To retrain a model from the zoo:
```bash
# Example: Retraining the Bull Sniper
python -m Model_trading_training.tools.train --config Model_trading_training/configs/zoo/bull_sniper_v13.json --epochs 50 --device cuda
```

### Evaluating a Model
To evaluate an existing model checkpoint:
```bash
python -m Model_trading_training.tools.evaluate \
    --config Model_trading_training/configs/zoo/bull_sniper_v13.json \
    --weights Model_trading_training/weights/bull_sniper_v13/model.pth
```

---

## 🦁 The Model Zoo

### 1. The Bull Sniper V13
*   **Role**: Aggressive Bull Market Entries
*   **Architecture**: V13 (Legacy V11 Base + High Capacity Heads)
*   **Features**: V13 Set (Includes Price Acceleration, RSI Slope, VIX Accelerations)
*   **Best Used For**: Catching the bottom of a pullback or a breakout in a strong uptrend. High precision (90%+ target).
*   **Config**: `configs/zoo/bull_sniper_v13.json`

### 2. The Bear Headhunter V11
*   **Role**: Crash Detection & Capital Preservation
*   **Architecture**: V11 (Standard Hybrid)
*   **Features**: V11 Set (Standard Returns, Macro, Volatility)
*   **Best Used For**: Exiting positions before significant downturns. Prioritizes Recall to ensure safety.
*   **Config**: `configs/zoo/bear_headhunter_v11.json`

### 3. The Super Model V12
*   **Role**: General Purpose Trend Following
*   **Architecture**: V11 (Standard Hybrid)
*   **Features**: V11 Set
*   **Best Used For**: Determining the broad market regime. Acts as the "Trend Filter" for the Sniper.
*   **Config**: `configs/zoo/super_model_v12.json`

### 4. Experimental V17 (Cutting Edge)
*   **Role**: Research & Next-Gen Forecasting
*   **Architecture**: V17 (Bi-LSTM + Transformer + KAN + RevIN)
*   **Features**: V15 Set (Sectors + Macro)
*   **Key Innovations**:
    *   **RevIN**: Reversible Instance Normalization to handle non-stationary price data.
    *   **Bi-LSTM**: Bidirectional temporal processing for fuller context.
    *   **KAN Experts**: Learnable activation functions (Kolmogorov-Arnold) replacing standard MLPs.
*   **Config**: `configs/templates/experimental_v17.json`

---

## 📚 Library Structure

*   `library/`: Core code.
    *   `network_architectures.py`: Contains `HybridJointNetwork`, the universal model structure.
    *   `processors.py`: Contains `ProcessorV11`, `ProcessorV13`, `ProcessorV15` handling feature engineering.
    *   `trainer.py`: Standardized training loop.
*   `weights/`: Stores the actual trained model artifacts (`.pth` files) migrated from Release Candidates.
*   `configs/`: JSON files that define the hyperparameters for each model type.

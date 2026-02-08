# 🧠 Model Integration Strategy: Phase 5

## 🎯 Objective
Generate consistent alpha by combining signals from **4 distinct AI strategies** (Momentum, Macro, Mean Reversion, Cycle) into a unified portfolio allocation. The strategy trades a diversified universe of assets including **QQQ, SPY, GLD, USO, ITA, and XLI**.

---

## 🌍 Traded Universe

| Strategy | Traded Tickers | Auxiliary Inputs | Rationale |
|----------|-----------------|-------------------|-----------|
| **1. Sector Alpha Rotator** | `QQQ`, `SQQQ` | `XLY`, `XLP`, `XLK`, `XLU` | **US Tech Momentum**: Detects "Risk-On" vs "Risk-Off" flows within the US economy. |
| **2. Real Economy Rotator** | `SPY`, `GLD`, `USO` | `XLI`, `^OVX` | **Inflation Rotation**: Rotates capital from Equities to Commodities when inflation heats up. |
| **3. Industrial Sniper** | `ITA`, `XLI` | `QQQ`, `VIX` | **Mean Reversion**: Buys Defense/Industrials on panic-driven oversold conditions ("Fat Pitch"). |
| **4. Deep Cycle Arbitrage** | `USO`, `GLD` | `^TNX` | **Macro Cycles**: Long-term trend following based on multi-year supply/demand supercycles. |

---

## 🤖 The AI Models

We use 3 distinct neural architectures, each purpose-built for its specific trading task. This is **not a "one model fits all"** approach.

### **1. PatchTST (Time Series Transformer)**
*   **Used By:** `Deep Cycle Arbitrage`
*   **Technical Logic:**
    *   **Patching:** Instead of processing 1 time step at a time (like LSTM), it breaks 256 days of data into "patches" (e.g., 16-day chunks).
    *   **Channel Independence:** Treats each variable (Oil Price, 10Y Yield) as a separate time series, learning unique patterns for each before mixing them.
    *   **Why it works:** Transformers struggle with long sequences. Patching reduces the sequence length by 16x, allowing the model to "see" year-long macro cycles that LSTMs miss.
*   **Integration:** We extract the **60-day regression target**. If the model predicts a >3% move over 60 days with high attention weight consistency, we enter a trend trade.

### **2. Experimental V17 (Bi-Directional LSTM + KAN)**
*   **Used By:** `Sector Alpha`, `Industrial Sniper`
*   **Technical Logic:**
    *   **Bi-Directional:** Looks at history *forwards and backwards* to understand cause-and-effect context.
    *   **KAN (Kolmogorov-Arnold Network):** Replaces standard dense layers with learnable spline functions. This allows the model to learn complex, non-linear relationships (like "If VIX > 30 AND RSI < 20, Buy") with fewer parameters.
    *   **Focal Loss:** Penalizes "boring" errors less and "critical" errors more, forcing the model to focus on difficult turning points.
*   **Integration:** 
    *   **Sector Alpha:** We use the **Classification Head** (Bull/Bear Probability).
    *   **Industrial Sniper:** We use the **Quantile Regression Head** (predicting the 10th and 90th percentile prices).

### **3. Hybrid V12 (LSTM + Transformer Encoder)**
*   **Used By:** `Real Economy Rotator`
*   **Technical Logic:**
    *   **LSTM Encoder:** Captures short-term immediate momentum (last 5 days).
    *   **Transformer Encoder:** Captures mid-term regime context (last 60 days).
    *   **Gating Mechanism:** Dynamically weights the LSTM vs Transformer output based on market volatility.
*   **Integration:** We use the **Regression Head** to predict the *relative performance spread* between Commodities and Equities.

---

## 🔌 Inference Architecture: How Models "Plug In"

The `InferenceEngine` acts as the universal adapter. Here is the technical data flow:

### 1. State Loading & Initialization
*   **Weights:** Load `model.pth` (PyTorch state dict).
*   **Architecture:** Load `config.json` to instantiate the correct class (PatchTST vs V17 vs V12).
*   **Metadata:** Load `model_info.json`. This contains:
    *   `scaler_stats`: Mean/Std for normalization.
    *   `feature_names`: Exact list of required input columns.

### 2. Tensor Construction (Pre-processing)
*   **Input:** Live DataFrame of market data.
*   **Scaling:** Apply `(Value - Mean) / Std` using the *training set's* statistics methods loaded from metadata.
*   **Reshaping:** Convert to 3D Tensor `[Batch=1, Lookback=Window, Features=N]`.
    *   *Note:* Different models have different windows (60 vs 256).

### 3. Forward Pass & Extraction
*   **Execution:** Run `model(input_tensor)` in `eval()` mode.
*   **Raw Output:** The model returns a dictionary of heads:
    ```python
    {
        "regression": tensor([[0.042]]),          # 5-day return
        "classification": tensor([[0.1, 0.8]]),   # Bull/Bear logits
        "quantile": tensor([[120.5, 125.0, 130.0]]) # 10th/50th/90th percentile
    }
    ```
*   **Post-Processing:**
    *   `Sigmoid` on Classification Logits → Probabilities.
    *   `InverseTransform` on Regression → Real % Returns.

### 4. Signal Normalization (The "Adapter" Layer)
*   All component outputs are normalized to a standard dictionary structure that the Ensemble can consume:
    ```python
    {
        "strategy": "sector_alpha_rotator",
        "signal": "LONG",          # Derived from logic below
        "strength": 0.85,          # Derived from probability/conviction
        "horizon": "21d",          # From config
        "expected_return": 0.042,  # Denormalized regression output
        "raw_outputs": { ... }     # Full diverse outputs for logging
    }
    ```

---

## ⚡ Alpha Generation Logic

### **Step 1: Signal Interpretation**
Each strategy has a specific `Interpreter` class to convert raw model outputs into a `(Direction, Conviction)` tuple.

| Strategy | Logic | Mathematical Threshold |
|----------|-------|------------------------|
| **Sector Alpha** | **Probability Entropy:** High probability with low entropy means high conviction. | `if p_bull > 0.65 and entropy < 0.5: LONG` |
| **Real Economy** | **Spread Expansion:** If the predicted spread between Commodities and SPY widens. | `if pred_commodity_ret - pred_spy_ret > 0.01: ROTATE_TO_COMMODITIES` |
| **Industrial Sniper** | **Quantile Breach:** If current price dips below the predicted 10th percentile (Oversold). | `if current_price < pred_10th_percentile: BUY` |
| **Deep Cycle** | **Trend Strength:** If the 60-day regression target is strongly positive. | `if pred_60d_return > 0.03: LONG` |

### **Step 2: The Meta-Ensemble**
We combine signals using an **Adaptive Weighted Ensemble**:

1.  **Performance Weighting:**
    *   Calculate rolling 60-day Sharpe Ratio for each strategy.
    *   `Weight = max(0, Sharpe_Ratio) / Sum(Positive Sharpe Ratios)`
    *   *Losers get zero capital.*

2.  **Vote Aggregation:**
    *   `Net Signal = Sum(Direction_i * Conviction_i * Weight_i)`
    *   Result is a value between -1 (Strong Short) and +1 (Strong Long).

3.  **Disagreement Penalty:**
    *   Calculate variance of signals.
    *   If variance is high (e.g., one model says Buy, another says Sell), apply a scalar penalty: `Net Signal *= (1 - Variance_Factor)`.

### **Step 3: Risk Guardrails**
Before execution, the aggregate signal must pass hard checks:

1.  **Volatility Check:** If `VIX > 30`, max position size is halved.
2.  **Stop Loss:** Hard 2% trailing stop on the *portfolio* level.

---

## 🛠️ Execution Roadmap

### **Phase 1: Infrastructure**
*   Create `portfolio/` module.
*   Build `ModelLoader` to ingest trained `.pth` files.
*   Build `InferenceEngine` to standardize raw model outputs.

### **Phase 2: Signal Logic**
*   Implement `SignalCombiner` with Weighted Voting.
*   Implement Strategy Interpreters (specific logic per model).

### **Phase 3: Deployment**
*   Create `tools/run_portfolio.py` CLI tool.
*   **Output:** Daily allocation targets (e.g., "70% QQQ, 30% Cash").

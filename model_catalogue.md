# Detailed Model Catalogue
A comprehensive technical and performance analysis of all available models.

## Executive Summary
| Model | Persona | Role | Status |
| :--- | :--- | :--- | :--- |
| **v21_real_economy_optimized_rank5** | Alpha Generator | SPY Strategy | ✅ Active |
| **v20_sector_alpha_optimized_rank1** | Alpha Generator | QQQ Strategy | ✅ Active |
| **v21_deep_cycle_optimized_rank2** | Market Tracker | BTC-USD Strategy | ✅ Active |
| **v23_fortress_master_final** | Super Model | Fortress (Unified Long) | ✅ Active |
| **bull_sniper_v13** | Sniper | Fortress (Aggressive Long) | ⚠️ Deprecating |
| **super_model_v12** | Super Model | Fortress (Moderate Long) | ⚠️ Deprecating |
| **bear_headhunter_v11** | Headhunter | Fortress (Crash Protection) | ✅ Active |
| **rank_3_v9** | Momentum | Legacy (V9 Strategy) | 🏛️ External |
| **sqqq_v2_1** | Bear Filter | Legacy (Backtesting) | 🏛️ External |
| **v11_bull** | Hunter | Legacy (Hunter-Killer) | 🏛️ External |
| **v12_bear** | Killer | Legacy (Hunter-Killer) | 🏛️ External |
| v21_sector_alpha_optimized_rank2 | Market Tracker | Archived | 📦 Archived |
| v21_sector_alpha_optimized_rank4 | Market Tracker | Archived | 📦 Archived |
| v21_real_economy_optimized_rank4 | Market Tracker | Archived | 📦 Archived |

---

## 🚀 Active Strategy Models

### 📦 v21_real_economy_optimized_rank5
**Persona**: **Alpha Generator** (Beats Market)
**Usage**: SPY Strategy

#### 📊 Performance Summary
- **H1 (Short Term)**: Acc: 60.5% | Return: 12.1%
- **H2 (Medium Term)**: Acc: 52.2% | Return: 76.0% (Market: 66.5%)
- **Bear Recall (Risk)**: 0.5%

#### 💪 Strengths & Weaknesses
- ✅ Strong Trend Capturing (Alpha: 9.4%)
- ⚠️ Blind to Bear Markets (High downside risk)

#### ⚙️ Technical Specifications
- **Architecture**: experimental_v17 (6 LSTM Layers)
- **Lookback Window**: 60 days
- **Input Features (16)**:
  - atr, tlt, underlying_close, volume, xle, xlf ...
- **Outputs**:
  - **target_1**: log_future_return (7 days)
  - **target_2**: log_future_return (21 days)

---

### 📦 v20_sector_alpha_optimized_rank1
**Persona**: **Alpha Generator** (Beats Market)
**Usage**: QQQ Strategy

#### 📊 Performance Summary
- **H1 (Short Term)**: Acc: 53.5% | Return: -7.4%
- **H2 (Medium Term)**: Acc: 46.0% | Return: 73.6% (Market: 66.5%)
- **Bear Recall (Risk)**: 13.4%

#### 💪 Strengths & Weaknesses
- ✅ Strong Trend Capturing (Alpha: 7.0%)
- (None Identified)

#### ⚙️ Technical Specifications
- **Architecture**: experimental_v17 (7 LSTM Layers)
- **Lookback Window**: 60 days
- **Input Features (16)**:
  - atr, ratio_growth, ratio_risk, underlying_close, volume ...
- **Outputs**:
  - **target_1**: log_future_return (7 days)
  - **target_2**: log_future_return (21 days)

---

### 📦 v21_deep_cycle_optimized_rank2
**Persona**: **Market Tracker** (Matches Market)
**Usage**: BTC-USD Strategy

#### 📊 Performance Summary
- **H1 (Short Term)**: Acc: 53.7% | Return: -2.6%
- **H2 (Medium Term)**: Acc: 46.7% | Return: 59.1% (Market: 59.9%)
- **Bear Recall (Risk)**: 15.5%

#### 💪 Strengths & Weaknesses
- (None Identified)
- (None Identified)

#### ⚙️ Technical Specifications
- **Architecture**: experimental_v17 (7 LSTM Layers)
- **Lookback Window**: 90 days
- **Input Features (17)**:
  - atr, eem, gld, log_ret, tlt, underlying_close, uup, volume ...
- **Outputs**:
  - **target_1**: log_future_return (7 days)
  - **target_2**: log_future_return (21 days)

---

## 🛡️ Fortress Strategy Ensemble (Bot 1)
Models located in `weights/` (previously `release_candidate_models`).

| Role | Model Name | Version | Condition |
| :--- | :--- | :--- | :--- |
| **Unified Master** (Long Focus) | `v23_fortress_master_final` | V23 | Replaces Sniper/SuperModel |
| **Sniper** (Aggressive Long) | `bull_sniper_v13` | V13 | Signal > 0.90 (To be replaced) |
| **Super Model** (Moderate Long) | `super_model_v12` | V12 | Signal > 0.60 (To be replaced) |
| **Headhunter** (Exit/Crash Protection) | `bear_headhunter_v11` | V11 | Bear Signal > 0.70 |

---

## 🏛️ Legacy & External Models
Models used in Backtesting App or specific legacy strategies.  
Located in `../backtesting-streamlit-app/BackTestStream/strategies/models/`

### 📦 rank_3_v9.pth
- **Role**: Momentum
- **Description**: The primary V9 Momentum Model (Rank 3).

### 📦 sqqq_v2_1.pth
- **Role**: Bear Filter
- **Description**: Optional SQQQ model used in backtesting (V9 Strategy) to filter out crashes.

### 📦 v11_bull.pth
- **Role**: The Hunter
- **Description**: V11 Bull Model (Rank 1). Provides high-precision buy signals.

### 📦 v12_bear.pth
- **Role**: The Killer
- **Description**: V12 Bear Model (Rank 1). Acts as a "Crash Killer" to force exits.

---

## 📦 Archived Models
Models preserved in `weights/archive/` for potential future use or analysis.

- **v21_sector_alpha_optimized_rank2**: Market Tracker (Matches Market)
- **v21_sector_alpha_optimized_rank4**: Market Tracker (Matches Market)
- **v21_real_economy_optimized_rank4**: Market Tracker (Matches Market)
- **rank_candidates**: Previous V19/V18 winners
- **model**: PatchTST Experiment

---

## 🔬 Generational Analysis: V20/V21 vs. V23

The core evolution of the active model families comes down to managing the tradeoff between Raw Directional Accuracy and Trading Sub-Simulation Returns.

### Legacy V20/V21 Strengths
The existing active models (e.g., `v21_real_economy`, `v20_sector_alpha`) were built heavily on purely recurrent loops. They historically struggle with high H2 (21-Day) Directional Accuracy (averaging **46 - 52%**), leaving them somewhat vulnerable during choppy macro shifts. However, their internal loss scaling naturally favored massive simulated momentum rallies, resulting in **Return profiles matching or beating the market (> 70%)**.

### Early V23 Benchmark Results
The V23 architecture introduces `KANLinear` activations and explicit Task-Specific routing (Crash vs Drift). 
In its first shallow configuration trials, V23 massively elevated raw prediction accuracy:
- `v23_fortress_master_final` achieved **61.6% H2 Directional Accuracy**.
- Initial trials of `v23_bear_headhunter` and `v23_sector_alpha` regularly hit **59-61% H2 Accuracy**.

**The Vulnerability (Why previous V23 iterations were cleared):**
Despite incredible directional accuracy, the early V23 configurations suffered in the mock backtest returns (-12% to +4%). The neural networks were too shallow (LSTMs capped at 2, limited hidden layers). They became overly cautious, correctly predicting individual minor dips but missing out on massive continuous multi-month market rallies.

**The Fix:**
V23 configurations are actively having their hyperparameter search spaces expanded to combine the massive 60%+ H2 accuracy of KAN Multi-Res layers with deep Transformer blocks (up to 8 layers, 16 attention heads, and 1024 hidden dimension capacity) to better memorize continuous temporal momentum.
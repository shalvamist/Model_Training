# 📊 V18 Models Evaluation Report
**Date:** 2026-02-10
**Status:** ✅ All 12 V18 Models Evaluated | Compared vs Previous Candidates

---

## 🏆 Executive Summary

The V18 architectural upgrades produced **major breakthroughs in two strategies** and mixed results in the other two.

| Strategy | Previous Best | V18 Best | Verdict |
| :--- | :--- | :--- | :--- |
| **Industrial Sniper** | 0.59 Sharpe (H2) | **0.78 Sharpe (H2)** | 🚀 **+32% improvement** |
| **Real Economy** | 0.62 Sharpe (H2) | 0.52 Sharpe (H2) | ⚠️ Slight regression |
| **Deep Cycle** | -0.81 Sharpe (H2) | **0.47 Sharpe (H2)** | 🚀 **Now profitable!** |
| **Sector Alpha** | **1.36 Sharpe (H2)** | 0.56 Sharpe (H2) | ❌ Regression |

---

## ⚔️ Head-to-Head: Best V18 vs Previous Candidates

### 1. 🏆 Industrial Sniper (V18 WINS)

| Metric | Previous (`rank1`) | V18 Best (`rank3`) | Δ |
| :--- | :--- | :--- | :--- |
| H2 Sharpe | 0.59 | **0.78** | +32% |
| H2 Return | 30.5% | **56.2%** | +84% |
| H2 DirAcc | 44.7% | **58.0%** | +30% |
| H1 DirAcc | 49.5% | **58.5%** | +18% |

> **Winner: V18 Rank 3.** The upgraded V17 with higher hidden_size (256) and n_experts (8) massively improved directional accuracy and returns.

### 2. 📈 Deep Cycle Arbitrage (V18 WINS)

| Metric | Previous (`rank2`) | V18 Best (`rank3`) | Δ |
| :--- | :--- | :--- | :--- |
| H2 Sharpe | -0.81 | **0.47** | ✅ Fixed |
| H2 Return | -41.4% | **26.9%** | ✅ Fixed |
| H1 DirAcc | 66.4% | 60.5% | -9% |
| H2 DirAcc | 40.5% | **50.4%** | +24% |

> **Winner: V18 Rank 3.** Huber Loss + 120-day lookback brought the macro model from "losing money" to profitable.

### 3. ⚠️ Real Economy Rotator (Previous WINS)

| Metric | Previous (`rank2`) | V18 Best (`rank1`) | Δ |
| :--- | :--- | :--- | :--- |
| H2 Sharpe | **0.62** | 0.52 | -17% |
| H2 Return | **39.1%** | 31.2% | -20% |
| H2 DirAcc | **67.3%** | 53.7% | -20% |

> **Winner: Previous Candidate.** The V12 architecture was a better fit for this spread-prediction task. The V17 upgrade didn't help here. Keep the original.

### 4. ❌ Sector Alpha (Previous WINS)

| Metric | Previous (`rank2`) | V18 Best (`rank1`) | Δ |
| :--- | :--- | :--- | :--- |
| H2 Sharpe | **1.36** | 0.56 | -59% |
| H2 Return | **99.6%** | 34.3% | -66% |
| H2 DirAcc | **56.5%** | **58.5%** | +3% |

> **Winner: Previous Candidate.** The Asymmetric Directional Loss improved accuracy (+3%) but the overall return dropped significantly. The original Focal Loss config was better tuned. Keep the original.

---

## 🎯 Final Portfolio Decision

| Slot | Model to Use | Source | H2 Sharpe |
| :--- | :--- | :--- | :--- |
| **Sector Alpha** | `sector_alpha_rotator_optimized_rank2` | Previous Candidate | **1.36** |
| **Industrial Sniper** | `industrial_sniper_v18_optimized_rank3` | 🆕 V18 | **0.78** |
| **Real Economy** | `real_economy_rotator_optimized_rank2` | Previous Candidate | **0.62** |
| **Deep Cycle** | `deep_cycle_arbitrage_v18_optimized_rank3` | 🆕 V18 | **0.47** |

**Portfolio Avg H2 Sharpe: 0.81** (up from 0.41 previously)

---

## 🚀 Recommended Actions
1. **Promote** `industrial_sniper_v18_optimized_rank3` → `rank_candidates/`
2. **Promote** `deep_cycle_arbitrage_v18_optimized_rank3` → `rank_candidates/`
3. **Keep** existing Sector Alpha and Real Economy candidates untouched
4. **Clean up** remaining V18 experimental runs from `weights/`

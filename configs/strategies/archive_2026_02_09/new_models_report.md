# 📊 New Models Evaluation Report
**Date:** 2026-02-09
**Status:** ✅ Re-Training Complete | 🔄 Comparison vs Candidates

## 🏆 Executive Summary
We compared the **Newly Trained Models** (using updated architectures) against the **Rank Candidates** (top performers from the previous batch).

**Key Findings:**
1.  **Industrial Sniper FIXED:** The switch to *Focal Loss* worked. The new `industrial_sniper_optimized_rank1` is finally profitable (+30% return, 0.59 Sharpe), whereas previous versions failed.
2.  **Sector Alpha Regression:** The newly trained Sector Alpha models underperformed the "Candidate" (Rank 2). The Candidate remains the gold standard.
3.  **Real Economy Regression:** The new models failed to replicate the 67% accuracy of the Candidate.

---

## ⚔️ Comparison Table

| Strategy | Metric | **Candidate (Saved)** | **New Best (Rank 1/2/3)** | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Sector Alpha** | H2 Sharpe | **1.36** | 0.74 (Rank 2) | 🏆 **Keep Candidate** |
| | H2 Return | **99.6%** | 38.1% | |
| **Real Economy** | H2 DirAcc | **67.3%** | 55.0% (Rank 2) | 🏆 **Keep Candidate** |
| | H2 Sharpe | **0.62** | 0.54 | |
| **Industrial Sniper**| H2 Sharpe | *Failed (<0)* | **0.59** (Rank 1) | 🚀 **Adopt New Model** |
| | H2 Return | *Failed* | **30.5%** | |
| **Deep Cycle** | H1 Sharpe | *Failed (<0)* | **0.54** (Rank 2) | 📈 **Improved (Mixed)** |
| | H1 Return | *Failed* | 31.1% | |

---

## 🔍 Deep Dive

### 1. Industrial Sniper (The Comeback)
*   **Change:** Switched from Quantile Regression -> **Focal Loss Classification**.
*   **Result:** Success. Rank 1 achieved a **0.59 Sharpe** and **30.5% Return**.
*   **Why:** The model can now effectively identify "Oversold" conditions without being forced to predict exact price levels (which is harder).
*   **Action:** Promote `weights/industrial_sniper_optimized_rank1` to `rank_candidates`.

### 2. Sector Alpha & Real Economy (Stick with Quality)
*   The **Candidate** models (saved in `weights/rank_candidates/`) are significantly better than the fresh runs.
*   This is common in deep learning; sometimes specific random seeds converge to better minima.
*   **Action:** Do NOT overwrite the candidates. Keep using the saved versions.

### 3. Deep Cycle Arbitrage
*   **Change:** Switched from PatchTST -> **Hybrid V12**.
*   **Result:** Improved from "losing money" to "making money" (Sharpe 0.54).
*   **Verdict:** It's now a "decent" model, but still not a star. Useful for diversification.

---

## 🚀 Final Decisions

1.  **Industrial Sniper:** Move `weights/industrial_sniper_optimized_rank1` -> `weights/rank_candidates/`.
2.  **Deep Cycle:** Move `weights/deep_cycle_arbitrage_optimized_rank2` -> `weights/rank_candidates/`.
3.  **Cleanup:** Delete the inferior re-trained runs.

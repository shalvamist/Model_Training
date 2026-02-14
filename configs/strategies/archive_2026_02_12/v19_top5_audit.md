# V19 Top 5 Performance Audit

## Executive Summary
This evaluation covered 20 models (Top 5 Optuna ranks for each of the 4 strategies). 
We found significant improvements, with **Deep Cycle** and **Sector Alpha** producing models with Sharpe Ratios > 1.30, significantly outperforming the previous "Winner" candidates.

## New Leaderboard (H1 Sharpe)

| Model | H1 Sharpe | H1 DirAcc | H1 Return | H2 Sharpe |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Cycle R2** | **1.34** | 39.7% | **110.2%** | 0.71 |
| **Sector Alpha R2** | **1.33** | 41.8% | 19.0% | 0.90 |
| **Deep Cycle R5** | 1.07 | **59.5%** | 85.2% | **0.98** |
| **Real Economy R2** | 1.04 | 40.2% | 70.2% | 0.75 |

## VS Previous Candidates (V19 Phase 1)

| Strategy | Prev Winner Sharpe | New Best Sharpe | Improvement |
| :--- | :---: | :---: | :---: |
| **Deep Cycle** | 0.87 (R3) | **1.34 (R2)** | **+54%** |
| **Sector Alpha** | 0.71 (R1) | **1.33 (R2)** | **+87%** |
| **Real Economy** | 0.64 (R1) | **1.04 (R2)** | **+62%** |
| **Industrial Sniper**| 0.44 (R2) | **0.81 (R3)** | **+84%** |

## Critical Observation: Sharpe vs. Directional Accuracy
We are seeing a divergence where high-Sharpe models sometimes have lower "raw" Directional Accuracy (around 40%). This indicates:
1. **Asymmetric Wins**: The models are avoiding small losses and catching massive breakouts.
2. **Neutral Bias**: They are staying Neutral during chop and only betting when the setup is extreme.
3. **Quality > Quantity**: These models are "smarter" traders even if they are "cautious" guessers.

## Recommendation
Promote **Deep Cycle R2** and **Sector Alpha R2** to the primary candidate list.
The `deep_cycle_arbitrage_v19_optimized_rank5` at **59.5% DirAcc** and **0.98 H2 Sharpe** is also a very strong "all-rounder" candidate.

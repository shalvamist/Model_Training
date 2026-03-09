# The V23 Multi-Res Deployment Fleet (March 2026)

This repository contains the configuration JSON files for the `V23_Deploy` Apex models.
The heavy PyTorch `.pth` binary weights are hosted entirely off-site in cold storage (Google Drive) and are **not** committed to this repository.

## Re-Assembling the Fleet 
To execute these models in the live trading environment:
1. Ensure your local `Model_trading_training/weights/V23_Deploy/` folder exists.
2. Download the heavy `.pth` weights from your GDrive backup (`G:\My Drive\projects\Models\V23_Trading_Backups\V23_WEIGHT_VAULT.zip`).
3. Extract the weights into `V23_Deploy`.
4. Run standard integration scripts.

---

## 🏆 How These Models Were Discovered
The original PyTorch training sweeps were consistently failing to reproduce legendary initial runs. Over the course of Iteration 1-11, we utilized a highly systematic funnel approach to force the models into perfection:

1. **Capacity Locking:** Optuna was previously down-sizing the network width. We permanently locked the network to *Titan Scale* (`hidden_size: 1024`, `batch_size: 32`) preventing the search algorithm from taking shallow, safer routes.
2. **Deterministic Seed Sweeping:** Once locked, we repeatedly trained identical networks using hundreds of parallel exact random seeds. This allowed us to map the initialization basins and find "monsters" that randomly started at the exact right math coordinates.
3. **Neighborhood Sweeps (Dropout):** Once a "monster" was found, we took its base and ran a tight neighborhood sweep specifically altering the `dropout` parameter, resulting in the "Low Dropout" masterpieces.
4. **Structural Depth Geometry:** Finally, we forcefully altered the geometry itself (e.g. going from 4 LSTM layers to 3 LSTM layers) while keeping the Titan scale intact, leading to the ultimate "Holy Grail."

---

## 🏷️ Naming Conventions
The final 6 Apex models use a specific tag system to denote exactly how they were discovered and what they do.

*   `..._PROTECTED_...`: Means this model was manually vetted and explicitly protected from automated archive deletion.
*   `..._SEED_SWEEP_...`: Means this model was discovered during the massive 100-seed deterministic brute-force passes.
*   `..._LOW_DROP_...`: Indicates the model was derived by scaling the dropout layers down on a known successful architecture.
*   `..._lstm3_...`: Indicates the model geometry was physically altered to use 3 LSTM layers instead of the default 4, reducing memory drag.
*   `..._87PERCENT_...`: The trailing number indicates the exact Directional Accuracy or Simulated Return achieved during training evaluation as a fast sanity check.

---

## The Model Hierarchy

**1. The Alpha Generators (The Core Index Predictors)**
- `v23_alpha_gen_lstm3_68302_PROTECTED_HOLY_GRAIL_87PERCENT_2026_03_07` 
  - *(H2 Base Index Predator: 61.1% Acc, +87% Return)*
- `v23_alpha_generation_rank5_84667_PROTECTED_H1_MONSTER_118SHARPE_2026_03_07` 
  - *(H1 Base Index Scalpel: 1.18 Sharpe)*

**2. The Sector Alphas (The Divergence Hunters)**
- `v23_sector_alpha_rank4_seed_25613_PROTECTED_SEED_SWEEP_135PERCENT_2026_03_06` 
  - *(The +135% Divergence Target)*
- `v23_sector_alpha_rank5_seed_88778_PROTECTED_ROTATIONAL_SWEEP_55PERCENT_2026_03_06` 
  - *(Momentum Confirmation Target)*

**3. The Risk Controllers**
- `v23_fortress_master_rank5_PROTECTED_UNIFIED_LONG_2026_03_04` 
  - *(Slow Macro Aggregator)*
- `v23_bear_headhunter_rank6_PROTECTED_CRASH_CATCHER_PRIMARY_2026_03_04` 
  - *(Extreme Downside Protection)*

For deeper configuration details and optimal Orchestrator blending logic, see the `fleet_analysis_report.md` artifact.

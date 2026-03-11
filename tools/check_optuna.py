import sys
import os
import json
import optuna

def main():
    config_path = r"c:\projects\QQQ_Reg_Models_v2\Model_trading_training\configs\v23_alpha_generation.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    opt_config = config.get("optuna", {})
    print("Optuna config:", opt_config)
    
    def mock_objective(trial):
        lr = trial.suggest_float("lr", opt_config["lr"][0], opt_config["lr"][1], log=True)
        bs = trial.suggest_categorical("batch_size", opt_config["batch_size"])
        hs = trial.suggest_categorical("hidden_size", opt_config["hidden_size"])
        dr = trial.suggest_float("dropout", opt_config["dropout"][0], opt_config["dropout"][1])
        return 1.0

    study = optuna.create_study()
    study.optimize(mock_objective, n_trials=10)
    for t in study.trials:
        print(t.params)

if __name__ == "__main__":
    main()

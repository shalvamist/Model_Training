import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, fbeta_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
import joblib
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    @staticmethod
    def get_f1_5_score(true, pred):
        """F1.5 Score favors Recall (Safety) over Precision"""
        return fbeta_score(true, pred, beta=1.5, average='macro', zero_division=0)
    
    @staticmethod
    def get_precision_recall(true, pred, focus_class=0):
        p = precision_score(true, pred, labels=[focus_class], average=None, zero_division=0)[0]
        r = recall_score(true, pred, labels=[focus_class], average=None, zero_division=0)[0]
        return p, r

    @staticmethod
    def get_regression_metrics(true, pred):
        """Calculate MSE, RMSE, MAE, R2"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    @staticmethod
    def get_directional_accuracy(true, pred):
        """Percentage of times prediction sign matches target sign"""
        # Align signs: +1 for >0, -1 for <0 (0 is tricky, treat as no-match or handle specifically)
        # Here we just check if (true * pred) > 0
        correct_direction = np.sum((true * pred) > 0)
        total = len(true)
        if total == 0: return 0.0
        return correct_direction / total

    @staticmethod
    def get_classification_metrics(true, pred):
        """Precision, Recall, F1, MCC for multiclass"""
        from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(true, pred)
        acc = accuracy_score(true, pred)
        return {"Precision": precision, "Recall": recall, "F1": f1, "MCC": mcc, "Accuracy": acc}

    @staticmethod
    def get_simple_trading_simulation(returns, pred_class, bull_class=2, bear_class=0):
        """
        Simulate a simple strategy:
        - Long if Bull
        - Short if Bear (or Cash if short not desirable, here we assume Short)
        - Neutral: Cash (0 return)
        
        returns: array of actual period returns
        pred_class: array of predicted classes
        """
        if len(returns) != len(pred_class):
            return {}

        strategy_returns = np.zeros_like(returns)
        
        # Long
        strategy_returns[pred_class == bull_class] = returns[pred_class == bull_class]
        # Short (inverse return)
        strategy_returns[pred_class == bear_class] = -1.0 * returns[pred_class == bear_class]
        # Neutral (0) - already 0
        
        # Cumulative
        cum_market = np.cumprod(1 + returns) - 1
        cum_strategy = np.cumprod(1 + strategy_returns) - 1
        
        total_market_return = cum_market[-1] if len(cum_market) > 0 else 0
        total_strategy_return = cum_strategy[-1] if len(cum_strategy) > 0 else 0
        
        # Sharpe (Annualized assuming daily data ~252)
        # If returns are not daily, this is just a 'ratio'
        std_strat = np.std(strategy_returns)
        mean_strat = np.mean(strategy_returns)
        sharpe = (mean_strat / std_strat * np.sqrt(252)) if std_strat > 1e-9 else 0
        
        # Win Rate (Positive returns / Total Trades (Bull+Bear))
        active_trades = (pred_class == bull_class) | (pred_class == bear_class)
        n_trades = np.sum(active_trades)
        if n_trades > 0:
            wins = np.sum(strategy_returns[active_trades] > 0)
            win_rate = wins / n_trades
        else:
            win_rate = 0.0
            
        return {
            "Total_Market_Return": total_market_return,
            "Total_Strategy_Return": total_strategy_return,
            "Sharpe_Ratio": sharpe,
            "Win_Rate": win_rate,
            "Num_Trades": int(n_trades)
        }

def robust_normalize(series, window=60):
    """Pandas rolling robust normalization"""
    roll_median = series.rolling(window=window).median()
    roll_q75 = series.rolling(window=window).quantile(0.75)
    roll_q25 = series.rolling(window=window).quantile(0.25)
    iqr = roll_q75 - roll_q25
    iqr = iqr.replace(0, 1e-6)
    return (series - roll_median) / iqr

def save_scalers(scaler_1, scaler_2, path_prefix):
    joblib.dump(scaler_1, f"{path_prefix}_scaler_1.pkl")
    joblib.dump(scaler_2, f"{path_prefix}_scaler_2.pkl")

def load_scalers(path_prefix):
    s1 = joblib.load(f"{path_prefix}_scaler_1.pkl")
    s2 = joblib.load(f"{path_prefix}_scaler_2.pkl")
    return s1, s2

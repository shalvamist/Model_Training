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
        return fbeta_score(true, pred, beta=1.5, labels=[0], average=None, zero_division=0)[0]
    
    @staticmethod
    def get_precision_recall(true, pred, focus_class=0):
        p = precision_score(true, pred, labels=[focus_class], average=None, zero_division=0)[0]
        r = recall_score(true, pred, labels=[focus_class], average=None, zero_division=0)[0]
        return p, r

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

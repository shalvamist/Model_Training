import pandas as pd
import numpy as np
import logging
from .utils import robust_normalize
from .financial_engineer import FinancialEngineer

logger = logging.getLogger(__name__)

class FeatureRegistry:
    """
    Registry of available transformation functions.
    Maps string names (from config) to executable functions.
    """
    _REGISTRY = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._REGISTRY[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name):
        if name not in cls._REGISTRY:
            raise ValueError(f"Transformation '{name}' not found in registry.")
        return cls._REGISTRY[name]

# ==========================================
# Standard Transformations
# ==========================================

@FeatureRegistry.register('log_return')
def step_log_return(df, config):
    col = config.get('input', 'underlying_close')
    out = config.get('output', 'log_ret')
    df[out] = np.log(df[col] / df[col].shift(1)).fillna(0)
    
    # Optional: Lags/Rolling Sums of this return
    lags = config.get('lags', [])
    for lag in lags:
        df[f"{out}_{lag}"] = df[out].rolling(lag).sum().fillna(0)
    return df

@FeatureRegistry.register('technical_indicator')
def step_tech_ind(df, config):
    ind = config['indicator']
    out = config.get('output', ind)
    
    fe = FinancialEngineer()
    
    if ind == 'rsi':
        period = config.get('period', 14)
        col = config.get('input', 'underlying_close')
        raw = fe.calculate_rsi(df[col], period)
        if config.get('normalize_100', True):
            df[out] = raw / 100.0
        else:
            df[out] = raw
            
    elif ind == 'atr':
        period = config.get('period', 14)
        df[out] = fe.calculate_atr(df['underlying_high'], df['underlying_low'], df['underlying_close'], period)
        
    elif ind == 'macd':
        fast = config.get('fast', 12)
        slow = config.get('slow', 26)
        col = config.get('input', 'underlying_close')
        df[out] = fe.calculate_macd(df[col], fast, slow)
    
    # Add more as needed
    return df

@FeatureRegistry.register('gap')
def step_gap(df, config):
    # (Open - PrevClose) / PrevClose
    open_col = config.get('open_col', 'underlying_open')
    close_col = config.get('close_col', 'underlying_close')
    out = config.get('output', 'gap')
    
    prev_close = df[close_col].shift(1)
    df[out] = (df[open_col] - prev_close) / (prev_close + 1e-6)
    df[out] = df[out].fillna(0)
    return df

@FeatureRegistry.register('difference')
def step_difference(df, config):
    # Col - Col.shift(lag)
    # Used for acceleration (Slope - Slope.shift)
    col = config['input']
    lag = config.get('lag', 1)
    out = config.get('output', f"{col}_diff_{lag}")
    
    df[out] = df[col] - df[col].shift(lag)
    df[out] = df[out].fillna(0)
    return df

@FeatureRegistry.register('rolling_stat')
def step_rolling(df, config):
    col = config['input']
    window = config['window']
    stat = config.get('stat', 'mean') # mean, std, skew, kurt, min, max
    out = config.get('output', f"{col}_{stat}_{window}")
    
    roll = df[col].rolling(window=window)
    if stat == 'mean': s = roll.mean()
    elif stat == 'std': s = roll.std()
    elif stat == 'skew': s = roll.skew()
    elif stat == 'min': s = roll.min()
    elif stat == 'max': s = roll.max()
    else: raise ValueError(f"Unknown stat: {stat}")
    
    df[out] = s.fillna(0)
    return df

@FeatureRegistry.register('lag_diff')
def step_lag_diff(df, config):
    # (Col - Col_Shift) / (Col_Shift or 1)
    col = config['input']
    lag = config.get('lag', 1)
    pct = config.get('pct_change', True)
    out = config.get('output', f"{col}_diff_{lag}")
    
    if pct:
        df[out] = (df[col] - df[col].shift(lag)) / (df[col].shift(lag) + 1e-6)
    else:
        df[out] = df[col] - df[col].shift(lag)
        
    df[out] = df[out].fillna(0)
    return df

@FeatureRegistry.register('arithmetic')
def step_arithmetic(df, config):
    # Simple col A op col B
    expr = config['formula'] # e.g. "colA / colB"
    out = config['output']
    # Safety: restricting eval scope? evaluating strictly on df columns
    # For now, simplistic eval
    try:
        df[out] = df.eval(expr).fillna(0)
    except Exception as e:
        logger.error(f"Arithmetic Error in {expr}: {e}")
        df[out] = 0.0
    return df

@FeatureRegistry.register('normalize')
def step_normalize(df, config):
    method = config.get('method', 'robust')
    cols = config.get('columns', [])
    window = config.get('window', 60)
    suffix = config.get('suffix', '_norm')
    
    for c in cols:
        if c not in df.columns: continue
        out_name = f"{c}{suffix}" if suffix else c
        
        if method == 'robust':
            df[out_name] = robust_normalize(df[c], window)
        elif method == 'zscore':
            roll = df[c].rolling(window)
            mu = roll.mean()
            sigma = roll.std()
            df[out_name] = ((df[c] - mu) / (sigma + 1e-6)).fillna(0)
            
    return df

@FeatureRegistry.register('custom_func')
def step_custom(df, config):
    # Hook for ad-hoc lambda if provided in code (not json safe usually, but good for extensibility)
    pass
    return df

@FeatureRegistry.register('greeks')
def step_greeks(df, config):
    col_S = config.get('col_S', 'underlying_close')
    col_K = config.get('col_K', 'underlying_close') 
    col_r = config.get('col_r', 'us_treasury_10y')
    col_sigma = config.get('col_sigma', 'iv')
    T_val = config.get('T_years', 65.0/365.0)
    
    fe = FinancialEngineer()
    
    # Calculate Greeks row-by-row to match V13 logic exactly (which used apply)
    # Note: V13 used: row['us_treasury_10y']/100.0, row['iv'], 'call'
    
    def calc_row(row):
        r_val = row[col_r] / 100.0 if config.get('rate_pct', True) else row[col_r]
        return fe.black_scholes_greeks(
            row[col_S], row[col_K], T_val, 
            r_val, row[col_sigma], 'call'
        )

    greeks = df.apply(lambda row: calc_row(row), axis=1, result_type='expand')
    greeks.columns = ['delta', 'gamma', 'theta', 'vega']
    
    for c in greeks.columns:
        df[c] = greeks[c]
        
    return df

@FeatureRegistry.register('fourier')
def step_fourier(df, config):
    col = config.get('input', 'log_ret')
    window = config.get('window', 60)
    out = config.get('output', f"{col}_entropy_{window}")
    
    fe = FinancialEngineer()
    # Ensure input is clean (though calc_spectral_entropy handles filling? No, we should fill here)
    series = df[col].fillna(0)
    df[out] = fe.calculate_spectral_entropy(series, window=window).fillna(0)
    return df

@FeatureRegistry.register('wavelet')
def step_wavelet(df, config):
    col = config.get('input', 'log_ret')
    widths = config.get('widths', [2, 5, 10])
    prefix = config.get('prefix', f"{col}_w")
    
    fe = FinancialEngineer()
    series = df[col].fillna(0)
    wave_df = fe.calculate_wavelet_energy(series, widths=widths)
    
    for c in wave_df.columns:
        # c is like "wavelet_2"
        w_val = c.split('_')[-1]
        out_name = f"{prefix}_{w_val}"
        df[out_name] = wave_df[c].fillna(0)
        
    return df

@FeatureRegistry.register('regime_gmm')
def step_regime_gmm(df, config):
    col = config.get('input', 'log_ret')
    n_components = config.get('n_components', 2)
    out = config.get('output', 'regime')
    
    fe = FinancialEngineer()
    series = df[col].fillna(0)
    
    # If len < n_components, return 0
    if len(series) < n_components + 10:
        df[out] = 0
        return df

    try:
        df[out] = fe.detect_regimes_gmm(series, n_components=n_components)
    except Exception as e:
        logger.error(f"Regime GMM failed: {e}")
        df[out] = 0
        
    return df

@FeatureRegistry.register('bollinger_position')
def step_bollinger(df, config):
    """Where price sits within Bollinger Bands (0=lower, 0.5=middle, 1=upper)."""
    col = config.get('input', 'underlying_close')
    window = config.get('window', 20)
    n_std = config.get('n_std', 2.0)
    out = config.get('output', f'bb_pos_{window}')
    
    sma = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    
    band_width = upper - lower
    df[out] = ((df[col] - lower) / (band_width + 1e-8)).clip(0, 1).fillna(0.5)
    return df

@FeatureRegistry.register('volume_ratio')
def step_volume_ratio(df, config):
    """Current volume relative to its moving average — detects volume surges."""
    col = config.get('input', 'volume')
    window = config.get('window', 20)
    out = config.get('output', f'vol_ratio_{window}')
    
    vol_sma = df[col].rolling(window).mean()
    df[out] = (df[col] / (vol_sma + 1e-8)).fillna(1.0).clip(0, 5)
    return df

@FeatureRegistry.register('close_vs_ma')
def step_close_vs_ma(df, config):
    """Percentage distance of close from its moving average."""
    col = config.get('input', 'underlying_close')
    window = config.get('window', 7)
    out = config.get('output', f'close_vs_ma{window}')
    
    ma = df[col].rolling(window).mean()
    df[out] = ((df[col] - ma) / (ma + 1e-8)).fillna(0)
    return df

@FeatureRegistry.register('roc')
def step_roc(df, config):
    """Rate of Change: (price - price_n) / price_n — pure momentum."""
    col = config.get('input', 'underlying_close')
    period = config.get('period', 3)
    out = config.get('output', f'roc_{period}')
    
    shifted = df[col].shift(period)
    df[out] = ((df[col] - shifted) / (shifted + 1e-8)).fillna(0)
    return df

class TransformationPipeline:
    def __init__(self, config_steps):
        self.steps = config_steps
        
    def run(self, df):
        if df.empty: return df
        
        for step_cfg in self.steps:
            step_name = step_cfg['step']
            try:
                func = FeatureRegistry.get(step_name)
                df = func(df, step_cfg)
            except Exception as e:
                logger.error(f"Pipeline Step Failed: {step_name} - {e}")
                # We might choose to crash or continue. For now, log.
                raise e
        return df

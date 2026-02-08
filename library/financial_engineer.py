import pandas as pd
import numpy as np
from scipy.stats import norm, entropy
from scipy import signal

class FinancialEngineer:
    @staticmethod
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(series, fast=12, slow=26):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2

    @staticmethod
    def calculate_atr(high, low, close, period=14):
        h_l = high - low
        h_c = np.abs(high - close.shift())
        l_c = np.abs(low - close.shift())
        ranges = pd.concat([h_l, h_c, l_c], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_width(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        width = (upper - lower) / (sma + 1e-6)
        return width

    @staticmethod
    def calculate_stochastic(high, low, close, period=14, smooth_k=3):
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        k = 100 * (close - low_min) / (high_max - low_min + 1e-6)
        d = k.rolling(window=smooth_k).mean()
        return k, d

    @staticmethod
    def calculate_williams_r(high, low, close, period=14):
        hh = high.rolling(window=period).max()
        ll = low.rolling(window=period).min()
        wr = -100 * (hh - close) / (hh - ll + 1e-6)
        return wr

    @staticmethod
    def calculate_roc(series, period=10):
        return ((series - series.shift(period)) / (series.shift(period) + 1e-6)) * 100

    @staticmethod
    def calculate_rolling_skew(series, window=60):
        return series.rolling(window=window).skew()

    @staticmethod
    def black_scholes_price_vectorized(S, K, T, r, sigma):
        S = np.array(S, dtype=np.float32)
        K = np.array(K, dtype=np.float32)
        r = np.array(r, dtype=np.float32)
        sigma = np.array(sigma, dtype=np.float32)
        
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-3)
        S = np.maximum(S, 1e-6) 
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price
    
    @staticmethod
    def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
        T = np.maximum(T, 1e-5)
        sigma = np.maximum(sigma, 1e-3)
        S = np.maximum(S, 1e-6)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1) / 100 
        return delta, gamma, theta, vega

    @staticmethod
    def calculate_spectral_entropy(series, window=60):
        """
        Calculates Spectral Entropy (Shanon entropy of power spectrum).
        High entropy -> Noise/Random. Low entropy -> Trend/Cyclic.
        """
        def _spec_ent(x):
            # 1. FFT
            # Detrend first to remove 0-freq dominance?
            x_detrend = signal.detrend(x)
            f_val = np.fft.rfft(x_detrend)
            # 2. Power Spectrum
            p_val = np.abs(f_val)**2
            # 3. Normalize to PDF
            p_sum = np.sum(p_val)
            if p_sum == 0: return 0.0
            p_norm = p_val / p_sum
            # 4. Entropy
            return entropy(p_norm)

        # Vectorized rolling apply is hard, standard rolling apply is slow but okay for now.
        return series.rolling(window=window).apply(_spec_ent, raw=True)

    @staticmethod
    def calculate_wavelet_energy(series, widths=[2, 5, 10, 20]):
        """
        Calculates Wavelet Energy using Ricker (Mexican Hat) wavelet.
        Since we lack PyWavelets, we use scipy.signal.cwt.
        Returns energy (sum of squared coeffs) at different scales.
        """
        # Scipy CWT is on whole series. Implementation needs care for look-ahead bias?
        # CWT is centered. To avoid look-ahead, we can't easily use standard CWT on whole history 
        # unless we shift? Or we use it only for "regime identification" in retrospective?
        # Standard approach for causal usage: 
        # Calculate CWT up to T, take value at T. This implies re-calculating CWT at every step -> Very Slow.
        # Approximation: CWT filter has finite support. 
        # Ricker of width A decays. If we effectively "convolve" with Ricker kernel.
        # Convolution is causal if we shift kernel.
        
        # Let's implementation simple convolution with truncated Ricker kernel for causality.
        results = {}
        data = series.values
        # Fill NaNs
        data = np.nan_to_num(data)
        
        for w in widths:
            # Manual Ricker Wavelet (Mexican Hat)
            # A = 2 / (sqrt(3*w) * pi^0.25)
            # t = vector points
            # psi = A * (1 - (t/w)^2) * exp(-(t^2)/(2*w^2))
            
            points = np.arange(-5*w, 5*w+1)
            A = 2 / (np.sqrt(3*w) * (np.pi**0.25))
            wsq = w**2
            vec = points**2
            kernel = A * (1 - vec/wsq) * np.exp(-vec / (2*wsq))
            
            # Normalize kernel energy to 1? Or sum abs to 1?
            # Standard convolution usually expects sum=1 for smoothing, but this is a bandpass.
            # sum(kernel) is near 0.
            # We want to extract energy.
            # Let's normalize L2 norm = 1.
            kernel = kernel / (np.linalg.norm(kernel) + 1e-6)
            
            # Convolve (Same mode, then shift to align for causality?)
            # Valid: only parts where kernel fits.
            # Full: produces output.
            # We want: Output[t] depends on Input[t-k...t].
            # Ricker is symmetric. Center is at 0.
            # So if we convolve, the result at t includes future data (t+half_kernel).
            # We must SHIFT the result by half_kernel length.
            # Or use 'valid' convolution on [Data + Padding].
            
            # Causal Convolution:
            # Convolve with [0, ..., 0, Kernel_Right_Half]? No.
            # Convolve with Kernel, then shift result right by (len(kernel)/2).
            padding = np.zeros(len(points)//2) 
            # Prepend padding to data to allow shift
            # Actually, standard convolution 'same':
            # c[n] = sum x[n-k]h[k]. 
            # If h is symmetric centered at 0, h[k] for k<0 is future?
            # Scipy convolve coords: 
            # To be causal, we need the impulse response to be 0 for t < 0.
            # Ricker is non-causal. We delayed-Ricker.
            
            # Implementation:
            # 1. Pad data at START with Zeros (len = len(kernel)-1)
            # 2. Convolve 'valid'. 
            # This effectively corresponds to a causal filter if kernel is "reversed" time?
            # Let's just use rolling window dot product.
            
            # Rolling Dot Product with Ricker Kernel is cleaner.
            # Kernel length L = 10*w.
            
            res = series.rolling(window=len(points)).apply(lambda x: np.dot(x, kernel), raw=True)
            results[f'wavelet_{w}'] = res
            
        return pd.DataFrame(results, index=series.index)

    @staticmethod
    def detect_regimes_gmm(series, n_components=2):
        """
        Detects market regimes using Gaussian Mixture Model on the series (e.g. Volatility).
        Returns regime labels (0, 1, ...).
        WARNING: Fits on entire series -> potential look-ahead bias if not careful.
        User should ideally fit on training data only, but for this generic processor,
        we process the provided dataframe.
        """
        from sklearn.mixture import GaussianMixture
        
        # Reshape for sklearn
        X = series.values.reshape(-1, 1)
        # Handle NaNs: fill with mean or 0? 
        mean_val = np.nanmean(X)
        if np.isnan(mean_val): mean_val = 0
        X = np.nan_to_num(X, nan=mean_val)
        
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        # Determine order of means to make labels consistent (0=Low, 1=High)
        means = gmm.means_.flatten()
        order = np.argsort(means) 
        
        labels = gmm.predict(X)
        
        # Remap labels
        remapped_labels = np.zeros_like(labels)
        for i, original_label in enumerate(order):
            remapped_labels[labels == original_label] = i
            
        return remapped_labels

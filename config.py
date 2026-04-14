"""Application configuration and defaults."""

import os

# Data provider
DATA_PROVIDER = os.environ.get("DATA_PROVIDER", "yfinance")

# Default ticker
DEFAULT_TICKER = "AAPL"

# Period and interval options
PERIOD_OPTIONS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
DEFAULT_PERIOD = "1y"

INTERVAL_OPTIONS = {
    "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
    "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
    "1mo": ["1d", "5d", "1wk"],
    "3mo": ["1d", "5d", "1wk", "1mo"],
    "6mo": ["1d", "5d", "1wk", "1mo"],
    "1y": ["1d", "5d", "1wk", "1mo"],
    "2y": ["1d", "5d", "1wk", "1mo"],
    "5y": ["1d", "5d", "1wk", "1mo"],
    "max": ["1d", "5d", "1wk", "1mo"],
}
DEFAULT_INTERVAL = "1d"

# Timeframe presets (label, period, interval)
TIMEFRAME_PRESETS = [
    ("1D", "1d", "5m"),
    ("5D", "5d", "15m"),
    ("1M", "1mo", "1d"),
    ("3M", "3mo", "1d"),
    ("6M", "6mo", "1d"),
    ("1Y", "1y", "1d"),
    ("5Y", "5y", "1wk"),
    ("Max", "max", "1mo"),
]

# Auto-refresh
AUTO_REFRESH_INTERVALS = {"10s": 10, "30s": 30, "1m": 60, "5m": 300}
DEFAULT_REFRESH_INTERVAL = "30s"

# Indicator defaults
SMA_PERIOD = 20
EMA_PERIOD = 12
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# Color palette
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#ffbb33",
    "info": "#17becf",
    "up": "#26a69a",
    "down": "#ef5350",
    "sma": "#ff7f0e",
    "ema": "#2ca02c",
    "bollinger_upper": "#9467bd",
    "bollinger_lower": "#9467bd",
    "bollinger_mid": "#8c564b",
    "macd": "#1f77b4",
    "macd_signal": "#ff7f0e",
    "macd_hist_pos": "#26a69a",
    "macd_hist_neg": "#ef5350",
    "volume_up": "#26a69a",
    "volume_down": "#ef5350",
    # Phase 2 indicators
    "vwap": "#e377c2",
    "ichimoku_a": "rgba(0, 128, 0, 0.15)",
    "ichimoku_b": "rgba(255, 0, 0, 0.15)",
    "ichimoku_tenkan": "#0496ff",
    "ichimoku_kijun": "#991515",
    "supertrend_up": "#26a69a",
    "supertrend_down": "#ef5350",
    "keltner_upper": "#bcbd22",
    "keltner_lower": "#bcbd22",
    "keltner_mid": "#7f7f7f",
    "stoch_k": "#1f77b4",
    "stoch_d": "#ff7f0e",
    "adx": "#1f77b4",
    "di_plus": "#26a69a",
    "di_minus": "#ef5350",
    "willr": "#9467bd",
    "atr": "#17becf",
    "support": "#26a69a",
    "resistance": "#ef5350",
    "fibonacci": "#ff7f0e",
    "squeeze_on": "#ff0000",
    "squeeze_off": "#00ff00",
}

# Comparison
MAX_COMPARISON_TICKERS = 5

# Cache TTLs (seconds)
CACHE_TTL_QUOTE = 300       # 5 minutes
CACHE_TTL_HISTORY = 3600    # 1 hour
CACHE_TTL_INFO = 86400      # 24 hours
CACHE_TTL_LIVE = 15         # 15 seconds for live mode

# Composite Signal Score weights (must sum to 1.0)
# Each category contributes this fraction to the final -100..+100 score.
# If a category has no computable indicators, its weight is redistributed
# proportionally to the remaining categories.
SIGNAL_WEIGHTS = {
    "trend":    0.25,   # Price vs SMA/EMA, MACD histogram, ADX direction
    "momentum": 0.25,   # RSI, Stochastic %K, MFI, ROC
    "volume":   0.10,   # CMF, OBV slope
    "pattern":  0.10,   # Candlestick pattern signal
    "ml":       0.30,   # XGBoost + LSTM probability consensus
}

# Label thresholds for the composite score
SIGNAL_THRESHOLDS = {
    "strong_buy":  50,
    "buy":         20,
    "neutral_low": -20,
    "sell":        -50,
    # below sell threshold → "Strong Sell"
}

# ML Models Configuration
# LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 150
LSTM_LEARNING_RATE = 2e-3
LSTM_BATCH_SIZE = 32
LSTM_PATIENCE = 15
LSTM_GRAD_CLIP = 1.0
LSTM_WEIGHT_DECAY = 1e-5

# XGBoost
XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 2
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.6
XGB_COLSAMPLE_BYTREE = 0.5
XGB_MIN_CHILD_WEIGHT = 8
XGB_REG_ALPHA = 1.0
XGB_REG_LAMBDA = 3.0
XGB_EVAL_METRIC = "logloss"
XGB_RANDOM_STATE = 42


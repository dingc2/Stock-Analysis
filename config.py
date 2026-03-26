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

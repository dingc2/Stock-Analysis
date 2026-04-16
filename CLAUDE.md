# CLAUDE.md

## Project

Stock analysis dashboard: Streamlit + Python. Interactive charts, 16+ technical indicators, candlestick patterns, ML direction prediction (XGBoost + LSTM), composite signal scoring.

## Architecture

```
data/           DataProvider ABC (yfinance impl), cache wrappers, Quote dataclass. VIX join via date-key lookup (preserves intraday index).
indicators/     technical.py (pandas-ta), ml_features.py (normalized ratios), composite.py (signal score)
charts/         Plotly: candlestick, volume, comparison
views/          Streamlit tabs: overview, technicals, financials, comparison, volume
ml/             ModelProvider ABC: direction_base.py (shared base), xgboost_direction.py, lstm_direction.py
config.py       All defaults, colors, indicator params, ML hyperparameters
app.py          Entry point: sidebar, tab routing
```

## Commands

```bash
pip install -r requirements.txt && streamlit run app.py   # local
docker compose up --build                                  # docker
pytest                                                     # all tests
pytest -m "not slow"                                       # skip ML
```

## ML Models

Both models predict next-day price direction (up/down). Output columns: `Pred_Direction`, `Pred_Probability`, `Prob_Up`. Metadata in `df.attrs`: `train_accuracy`, `validation_accuracy`, `train_size`, `test_size`, `feature_cols`.

**Architecture:** Shared `BaseDirectionModel(ModelProvider)` in `ml/direction_base.py` handles feature engineering, train/val split, prediction assignment, attrs, and session-scoped caching. Subclasses implement only model-specific training logic.

**Features (shared by both):** 15 indicator columns (RSI, Stochastic, ADX/DI, ROC, MFI, BBB, BBP, VIX + MAs, Fear_Greed_Proxy) + 16 normalized derived features from `ml_features.py` (Price_vs_SMA, Price_vs_EMA, MACD_Norm, MACDh_Norm, MACDs_Norm, ATR_Norm, Return_1d/2d/3d/5d/10d/20d, Volatility_10d, Volume_Ratio, Gap_Return, Daily_Range).

**Key design rules:**
- All features are unitless ratios — no raw prices (enables cross-ticker comparability)
- Predictions assigned to holdout rows only (no data leakage)
- Final model retrains on all labeled data for last-row live forecast
- Features computed via `indicators/technical.py` + `indicators/ml_features.py` (shared with charts)
- Registry (`ml/__init__.py`) catches only `ImportError` — real bugs surface
- Session-scoped cache: identical OHLCV data returns cached predictions for 1 hour (avoids redundant retraining across tab renders)

**XGBoost:** max_depth=2, n_estimators=150, lr=0.03, subsample=0.6, colsample=0.5, early stopping (10 rounds, frac=0.15 from config).
**LSTM:** hidden=64, layers=1, dropout=0.2, seq_len=20, epochs=150, lr=2e-3, patience=15, ReduceLROnPlateau scheduler.

## Key Conventions

- `views/` modules export `render()`. Shared `render_timeframe_buttons()` in `views/__init__.py`.
- `@st.cache_data` TTLs: 5min quotes, 1hr history, 24hr info, 15s live.
- **Intervals:** `config.INTERVAL_OPTIONS` aligned to yfinance limits: `1m` ≤ 7d, `2m`–`90m` ≤ 60d, `1d`+ unlimited. Never `.normalize()` the main df index — it destroys intraday timestamps.
- Sidebar state: `selected_ticker`, `period`, `interval`, `auto_refresh`, `auto_refresh_seconds`.
- Auto-refresh via `@st.fragment(run_every=...)`.
- Tests: XGBoost must run before LSTM (BLAS segfault on macOS/Python 3.13). Handled by `tests/ml/conftest.py` + `OMP_NUM_THREADS=1` in root conftest.

## Adding New Components

**Data provider:** Implement `DataProvider` in `data/base.py` → register in `data/__init__.py`.
**ML model:** Implement `ModelProvider` in `ml/base.py` → register in `ml/__init__.py`.
**View:** Create `views/new_tab.py` with `render()` → add tab in `app.py`.

## Roadmap

- ~~Model persistence/caching (avoid retraining every page load)~~ — done (session-scoped OHLCV hash cache, 1hr TTL)
- Walk-forward cross-validation for reported metrics
- Feature importance display, probability calibration
- LightGBM/CatBoost, stacking ensemble, Temporal Fusion Transformer
- Risk analytics (Sharpe, VaR, GARCH), portfolio optimization
- Sentiment (FinBERT, news, Reddit), options chain, backtesting

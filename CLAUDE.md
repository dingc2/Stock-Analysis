# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Vision

A personal, extensible stock analysis platform built with Python and Streamlit, containerized with Docker. The app pulls free real-time and historical market data, presents interactive visualizations with technical indicators, and is architected from day one to support pluggable ML models (price prediction, signal detection, anomaly detection, etc.) without modifying existing code.

### Goals
1. **Barebones first** -- ship a working dashboard with charts, indicators, and multi-ticker comparison before adding ML
2. **Zero vendor lock-in** -- data providers are swappable via config (yfinance today, Finnhub/Alpha Vantage tomorrow)
3. **ML-ready architecture** -- indicators layer is shared between charts and models so features are always consistent
4. **Dockerized** -- `docker compose up` and it works, no local Python setup required

## Architecture

### Data Layer (Strategy Pattern)
- `data/base.py` -- `DataProvider` ABC: `get_quote()`, `get_history()`, `get_info()`, `search_ticker()`, `validate_ticker()`. Also contains `Quote` dataclass.
- `data/yfinance_provider.py` -- Concrete implementation using yfinance (free, no API key)
- `data/cache.py` -- Streamlit `@st.cache_data` wrappers with TTLs (5min quotes, 1hr history, 24hr company info, 15s live mode)
- `data/__init__.py` -- `get_provider()` factory that reads `DATA_PROVIDER` from config/env

**To add a new data provider:**
1. Create `data/new_provider.py` implementing `DataProvider`
2. Add an `elif` branch in `data/__init__.py`
3. Set `DATA_PROVIDER = "new_provider"` in `config.py` or via env var

### Indicators Layer (Stateless Pure Functions)
`indicators/technical.py` contains pure functions that take an OHLCV DataFrame and return it with added columns. No Streamlit dependency, no side effects. Uses `pandas-ta` (pure Python, no C deps).

Core: `add_sma()`, `add_ema()`, `add_rsi()`, `add_macd()`, `add_bollinger()`, `add_all()`
Advanced: VWAP, Ichimoku, Supertrend, ADX, Stochastic, Williams %R, Keltner, ATR, OBV, A/D Line, CMF, ROC, MFI
`add_all(df, include_advanced=True)` adds core + Stochastic, ADX, ATR, ROC, MFI (used by ML models)
Patterns: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star (pure Python, no TA-Lib)
Levels: Support/Resistance (scipy peak detection), Fibonacci retracement, Bollinger Squeeze

**Critical design rule:** ML models MUST import from `indicators/technical.py` to compute their features. This guarantees the model trains on the exact same indicator calculations the user sees in the charts.

### Charts Layer (Plotly)
- `charts/price.py` -- Candlestick chart with indicator overlays, RSI/MACD subplots
- `charts/volume.py` -- Color-coded volume bars (green=up, red=down), volume profile
- `charts/comparison.py` -- Normalized percentage-change overlay for multi-ticker comparison

### Views (Streamlit Tabs)
Each view module exports a `render()` function called by `app.py` inside the appropriate tab context:
- `views/__init__.py` -- Shared `render_timeframe_buttons()` widget used by all chart views
- `views/overview.py` -- Current quote, key metrics, mini sparkline chart
- `views/technicals.py` -- Full interactive chart, indicator checkboxes/sliders, RSI/MACD subplots, ML overlay hook
- `views/financials.py` -- P/E, EPS, dividend yield, beta, 52-week range, revenue, profit margin
- `views/comparison.py` -- Multi-ticker selector (up to 5), normalized price comparison, metrics table
- `views/volume.py` -- Volume bars with MA overlay, volume-price correlation, day-of-week patterns, OBV/AD/CMF

All chart pages include timeframe preset buttons (1D/5D/1M/3M/6M/1Y/5Y/Max) and auto-refresh support via `@st.fragment(run_every=...)`.

### ML Models
- `ml/base.py` -- `ModelProvider` ABC: `predict(df) -> df`, `get_name()`, `get_description()`
- `ml/__init__.py` -- `get_available_models()` returns registered models; each import is isolated so one backend failure doesn't block others
- `ml/xgboost_direction.py` -- XGBoost 1-day price direction classifier (30 features)
- `ml/lstm_direction.py` -- PyTorch LSTM 1-day price direction classifier with early stopping, gradient clipping, dropout regularization (26 features)

Both models use `add_all(df, include_advanced=True)` to compute indicators from `indicators/technical.py`.
Both share 18 indicator columns: SMA, EMA, RSI, MACD (3), Bollinger (5), Stochastic (2), ADX+DI (3), ATR, ROC, MFI.
XGBoost adds 12 derived features: Return_1d/2d/3d/5d/10d/20d, Volatility_10d, Price_vs_SMA, Volume_Ratio, Gap_Return, Daily_Range.
LSTM adds 7 derived features: Return_1d/5d, Volatility_10d, Price_vs_SMA, Volume_Ratio, Gap_Return, Daily_Range.
(XGBoost gets more lag returns because it has no sequence memory; LSTM learns temporal patterns from its 20-step window.)

XGBoost hyperparameters: max_depth=2, n_estimators=150, lr=0.03, subsample=0.6, colsample_bytree=0.5, min_child_weight=8, reg_alpha=1.0, reg_lambda=3.0. Uses early stopping (10 rounds, 15% eval split) to prevent overfitting. HOLDOUT_ROWS=30.
LSTM hyperparameters: hidden_size=48, num_layers=1, dropout=0.3, seq_len=20, epochs=50, lr=1e-3, patience=8, weight_decay=1e-4, grad_clip=1.0, batch_size=32.

Both models output: `Pred_Direction` (1=Up, 0=Down), `Pred_Probability` (confidence), `Prob_Up` (raw P(Up)).
Both set `df.attrs`: `train_accuracy`, `validation_accuracy`, `train_size`, `test_size`, `feature_cols`.
Both warn via `warnings.warn` when indicator columns are missing from the feature set.

**To add an ML model:**
1. Create `ml/your_model.py` implementing `ModelProvider`
2. Register it in `ml/__init__.py`
3. No other code changes needed -- the UI auto-discovers registered models

### Data Flow
```
yfinance API (or other provider)
       |
       v  data/yfinance_provider.py (DataProvider ABC)
       |
       v  data/cache.py (@st.cache_data wrappers, 15s TTL in live mode)
       |
       +------> views/overview.py -----> charts/price.py (line chart)
       |
       +------> indicators/technical.py (shared feature engineering)
       |              |
       |              +------> views/technicals.py --> charts/price.py (candlestick + overlays)
       |              |
       |              +------> ml/xgboost_direction.py, ml/lstm_direction.py
       |
       +------> views/financials.py (ticker.info dict)
       |
       +------> views/comparison.py --> charts/comparison.py (normalized overlay)
       |
       +------> views/volume.py -----> charts/volume.py (volume bars + profile)
```

## Project Structure

```
stock-analysis/
├── .gitignore
├── .dockerignore
├── Dockerfile                 # python:3.12-slim, no C deps needed
├── docker-compose.yml         # port 8501, volume mount for dev hot-reload
├── requirements.txt           # streamlit, yfinance, pandas, pandas-ta, plotly, numpy, xgboost, torch
├── config.py                  # Defaults, color palette, indicator params, ML hyperparameters, DATA_PROVIDER, timeframe presets
├── app.py                     # Streamlit entry: page config, sidebar (incl. live refresh), st.tabs routing
├── CLAUDE.md
├── data/
│   ├── __init__.py            # get_provider() factory
│   ├── base.py                # DataProvider ABC + Quote dataclass
│   ├── yfinance_provider.py   # yfinance implementation
│   └── cache.py               # @st.cache_data wrappers (5min/1hr/24hr TTLs + 15s live)
├── indicators/
│   ├── __init__.py
│   ├── technical.py           # All indicators via pandas-ta + pure Python patterns
│   ├── composite.py           # Composite Signal Score calculator
│   └── ml_features.py         # ML-specific feature engineering (Fear/Greed, volatility, etc.)
├── charts/
│   ├── __init__.py
│   ├── price.py               # Candlestick + indicator overlays + subplots
│   ├── volume.py              # Volume bars, volume profile
│   └── comparison.py          # Normalized multi-ticker chart
├── views/
│   ├── __init__.py            # Shared render_timeframe_buttons() widget
│   ├── overview.py            # Quote + metrics + mini chart
│   ├── technicals.py          # Full chart + indicator controls + ML overlay hook
│   ├── financials.py          # Key financial metrics table
│   ├── comparison.py          # Multi-ticker normalized comparison
│   └── volume.py              # Volume analysis + patterns
└── ml/
    ├── __init__.py            # get_available_models() registry
    ├── base.py                # ModelProvider ABC
    ├── xgboost_direction.py   # XGBoost 1-day price direction classifier
    └── lstm_direction.py      # PyTorch LSTM 1-day price direction classifier
```

## Commands

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Run with Docker
```bash
docker compose up --build       # builds and starts at http://localhost:8501
docker compose up               # subsequent runs (no rebuild)
docker compose down             # stop
```

### Switch data provider
```bash
DATA_PROVIDER=yfinance          # default, free, no API key
# DATA_PROVIDER=finnhub         # future: requires FINNHUB_API_KEY env var
# DATA_PROVIDER=alphavantage    # future: requires ALPHA_VANTAGE_API_KEY env var
```

## Dependencies

```
streamlit==1.41.0
yfinance>=1.2.0
pandas>=2.3.2
pandas-ta>=0.4.71b0
plotly==5.24.1
numpy>=1.26.4
scipy>=1.11.0
xgboost>=2.0.0
scikit-learn>=1.3.0
torch>=2.3.0
```

- Python 3.13 (dev machine), Dockerfile uses python:3.12-slim
- pandas-ta chosen over TA-Lib specifically because it's pure Python (no C library = trivial Docker builds)
- On macOS, xgboost may require `libomp` (`brew install libomp`) for `libxgboost.dylib` to load
- yfinance 1.2.0+ required (0.2.x had broken Yahoo Finance API compatibility)
- `ticker.fast_info` for quick price lookups (attribute access, not `.get()`), `ticker.info` for full fundamentals
- `yfinance.Search(query).quotes` for ticker search (1.x+ API)

## Key Design Decisions

- Uses `st.tabs` (not Streamlit multipage) -- all tabs share sidebar state via `st.session_state`
- Sidebar state keys: `selected_ticker`, `period`, `interval`, `comparison_tickers`, `auto_refresh`, `auto_refresh_seconds`
- Auto-refresh uses `@st.fragment(run_every=timedelta(...))` to re-execute chart sections without full page rerun
- Config defaults for indicators: SMA(20), EMA(12), RSI(14), MACD(12/26/9), Bollinger(20, 2.0)
- ML models retrain from scratch on every Streamlit rerun (no persistence/checkpointing yet)

## Completed Phases

- **Phase 1** -- Dashboard with charts, indicators, multi-ticker comparison
- **Phase 2** -- Advanced indicators, candlestick patterns, support/resistance, XGBoost direction model
- **Phase 2.5** -- Timeframe presets, live auto-refresh, market status indicator
- **Phase 3C** -- LSTM baseline with early stopping, gradient clipping, dropout, isolated RNG, NaN-safe last-row prediction
- **Phase 3C.1** -- Expanded feature sets (Stochastic, ADX, ATR, ROC, MFI + price-derived features), differentiated XGBoost (30 features) vs LSTM (26 features), XGBoost overfitting fix via early stopping + regularization tuning, LSTM hyperparameter calibration

## Phase 2.6: Tab-Specific Sidebar Controls (NOT STARTED)

Restructure sidebar so global controls stay at top and tab-specific controls appear below based on active tab. Each view module exports `render_sidebar()`. Tab detection via `set_active_page()` since `st.tabs()` has no native selection state API. All tab-specific session state keys use prefixes (`tech_`, `volume_`, `overview_`, `financials_`, `compare_`).

Trade-off: sidebar shows previous tab's controls for one rerun cycle after switching (no `st.tabs` selection callback).

## Remaining ML Work

- Model persistence/checkpointing to avoid retraining on every rerun
- Configurable sequence length/epochs from UI or config
- Out-of-sample evaluation improvements (ROC-AUC, confusion matrix, rolling validation)
- Feature importance display in UI (XGBoost supports this natively via `feature_importances_`)

## Future Roadmap

### Phase 3: ML Model Suite
- **3A** -- LightGBM + CatBoost direction classifiers
- **3B** -- Stacking ensemble (XGBoost meta-learner over base models)
- **3D** -- Temporal Fusion Transformer (`pytorch-forecasting` or `darts`)

### Phase 4: Risk & Portfolio Analytics
- **4A** -- Risk metrics tab (Sharpe, Sortino, VaR, CVaR, drawdown)
- **4B** -- GJR-GARCH volatility forecasting (`arch`)
- **4C** -- Portfolio optimization (efficient frontier, Mean-Variance, HRP, Black-Litterman)
- **4D** -- Correlation heatmap + rolling correlation

### Phase 5: Sentiment Analysis
- **5A** -- FinBERT news sentiment (`transformers` + `ProsusAI/finbert`)
- **5B** -- News headlines integration (newsapi or RSS)
- **5C** -- Reddit/StockTwits sentiment (`praw`)
- **5D** -- SEC filing analysis (`edgartools`)

### Phase 6: Alternative Data & Options
- **6A** -- Options chain (put/call ratio, volume by strike)
- **6B** -- Unusual options activity detection
- **6C** -- Insider & institutional data (13F, Form 4)
- **6D** -- Short interest tracking

### Phase 7: Advanced ML & Automation
- **7A** -- Time series foundation models (Chronos, TimeGPT, Lag-Llama)
- **7B** -- RL trading agent (FinRL, A2C/PPO)
- **7C** -- Automated chart pattern detection (head & shoulders, wedges, channels)
- **7D** -- Backtesting framework (`vectorbt` or `backtesting.py`)

### Phase 8: Infrastructure & Scale
- **8A** -- Additional data providers (Finnhub, Alpha Vantage, Polygon.io)
- **8B** -- Redis caching for multi-user
- **8C** -- PostgreSQL persistence (watchlists, portfolios, alerts)
- **8D** -- FastAPI backend
- **8E** -- CI/CD (GitHub Actions)

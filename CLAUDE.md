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
The data layer uses the Strategy pattern to decouple the app from any specific stock API.

- `data/base.py` -- `DataProvider` ABC defines the contract: `get_quote()`, `get_history()`, `get_info()`, `search_ticker()`, `validate_ticker()`. Also contains the `Quote` dataclass for current price snapshots.
- `data/yfinance_provider.py` -- Concrete implementation using yfinance (free, no API key)
- `data/cache.py` -- Streamlit `@st.cache_data` wrappers with TTLs (5min quotes, 1hr history, 24hr company info)
- `data/__init__.py` -- `get_provider()` factory that reads `DATA_PROVIDER` from config/env

**To add a new data provider:**
1. Create `data/new_provider.py` implementing `DataProvider`
2. Add an `elif` branch in `data/__init__.py`
3. Set `DATA_PROVIDER = "new_provider"` in `config.py` or via env var

### Indicators Layer (Stateless Pure Functions)
`indicators/technical.py` contains pure functions that take an OHLCV DataFrame and return it with added columns. No Streamlit dependency, no side effects. Uses `pandas-ta` (pure Python, no C deps).

Functions: `add_sma()`, `add_ema()`, `add_rsi()`, `add_macd()`, `add_bollinger()`, `add_all()`

**Critical design rule:** ML models MUST import from `indicators/technical.py` to compute their features. This guarantees the model trains on the exact same indicator calculations the user sees in the charts.

### Charts Layer (Plotly)
- `charts/price.py` -- Candlestick chart with indicator overlays, RSI/MACD subplots
- `charts/volume.py` -- Color-coded volume bars (green=up, red=down), volume profile
- `charts/comparison.py` -- Normalized percentage-change overlay for multi-ticker comparison

### Pages (Streamlit Tabs)
Each page module exports a `render()` function called by `app.py` inside the appropriate tab context:
- `pages/overview.py` -- Current quote, key metrics, mini sparkline chart
- `pages/technicals.py` -- Full interactive chart, indicator checkboxes/sliders, RSI/MACD subplots
- `pages/financials.py` -- P/E, EPS, dividend yield, beta, 52-week range, revenue, profit margin
- `pages/comparison.py` -- Multi-ticker selector (up to 5), normalized price comparison, metrics table
- `pages/volume.py` -- Volume bars with MA overlay, volume-price correlation, day-of-week patterns

### ML Extension Points (Stubs)
- `ml/base.py` -- `ModelProvider` ABC: `predict(df) -> df`, `get_name()`, `get_description()`
- `ml/__init__.py` -- `get_available_models()` returns registered models (empty list for now)
- `pages/technicals.py` checks this list and renders prediction overlays if any models exist

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
       v  data/cache.py (@st.cache_data wrappers)
       |
       +------> pages/overview.py -----> charts/price.py (line chart)
       |
       +------> indicators/technical.py (add SMA, EMA, RSI, MACD, Bollinger)
       |              |
       |              +------> pages/technicals.py --> charts/price.py (candlestick + overlays)
       |              |
       |              +------> ml/models (future: shared feature engineering)
       |
       +------> pages/financials.py (ticker.info dict)
       |
       +------> pages/comparison.py --> charts/comparison.py (normalized overlay)
       |
       +------> pages/volume.py -----> charts/volume.py (volume bars + profile)
```

## Project Structure

```
stock-analysis/
├── .gitignore
├── .dockerignore
├── Dockerfile                 # python:3.12-slim, no C deps needed
├── docker-compose.yml         # port 8501, volume mount for dev hot-reload
├── requirements.txt           # streamlit, yfinance, pandas, pandas-ta, plotly, numpy
├── config.py                  # Defaults, color palette, indicator params, DATA_PROVIDER
├── app.py                     # Streamlit entry: page config, sidebar, st.tabs routing
├── CLAUDE.md
│
├── data/
│   ├── __init__.py            # get_provider() factory
│   ├── base.py                # DataProvider ABC + Quote dataclass
│   ├── yfinance_provider.py   # yfinance implementation
│   └── cache.py               # @st.cache_data wrappers (5min/1hr/24hr TTLs)
│
├── indicators/
│   ├── __init__.py
│   └── technical.py           # SMA, EMA, RSI, MACD, Bollinger via pandas-ta
│
├── charts/
│   ├── __init__.py
│   ├── price.py               # Candlestick + indicator overlays + subplots
│   ├── volume.py              # Volume bars, volume profile
│   └── comparison.py          # Normalized multi-ticker chart
│
├── pages/
│   ├── __init__.py
│   ├── overview.py            # Quote + metrics + mini chart
│   ├── technicals.py          # Full chart + indicator controls + ML overlay hook
│   ├── financials.py          # Key financial metrics table
│   ├── comparison.py          # Multi-ticker normalized comparison
│   └── volume.py              # Volume analysis + patterns
│
└── ml/
    ├── __init__.py            # get_available_models() registry
    ├── base.py                # ModelProvider ABC
    └── xgboost_direction.py   # XGBoost 1-day price direction classifier
```

## Commands

### Run locally
```bash
cd stock-analysis
pip install -r requirements.txt
streamlit run app.py
```

### Run with Docker
```bash
cd stock-analysis
docker compose up --build       # builds and starts at http://localhost:8501
docker compose up               # subsequent runs (no rebuild)
docker compose down             # stop
```

### Switch data provider
```bash
# In docker-compose.yml or shell:
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
xgboost>=2.0.0
scikit-learn>=1.3.0
```

- Python 3.13 (dev machine), Dockerfile uses python:3.12-slim
- yfinance upgraded to 1.2.0+ (0.2.51 had broken Yahoo Finance API compatibility; `fast_info` requires attribute access not `.get()`)
- pandas-ta 0.4.x requires pandas>=2.3.2 (older 0.3.14b1 is not available on Python 3.13)
- pandas-ta chosen over TA-Lib specifically because it's pure Python (no C library = trivial Docker builds)
- numpy unpinned to allow Python 3.13 compatibility
- xgboost for ML price direction classifier (Phase 2)

## Notes

- Uses `st.tabs` (not Streamlit multipage) -- all tabs share sidebar state via `st.session_state`
- Sidebar state keys: `selected_ticker`, `period`, `interval`, `comparison_tickers`
- Config defaults for indicators: SMA(20), EMA(12), RSI(14), MACD(12/26/9), Bollinger(20, 2.0)
- yfinance search API: `yfinance.Search(query).quotes` (available in 1.x+)
- Use `ticker.fast_info` for quick price lookups (attribute access, not `.get()`), `ticker.info` for full fundamentals

## Phase 1 Status: COMPLETE

All Phase 1 code is implemented and the app launches. yfinance upgraded to 1.2.0 which resolved the Yahoo Finance 429 / JSON decode errors.

## Phase 2 Status: COMPLETE — Advanced Indicators & Pattern Recognition

### What was built
- **XGBoost ML model** (`ml/xgboost_direction.py`) — 1-day price direction classifier, auto-discovered by UI
- **11 new indicators** in `indicators/technical.py`: VWAP, Ichimoku Cloud, Supertrend, ADX, Stochastic, Williams %R, Keltner Channels, ATR, OBV, A/D Line, CMF
- **Candlestick pattern detection** — pure Python (no TA-Lib): Doji, Hammer, Shooting Star, Bull/Bear Engulfing, Morning/Evening Star
- **Support/Resistance detection** — scipy peak/trough detection with clustering
- **Fibonacci retracement** — auto-computed from swing high/low over lookback window
- **Bollinger Squeeze** — detects Bollinger Bands inside Keltner Channels
- **Technicals page** reorganized with expanders: Trend Overlays, Bands & Channels, Momentum Oscillators, Trend Strength & Volatility, Patterns & Levels
- **Volume page** enhanced with OBV, A/D Line, and CMF charts
- All indicators are pure functions in `indicators/technical.py`, all chart rendering in `charts/price.py`

---

## Phase 3: ML Model Suite

### 3A: Gradient Boosted Trees
- Add LightGBM and CatBoost direction classifiers alongside XGBoost
- Same `ModelProvider` ABC, same feature set from `indicators/technical.py`
- CatBoost with categorical features (sector, day-of-week)
- New deps: `lightgbm`, `catboost`

### 3B: Ensemble Model
- Stacking meta-learner combining XGBoost + LightGBM + CatBoost
- XGBoost as meta-learner over base model predictions
- Display individual and ensemble accuracy

### 3C: LSTM Sequence Model
- PyTorch LSTM for price direction/magnitude prediction
- Sliding window features from `indicators/technical.py`
- New dep: `torch`

### 3D: Temporal Fusion Transformer
- Multi-horizon forecasting with interpretability
- Via `pytorch-forecasting` or `darts` library
- Variable importance visualization

---

## Phase 4: Risk & Portfolio Analytics

### 4A: Risk Metrics Tab
- New "Risk" tab in the app
- Sharpe, Sortino, Calmar ratios
- Maximum drawdown analysis with chart
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Beta and Alpha vs benchmark (SPY)
- Rolling volatility chart
- New dep: `quantstats`

### 4B: GARCH Volatility Forecasting
- GJR-GARCH model via `arch` library
- Volatility forecast overlay on price chart

### 4C: Portfolio Optimization
- New "Portfolio" tab
- Multi-ticker portfolio input
- Efficient frontier visualization
- Mean-Variance, Hierarchical Risk Parity, Black-Litterman
- Allocation pie chart
- New dep: `riskfolio-lib` or `skfolio`

### 4D: Correlation Analysis
- Correlation heatmap for selected tickers
- Rolling correlation chart
- Sector/industry correlation grouping

---

## Phase 5: Sentiment Analysis

### 5A: FinBERT News Sentiment
- `transformers` + `ProsusAI/finbert` for financial text sentiment scoring
- Sentiment score for recent news headlines
- Sentiment trend chart over time
- New deps: `transformers`, `torch` (shared with Phase 3C)

### 5B: News Integration
- Pull financial news headlines via free API (newsapi.org or RSS feeds)
- Display headlines with sentiment color-coding in new "Sentiment" tab
- New dep: `newsapi-python` or `feedparser`

### 5C: Social Media Sentiment
- Reddit via `praw` library, StockTwits API integration
- Aggregate sentiment gauge (bullish/bearish meter)
- New dep: `praw`

### 5D: SEC Filing Analysis
- `edgartools` for SEC EDGAR data (10-K, 10-Q, Form 4)
- FinBERT sentiment on management discussion sections
- Insider transaction display
- New dep: `edgartools`

---

## Phase 6: Alternative Data & Options

### 6A: Options Chain Analysis
- Put/call ratio, options volume by strike
- yfinance option chain data (already available, no new deps)
- New "Options" tab with put/call ratio chart

### 6B: Unusual Options Activity
- Volume/OI ratio screening for abnormal flow
- OTM call/put sweeps detection
- Alert-style display for unusual activity

### 6C: Insider & Institutional Data
- 13F holdings, Form 4 transactions via `edgartools` (from Phase 5D)
- Top institutional holders table
- Insider buy/sell timeline chart

### 6D: Short Interest
- Short interest ratio and trends
- Data via Finnhub API or similar
- Short interest vs price chart

---

## Phase 7: Advanced ML & Automation

### 7A: Time Series Foundation Models
- Chronos (Amazon), TimeGPT (Nixtla), or Lag-Llama
- Zero-shot forecasting without training
- Confidence intervals on predictions
- New dep: `chronos-forecasting` or `nixtla`

### 7B: Reinforcement Learning Agent
- FinRL-based trading simulation (A2C/PPO)
- Buy/sell signal overlay on chart
- Simulated P&L display
- New deps: `finrl`, `stable-baselines3`

### 7C: Automated Chart Pattern Detection
- Head & shoulders, double tops/bottoms, wedges, channels
- scipy peak detection + geometric rules
- Pattern annotations on candlestick chart

### 7D: Backtesting Framework
- Test ML signals and indicator strategies on historical data
- Equity curve, trade log, performance metrics
- New dep: `vectorbt` or `backtesting.py`

---

## Phase 8: Infrastructure & Scale

### 8A: Additional Data Providers
- Finnhub, Alpha Vantage, Polygon.io
- Implement `DataProvider` ABC for each
- API key management via env vars

### 8B: Redis Caching
- Replace `@st.cache_data` for multi-user support
- New dep: `redis`

### 8C: Persistent Storage
- PostgreSQL for watchlists, portfolios, alerts
- New deps: `sqlalchemy`, `psycopg2`

### 8D: API Backend
- FastAPI service layer to decouple data/ML from Streamlit
- REST API for programmatic access
- New deps: `fastapi`, `uvicorn`

### 8E: CI/CD
- GitHub Actions pipeline: lint, test, Docker build on PR
- Auto-deploy on merge to main

# Stock Analysis Platform

A personal, extensible stock analysis dashboard built with Python and Streamlit. Pulls free real-time and historical market data, presents interactive visualizations with 16+ technical indicators, candlestick pattern detection, and an ML-powered price direction predictor.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Sidebar Controls](#sidebar-controls)
  - [Overview Tab](#overview-tab)
  - [Technicals Tab](#technicals-tab)
  - [Financials Tab](#financials-tab)
  - [Comparison Tab](#comparison-tab)
  - [Volume Tab](#volume-tab)
- [Technical Indicators](#technical-indicators)
  - [Trend Overlays](#trend-overlays)
  - [Bands and Channels](#bands-and-channels)
  - [Momentum Oscillators](#momentum-oscillators)
  - [Trend Strength and Volatility](#trend-strength-and-volatility)
  - [Volume Indicators](#volume-indicators)
- [Candlestick Pattern Detection](#candlestick-pattern-detection)
- [Support/Resistance and Fibonacci](#supportresistance-and-fibonacci)
- [Bollinger Squeeze Detection](#bollinger-squeeze-detection)
- [ML Price Direction Predictor](#ml-price-direction-predictor)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Adding a New Data Provider](#adding-a-new-data-provider)
- [Adding a New ML Model](#adding-a-new-ml-model)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Dependencies](#dependencies)

---

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd stock-analysis
pip install -r requirements.txt

# Run
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Prerequisites

- **Python 3.12+** (tested on 3.12 and 3.13)
- **pip** (Python package manager)
- **Docker** (optional, for containerized setup)

---

## Installation

### Local Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open **http://localhost:8501** in your browser.

### Docker Setup

```bash
# Build and start
docker compose up --build

# Subsequent runs (no rebuild needed)
docker compose up

# Stop
docker compose down
```

The app is available at **http://localhost:8501**.

The Docker setup mounts the project directory as a volume, so code changes are reflected immediately (Streamlit hot-reload).

**Environment variables** can be set in `docker-compose.yml`:

```yaml
environment:
  - DATA_PROVIDER=yfinance    # default, free, no API key
```

---

## Usage

### Sidebar Controls

The sidebar (left panel) controls all tabs:

| Control | Description |
|---------|-------------|
| **Ticker Symbol** | Type a ticker (e.g., `AAPL`) and press Enter. Auto-complete suggestions appear as you type. |
| **Period** | Historical data range: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max` |
| **Interval** | Candle size. Options change based on period. Intraday periods (`1d`, `5d`) support `1m` to `90m`. Longer periods support `1d`, `5d`, `1wk`, `1mo`. |

The default view is **AAPL** with a **1-year** period at **daily** intervals.

---

### Overview Tab

Displays a snapshot of the selected ticker:

- **Price** with daily change (absolute and percentage)
- **Day High / Day Low**
- **Open / Previous Close / Volume / Market Cap**
- **Price History** sparkline chart for the selected period

Market cap is auto-formatted (e.g., `$3.71T`, `$245.00B`).

---

### Technicals Tab

The main analysis view with an interactive candlestick chart and configurable indicators.

Indicators are organized into collapsible sections:

1. **Trend Overlays** - SMA, EMA, VWAP, Supertrend, Ichimoku Cloud
2. **Bands & Channels** - Bollinger Bands, Keltner Channels, Bollinger Squeeze
3. **Momentum Oscillators** - RSI, MACD, Stochastic, Williams %R
4. **Trend Strength & Volatility** - ADX, ATR
5. **Patterns & Levels** - Candlestick Patterns, Support/Resistance, Fibonacci Retracement

Toggle each indicator with a checkbox. Some indicators have adjustable parameters via sliders (e.g., SMA period 5-200).

Subplots (RSI, MACD, Stochastic, Williams %R, ADX, ATR) appear as separate rows below the main price chart. The chart height adjusts automatically.

At the bottom, the **ML Predictions** section shows the XGBoost model's next-day price direction forecast (see [ML Price Direction Predictor](#ml-price-direction-predictor)).

---

### Financials Tab

Key fundamental data for the selected ticker:

| Section | Metrics |
|---------|---------|
| **Valuation** | P/E (TTM), Forward P/E, P/B Ratio, P/S Ratio |
| **Earnings** | EPS (TTM), Forward EPS, Revenue, Profit Margin |
| **Dividends & Risk** | Dividend Yield, Beta, 52-Week High, 52-Week Low |
| **About** | Company description |

---

### Comparison Tab

Compare up to 5 tickers side-by-side:

1. Enter comma-separated tickers in the text field (e.g., `MSFT, GOOGL, AMZN`)
2. The currently selected ticker from the sidebar is always included
3. View a **normalized percentage-change chart** showing relative performance
4. A **metrics table** shows Price, Change %, Period Return, and Volume for each ticker

---

### Volume Tab

Dedicated volume analysis:

- **Volume Chart** - Color-coded bars (green = up day, red = down day) with a 20-day moving average overlay
- **Volume Profile** - Horizontal histogram showing volume distribution across price levels (30 bins)
- **Price & Volume** - Dual-axis chart correlating price movement with volume
- **Day-of-Week Patterns** - Average volume by weekday (shown for `1d` and `5d` intervals only)
- **Volume Indicators**:
  - **OBV** (On-Balance Volume) - Cumulative volume flow
  - **A/D Line** (Accumulation/Distribution) - Volume weighted by close position within daily range
  - **CMF** (Chaikin Money Flow) - 20-period buying/selling pressure oscillator
- **Volume Statistics** - Average, Max, Min volume, and Latest vs. Average ratio

---

## Technical Indicators

All indicators are computed using [pandas-ta](https://github.com/twopirllc/pandas-ta) (pure Python, no C dependencies).

### Trend Overlays

These are drawn directly on the price chart:

| Indicator | Description | Default Parameters |
|-----------|-------------|-------------------|
| **SMA** | Simple Moving Average | Period: 20 (adjustable 5-200) |
| **EMA** | Exponential Moving Average | Period: 12 (adjustable 5-200) |
| **VWAP** | Volume Weighted Average Price | None (auto-computed) |
| **Supertrend** | ATR-based trend follower. Green line = uptrend support, red line = downtrend resistance. | Length: 7, Multiplier: 3.0 |
| **Ichimoku Cloud** | Shows Tenkan-sen (conversion), Kijun-sen (base), and the cloud (Senkou Span A & B) as a shaded area. | Tenkan: 9, Kijun: 26 |

### Bands and Channels

| Indicator | Description | Default Parameters |
|-----------|-------------|-------------------|
| **Bollinger Bands** | Upper/lower bands at standard deviations from SMA. Filled area between bands. | Period: 20 (adjustable 5-50), Std Dev: 2.0 (adjustable 1.0-3.0, step 0.5) |
| **Keltner Channels** | EMA-based bands using ATR for width. | Period: 20, Multiplier: 2.0 |
| **Bollinger Squeeze** | Detects when Bollinger Bands contract inside Keltner Channels, signaling low volatility before a potential breakout. | Requires both Bollinger and Keltner to be computed |

### Momentum Oscillators

These are rendered as separate subplots below the price chart:

| Indicator | Description | Default Parameters | Levels |
|-----------|-------------|-------------------|--------|
| **RSI** | Relative Strength Index (0-100) | Period: 14 (adjustable 5-50) | Overbought: 70, Oversold: 30 |
| **MACD** | Moving Average Convergence Divergence with signal line and histogram | Fast: 12, Slow: 26, Signal: 9 | Zero line |
| **Stochastic** | %K and %D lines (0-100) | K: 14, D: 3, Smooth K: 3 | Overbought: 80, Oversold: 20 |
| **Williams %R** | Momentum oscillator (-100 to 0) | Period: 14 | Overbought: -20, Oversold: -80 |

### Trend Strength and Volatility

| Indicator | Description | Default Parameters |
|-----------|-------------|-------------------|
| **ADX** | Average Directional Index with +DI/-DI. ADX > 25 = strong trend. | Period: 14 |
| **ATR** | Average True Range. Measures volatility in price units. | Period: 14 |

### Volume Indicators

These are displayed on the **Volume** tab:

| Indicator | Description |
|-----------|-------------|
| **OBV** | On-Balance Volume. Cumulative: adds volume on up days, subtracts on down days. Divergence from price signals potential reversals. |
| **A/D Line** | Accumulation/Distribution. Like OBV but weights volume by where close falls within the day's high-low range. |
| **CMF** | Chaikin Money Flow (20-period). Oscillates between -1 and 1. Positive = buying pressure, negative = selling pressure. |

---

## Candlestick Pattern Detection

The app detects 7 common candlestick patterns using a pure Python implementation (no TA-Lib required):

| Pattern | Signal | Detection Logic |
|---------|--------|-----------------|
| **Doji** | Neutral | Body < 10% of total range |
| **Hammer** | Bullish | Lower shadow >= 2x body, small upper shadow |
| **Shooting Star** | Bearish | Upper shadow >= 2x body, small lower shadow |
| **Bullish Engulfing** | Bullish | Green candle fully engulfs previous red candle |
| **Bearish Engulfing** | Bearish | Red candle fully engulfs previous green candle |
| **Morning Star** | Bullish | 3-candle pattern: large red, small body, large green recovering above midpoint |
| **Evening Star** | Bearish | 3-candle pattern: large green, small body, large red dropping below midpoint |

When enabled, patterns appear as diamond markers on the chart with labels. A summary table of recent detections is shown below the chart.

---

## Support/Resistance and Fibonacci

### Support/Resistance

Automatically detects significant price levels using `scipy.signal.argrelextrema`:

- Finds local highs (resistance) and lows (support) with a 20-bar lookback window
- Clusters nearby levels within 1.5% of each other
- Displays the 5 levels closest to the current price
- Green dashed lines = support, red dashed lines = resistance

### Fibonacci Retracement

Computes Fibonacci levels from the swing high and swing low over the last 60 bars:

| Level | Ratio |
|-------|-------|
| 0.0% | Swing High |
| 23.6% | |
| 38.2% | |
| 50.0% | |
| 61.8% | |
| 78.6% | |
| 100.0% | Swing Low |

Displayed as orange dotted lines with price annotations.

---

## Bollinger Squeeze Detection

The **Bollinger Squeeze** identifies periods of low volatility (potential breakout setups):

- **Squeeze ON**: Bollinger Bands are completely inside Keltner Channels (both upper BB < upper KC and lower BB > lower KC)
- **Squeeze OFF**: Bands have expanded beyond the channels

When active, red square markers appear at the lowest price level on the chart, and a status message is displayed below.

---

## ML Price Direction Predictor

The app includes an **XGBoost classifier** that predicts whether tomorrow's close will be **higher (Up)** or **lower (Down)** than today's close.

### How it works

1. **Features** (16 total):
   - 11 technical indicators: SMA(20), EMA(12), RSI(14), MACD, MACD Histogram, MACD Signal, Bollinger Lower/Mid/Upper/Bandwidth/%-B
   - 5 derived features: 1-day return, 5-day return, 10-day rolling volatility, price vs. SMA ratio, volume vs. 20-day average ratio

2. **Training**: Trains on all available data except the last 5 rows. Uses `XGBClassifier` with 100 trees, max depth 4, learning rate 0.1.

3. **Output**: Displays in the Technicals tab:
   - **Next-Day Prediction**: Up or Down
   - **Confidence**: Probability (0-100%)
   - **Train Accuracy**: Model's in-sample accuracy
   - **Recent Predictions Table**: Last 10 predictions with direction and confidence

### Requirements

- Minimum **60 data rows** after indicator warm-up (use a 3-month or longer period)
- Short periods (1 day, 5 days, 1 month) will show an error message explaining the requirement

### Important notes

- This is a **demonstration model**, not financial advice
- Train accuracy will be high (the model overfits training data by design for demonstration)
- The model retrains from scratch on every page load (no saved weights)

---

## Configuration

All defaults are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DATA_PROVIDER` | `"yfinance"` | Data source (overridable via `DATA_PROVIDER` env var) |
| `DEFAULT_TICKER` | `"AAPL"` | Initial ticker on app load |
| `DEFAULT_PERIOD` | `"1y"` | Initial period |
| `DEFAULT_INTERVAL` | `"1d"` | Initial interval |
| `SMA_PERIOD` | `20` | Default SMA lookback |
| `EMA_PERIOD` | `12` | Default EMA lookback |
| `RSI_PERIOD` | `14` | Default RSI lookback |
| `MACD_FAST/SLOW/SIGNAL` | `12/26/9` | Default MACD parameters |
| `BOLLINGER_PERIOD` | `20` | Default Bollinger lookback |
| `BOLLINGER_STD` | `2.0` | Default Bollinger standard deviations |
| `MAX_COMPARISON_TICKERS` | `5` | Max tickers in comparison tab |
| `CACHE_TTL_QUOTE` | `300` | Quote cache: 5 minutes |
| `CACHE_TTL_HISTORY` | `3600` | History cache: 1 hour |
| `CACHE_TTL_INFO` | `86400` | Company info cache: 24 hours |

The color palette for all charts is defined in `config.COLORS`.

---

## Architecture

```
app.py (Streamlit entry point)
  |
  +-- Sidebar: ticker search, period/interval selection
  |
  +-- st.tabs: Overview | Technicals | Financials | Comparison | Volume
        |
        v
      pages/*.py  -->  data/cache.py  -->  data/yfinance_provider.py  -->  Yahoo Finance API
        |                                       (DataProvider ABC)
        v
      indicators/technical.py  -->  charts/price.py (Plotly)
        |
        v
      ml/xgboost_direction.py  -->  (auto-discovered by technicals page)
         (ModelProvider ABC)
```

**Key design principles:**

- **Strategy Pattern** for data providers: swap yfinance for any other source by implementing `DataProvider`
- **Pure functions** for indicators: no Streamlit dependency, testable in isolation
- **Shared indicators**: ML models import from `indicators/technical.py` to ensure feature consistency with charts
- **Auto-discovery** for ML models: register in `ml/__init__.py`, UI picks them up automatically
- **Caching**: `@st.cache_data` with TTLs to minimize API calls

---

## Adding a New Data Provider

1. Create `data/your_provider.py` implementing the `DataProvider` ABC from `data/base.py`:
   - `get_quote(ticker) -> Quote`
   - `get_history(ticker, period, interval) -> pd.DataFrame`
   - `get_info(ticker) -> dict`
   - `search_ticker(query) -> list[dict]`
   - `validate_ticker(ticker) -> bool`

2. Add an `elif` branch in `data/__init__.py`:
   ```python
   elif name == "your_provider":
       from data.your_provider import YourProvider
       return YourProvider()
   ```

3. Set the environment variable:
   ```bash
   DATA_PROVIDER=your_provider streamlit run app.py
   ```

---

## Adding a New ML Model

1. Create `ml/your_model.py` implementing the `ModelProvider` ABC from `ml/base.py`:
   ```python
   class YourModel(ModelProvider):
       def get_name(self) -> str:
           return "Your Model Name"

       def get_description(self) -> str:
           return "What it does"

       def predict(self, df: pd.DataFrame) -> pd.DataFrame:
           # Add prediction columns to df and return it
           ...
   ```

2. Register it in `ml/__init__.py`:
   ```python
   from ml.your_model import YourModel
   models.append(YourModel())
   ```

3. No other code changes needed. The Technicals tab auto-discovers registered models.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Failed to get ticker" / "possibly delisted"** | Yahoo Finance rate limiting. Wait a moment and reload. The cache prevents this on subsequent loads. |
| **Empty charts** | Check that the ticker is valid and the period/interval combination returns data. Very short intraday periods may have no data outside market hours. |
| **ML model shows "Model error"** | Use a period of 3 months or longer. The model needs at least 60 rows of data after indicator warm-up. |
| **Ichimoku Cloud not showing** | Ichimoku requires approximately 52+ bars of data. Use a 3-month or longer period. |
| **Port 8501 already in use** | Stop other Streamlit instances: `pkill -f "streamlit run"` or use `streamlit run app.py --server.port 8502` |
| **Docker build fails** | Ensure Docker Desktop is running. Run `docker compose up --build` to rebuild. |
| **Import errors** | Run `pip install -r requirements.txt` to install all dependencies. |

---

## Roadmap

See `CLAUDE.md` for the full phased roadmap. Upcoming phases:

- **Phase 3**: ML Model Suite (LightGBM, CatBoost, Ensemble, LSTM, Temporal Fusion Transformer)
- **Phase 4**: Risk & Portfolio Analytics (Sharpe, VaR, GARCH, portfolio optimization)
- **Phase 5**: Sentiment Analysis (FinBERT, news, social media, SEC filings)
- **Phase 6**: Alternative Data (options flow, insider trading, short interest)
- **Phase 7**: Advanced ML (foundation models, reinforcement learning, backtesting)
- **Phase 8**: Infrastructure (Redis, PostgreSQL, FastAPI, CI/CD)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.41.0 | Web dashboard framework |
| yfinance | >= 1.2.0 | Yahoo Finance data provider |
| pandas | >= 2.3.2 | Data manipulation |
| pandas-ta | >= 0.4.71b0 | Technical indicators (pure Python) |
| plotly | 5.24.1 | Interactive charts |
| numpy | >= 1.26.4 | Numerical computation |
| scipy | >= 1.11.0 | Support/resistance peak detection |
| xgboost | >= 2.0.0 | ML price direction model |
| scikit-learn | >= 1.3.0 | ML utilities |

---

## License

Personal project. Not financial advice.

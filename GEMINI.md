# GEMINI.md - Stock Analysis Platform

Stock analysis dashboard: Python/Streamlit, 16+ technical indicators, candlestick patterns, ML price direction prediction (XGBoost + LSTM).

## Architecture

- **Data (`data/`):** `DataProvider` ABC (Strategy Pattern). Current: yfinance. VIX fetched at data layer via `include_vix=True`. VIX join uses a date-only lookup key to preserve intraday timestamps.
- **Indicators (`indicators/`):** Stateless pure functions. `technical.py` (pandas-ta), `ml_features.py` (normalized ratios, returns, volatility), `composite.py` (signal scoring).
- **Charts (`charts/`):** Plotly candlesticks, volume profiles, comparison overlays.
- **Views (`views/`):** Streamlit tab modules, each exports `render()`.
- **ML (`ml/`):** `ModelProvider` ABC. `direction_base.py` provides shared training/prediction logic. XGBoost + LSTM direction classifiers with session-scoped caching. Out-of-sample predictions only (no data leakage). Hyperparameters in `config.py`.

## Running

```bash
pip install -r requirements.txt
streamlit run app.py              # http://localhost:8501
docker compose up --build         # or via Docker
pytest                            # full suite
pytest -m "not slow"              # skip ML training tests
```

## Development Rules

- **Features:** All ML features must be unitless ratios (e.g. `Price_vs_SMA`, `ATR_Norm`), never raw prices. Computed in `indicators/ml_features.py`.
- **Indicators:** ML models import from `indicators/technical.py` — same calculations as charts.
- **Data leakage:** Predictions are only assigned to holdout (out-of-sample) rows. Final model retrains on all data for the live last-row forecast only.
- **New data source:** Implement `DataProvider` in `data/base.py`, register in `data/__init__.py`.
- **New ML model:** Implement `ModelProvider` in `ml/base.py`, register in `ml/__init__.py`.
- **Caching:** `@st.cache_data` with TTLs in `data/cache.py`.
- **Intervals:** `config.INTERVAL_OPTIONS` aligned to yfinance limits: `1m` ≤ 7d, `2m`–`90m` ≤ 60d, `1d`+ unlimited. Never call `.normalize()` on `df.index` — it destroys intraday timestamps.
- **Testing:** XGBoost tests must run before LSTM tests (BLAS conflict on macOS). Handled by `tests/ml/conftest.py`.

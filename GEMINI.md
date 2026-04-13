# GEMINI.md - Stock Analysis Platform

This project is a personal, extensible stock analysis dashboard built with Python and Streamlit. It provides interactive visualizations, 16+ technical indicators, candlestick pattern detection, and ML-powered price direction prediction using real-time and historical market data.

## Project Overview

*   **Main Technologies:** Python (3.12+), Streamlit, pandas-ta, Plotly, yfinance, XGBoost, PyTorch (LSTM).
*   **Architecture:**
    *   **Data Layer (`data/`):** Uses a Strategy Pattern (`DataProvider` ABC) to allow swappable data sources. Current implementation: `yfinance_provider.py`. The data layer handles auxiliary data fetching (e.g., VIX via `include_vix` in `get_history`) to keep ML and View layers pure.
    *   **Indicators Layer (`indicators/`):** Contains stateless pure functions for computing technical indicators (`technical.py`) and ML-specific features (`ml_features.py`). Shared between visualization and ML models to ensure feature consistency.
    *   **Charts Layer (`charts/`):** Interactive visualizations using Plotly (candlesticks, volume profiles, etc.).
    *   **Views (`views/`):** Modular components for Streamlit tabs (Overview, Technicals, Financials, etc.).
    *   **ML Layer (`ml/`):** Pluggable model architecture (`ModelProvider` ABC). Includes XGBoost and LSTM direction classifiers. Hyperparameters are centralized in `config.py`.

## Building and Running

### Prerequisites
- Python 3.12 or higher.
- Docker (optional).

### Local Setup
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The app will be available at [http://localhost:8501](http://localhost:8501).

### Docker Setup
1.  **Build and start:**
    ```bash
    docker compose up --build
    ```
    The project directory is mounted as a volume for hot-reload.

### Testing
- Run tests using `pytest`:
  ```bash
  pytest
  ```
- Deselect slow tests (e.g., ML training):
  ```bash
  pytest -m "not slow"
  ```

## Development Conventions

### Architecture & Patterns
- **Strategy Pattern:** Implement `DataProvider` in `data/base.py` for new data sources or `ModelProvider` in `ml/base.py` for new ML models.
- **Indicator Consistency:** Always use `indicators/technical.py` and `indicators/ml_features.py` for feature engineering in both charts and ML models.
- **Pure Functions:** Keep indicator logic stateless and side-effect-free.
- **Data Layer Delegation:** Auxiliary data (like VIX) is fetched at the data layer (`include_vix=True`), not inside ML models.
- **Caching:** Use Streamlit's `@st.cache_data` with appropriate TTLs (defined in `data/cache.py`) to minimize API calls.

### Coding Style
- Follow standard Python (PEP 8) conventions.
- Type annotations are preferred for `DataProvider` and `ModelProvider` implementations.
- UI components should be placed in `views/` and follow the `render()` function pattern.

### Key Data Structures
- **Quote (dataclass):** Found in `data/base.py`.
- **OHLCV DataFrame:** Standard format used across the app (Open, High, Low, Close, Volume).

## Configuration
- Default settings (ticker, periods, colors, indicator parameters) and ML hyperparameters are managed in `config.py`.
- Data providers can be switched via the `DATA_PROVIDER` environment variable.

"""Abstract base class for data providers and shared data types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Quote:
    """Current price snapshot for a ticker."""

    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None


class DataProvider(ABC):
    """Contract for all stock data providers."""

    @abstractmethod
    def get_quote(self, ticker: str) -> Quote:
        """Get the current price quote for a ticker."""

    @abstractmethod
    def get_history(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical OHLCV data. Returns a DataFrame with columns:
        Open, High, Low, Close, Volume (and optionally Adj Close).
        Index is DatetimeIndex.
        """

    @abstractmethod
    def get_info(self, ticker: str) -> dict:
        """Get company/ticker information (fundamentals, description, etc.)."""

    @abstractmethod
    def search_ticker(self, query: str) -> list[dict]:
        """Search for tickers matching a query string.
        Returns list of dicts with at least 'symbol' and 'name' keys.
        """

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker symbol is valid."""

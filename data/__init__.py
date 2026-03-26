"""Data provider factory."""

import config
from data.base import DataProvider


def get_provider() -> DataProvider:
    """Return the configured data provider instance."""
    name = config.DATA_PROVIDER.lower()

    if name == "yfinance":
        from data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    else:
        raise ValueError(
            f"Unknown data provider: {name!r}. "
            f"Supported: 'yfinance'"
        )

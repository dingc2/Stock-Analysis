"""yfinance implementation of DataProvider."""

import yfinance as yf
import pandas as pd

from data.base import DataProvider, Quote


class YFinanceProvider(DataProvider):
    """Data provider using the yfinance library (free, no API key)."""

    def get_quote(self, ticker: str) -> Quote:
        t = yf.Ticker(ticker)
        fi = t.fast_info

        price = getattr(fi, "last_price", None) or 0.0
        prev_close = getattr(fi, "previous_close", None) or 0.0
        change = price - prev_close if prev_close else 0.0
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return Quote(
            symbol=ticker.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=int(getattr(fi, "last_volume", None) or 0),
            market_cap=getattr(fi, "market_cap", None),
            day_high=getattr(fi, "day_high", None),
            day_low=getattr(fi, "day_low", None),
            open=getattr(fi, "open", None),
            previous_close=prev_close,
        )

    def get_history(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        return df

    def get_info(self, ticker: str) -> dict:
        t = yf.Ticker(ticker)
        return dict(t.info)

    def search_ticker(self, query: str) -> list[dict]:
        try:
            search = yf.Search(query)
            results = []
            for q in search.quotes:
                results.append(
                    {
                        "symbol": q.get("symbol", ""),
                        "name": q.get("shortname", q.get("longname", "")),
                        "exchange": q.get("exchange", ""),
                        "type": q.get("quoteType", ""),
                    }
                )
            return results
        except Exception:
            return []

    def validate_ticker(self, ticker: str) -> bool:
        try:
            t = yf.Ticker(ticker)
            return getattr(t.fast_info, "last_price", None) is not None
        except Exception:
            return False

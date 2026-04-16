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
        self, ticker: str, period: str = "1y", interval: str = "1d", include_vix: bool = False
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        
        if include_vix:
            try:
                # Fetch long history to ensure we cover the input df
                vix_t = yf.Ticker("^VIX")
                vix_df = vix_t.history(period="5y", interval="1d")
                if not vix_df.empty:
                    vix_close = vix_df["Close"].rename("VIX_Close")

                    # Strip timezones from VIX index and normalize to date
                    if vix_close.index.tz is not None:
                        vix_close.index = vix_close.index.tz_convert(None).normalize()
                    else:
                        vix_close.index = vix_close.index.normalize()

                    # Build a date-only key for the main df WITHOUT mutating its index.
                    # This preserves intraday timestamps (5m, 15m, etc.) that would
                    # otherwise be destroyed by .normalize().
                    original_index = df.index
                    if df.index.tz is not None:
                        date_key = df.index.tz_convert(None).normalize()
                    else:
                        date_key = df.index.normalize()

                    # Map VIX values by date
                    vix_map = vix_close.to_dict()
                    df["VIX_Close"] = [vix_map.get(d, float("nan")) for d in date_key]
                    df["VIX_Close"] = df["VIX_Close"].ffill()

                    # Strip timezone from the main index (keeps full timestamps)
                    if original_index.tz is not None:
                        df.index = original_index.tz_convert(None)

                    df["VIX_10d_MA"] = df["VIX_Close"].rolling(10).mean()
                    df["VIX_20d_MA"] = df["VIX_Close"].rolling(20).mean()
                    df["VIX_30d_MA"] = df["VIX_Close"].rolling(30).mean()
            except Exception as e:
                import warnings
                warnings.warn(f"YFinanceProvider: Failed to fetch VIX data: {e}", stacklevel=2)

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

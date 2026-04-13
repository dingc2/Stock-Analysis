"""Streamlit cache wrappers for data provider calls."""

import streamlit as st
import pandas as pd

import config
from data.base import DataProvider, Quote


def get_quote_cached(provider: DataProvider, ticker: str) -> Quote:
    """Cached quote lookup (5 min TTL)."""
    return _get_quote(provider, ticker)


@st.cache_data(ttl=config.CACHE_TTL_QUOTE, show_spinner=False)
def _get_quote(_provider: DataProvider, ticker: str) -> Quote:
    return _provider.get_quote(ticker)


def get_history_cached(
    provider: DataProvider, ticker: str, period: str = "1y", interval: str = "1d", include_vix: bool = False
) -> pd.DataFrame:
    """Cached history lookup (1 hr TTL)."""
    return _get_history(provider, ticker, period, interval, include_vix)


@st.cache_data(ttl=config.CACHE_TTL_HISTORY, show_spinner=False)
def _get_history(
    _provider: DataProvider, ticker: str, period: str, interval: str, include_vix: bool = False
) -> pd.DataFrame:
    return _provider.get_history(ticker, period=period, interval=interval, include_vix=include_vix)


def get_history_live(
    provider: DataProvider, ticker: str, period: str = "1y", interval: str = "1d", include_vix: bool = False
) -> pd.DataFrame:
    """Live history lookup with short TTL for auto-refresh."""
    return _get_history_live(provider, ticker, period, interval, include_vix)


@st.cache_data(ttl=config.CACHE_TTL_LIVE, show_spinner=False)
def _get_history_live(
    _provider: DataProvider, ticker: str, period: str, interval: str, include_vix: bool = False
) -> pd.DataFrame:
    return _provider.get_history(ticker, period=period, interval=interval, include_vix=include_vix)


def get_quote_live(provider: DataProvider, ticker: str) -> Quote:
    """Live quote lookup with short TTL for auto-refresh."""
    return _get_quote_live(provider, ticker)


@st.cache_data(ttl=config.CACHE_TTL_LIVE, show_spinner=False)
def _get_quote_live(_provider: DataProvider, ticker: str) -> Quote:
    return _provider.get_quote(ticker)


def get_info_cached(provider: DataProvider, ticker: str) -> dict:
    """Cached info lookup (24 hr TTL)."""
    return _get_info(provider, ticker)


@st.cache_data(ttl=config.CACHE_TTL_INFO, show_spinner=False)
def _get_info(_provider: DataProvider, ticker: str) -> dict:
    return _provider.get_info(ticker)

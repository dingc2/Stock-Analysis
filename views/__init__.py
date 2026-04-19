"""Shared page utilities."""

import streamlit as st
import config
from ml import get_available_models
from ml.base import ModelProvider


@st.cache_resource(show_spinner=False)
def get_cached_models() -> list[ModelProvider]:
    """Return ML model instances that persist across Streamlit reruns.

    Why: each BaseDirectionModel holds a per-instance prediction cache
    keyed by OHLCV hash. If we rebuilt instances on every rerun, the
    cache would always be empty and both models would retrain on every
    tab switch / slider change (60-180s each).
    """
    return get_available_models()


def render_timeframe_buttons(period: str, interval: str, key_prefix: str = ""):
    """Render timeframe preset buttons. Active preset is highlighted."""
    cols = st.columns(len(config.TIMEFRAME_PRESETS))
    for i, (label, p, intv) in enumerate(config.TIMEFRAME_PRESETS):
        with cols[i]:
            is_active = period == p and interval == intv
            if st.button(
                label,
                key=f"{key_prefix}tf_{label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["period"] = p
                st.session_state["interval"] = intv
                st.rerun()

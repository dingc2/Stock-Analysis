"""Shared page utilities."""

import streamlit as st
import config


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

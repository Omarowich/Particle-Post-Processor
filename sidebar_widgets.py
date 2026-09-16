"""Shared Streamlit sidebar widget blocks used by more than one plot mode.

Extracted from app.py (Phase 2 de-spaghetti pass): before this, the
"cluster detection mode + DBSCAN eps/min_samples + min cluster size" widget
group was copy-pasted between the Multiple-plots sidebar and the
Summary-plots sidebar, so a tweak to one had to be repeated in the other by
hand. Now there is exactly one place that defines these widgets.
"""

import streamlit as st


def render_cluster_params_widgets(key_prefix: str) -> dict:
    """
    Render the "Cluster metrics parameters" widget group (detection mode,
    DBSCAN eps, DBSCAN min_samples, min cluster size).

    `key_prefix` namespaces the underlying Streamlit widget keys so this can
    be rendered more than once per run (e.g. "multi", "summary").

    Returns {"mode": str, "eps": float, "min_samples": int, "min_cluster_size": int}.
    """
    st.subheader("Cluster metrics parameters")

    mode = st.selectbox(
        "Cluster detection mode",
        ["per_cluster", "global"],
        index=0,
        key=f"{key_prefix}_cluster_mode",
    )
    eps = st.number_input(
        "DBSCAN eps",
        value=3.5,
        step=0.1,
        key=f"{key_prefix}_cluster_eps",
    )
    min_samples = st.number_input(
        "DBSCAN min_samples",
        value=3,
        step=1,
        min_value=1,
        key=f"{key_prefix}_cluster_min_samples",
    )
    min_cluster_size = st.number_input(
        "Min cluster size",
        value=3,
        step=1,
        min_value=1,
        key=f"{key_prefix}_cluster_min_cluster_size",
    )

    return {
        "mode": mode,
        "eps": float(eps),
        "min_samples": int(min_samples),
        "min_cluster_size": int(min_cluster_size),
    }

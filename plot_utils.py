"""Matplotlib figure styling and small file-output helpers shared across
all plot modes (global legend/axis/title styling, grid toggling, filename
slugification, saving a figure to disk, saving an uploaded file to a temp path).

Extracted from app.py (Phase 1 de-spaghetti pass).
"""

import os
import pathlib
import re
import tempfile

import matplotlib.pyplot as plt
import streamlit as st


def save_uploaded_file(uploaded_file, filename):
    temp_dir = tempfile.gettempdir()
    path = pathlib.Path(temp_dir) / filename
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)

def apply_global_styling(
    fig,
    legend_on=True,
    legend_fontsize=16,
    axis_fontsize=18,
    title_on=True,
    title_fontsize=20,
):
    """
    Apply legend / axis / title styling to all axes in a figure.
    """
    for ax in fig.get_axes():
        # Axis tick font size
        ax.tick_params(axis="both", labelsize=axis_fontsize)

        # Axis labels
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=axis_fontsize)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=axis_fontsize)

        # Title
        if not title_on:
            ax.set_title("")
        else:
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=title_fontsize)

        # Legend
        leg = ax.get_legend()
        if leg is not None:
            leg.set_visible(legend_on)
            if legend_on and legend_fontsize is not None:
                for text in leg.get_texts():
                    text.set_fontsize(legend_fontsize)

def apply_grid(ax, show_grid: bool):
    if show_grid:
        ax.grid(True, which="major", alpha=0.3)
    else:
        ax.grid(False, which="both")
        ax.minorticks_off()

def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")

def render_smoothing_preview(curves, smooth_win, *, title="Smoothing preview"):
    """
    Small raw-vs-smoothed line plot so a smoothing-window slider's effect is
    visible immediately, instead of having to run the full analysis to see it.

    `curves` is a list of {"label": str, "x": array, "y_raw": array,
    "y_smoothed": array} -- pass a single-item list to preview one curve, or
    every curve to preview them all at once. Caller supplies the
    already-smoothed array so this works regardless of which smoothing
    function a given section actually uses downstream.
    """
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for c in curves:
        ax.plot(c["x"], c["y_raw"], color="#bbbbbb", linewidth=1, alpha=0.6)
    for c in curves:
        ax.plot(c["x"], c["y_smoothed"], linewidth=1.5, label=c["label"])
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{title} (window={smooth_win})", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)


def save_figure_if_requested(fig, base_name: str, save_enabled: bool, output_dir: str):
    if not save_enabled:
        return
    if not output_dir:
        st.warning("Save enabled but no output folder provided.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = slugify(base_name) or "plot"
        path = os.path.join(output_dir, f"{filename}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        st.info(f"Saved plot to: `{path}`")
    except Exception as e:
        st.warning(f"Could not save figure to `{output_dir}`: {e}")

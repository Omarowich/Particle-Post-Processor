# app.py

import math
import os
import pathlib
import re
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from area_fraction_over_time import area_fraction_over_time
from bond_orientation_over_time import bond_orientational_order_over_time
from data_reader_csv import read_particle_data_csv
from movement_change_calculator_time_study import (
    particle_displacement_from_inintal_position_over_time,
    particle_displacement_over_time,
    median_total_path_distance_over_time,
)
from cluster_post_processor import num_clusters_over_time, avg_cluster_size_over_time
from particle_distance_over_time import particle_distance_over_time
from plot_bond_orientation_at_time_step import (
    plot_bond_orientational_order_at_timestep,
)
from plot_crystal_at_timestep import plot_crystals_at_timestep
from plot_fourier_analysis import fourier_of_metric
from plot_together_tkb_func import plots_together
from radial_distribution_function import rdf_over_time
from rg_over_time import average_rg_over_time
from track_cluster_movement import track_cluster_centroids
from track_crystal_formation_over_time import (
    detect_crystals_over_time,
    crystals_hex_over_time,
    crystals_square_over_time,
    crystals_tri_over_time,
    crystals_all_over_time,
    crystals_none_over_time,
)
from phase_snapshot_fields import (
    snapshot_local_density,
    snapshot_displacement_vectors,
    snapshot_alignment_field,
    snapshot_displacement_vectors_voronoi
)

import ast
import operator as op

def post_analysis_block_generic(
    key: str,
    curves: list[dict],
    ylabel: str,
    xlabel: str,
    legend_on=True,
    legend_fontsize=14,
    axis_fontsize=16,
    title_on=True,
    title_fontsize=18,
    extra_ui_fn=None,
    sweep_rows=None,
    x_choice=None,
):

    with st.expander("🔎 Post-analysis (knee / turning / threshold / peaks / breakpoint)", expanded=False):
        if not curves:
            st.info("No curves available.")
            return

        # ALWAYS rendered (state won’t reset)


        with st.form(key=f"post_form_{key}", clear_on_submit=False):
            run_opts = ["All"] + [c["run"] for c in curves]
            pick = st.selectbox("Analyze curve", run_opts, index=0, key=f"{key}_pick")

            smooth_win = st.slider("Smoothing window", 1, 51, 9, 2, key=f"{key}_smooth")
            sustain = st.slider("Flatten sustain (% of points)", 1, 30, 8, 1, key=f"{key}_sustain")
            slope_eps_user = st.number_input(
                "Flatten slope threshold (leave 0 for auto)",
                value=0.0, step=0.0001, format="%.6f", key=f"{key}_slopeeps"
            )

            do_knee = st.checkbox("Mark knee (triangle method)", True, key=f"{key}_knee")
            do_turn = st.checkbox("Mark turning point (curvature)", True, key=f"{key}_turn")
            do_flat = st.checkbox("Mark flattening point", True, key=f"{key}_flat")
            label_markers = st.checkbox("Label markers on plot", value=True, key=f"{key}_labels")

            st.markdown("#### Extra post-analysis")
            do_thr = st.checkbox("X-to-threshold", value=False, key=f"{key}_thr")
            thr_mode = st.selectbox("Threshold mode", ["absolute", "% of max (per curve)"], index=1, key=f"{key}_thrmode")
            thr_val = st.number_input("Threshold value (abs or fraction)", value=0.6, step=0.01, format="%.4f", key=f"{key}_thrval")
            thr_dir = st.selectbox("Threshold direction", ["rising", "falling"], index=0, key=f"{key}_thrdir")
            thr_sustain = st.number_input("Threshold sustain (points)", value=3, step=1, min_value=1, key=f"{key}_thrsus")

            do_auc = st.checkbox("AUC + mean over x-window", value=False, key=f"{key}_auc")
            do_maxslope = st.checkbox("Max |slope| marker", value=False, key=f"{key}_maxslope")

            do_peaks = st.checkbox("Peak detection", value=False, key=f"{key}_peaks")
            peak_min_prom = st.number_input("Min peak prominence", value=0.0, step=0.01, format="%.4f", key=f"{key}_pprom")
            peak_min_dist = st.number_input("Min distance between peaks (points)", value=5, step=1, min_value=1, key=f"{key}_pdist")

            do_pwlin = st.checkbox("Piecewise-linear breakpoint", value=False, key=f"{key}_pwlin")
            pw_min_seg = st.slider("Min segment size (fraction)", 0.05, 0.4, 0.1, 0.05, key=f"{key}_pwseg")

            do_prog = st.checkbox("Prognosis (fit surface & predict)", value=False, key=f"{key}_prog")
            tk_query = st.text_input("Predict at TkB values (comma)", value="0,5,10,20,30,40,50", key=f"{key}_prog_tk")
            ta_query = st.text_input("Predict at TauB values (comma)", value="30", key=f"{key}_prog_ta")

            show_locus = st.checkbox("Show locus (connect feature points)", value=True, key=f"{key}_show_locus")
            locus_type = st.selectbox("Locus type", ["turn", "knee", "flat"], index=0, key=f"{key}_locus_type")

            # ✅ optional extra content INSIDE the expander
            if extra_ui_fn is not None:
                st.markdown("---")
                extra_ui_fn()
                st.markdown("---")

            run_post = st.form_submit_button("▶ Run post-analysis")

        if not run_post:
            st.caption("Adjust settings, then click **Run post-analysis**.")
            return

        # ---- run ----
        plt.close("all")
        plt.figure()
        ax2 = plt.gca()




        # plot selected curves
        ################
        #NOT DUPLICATE: JUST PLOTTING FIRST FOR OVERLAY##########
        ################

        for c in curves:
            if pick != "All" and c["run"] != pick:
                continue
            x = np.asarray(c["x"], float)
            y = np.asarray(c["y"], float)

            # sort by x (important for summary sweeps!)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size < 2:
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]

            ax2.plot(x, y, label=c["run"])

        slope_eps = None if slope_eps_user == 0 else float(slope_eps_user)
        sustain_frac = float(sustain) / 100.0

        results = []
        turn_pts = []
        knee_pts = []
        flat_pts = []

        y_level = 0

        for c in curves:
            if pick != "All" and c["run"] != pick:
                continue

            x = np.asarray(c["x"], float)
            y = np.asarray(c["y"], float)

            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size < 3:
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            sw = _clamp_smooth_win(len(x), smooth_win)

            row = {"curve": c["run"]}

            if do_knee:
                xk_val, ik = detect_knee_point_triangle(x, y, smooth_win=smooth_win)
                xk, yk = _feature_xy_from_detector(x, y, xk_val, ik)

                row["x_knee"] = xk
                if np.isfinite(xk):
                    ax2.axvline(xk, linestyle="--", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xk, "knee", y_level=y_level);
                        y_level += 1
                    knee_pts.append((xk, yk))

            if do_turn:
                xt_val, it = detect_turning_point_curvature(x, y, smooth_win=smooth_win)
                xt, yt = _feature_xy_from_detector(x, y, xt_val, it)

                row["x_turn"] = xt
                if np.isfinite(xt):
                    ax2.axvline(xt, linestyle=":", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xt, "turn", y_level=y_level);
                        y_level += 1
                    turn_pts.append((xt, yt))

            if do_flat:
                xf_val, iflat = detect_flattening_point(
                    x, y, smooth_win=smooth_win, slope_eps=slope_eps, sustain_frac=sustain_frac
                )
                xf, yf = _feature_xy_from_detector(x, y, xf_val, iflat)

                row["x_flat"] = xf
                if np.isfinite(xf):
                    ax2.axvline(xf, linestyle="-.", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xf, "flat", y_level=y_level);
                        y_level += 1
                    flat_pts.append((xf, yf))

            if do_thr:
                if "max" in thr_mode:
                    yref = np.nanmax(y)
                    thr = float(thr_val) * yref
                else:
                    thr = float(thr_val)

                xthr, _ = first_sustained_crossing(x, y, thr, direction=thr_dir, sustain=int(thr_sustain))
                row["thr"] = thr
                row["x_thr"] = xthr
                if np.isfinite(xthr):
                    ax2.axvline(xthr, linestyle="-", linewidth=2.0)
                    if label_markers:
                        _label_vline(ax2, xthr, "thr", y_level=y_level); y_level += 1

            if do_auc:
                ok = np.isfinite(x) & np.isfinite(y)
                if np.sum(ok) >= 2:
                    auc = float(np.trapz(y[ok], x[ok]))
                    span = float(x[ok][-1] - x[ok][0])
                    mean = auc / span if span != 0 else np.nan
                else:
                    auc, mean = np.nan, np.nan
                row["auc"] = auc
                row["mean"] = mean

            if do_maxslope:
                dy = np.gradient(_moving_average(y, sw), x)
                i = int(np.nanargmax(np.abs(dy)))
                row["x_maxabs_slope"] = float(x[i])
                row["maxabs_slope"] = float(dy[i])
                ax2.axvline(x[i], linestyle="--", linewidth=1.2)
                if label_markers:
                    _label_vline(ax2, x[i], "max|slope|", y_level=y_level); y_level += 1

            if do_peaks:
                idxs = simple_peaks(y, min_prom=float(peak_min_prom), min_dist=int(peak_min_dist))
                row["n_peaks"] = len(idxs)
                for j, i in enumerate(idxs[:10]):
                    ax2.axvline(x[i], linestyle=":", linewidth=1.0)
                    if label_markers and j == 0:
                        _label_vline(ax2, x[i], "peaks", y_level=y_level); y_level += 1


            if do_pwlin:
                xb, _, sse = piecewise_linear_breakpoint(x, y, min_seg_frac=float(pw_min_seg))
                row["x_break"] = xb
                row["pw_sse"] = sse
                if np.isfinite(xb):
                    ax2.axvline(xb, linestyle="-", linewidth=2.5)
                    if label_markers:
                        _label_vline(ax2, xb, "break", y_level=y_level); y_level += 1

            results.append(row)


        if do_prog:
            _overlay_prognosis_surface(
                ax2,
                sweep_rows=sweep_rows,
                x_choice=x_choice,
                tk_query=tk_query,
                ta_query=ta_query,
            )

        if show_locus:
            _overlay_feature_locus(ax2, curves, feature=locus_type, smooth_win=smooth_win, label=f"{locus_type} locus")

        ax2.set_title("Post-analysis (summary sweep)")
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig2 = plt.gcf()
        apply_global_styling(
            fig2,
            legend_on=legend_on,
            legend_fontsize=legend_fontsize,
            axis_fontsize=axis_fontsize,
            title_on=title_on,
            title_fontsize=title_fontsize,
        )
        st.pyplot(fig2)

        import pandas as pd
        st.dataframe(pd.DataFrame(results))




def post_analysis_block(
    key: str,
    curves: list[dict],
    ylabel: str,
    x_axis_mode: str,
    legend_on=True,
    legend_fontsize=14,
    axis_fontsize=16,
    title_on=True,
    title_fontsize=18,
):
    """
    curves: list of dicts with at least:
      - run (str)
      - x (np.ndarray)
      - y (np.ndarray)
    optional:
      - tkb, taub, kind, etc.
    """
    with st.expander("🔎 Post-analysis (turning point / flattening / thresholds)", expanded=False):

        if not curves:
            st.info("No curves available.")
            return

        with st.form(key=f"post_form_{key}", clear_on_submit=False):
            run_opts = ["All"] + sorted({c.get("run", "run") for c in curves})
            pick = st.selectbox("Analyze run", run_opts, index=0, key=f"{key}_pick")

            smooth_win = st.slider("Smoothing window", 1, 51, 9, 2, key=f"{key}_smooth")
            sustain = st.slider("Flatten sustain (% of points)", 1, 30, 8, 1, key=f"{key}_sustain")
            slope_eps_user = st.number_input(
                "Flatten slope threshold (leave 0 for auto)",
                value=0.0, step=0.0001, format="%.6f",
                key=f"{key}_slopeeps",
            )

            do_knee = st.checkbox("Mark knee (triangle method)", True, key=f"{key}_knee")
            do_turn = st.checkbox("Mark turning point (curvature)", True, key=f"{key}_turn")
            do_flat = st.checkbox("Mark flattening point", True, key=f"{key}_flat")
            label_markers = st.checkbox("Label markers on plot", value=True, key=f"{key}_labels")

            st.markdown("#### Extra post-analysis")
            do_thr = st.checkbox("Time-to-threshold", value=False, key=f"{key}_thr")
            thr_mode = st.selectbox("Threshold mode", ["absolute", "% of max (per curve)"], index=1, key=f"{key}_thrm")
            thr_val = st.number_input("Threshold value (abs or fraction)", value=0.6, step=0.01, format="%.4f", key=f"{key}_thrv")
            thr_dir = st.selectbox("Threshold direction", ["rising", "falling"], index=0, key=f"{key}_thrd")
            thr_sustain = st.number_input("Threshold sustain (points)", value=3, step=1, min_value=1, key=f"{key}_thrs")

            do_auc = st.checkbox("AUC + mean over window", value=False, key=f"{key}_auc")
            do_maxslope = st.checkbox("Max |slope| marker", value=False, key=f"{key}_ms")

            do_peaks = st.checkbox("Peak detection", value=False, key=f"{key}_peaks")
            peak_min_prom = st.number_input("Min peak prominence", value=0.0, step=0.01, format="%.4f", key=f"{key}_pp")
            peak_min_dist = st.number_input("Min distance between peaks (points)", value=5, step=1, min_value=1, key=f"{key}_pd")

            do_pwlin = st.checkbox("Piecewise-linear breakpoint", value=False, key=f"{key}_pw")
            pw_min_seg = st.slider("Min segment size (fraction)", 0.05, 0.4, 0.1, 0.05, key=f"{key}_pwseg")

            run_post = st.form_submit_button("▶ Run post-analysis")

        if not run_post:
            st.caption("Adjust settings, then click **Run post-analysis**.")
            return

        # -------- run analysis + plot ----------
        plt.close("all")
        plt.figure()
        ax2 = plt.gca()

        # re-plot selected curves
        for c in curves:
            if pick != "All" and c.get("run") != pick:
                continue
            ax2.plot(c["x"], c["y"], label=c.get("run", "run"))

        slope_eps = None if slope_eps_user == 0 else float(slope_eps_user)
        sustain_frac = float(sustain) / 100.0

        results = []
        y_level = 0

        for c in curves:
            if pick != "All" and c.get("run") != pick:
                continue

            x = np.asarray(c["x"], float)
            y = np.asarray(c["y"], float)

            m = np.isfinite(x) & np.isfinite(y)
            row = {"run": c.get("run"), "tkb": c.get("tkb"), "taub": c.get("taub"), "kind": c.get("kind")}
            if np.sum(m) < 5:
                row["note"] = "Skipped: <5 finite points"
                results.append(row)
                continue

            x = x[m]
            y = y[m]

            row = {
                "run": c.get("run"),
                "tkb": c.get("tkb"),
                "taub": c.get("taub"),
                "kind": c.get("kind"),
            }

            if do_knee:
                xk, _ = detect_knee_point_triangle(x, y, smooth_win=smooth_win)
                row["t_knee"] = xk
                if np.isfinite(xk):
                    ax2.axvline(xk, linestyle="--", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xk, "knee", y_level=y_level); y_level += 1

            if do_turn:
                xt, _ = detect_turning_point_curvature(x, y, smooth_win=smooth_win)
                row["t_turn"] = xt
                if np.isfinite(xt):
                    ax2.axvline(xt, linestyle=":", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xt, "turn", y_level=y_level); y_level += 1

            if do_flat:
                xf, _ = detect_flattening_point(x, y, smooth_win=smooth_win, slope_eps=slope_eps, sustain_frac=sustain_frac)
                row["t_flat"] = xf
                if np.isfinite(xf):
                    ax2.axvline(xf, linestyle="-.", linewidth=1.5)
                    if label_markers:
                        _label_vline(ax2, xf, "flat", y_level=y_level); y_level += 1

            if do_thr:
                if "max" in thr_mode:
                    thr = float(thr_val) * float(np.nanmax(y))
                else:
                    thr = float(thr_val)

                xthr, _ = first_sustained_crossing(x, y, thr, direction=thr_dir, sustain=int(thr_sustain))
                row["thr"] = thr
                row["t_thr"] = xthr
                if np.isfinite(xthr):
                    ax2.axvline(xthr, linestyle="-", linewidth=2.0)
                    if label_markers:
                        _label_vline(ax2, xthr, "thr", y_level=y_level); y_level += 1

            if do_auc:
                m = np.isfinite(x) & np.isfinite(y)
                if np.sum(m) >= 2:
                    auc = float(np.trapz(y[m], x[m]))
                    span = float(x[m][-1] - x[m][0])
                    mean = auc / span if span != 0 else np.nan
                else:
                    auc, mean = np.nan, np.nan
                row["auc"] = auc
                row["mean"] = mean

            if do_maxslope:
                dy = np.gradient(_moving_average(y, smooth_win), x)
                i = int(np.nanargmax(np.abs(dy)))
                row["t_maxabs_slope"] = float(x[i])
                row["maxabs_slope"] = float(dy[i])
                ax2.axvline(x[i], linestyle="--", linewidth=1.2)
                if label_markers:
                    _label_vline(ax2, x[i], "max|slope|", y_level=y_level); y_level += 1

            if do_peaks:
                idxs = simple_peaks(y, min_prom=float(peak_min_prom), min_dist=int(peak_min_dist))
                row["n_peaks"] = len(idxs)
                for j, ii in enumerate(idxs[:10]):
                    ax2.axvline(x[ii], linestyle=":", linewidth=1.0)
                    if label_markers and j == 0:
                        _label_vline(ax2, x[ii], "peaks", y_level=y_level); y_level += 1

            if do_pwlin:
                xb, _, sse = piecewise_linear_breakpoint(x, y, min_seg_frac=float(pw_min_seg))
                row["t_break"] = xb
                row["pw_sse"] = sse
                if np.isfinite(xb):
                    ax2.axvline(xb, linestyle="-", linewidth=2.5)
                    if label_markers:
                        _label_vline(ax2, xb, "break", y_level=y_level); y_level += 1

            results.append(row)

        ax2.set_title(f"")
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel("Time (τB)" if x_axis_mode == "Brownian time τ_B" else "Time (s)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig2 = plt.gcf()
        apply_global_styling(
            fig2,
            legend_on=legend_on,
            legend_fontsize=legend_fontsize,
            axis_fontsize=axis_fontsize,
            title_on=title_on,
            title_fontsize=title_fontsize,
        )
        st.pyplot(fig2)

        import pandas as pd
        st.dataframe(pd.DataFrame(results))

_num_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def _try_float_from_label(s: str):
    m = _num_re.search(str(s))
    return float(m.group(1)) if m else None

def _overlay_feature_locus(ax, curves, feature: str, smooth_win: int, label: str):
    pts = []
    for c in curves:
        x = np.asarray(c["x"], float)
        y = np.asarray(c["y"], float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 3:
            continue
        o = np.argsort(x)
        x, y = x[o], y[o]

        if feature == "turn":
            xf, _ = detect_turning_point_curvature(x, y, smooth_win=smooth_win)
        elif feature == "knee":
            xf, _ = detect_knee_point_triangle(x, y, smooth_win=smooth_win)
        elif feature == "flat":
            xf, _ = detect_flattening_point(x, y, smooth_win=smooth_win, slope_eps=None, sustain_frac=0.08)
        else:
            continue

        if not np.isfinite(xf):
            continue

        # SNAP: use nearest sampled point, not interpolation
        i = int(np.nanargmin(np.abs(x - xf)))
        pts.append({
            "x": float(x[i]),
            "y": float(y[i]),
            "order": _try_float_from_label(c.get("run", "")),
        })

    if len(pts) < 2:
        return

    # Connect in parameter order if possible (TauB 2, TauB 10, ...)
    if all(p["order"] is not None for p in pts):
        pts.sort(key=lambda p: p["order"])
    else:
        # fallback: connect left-to-right
        pts.sort(key=lambda p: p["x"])

    ax.plot([p["x"] for p in pts], [p["y"] for p in pts],
            linestyle="--", marker="x", linewidth=2, label=label)

def _feature_xy_from_detector(x, y, x_or_idx, idx):
    """
    Many detectors return (x_value, idx) or (idx, something).
    We accept both and try to produce a valid (xf, yf) on the curve.
    """
    # prefer index if provided
    if idx is not None and np.isfinite(idx):
        i = int(np.clip(int(idx), 0, len(x) - 1))
        return float(x[i]), float(y[i])

    # otherwise treat first output as x-value
    xv = float(x_or_idx)
    if not np.isfinite(xv):
        return np.nan, np.nan

    return xv, float(np.interp(xv, x, y))

def _clamp_smooth_win(n: int, w: int) -> int:
    w = int(w)

    # must be at least 3 if possible
    if n >= 3:
        w = max(3, w)
    else:
        return max(1, min(w, n))

    # can't exceed n (and prefer odd <= n)
    w = min(w, n)

    # force odd
    if w % 2 == 0:
        w -= 1

    # if we accidentally dropped below 3, bump back (only happens for very small n)
    if w < 3 and n >= 3:
        w = 3

    return w



def _overlay_feature_locus(ax, curves, *, feature="turn", smooth_win=9, label=None):
    pts = []

    for c in curves:
        x = np.asarray(c["x"], float)
        y = np.asarray(c["y"], float)

        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 3:
            continue

        o = np.argsort(x)
        x, y = x[o], y[o]

        # detect feature x-position
        if feature == "turn":
            xf, _ = detect_turning_point_curvature(x, y, smooth_win=smooth_win)
        elif feature == "flat":
            xf, _ = detect_flattening_point(x, y, smooth_win=smooth_win)
        else:  # "knee"
            xf, _ = detect_knee_point_triangle(x, y, smooth_win=smooth_win)

        if not np.isfinite(xf):
            continue

        # IMPORTANT: y value at that x, not y[idx]
        yf = float(np.interp(xf, x, y))
        pts.append((float(xf), float(yf)))

    if len(pts) >= 2:
        pts.sort(key=lambda p: p[0])
        if label is None:
            label = f"{feature} locus"

        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            linestyle="--",
            marker="x",
            linewidth=2,
            label=label,
        )


def _overlay_prognosis_surface(
    ax,
    *,
    sweep_rows,
    x_choice,
    tk_query,
    ta_query,
    min_points=6,
):
    """
    Fit a quadratic surface y = f(TkB, TauB) to summary sweep rows,
    then overlay predicted curves on the current axes.

    sweep_rows: [(run, tkb, taub, scalar), ...]
    x_choice: "TkB" or "TauB"  (this is what the current plot uses on x-axis)
    tk_query/ta_query: comma-separated strings of query points.
    """
    if not sweep_rows or x_choice not in ("TkB", "TauB"):
        return

    tk = np.array([r[1] for r in sweep_rows], float)
    ta = np.array([r[2] for r in sweep_rows], float)
    yy = np.array([r[3] for r in sweep_rows], float)

    m = np.isfinite(tk) & np.isfinite(ta) & np.isfinite(yy)
    tk, ta, yy = tk[m], ta[m], yy[m]

    if tk.size < int(min_points):
        return  # not enough data to fit

    # quadratic surface in (tk, ta)
    X = np.column_stack([np.ones_like(tk), tk, ta, tk**2, ta**2, tk * ta])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)

    def predict(tk_new, ta_new):
        tk_new = np.asarray(tk_new, float)
        ta_new = np.asarray(ta_new, float)
        Xn = np.column_stack([np.ones_like(tk_new), tk_new, ta_new, tk_new**2, ta_new**2, tk_new * ta_new])
        return Xn @ coef

    # parse queries safely
    try:
        tk_list = [float(s.strip()) for s in str(tk_query).split(",") if s.strip()]
    except Exception:
        tk_list = []
    try:
        ta_list = [float(s.strip()) for s in str(ta_query).split(",") if s.strip()]
    except Exception:
        ta_list = []

    if x_choice == "TkB":
        if not tk_list or not ta_list:
            return
        # y vs TkB for each requested TauB
        for taub_val in ta_list:
            yhat = predict(tk_list, [taub_val] * len(tk_list))
            order = np.argsort(tk_list)
            xs = np.asarray(tk_list, float)[order]
            ys = np.asarray(yhat, float)[order]
            ax.plot(xs, ys, linestyle="--", marker=".", label=f"pred TauB {taub_val:g}")

    else:  # x_choice == "TauB"
        if not ta_list or not tk_list:
            return
        # y vs TauB for each requested TkB
        for tkb_val in tk_list:
            yhat = predict([tkb_val] * len(ta_list), ta_list)
            order = np.argsort(ta_list)
            xs = np.asarray(ta_list, float)[order]
            ys = np.asarray(yhat, float)[order]
            ax.plot(xs, ys, linestyle="--", marker=".", label=f"pred TkB {tkb_val:g}")


def first_sustained_crossing(x, y, thr, direction="rising", sustain=3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < sustain:
        return np.nan, None

    if direction == "rising":
        ok = y >= thr
    else:
        ok = y <= thr

    for i in range(0, len(ok) - sustain + 1):
        if np.all(ok[i:i+sustain]):
            return float(x[i]), i
    return np.nan, None




def simple_peaks(y, min_prom=0.0, min_dist=5):
    """
    Simple peak detector.
    Returns indices of peaks.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        return []

    # find local maxima
    cand = []
    for i in range(1, n - 1):
        if np.isfinite(y[i-1]) and np.isfinite(y[i]) and np.isfinite(y[i+1]):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                cand.append(i)

    # prominence filter
    def prominence(i):
        left = np.nanmin(y[max(0, i - min_dist): i + 1])
        right = np.nanmin(y[i: min(n, i + min_dist + 1)])
        return y[i] - max(left, right)

    cand = [i for i in cand if prominence(i) >= min_prom]

    # enforce minimum distance
    kept = []
    for i in sorted(cand, key=lambda j: y[j], reverse=True):
        if all(abs(i - k) >= min_dist for k in kept):
            kept.append(i)

    return sorted(kept)

def first_threshold_crossing(x, y, thr, *, direction="rising", sustain_pts=1):
    """
    Return (x_cross, idx_cross) of first sustained threshold crossing.
    direction: "rising" -> y >= thr, "falling" -> y <= thr
    sustain_pts: must hold condition for this many consecutive points
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 2:
        return np.nan, None

    sustain_pts = int(max(1, sustain_pts))
    if direction == "falling":
        cond = (y <= thr)
    else:
        cond = (y >= thr)

    for i in range(0, len(cond) - sustain_pts + 1):
        if np.all(cond[i:i+sustain_pts]):
            return float(x[i]), int(i)
    return np.nan, None


def auc_and_mean(x, y):
    """Return (auc, mean) over x using trapezoid AUC and time-average."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 2:
        return np.nan, np.nan

    auc = float(np.trapz(y, x))
    dt = float(x[-1] - x[0])
    mean = float(auc / dt) if dt != 0 else np.nan
    return auc, mean


def max_slope_time(x, y, *, smooth_win=9):
    """
    Return (x_at_max_abs_slope, idx, slope_value) using smoothed derivative.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 3:
        return np.nan, None, np.nan

    ys = _moving_average(y, smooth_win)
    dy = np.gradient(ys, x)
    idx = int(np.nanargmax(np.abs(dy)))
    return float(x[idx]), idx, float(dy[idx])




def piecewise_linear_breakpoint(x, y, min_seg_frac=0.1):
    """
    Fit two line segments with a breakpoint k and choose k minimizing SSE.
    Returns (x_break, idx_break, sse).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = x.size

    if n < 10:
        return np.nan, None, np.nan

    min_seg = max(3, int(round(min_seg_frac * n)))
    best_sse = np.inf
    best_k = None

    for k in range(min_seg, n - min_seg):
        p1 = np.polyfit(x[:k], y[:k], 1)
        p2 = np.polyfit(x[k:], y[k:], 1)

        y1 = np.polyval(p1, x[:k])
        y2 = np.polyval(p2, x[k:])

        sse = np.nansum((y[:k] - y1) ** 2) + np.nansum((y[k:] - y2) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_k = k

    if best_k is None:
        return np.nan, None, np.nan

    return float(x[best_k]), int(best_k), float(best_sse)

def _label_vline(ax, x, text, *, rotation=90, fontsize=10, pad_frac=0.02, y_level=0):
    """
    Put a small label near the top of the axes at x.
    y_level lets you stagger labels (0,1,2...) to reduce overlap.
    """
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if (y1 != y0) else 1.0
    y = y1 - (pad_frac + 0.06 * y_level) * yr
    ax.text(x, y, text, rotation=rotation, va="top", ha="left", fontsize=fontsize)


def _moving_average(y, win: int):
    y = np.asarray(y, float)
    win = int(max(1, win))
    if win <= 1 or y.size < 3:
        return y
    # pad edges to avoid shrinking
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    k = np.ones(win, float) / win
    return np.convolve(ypad, k, mode="valid")

def detect_flattening_point(x, y, *, smooth_win=9, slope_eps=None, sustain_frac=0.08):
    """
    Flattening = first time where |dy/dx| stays below slope_eps for a sustained window.
    If slope_eps is None, it is set relative to typical slope magnitude.
    Returns (x_flat, idx_flat) or (np.nan, None).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return np.nan, None

    ys = _moving_average(y, smooth_win)
    dy = np.gradient(ys, x)

    if slope_eps is None:
        # relative threshold: small fraction of typical slope
        ref = np.nanmedian(np.abs(dy))
        slope_eps = 0.08 * ref if np.isfinite(ref) and ref > 0 else 1e-12

    sustain = int(max(2, round(sustain_frac * x.size)))
    ok = np.abs(dy) <= slope_eps

    # find first index i such that ok[i:i+sustain] all True
    for i in range(0, len(ok) - sustain):
        if np.all(ok[i:i + sustain]):
            return float(x[i]), int(i)

    return np.nan, None

def detect_knee_point_triangle(x, y, *, smooth_win=9):
    """
    Knee/turning point via "triangle method":
    Normalize curve, then find point with max distance to line from start->end.
    Returns (x_knee, idx_knee) or (np.nan, None).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # keep only finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan, None

    ys = _moving_average(y, smooth_win)

    # normalize x to [0,1]
    x0, x1 = float(x[0]), float(x[-1])
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 == x0:
        return np.nan, None
    xn = (x - x0) / (x1 - x0)

    # normalize y to [0,1]
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
        return np.nan, None
    yn = (ys - y0) / (y1 - y0)

    # line from start to end
    p0 = np.array([0.0, yn[0]])
    p1 = np.array([1.0, yn[-1]])
    v = p1 - p0
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv == 0:
        return np.nan, None

    pts = np.column_stack([xn, yn])
    cross = np.abs((pts[:, 0] - p0[0]) * v[1] - (pts[:, 1] - p0[1]) * v[0])
    dist = cross / nv

    # NEW: guard against all-NaN dist
    if not np.any(np.isfinite(dist)):
        return np.nan, None

    idx = int(np.nanargmax(dist))
    return float(x[idx]), idx

def detect_turning_point_curvature(x, y, *, smooth_win=9):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan, None

    ys = _moving_average(y, smooth_win)
    d1 = np.gradient(ys, x)
    d2 = np.gradient(d1, x)

    if not np.any(np.isfinite(d2)):
        return np.nan, None

    idx = int(np.nanargmax(np.abs(d2)))
    return float(x[idx]), idx

def sidebar_runs_common_ui(
    prefix: str,
    source_root: str,
    metric_options: list[str],
    *,
    include_metric_select: bool = False,
    default_metrics: list[str] | None = None,
    include_crystal_multiselect: bool = False,
):
    """
    Common sidebar UI.
    - If include_metric_select=True: shows metric multiselect right after run selection.
    - If include_crystal_multiselect=True: shows crystal-type multiselect ONLY when Detect crystals is selected.
    - "File names" are placed directly under the folder chooser and collapsed in an expander.
    """

    st.sidebar.header("2. Data folder")

    folder_source = st.sidebar.radio(
        "Folder source:",
        ["HIWI root subfolder", "Custom path"],
        key=f"{prefix}_folder_source",
    )

    hiwi_subdirs = []
    if os.path.isdir(source_root):
        try:
            hiwi_subdirs = [
                d for d in os.listdir(source_root)
                if os.path.isdir(os.path.join(source_root, d))
            ]
        except Exception as e:
            st.sidebar.error(f"Could not list subfolders under source root: {e}")

    if folder_source == "HIWI root subfolder":
        subchoice = st.sidebar.selectbox(
            "Choose subfolder under source root:",
            hiwi_subdirs,
            key=f"{prefix}_hiwi_sub",
        )
        base_dir = os.path.join(source_root, subchoice) if subchoice else ""
    else:
        base_dir = st.sidebar.text_input(
            "Custom base folder path:",
            value="",
            key=f"{prefix}_folder_custom",
        )

    base_dir = base_dir.strip().strip('"').strip("'")
    dataset_tag = os.path.basename(base_dir.rstrip("\\/")) if base_dir else ""

    # --- File names: collapsed right under folder chooser (and compact) ---
    with st.sidebar.expander("4. File names", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            x_name = st.text_input(
                "X",
                value="datax.csv",
                key=f"{prefix}_xname",
                help="X filename inside each run folder",
            )
        with c2:
            y_name = st.text_input(
                "Y",
                value="datay.csv",
                key=f"{prefix}_yname",
                help="Y filename inside each run folder",
            )
        c3, c4 = st.columns(2)
        with c3:
            skip = st.number_input(
                "Skip (frame stride)",
                value=1,
                step=1,
                min_value=0,
                key=f"{prefix}_skip",
            )
        with c4:
            normY = st.checkbox(
                "Normalize Y axis",
                value=True,
                key=f"{prefix}_normY",
            )

    # ---- runs detection ----
    st.sidebar.header("3. Runs (TkB / TauB)")

    available_runs = []
    if base_dir and os.path.isdir(base_dir):
        try:
            folder_re = re.compile(
                r"^([0-9]+(?:\.[0-9]+)?)Tkb[ _]*([0-9]+(?:\.[0-9]+)?)TauB$",
                re.IGNORECASE,
            )
            for fn in os.listdir(base_dir):
                m = folder_re.match(fn)
                if not m:
                    continue
                tkb_val = float(m.group(1))
                tau_val = float(m.group(2))
                folder_path = os.path.join(base_dir, fn)
                available_runs.append((folder_path, tkb_val, tau_val))
        except Exception as e:
            st.sidebar.error(f"Could not scan TkB/TauB folders in {base_dir}: {e}")

    available_runs = sorted(available_runs, key=lambda r: (r[1], r[2]))

    def format_run_option(run) -> str:
        _, tkb, tau = run
        return f"{tkb:g} Tkb (TauB: {tau:g})"

    if available_runs:
        selected_runs = st.sidebar.multiselect(
            "Select TkB / TauB combinations (from folders):",
            options=available_runs,
            default=available_runs,
            key=f"{prefix}_runs",
            format_func=format_run_option,
        )
    else:
        selected_runs = []
        st.sidebar.warning("No Tkb_*TauB* subfolders detected. Please check your base folder.")

    # ---- metrics selection (NOW right after run selection) ----
    metrics_selected = None
    if include_metric_select:
        st.sidebar.header("4. Metrics")
        if default_metrics is None:
            default_metrics = [metric_options[0]] if metric_options else []
        metrics_selected = st.sidebar.multiselect(
            "Select metric function(s) (y-axis):",
            metric_options,
            default=default_metrics,
            key=f"{prefix}_metric_select",
        )

    # ---- Crystal metrics parameters (ONLY if Detect crystals selected) ----
    crystal_types = []
    if include_crystal_multiselect and metrics_selected and ("Detect crystals" in metrics_selected):
        st.sidebar.header("Crystal metrics parameters")
        crystal_types = st.sidebar.multiselect(
            "Crystal types (Detect crystals)",
            ["Hexagonal", "Square", "Triangular", "All", "None"],
            default=["Hexagonal"],
            key=f"{prefix}_crystal_types",
        )

    # ---- common params ----
    with st.sidebar.expander("Common parameters", expanded=False):

        # cluster params
        st.subheader("Cluster metrics parameters")

        area_fraction_mode = st.selectbox(
            "Cluster detection mode",
            ["per cluster", "global"],
            index=0,
            key=f"{prefix}_area_fraction_mode",
        )

        cluster_eps = st.number_input(
            "DBSCAN eps",
            value=3.5,
            step=0.1,
            key=f"{prefix}_cluster_eps",
        )
        cluster_min_samples = st.number_input(
            "DBSCAN min_samples",
            value=3,
            step=1,
            min_value=1,
            key=f"{prefix}_cluster_min_samples",
        )
        cluster_min_cluster_size = st.number_input(
            "Min cluster size",
            value=3,
            step=1,
            min_value=1,
            key=f"{prefix}_cluster_min_cluster_size",
        )


        st.subheader("RDF parameters")
        rdf_r_max = st.number_input(
            "RDF r_max (µm)", value=10.0, step=0.5, key=f"{prefix}_rdf_rmax"
        )
        rdf_dr = st.number_input(
            "RDF dr (bin width, µm)", value=0.1, step=0.05, format="%.3f", key=f"{prefix}_rdf_dr"
        )


    # styling
    st.sidebar.header("Styling")
    c_left, c_right = st.sidebar.columns([1, 1])
    with c_left:
        legend_on = st.checkbox("Show legend", value=True, key=f"{prefix}_legend_on")
        show_taub_in_legend = st.checkbox("Show TauB in legend", value=False, key="multi_show_taub")
        title_on = st.checkbox("Show title", value=False, key=f"{prefix}_title_on")
    with c_right:
        legend_fontsize = st.number_input(
            "Legend font size", value=16, min_value=1, max_value=40, key=f"{prefix}_legend_fs"
        )
        axis_fontsize = st.number_input(
            "Axis font size", value=18, min_value=1, max_value=40, key=f"{prefix}_axis_fs"
        )
        title_fontsize = st.number_input(
            "Title font size", value=20, min_value=1, max_value=60, key=f"{prefix}_title_fs"
        )

    # x-axis mode
    st.sidebar.subheader("X-Axis")
    c11, c12 = st.sidebar.columns(2)
    with c11:
        x_axis_mode = st.radio(
            "X-axis units",
            ["Brownian time τ_B", "Seconds"],
            key=f"{prefix}_xaxis_mode",
        )
    with c12:
        TauB = st.number_input(
            "TauB (Brownian time)",
            value=2.0,
            step=0.5,
            key=f"{prefix}_TauB",
        )

    # y-axis mode
    st.sidebar.subheader("Y-Axis")
    c21, c22 = st.sidebar.columns(2)
    with c21:
        y_scale = st.selectbox(
            "Scale",
            ["Linear", "Log"],
            index=0,
            key=f"{prefix}_y_scale",
        )
    with c22:
        y_unit = st.text_input(
            "Unit label",
            placeholder="μm, mm, 1/s, …",
            key=f"{prefix}_y_unit",
            help="This is only the label shown on the plot (does not change the data).",
        )
    y_range_custom = st.sidebar.checkbox("Custom Y range", value=False, key=f"{prefix}_y_range_custom")
    y_range_min = st.sidebar.number_input("Y min", value=0.0, key=f"{prefix}_y_range_min") if y_range_custom else None
    y_range_max = st.sidebar.number_input("Y max", value=1.0, key=f"{prefix}_y_range_max") if y_range_custom else None

    # export / recompute
    st.sidebar.header("Caching / export")
    export_data = st.sidebar.checkbox(
        "Export (recompute + overwrite saved run data)",
        value=False,
        key=f"{prefix}_export_data",
    )
    export_dir = st.sidebar.text_input(
        "Optional CSV export folder (leave empty to skip CSV export):",
        value=DEFAULT_SAVE_DIR,
        key=f"{prefix}_export_dir",
    )

    out = {
        "base_dir": base_dir,
        "dataset_tag": dataset_tag,
        "selected_runs": selected_runs,
        "x_name": x_name,
        "y_name": y_name,
        "TauB": TauB,
        "skip": int(skip),
        "normY": int(normY),
        "cluster_eps": float(cluster_eps),
        "cluster_min_samples": int(cluster_min_samples),
        "cluster_min_cluster_size": int(cluster_min_cluster_size),
        "legend_on": legend_on,
        "show_taub_in_legend": show_taub_in_legend,
        "legend_fontsize": int(legend_fontsize),
        "axis_fontsize": int(axis_fontsize),
        "title_on": title_on,
        "title_fontsize": int(title_fontsize),
        "x_axis_mode": x_axis_mode,
        "export_data": export_data,
        "export_dir": export_dir,
        "crystal_types": crystal_types,
        "metrics_selected": metrics_selected,
        "y_scale": y_scale,
        "y_unit": y_unit,
        "rdf_r_max": float(rdf_r_max),
        "rdf_dr": float(rdf_dr),
        "y_range_custom": y_range_custom,
        "y_range_min": y_range_min,
        "y_range_max": y_range_max,
        "area_fraction_mode": area_fraction_mode,

    }
    return out



def slice_by_time_window(t: np.ndarray, y: np.ndarray, start_frac: float, end_frac: float):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 2:
        return t, y

    t0, t1 = float(t.min()), float(t.max())
    a = t0 + (t1 - t0) * float(start_frac)
    b = t0 + (t1 - t0) * float(end_frac)

    sel = (t >= a) & (t <= b)
    return t[sel], y[sel]


def avg_slope_linear_fit(t: np.ndarray, y: np.ndarray, t_start_frac=0.0, t_end_frac=1.0) -> float:
    """
    Returns slope dy/dt from a least-squares linear fit of y(t)
    between fractions of the time span.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # finite mask
    m = np.isfinite(t) & np.isfinite(y)
    t = t[m]
    y = y[m]
    if t.size < 2:
        return float("nan")

    t0, t1 = float(t.min()), float(t.max())
    if t1 <= t0:
        return float("nan")

    a = t0 + (t1 - t0) * float(t_start_frac)
    b = t0 + (t1 - t0) * float(t_end_frac)

    sel = (t >= a) & (t <= b)
    t_fit = t[sel]
    y_fit = y[sel]

    if t_fit.size < 2:
        return float("nan")

    # slope from polyfit (degree 1)
    slope, intercept = np.polyfit(t_fit, y_fit, 1)
    return float(slope)


def _safe_dt(t):
    t = np.asarray(t, float)
    if len(t) < 2:
        return 1.0
    dt = float(t[1] - t[0])
    return dt if dt != 0 else 1.0

def deriv(y, t=None):
    """Numerical derivative dy/dt using numpy gradient on uniform grid."""
    y = np.asarray(y, float)
    if t is None:
        return np.gradient(y)
    dt = _safe_dt(t)
    return np.gradient(y, dt)

def integ(y, t=None):
    """Cumulative integral ∫ y dt using cumulative trapezoid (no scipy)."""
    y = np.asarray(y, float)
    if len(y) < 2:
        return np.zeros_like(y)
    if t is None:
        # assume dt=1
        return np.cumsum((y[:-1] + y[1:]) * 0.5)  # length n-1
    t = np.asarray(t, float)
    dt = _safe_dt(t)
    out = np.zeros_like(y)
    out[1:] = np.cumsum((y[:-1] + y[1:]) * 0.5 * dt)
    return out

def fft_amp(y):
    """One-sided FFT amplitude spectrum (positive frequencies)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return y
    Y = np.fft.rfft(y - np.nanmean(y))
    return np.abs(Y)

def fft_power(y):
    """One-sided FFT power spectrum."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        return y
    Y = np.fft.rfft(y - np.nanmean(y))
    return (np.abs(Y) ** 2)

def fft_freqs(t):
    """FFT frequency axis (Hz) for rfft based on time grid t (seconds)."""
    t = np.asarray(t, float)
    n = len(t)
    dt = _safe_dt(t)
    return np.fft.rfftfreq(n, d=dt)

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

# Allowed functions (vectorized via numpy)
_ALLOWED_FUNCS = {
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    "clip": np.clip,
    "where": np.where,
    "min": np.minimum,
    "max": np.maximum,
    "deriv": deriv,  # deriv(A) or deriv(A, t)
    "integ": integ,  # integ(A) or integ(A, t)
    "fft_amp": fft_amp,  # fft_amp(A)
    "fft_power": fft_power,
    "amax": np.max,
    "amin": np.min,
    "mean": np.mean,
    "median": np.median,
}

_ALLOWED_NAMES = {"A", "B", "pi", "e", "t"}

def safe_eval_expr(expr: str, env: dict):
    """
    Safely evaluate a math expression using AST.
    env must provide A and B as numpy arrays, plus optional constants.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Only numeric constants are allowed.")

        # names
        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES:
                raise ValueError(f"Unknown variable '{node.id}'. Allowed: A, B.")
            return env[node.id]

        # binary ops
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINOPS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))

        # unary ops
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARYOPS:
                raise ValueError("Unary operator not allowed.")
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))

        # function calls
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed.")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(
                    f"Function '{fname}' not allowed. Allowed: {', '.join(sorted(_ALLOWED_FUNCS.keys()))}"
                )
            args = [_eval(a) for a in node.args]
            return _ALLOWED_FUNCS[fname](*args)

        raise ValueError("Expression contains unsupported syntax.")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)



import json, hashlib
from pathlib import Path

RUN_DATA_DIRNAME = ".run_data"  # where saved per-run metric arrays live

def rundata_dir(base_dir: str) -> Path:
    d = Path(base_dir) / RUN_DATA_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_rundata_stem(dataset_tag: str, tkb: float, taub: float, metric_label: str) -> str:
    # Your naming: NAF_tkb_taub_metric
    # Add units text to avoid ambiguity and keep filenames safe
    return slugify(f"{dataset_tag}_{tkb:g}tkb_{taub:g}taub_{metric_label}")

def rundata_npz_path(base_dir: str, dataset_tag: str, tkb: float, taub: float, metric_label: str) -> Path:
    return rundata_dir(base_dir) / (make_rundata_stem(dataset_tag, tkb, taub, metric_label) + ".npz")

def save_rundata_npz(path: Path, y: np.ndarray):
    y = np.asarray(y, float)
    np.savez_compressed(path, y=y)

def load_rundata_npz(path: Path) -> np.ndarray:
    z = np.load(path)
    return np.asarray(z["y"], float)

def export_rundata_csv(path_csv: Path, y: np.ndarray):
    # No time column, just index and y
    import csv
    y = np.asarray(y, float)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y"])
        for i, val in enumerate(y):
            w.writerow([i, "" if not np.isfinite(val) else float(val)])


DATASET_CACHE_DIRNAME = ".dataset_cache"   # separate from .metric_cache

def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def dataset_cache_path(base_dir: str, metric_label: str, metric_func_name: str, runs: list, x_name: str, y_name: str, metric_kwargs_base: dict) -> Path:
    """
    Cache key includes:
      - metric identity
      - exact selected runs (folder names + tkb + taub)
      - file names (datax.csv, datay.csv)
      - common kwargs (skip, normY, etc.)
    """
    cache_dir = Path(base_dir) / DATASET_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs_key = [
        {"folder": os.path.basename(fp), "tkb": float(tkb), "taub": float(taub)}
        for (fp, tkb, taub) in runs
    ]

    key_obj = {
        "metric_label": metric_label,
        "metric_func": metric_func_name,
        "runs": runs_key,
        "x_name": x_name,
        "y_name": y_name,
        "metric_kwargs_base": metric_kwargs_base,
    }
    key = _stable_hash(key_obj)
    return cache_dir / f"{slugify(metric_label)}__{key}.npz"

def save_dataset_npz(path: Path, series: list[dict]):
    """
    series entries must be dicts with keys: run, tkb, taub, t, y
    Stored as arrays in NPZ (object arrays for variable-length t/y).
    """
    runs = np.array([s["run"] for s in series], dtype=object)
    tkb  = np.array([float(s["tkb"]) for s in series], dtype=float)
    taub = np.array([float(s["taub"]) for s in series], dtype=float)
    t_arr = np.array([np.asarray(s["t"], float) for s in series], dtype=object)
    y_arr = np.array([np.asarray(s["y"], float) for s in series], dtype=object)

    np.savez_compressed(path, runs=runs, tkb=tkb, taub=taub, t=t_arr, y=y_arr)

def get_series_for_metric(
    metric_label: str,
    metric_func,
    selected_runs,
    base_dir: str,
    dataset_tag: str,
    x_name: str,
    y_name: str,
    skip: int,
    normY: int,
    export_data: bool,
    export_dir: str,
    extra_kwargs= None,
):
    """
    Returns series = [{"run","tkb","taub","y"}...] for one metric_label.
    Uses NPZ cache if export_data is False and file exists.
    Otherwise computes + saves NPZ (and optionally exports CSV).
    """
    series = []
    missing = []

    for folder_path, tkb_val, tau_val in selected_runs:
        run_name = os.path.basename(folder_path)
        tkb_val = float(tkb_val)
        tau_val = float(tau_val)

        npz_path = rundata_npz_path(base_dir, dataset_tag, tkb_val, tau_val, metric_label)

        y = None
        loaded = False

        if (not export_data) and npz_path.exists():
            try:
                y = load_rundata_npz(npz_path)
                loaded = True
            except Exception:
                y = None

        if y is None:
            fx = os.path.join(folder_path, x_name)
            fy = os.path.join(folder_path, y_name)
            if not (os.path.isfile(fx) and os.path.isfile(fy)):
                missing.append(run_name)
                continue

            kwargs_run = dict(skip=int(skip), normY=int(normY), TauB=float(tau_val))
            if extra_kwargs:
                kwargs_run.update(extra_kwargs)

            # Bust the .metric_cache entry when export_data is ticked
            if export_data:
                p = metric_cache_path(base_dir, run_name, metric_func.__name__, kwargs_run)
                if p.exists():
                    p.unlink(missing_ok=True)

            t, y_calc = load_or_compute_metric_cached(
                base_dir=base_dir,
                run_folder=run_name,
                fx=fx,
                fy=fy,
                metric_func=metric_func,
                metric_kwargs=kwargs_run,
            )
            y = np.asarray(y_calc, float)

            try:
                save_rundata_npz(npz_path, y)
            except Exception as e:
                st.warning(f"Could not save NPZ for {run_name}: {e}")

            # Optional CSV export (index+y only)
            if export_dir.strip():
                try:
                    out = Path(export_dir.strip())
                    out.mkdir(parents=True, exist_ok=True)
                    csv_path = out / (npz_path.stem + ".csv")
                    export_rundata_csv(csv_path, y)
                except Exception as e:
                    st.warning(f"Could not export CSV for {run_name}: {e}")

        series.append({"run": run_name, "tkb": tkb_val, "taub": tau_val, "y": np.asarray(y, float)})

    return series, missing


def load_dataset_npz(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=True)
    runs = z["runs"]
    tkb  = z["tkb"]
    taub = z["taub"]
    t_arr = z["t"]
    y_arr = z["y"]

    series = []
    for i in range(len(runs)):
        series.append({
            "run": str(runs[i]),
            "tkb": float(tkb[i]),
            "taub": float(taub[i]),
            #"t": np.asarray(t_arr[i], float),
            "y": np.asarray(y_arr[i], float),
        })
    return series



def export_series_to_csv(out_dir: str, dataset_tag: str, metric_label: str, series: list[dict]):
    import csv
    os.makedirs(out_dir, exist_ok=True)

    written_paths = []
    for s in series:
        tkb = float(s["tkb"])
        taub = float(s["taub"])
        run = s["run"]
        y = np.asarray(s["y"], float)

        filename = slugify(f"{dataset_tag}_{tkb:g}tkb_{taub:g}taub_{metric_label}") + ".csv"
        path = os.path.join(out_dir, filename)

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "y", "run_folder"])
            for i, val in enumerate(y):
                w.writerow([i, "" if not np.isfinite(val) else float(val), run])

        written_paths.append(path)

    return written_paths

def align_y_by_taub_length(y_a: np.ndarray, y_b: np.ndarray, taub: float):
    """
    Align two y arrays to same length by interpolating along a synthetic time axis
    spanning [0, taub*13.513] seconds.
    Returns (t_grid_sec, yA_grid, yB_grid).
    """
    y_a = np.asarray(y_a, float)
    y_b = np.asarray(y_b, float)

    n = max(len(y_a), len(y_b))
    t_end_sec = float(taub) * 13.513
    t_grid = np.linspace(0.0, t_end_sec, n)

    # create each array's own x axis then interp onto grid
    xa = np.linspace(0.0, t_end_sec, len(y_a))
    xb = np.linspace(0.0, t_end_sec, len(y_b))

    ya = np.interp(t_grid, xa, y_a, left=y_a[0], right=y_a[-1]) if len(y_a) > 1 else np.full(n, y_a[0] if len(y_a) else np.nan)
    yb = np.interp(t_grid, xb, y_b, left=y_b[0], right=y_b[-1]) if len(y_b) > 1 else np.full(n, y_b[0] if len(y_b) else np.nan)

    return t_grid, ya, yb




CACHE_DIRNAME = ".metric_cache"

def _stable_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def metric_cache_path(base_dir: str, run_folder: str, metric_name: str, metric_kwargs: dict) -> Path:
    cache_dir = Path(base_dir) / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _stable_hash({"run": run_folder, "metric": metric_name, "kwargs": metric_kwargs})
    return cache_dir / f"{key}.npz"

def load_or_compute_metric_cached(base_dir: str, run_folder: str, fx: str, fy: str, metric_func, metric_kwargs: dict, *, force_recompute=False):
    p = metric_cache_path(base_dir, run_folder, metric_func.__name__, metric_kwargs)
    if p.exists() and not force_recompute:
        z = np.load(p)
        return z["t"], z["y"]
    t, y = metric_func(fx, fy, **metric_kwargs)
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    np.savez_compressed(p, t=t, y=y)
    return t, y

def resample_to_grid(t, y, t_grid):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t_ok, y_ok = t[ok], y[ok]
    if len(t_ok) < 2:
        return np.full_like(t_grid, np.nan, dtype=float)
    return np.interp(t_grid, t_ok, y_ok, left=y_ok[0], right=y_ok[-1])



# ---------------------------------------------------------------------
# Default root directory (used as default value for the user-editable root)
# ---------------------------------------------------------------------
HIWI_ROOT = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data"
# app.py

DEFAULT_SAVE_DIR = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Saved Data for Streamlit"


# ---------------------------------------------------------------------
# Helper: save uploaded files
# ---------------------------------------------------------------------
def save_uploaded_file(uploaded_file, filename):
    temp_dir = tempfile.gettempdir()
    path = pathlib.Path(temp_dir) / filename
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(path)


# ---------------------------------------------------------------------
# Helper: global styling for any matplotlib figure
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Helper: slugify for filenames + save figure if requested
# ---------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


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


# ---------------------------------------------------------------------
# Helper: build PPTX from image folders
# ---------------------------------------------------------------------
def build_pptx_from_images(
    images_root: Path,
    output_pptx: Optional[Path] = None,
    cols: int = 4,
    margin_in=0.2,
    title_h_in=0.5,
    label_h_in=0.6,
    row_gap_in=0.4,
) -> Path:
    """
    Walk a folder structure like:
        images_root/
            0Tkb_.../
                metric1.png
                metric2.png
            5Tkb_.../
                metric1.png
                metric2.png
            ...

    and build a PPTX where each slide is one 'metric' (stem name),
    with columns = different Tkb folders.
    """

    margin  = Inches(margin_in)
    title_h = Inches(title_h_in)
    label_h = Inches(label_h_in)
    row_gap = Inches(row_gap_in)

    def parse_tkb(fn: str) -> float:
        m = re.match(r"([\d\.]+)Tkb", fn)
        return float(m.group(1)) if m else float("inf")

    if output_pptx is None:
        output_pptx = images_root / "plots_by_type_Tkb.pptx"

    # 1) collect & sort your Tkb folders
    param_dirs = sorted(
        [d for d in images_root.iterdir() if d.is_dir()],
        key=lambda d: parse_tkb(d.name),
    )
    if not param_dirs:
        raise RuntimeError(f"No Tkb folders found inside: {images_root}")

    # 2) collect all plot‐stems
    plot_stems = sorted(
        {
            img.stem
            for d in param_dirs
            for img in d.glob("*.png")
        }
    )
    if not plot_stems:
        raise RuntimeError(f"No .png images found under: {images_root}")

    n_params = len(param_dirs)
    n_rows   = math.ceil(n_params / cols)

    prs = Presentation()

    usable_w = prs.slide_width - 2*margin
    img_w    = (usable_w - (cols-1)*margin) / cols
    img_h    = img_w

    # total height = top margin + title + gap + rows*(label + plot + gap) + bottom margin
    total_h = (
        margin
        + title_h
        + margin
        + n_rows * (label_h + img_h + row_gap)
        + margin
    )
    prs.slide_height = int(round(total_h))

    blank = prs.slide_layouts[6]

    # 4) build slides
    for stem in plot_stems:
        slide = prs.slides.add_slide(blank)

        # slide title
        tb = slide.shapes.add_textbox(
            margin,
            margin / 2,
            prs.slide_width - 2 * margin,
            title_h,
        )
        p  = tb.text_frame.add_paragraph()
        p.text      = stem.replace("_", " ").capitalize()
        p.font.size = Pt(28)
        p.alignment = PP_ALIGN.CENTER

        # place each Tkb in ascending order
        for idx, d in enumerate(param_dirs):
            img_path = d / f"{stem}.png"
            if not img_path.exists():
                continue

            row, col = divmod(idx, cols)
            left     = margin + col * (img_w + margin)
            top_base = margin + title_h + margin + row * (label_h + img_h + row_gap)

            # label ABOVE the plot
            lbl_tb = slide.shapes.add_textbox(left, top_base, img_w, label_h)
            lbl_p  = lbl_tb.text_frame.add_paragraph()
            lbl_p.text      = f"{parse_tkb(d.name)} Tkb"
            lbl_p.font.size = Pt(12)
            lbl_p.alignment = PP_ALIGN.LEFT

            # picture immediately below that label
            slide.shapes.add_picture(
                str(img_path),
                left,
                top_base + label_h,
                width=img_w,
            )

    prs.save(output_pptx)
    return output_pptx


# ---------------------------------------------------------------------
# Phase-diagram snapshot panel
# ---------------------------------------------------------------------
def make_snapshot_phase_panel(
    base_dir,
    tkb_rows,
    column_times_s,
    taub,
    x_name="datax.csv",
    y_name="datay.csv",
    figure_size=(12, 14),
    dpi=200,
    plot_func=None,
    plot_kwargs=None,
    show_xy_labels=False,
    colorbar_cmap=None,   # NEW
    colorbar_label=None,  # NEW
):

    if plot_kwargs is None:
        plot_kwargs = {}
    if plot_func is None:
        raise ValueError("plot_func must be provided")

    total_time_sec = taub * 13.514

    folder_re = re.compile(r"^([0-9]+(?:\.[0-9]+)?)Tkb_")
    tkb_rows = [float(t) for t in tkb_rows]
    tkb_to_folder = {}
    for fname in sorted(os.listdir(base_dir)):
        m = folder_re.match(fname)
        if m:
            tkb = float(m.group(1))
            if tkb in tkb_rows:
                tkb_to_folder[tkb] = os.path.join(base_dir, fname)

    nrows, ncols = len(tkb_rows), len(column_times_s)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figure_size,
        constrained_layout=True,
        dpi=dpi
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for r, tkb in enumerate(tkb_rows):
        folder = tkb_to_folder.get(tkb)
        for c, t_req in enumerate(column_times_s):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])

            if not folder:
                ax.axis("off")
                continue

            fx = os.path.join(folder, x_name)
            fy = os.path.join(folder, y_name)

            skip = int(plot_kwargs.get("skip", 0))
            X = read_particle_data_csv(fx)[skip::2, 1:][1:, :]
            n_frames = X.shape[0]

            t_sec = max(0.0, min(float(total_time_sec), float(t_req)))
            dt = float(total_time_sec) / max(n_frames - 1, 1)
            frame_idx = int(round(t_sec / dt))
            frame_idx = max(0, min(n_frames - 1, frame_idx))

            plot_func(fx, fy, timestep=frame_idx, ax=ax, **plot_kwargs)

            if not show_xy_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")

            if r == 0:
                ax.set_title(f"{t_sec:.1f} s", fontsize=14)

            if c == 0:
                label = rf"{int(tkb) if tkb.is_integer() else tkb}$T_{{kB}}$"
                ax.set_ylabel(label, fontsize=14)


    # --- optional shared colorbar on the side ---
    if colorbar_cmap is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                            cmap=plt.get_cmap(colorbar_cmap))
        sm.set_array([])

        fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            label=colorbar_label if colorbar_label else "",
            fraction=0.03,
            pad=0.02,
        )

    return fig, axes



# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------



st.set_page_config(page_title="Particle Post Processor and Analyzer", layout="wide")
st.title("Acoustic Particles Analyzer")

# ========== STEP 0: Source root ==========
st.sidebar.header("0. Source root")
source_root = st.sidebar.text_input(
    "Base root folder (where ARF / ASF / NAF live):",
    value=HIWI_ROOT,
    key="source_root",
)
source_root = source_root.strip().strip('"').strip("'")

# ========== STEP 0.5: Global saving options ==========
with st.sidebar.expander("0.5 Output saving", expanded=False):
    save_plots = st.checkbox(
        "Save all generated plots to folder", value=False, key="save_plots"
    )
    output_dir = st.text_input(
        "Output folder path (for saving plots):",
        value="",
        key="output_dir",
    )

    # ========== EXTRA: PowerPoint export (collapsible, nested inside the above) ==========
    with st.expander("📊 PowerPoint export (from existing PNGs)", expanded=False):
        images_root_str = st.text_input(
            "Image root (folder with Tkb_* subfolders & .png plots):",
            value="",
            key="pptx_images_root",
        )

        cols_pptx = st.number_input(
            "Number of columns per slide",
            value=4,
            min_value=1,
            max_value=10,
            step=1,
            key="pptx_cols",
        )

        custom_pptx_name = st.text_input(
            "Output PPTX file name:",
            value="plots_by_type_Tkb.pptx",
            key="pptx_name",
        )

        run_pptx = st.button("Build PPTX", key="run_pptx")

        if run_pptx:
            try:
                img_root = Path(images_root_str.strip().strip('"').strip("'"))
                if not img_root.is_dir():
                    st.error(f"Image root is not a folder: {img_root}")
                else:
                    out_path = img_root / custom_pptx_name if custom_pptx_name else None
                    result_path = build_pptx_from_images(
                        images_root=img_root,
                        output_pptx=out_path,
                        cols=int(cols_pptx),
                    )
                    st.success(f"✅ PPTX created: `{result_path}`")
            except Exception as e:
                st.error(f"Error while building PPTX: {e}")
# ========== STEP 1: Output type ==========
st.sidebar.header("1. Output type")
plot_mode = st.sidebar.radio(
    "Select output type:",
    ["Single plot", "Multiple plots", "Function mixer", "Summary plots", "Phase diagram"],
    key="plot_mode",
    index=1,
)

metric_options = [
    "Radius of gyration",
    "Area fraction",
    "Bond orientational order",
    "Detect crystals",
    "Particle distance",
    "Radial distribution function",
    "Median total path distance",
    "Particle displacement (over time)",
    "Particle displacement (from initial)",
    "Number of clusters",
    "Average cluster size",
    # "Crystals: Hexagonal",
    # "Crystals: Square",
    # "Crystals: Triangular",
    # "Crystals: All",
    # "Crystals: None",
]

metric_map = {
    "Radius of gyration": (average_rg_over_time, "Radius of gyration"),
    "Area fraction": (area_fraction_over_time, "Area fraction"),
    "Bond orientational order": (bond_orientational_order_over_time,"Bond orientational order",),
    "Detect crystals": (detect_crystals_over_time, "Crystal metric"),
    "Particle distance": (particle_distance_over_time, "Particle distance"),
    "Radial distribution function":(rdf_over_time, "Radial distribution g(r)"),
    "Median total path distance": (median_total_path_distance_over_time,"Median path distance",),
    "Particle displacement (over time)": (particle_displacement_over_time,"Displacement"),
    "Particle displacement (from initial)": (particle_displacement_from_inintal_position_over_time,"Displacement from initial",),
    "Number of clusters": (num_clusters_over_time, "Number of clusters"),
    "Average cluster size": (avg_cluster_size_over_time, "Average cluster size"),
    # "Crystals: Hexagonal": (crystals_hex_over_time, "Hexagonal crystal fraction"),
    # "Crystals: Square": (crystals_square_over_time, "Square crystal fraction"),
    # "Crystals: Triangular": (crystals_tri_over_time, "Triangular crystal fraction"),
    # "Crystals: All": (crystals_all_over_time, "All crystalline fraction"),
    # "Crystals: None": (crystals_none_over_time, "Non-crystalline fraction"),

}

# ---------------------------------------------------------------------
# SINGLE PLOT MODE  (upload X & Y)
# ---------------------------------------------------------------------
if plot_mode == "Single plot":
    st.sidebar.header("2. Upload X & Y data")

    uploaded_files = st.sidebar.file_uploader(
        "Upload coordDynX and coordDynY (exactly 2 CSV files)",
        type=["csv"],
        accept_multiple_files=True,
        key="single_files",
    )

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("⬅️ Please upload **two** CSV files (X and Y) at the same time.")
        st.stop()

    if len(uploaded_files) > 2:
        st.warning("You uploaded more than 2 files. Using the first two only.")
        uploaded_files = uploaded_files[:2]

    uploaded_x = None
    uploaded_y = None
    for f in uploaded_files:
        name = f.name.lower()
        if ("x" in name or "coorddynx" in name) and uploaded_x is None:
            uploaded_x = f
        elif ("y" in name or "coorddyny" in name) and uploaded_y is None:
            uploaded_y = f

    if uploaded_x is None or uploaded_y is None:
        uploaded_x, uploaded_y = uploaded_files[0], uploaded_files[1]

    st.sidebar.write(f"**X file:** {uploaded_x.name}")
    st.sidebar.write(f"**Y file:** {uploaded_y.name}")

    coordDynX_path = save_uploaded_file(uploaded_x, "datax.csv")
    coordDynY_path = save_uploaded_file(uploaded_y, "datay.csv")

    st.sidebar.header("3. Single-plot options")

    analysis_options = [
        "Cluster movement over time",
        "Radius of gyration over time",
        "Crystals at a given timestep",
        "Bond orientational order at a timestep",
        "Fourier of particle displacement",
    ]

    analyses_selected = st.sidebar.multiselect(
        "Select analysis type(s):",
        analysis_options,
        default=[analysis_options[0]],
        key="single_analysis_multi",
    )

    if not analyses_selected:
        st.sidebar.warning("Select at least one analysis.")
        st.stop()

    # --- Styling options for single plot ---
    st.sidebar.header("4. Styling (single plot)")
    legend_on_single = st.sidebar.checkbox(
        "Show legend", value=True, key="legend_on_single"
    )
    legend_fontsize_single = st.sidebar.number_input(
        "Legend font size", value=16, min_value=1, max_value=40, key="legend_fs_single"
    )
    axis_fontsize_single = st.sidebar.number_input(
        "Axis label/tick font size",
        value=18,
        min_value=1,
        max_value=40,
        key="axis_fs_single",
    )
    title_on_single = st.sidebar.checkbox(
        "Show title", value=False, key="title_on_single"
    )
    title_fontsize_single = st.sidebar.number_input(
        "Title font size", value=20, min_value=1, max_value=60, key="title_fs_single"
    )

    st.sidebar.markdown("---")
    TauB = st.sidebar.number_input(
        "TauB (Brownian time)", value=20.0, step=1.0, key="TauB_single"
    )
    skip = st.sidebar.number_input(
        "Skip (frame stride)", value=1, step=1, min_value=0, key="skip_single"
    )
    normY = st.sidebar.checkbox(
        "Normalize Y axis", value=True, key="normY_single"
    )

    # Extra parameter UI only if exactly one analysis is chosen
    analysis_params = {}
    if len(analyses_selected) == 1:
        analysis = analyses_selected[0]

        if analysis == "Cluster movement over time":
            eps = st.sidebar.number_input(
                "DBSCAN eps (distance threshold)",
                value=10.0,
                step=0.5,
                key="eps_cluster",
            )
            min_samples = st.sidebar.number_input(
                "DBSCAN min_samples",
                value=2,
                step=1,
                min_value=1,
                key="min_samples_cluster",
            )
            analysis_params["eps"] = eps
            analysis_params["min_samples"] = min_samples

        elif analysis == "Crystals at a given timestep":
            timestep = st.sidebar.number_input(
                "Timestep index (0 = first, -1 = last)",
                value=-1,
                step=1,
                key="timestep_crystals",
            )
            eps = st.sidebar.number_input(
                "Distance cutoff eps",
                value=3.043,
                step=0.1,
                key="eps_crystals",
            )
            min_samples = st.sidebar.number_input(
                "DBSCAN min_samples (for filtering)",
                value=5,
                step=1,
                min_value=1,
                key="min_samples_crystals",
            )
            analysis_params["timestep"] = timestep
            analysis_params["eps"] = eps
            analysis_params["min_samples"] = min_samples

        elif analysis == "Bond orientational order at a timestep":
            timestep = st.sidebar.number_input(
                "Timestep index (0 = first, -1 = last)",
                value=-1,
                step=1,
                key="timestep_bond",
            )
            neighbor_cutoff = st.sidebar.number_input(
                "Neighbor cutoff distance",
                value=3.5,
                step=0.1,
                key="neighbor_cutoff_bond",
            )
            n = st.sidebar.number_input(
                "n for ψₙ (e.g. 6 for hexagonal)",
                value=6,
                step=1,
                min_value=1,
                key="n_bond",
            )
            analysis_params["timestep"] = timestep
            analysis_params["neighbor_cutoff"] = neighbor_cutoff
            analysis_params["n"] = n

        # Fourier uses only global TauB & skip → no extra UI
    else:
        st.sidebar.info(
            "Multiple analyses selected – using default parameters for each "
            "(e.g. eps, min_samples, timesteps)."
        )

    run = st.sidebar.button("▶ Run single plot(s)", key="run_single")

    if not run:
        st.stop()

    # ----- RUN ALL SELECTED ANALYSES -----
    for analysis in analyses_selected:
        plt.close("all")

        if analysis == "Cluster movement over time":
            st.subheader("Cluster movement over time")

            eps = analysis_params.get("eps", 10.0)
            min_samples = analysis_params.get("min_samples", 2)

            time_axis, cluster_movements = track_cluster_centroids(
                coordDynX_path,
                coordDynY_path,
                eps=eps,
                min_samples=min_samples,
                TauB=TauB,
                skip=skip,
                normY=int(normY),
            )

        elif analysis == "Radius of gyration over time":
            st.subheader("Radius of gyration over time")

            time_axis, rg_list = average_rg_over_time(
                coordDynX_path,
                coordDynY_path,
                TauB=TauB,
                skip=skip,
                normY=int(normY),
            )

        elif analysis == "Crystals at a given timestep":
            st.subheader("Crystal structure at timestep")

            timestep = analysis_params.get("timestep", -1)
            eps = analysis_params.get("eps", 3.043)
            min_samples = analysis_params.get("min_samples", 5)

            plot_crystals_at_timestep(
                coordDynX_path,
                coordDynY_path,
                timestep=None if timestep == -1 else int(timestep),
                eps=eps,
                min_samples=min_samples,
                skip=skip,
                normY=int(normY),
            )

        elif analysis == "Bond orientational order at a timestep":
            st.subheader("Bond orientational order ψₙ at a timestep")

            timestep = analysis_params.get("timestep", -1)
            neighbor_cutoff = analysis_params.get("neighbor_cutoff", 3.5)
            n = analysis_params.get("n", 6)

            plot_bond_orientational_order_at_timestep(
                coordDynX_path,
                coordDynY_path,
                timestep=None if timestep == -1 else int(timestep),
                n=int(n),
                neighbor_cutoff=neighbor_cutoff,
                skip=skip,
                normY=int(normY),
            )


        elif analysis == "Fourier of particle displacement":
            st.subheader("Fourier transform of median displacement over time")

            freqs, amps = fourier_of_metric(
                coordDynX=coordDynX_path,
                coordDynY=coordDynY_path,
                metric_func=particle_displacement_from_inintal_position_over_time,
                metric_kwargs={"TauB": TauB, "skip": skip},
                window=None,
                normalize=True,
                title="FFT of median displacement",
            )

        fig = plt.gcf()
        apply_global_styling(
            fig,
            legend_on=legend_on_single,
            legend_fontsize=legend_fontsize_single,
            axis_fontsize=axis_fontsize_single,
            title_on=title_on_single,
            title_fontsize=title_fontsize_single,
        )
        st.pyplot(fig)
        save_figure_if_requested(
            fig,
            base_name=f"single_{analysis}",
            save_enabled=save_plots,
            output_dir=output_dir,
        )

    st.success("Single plot(s) done ✅")


# ---------------------------------------------------------------------
# MULTIPLE PLOTS MODE  (folder-based, uses plots_together)
# ---------------------------------------------------------------------
elif plot_mode == "Multiple plots":
    ui = sidebar_runs_common_ui(
        prefix="multi",
        source_root=source_root,
        metric_options=metric_options,
        include_metric_select=True,
        default_metrics=[metric_options[0]],
        include_crystal_multiselect=True,
    )

    metrics_selected = ui["metrics_selected"]
    crystal_types_multi = ui["crystal_types"]
    base_dir = ui["base_dir"]
    dataset_tag = ui["dataset_tag"]
    selected_runs = ui["selected_runs"]
    x_name = ui["x_name"]
    y_name = ui["y_name"]
    skip = ui["skip"]
    normY = ui["normY"]
    cluster_eps_multi = ui["cluster_eps"]
    cluster_min_samples_multi = ui["cluster_min_samples"]
    cluster_min_cluster_size_multi = ui["cluster_min_cluster_size"]
    legend_on_multi = ui["legend_on"]
    show_taub_in_legend = ui["show_taub_in_legend"]
    legend_fontsize_multi = ui["legend_fontsize"]
    axis_fontsize_multi = ui["axis_fontsize"]
    title_on_multi = ui["title_on"]
    title_fontsize_multi = ui["title_fontsize"]
    x_axis_mode_multi = ui["x_axis_mode"]
    y_scale_multi = ui.get("y_scale", "Linear")
    y_unit_multi = ui.get("y_unit", "")
    export_data = ui["export_data"]
    export_dir = ui["export_dir"]
    TauB = ui["TauB"]
    rdf_r_max = ui.get("rdf_r_max", 10.0)
    rdf_dr = ui.get("rdf_dr", 0.1)
    y_range_custom = ui.get("y_range_custom", False)
    y_range_min = ui.get("y_range_min", None)
    y_range_max = ui.get("y_range_max", None)
    area_fraction_mode = ui["area_fraction_mode"]




    # ---- Color mode UI ----
    st.sidebar.header("Curve colors")
    color_mode_multi = st.sidebar.radio(
        "Curve colors",
        ["Matplotlib default", "Gradient over TkB"],
        key="multi_color_mode",  # unique key
    )

    if color_mode_multi == "Gradient over TkB":
        st.sidebar.write("Gradient colors:")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            color_start_multi = st.sidebar.color_picker(
                "Start",
                value="#000000",
                key="multi_color_start",
            )
        with col2:
            color_end_multi = st.sidebar.color_picker(
                "End",
                value="#CCCCCC",
                key="multi_color_end",
            )
    else:
        color_start_multi = "#000000"
        color_end_multi = "#CCCCCC"


    run = st.sidebar.button("▶ Run multiple-plot routine", key="run_multi")
    have_cached = st.session_state.get("multi_cached", False)


    if (not run) and (not have_cached):
        st.info("⬅️ Click **Run multiple-plot routine** to compute plots.")
        st.stop()

    if not base_dir:
        st.warning("Please select or enter a base folder path.")
        st.stop()

    if not selected_runs:
        st.warning("No TkB/TauB runs selected.")
        st.stop()

    st.subheader("Multiple plots over TkB")
    st.write(f"Base folder: `{base_dir}`")
    st.write(f"Metrics: **{metrics_selected}**")
    st.write("Selected runs (TkB, TauB):")


    readable_runs = [
        f"{tkb:g} Tkb  {tau:g} TauB"  # or f"{tkb:g}Tkb_{tau:g}TauB"
        for _, tkb, tau in selected_runs
    ]

    st.write("Selected runs:")
    for txt in readable_runs:
        st.write(f"- {txt}")

    metric_kwargs = dict(TauB=TauB, skip=skip, normY=int(normY))

    # ----- RUN ALL SELECTED METRICS -----
    for metric_label in metrics_selected:
        plt.close("all")
        metric_func, default_ylabel = metric_map[metric_label]

        # --- Detect crystals: plot one curve per selected crystal type ---
        if metric_label == "Detect crystals":
            plt.figure()
            ax = plt.gca()

            post_curves = []  # define this once before the TauB grouping loop

            for kind in crystal_types_multi:
                # compute one series per kind (same cache system, because kwargs differ)
                series_kind = []
                missing = []

                for folder_path, tkb_val, tau_val in selected_runs:
                    run_name = os.path.basename(folder_path)

                    fx = os.path.join(folder_path, x_name)
                    fy = os.path.join(folder_path, y_name)
                    if not (os.path.isfile(fx) and os.path.isfile(fy)):
                        missing.append(run_name)
                        continue

                    # IMPORTANT: pass the chosen kind into kwargs
                    kwargs_run = dict(skip=int(skip), normY=int(normY), TauB=float(tau_val))
                    kwargs_run["type"] = kind  # <-- use this instead if your function param is named type

                    p = metric_cache_path(base_dir, run_name, detect_crystals_over_time.__name__, kwargs_run)
                    if export_data and p.exists():
                        p.unlink(missing_ok=True)  # force recompute when export_data is ticked

                    t, y = load_or_compute_metric_cached(
                        base_dir=base_dir,
                        run_folder=run_name,
                        fx=fx,
                        fy=fy,
                        metric_func=detect_crystals_over_time,
                        metric_kwargs=kwargs_run,
                        force_recompute=export_data,
                    )

                    series_kind.append({
                        "run": run_name,
                        "tkb": float(tkb_val),
                        "taub": float(tau_val),
                        "y": np.asarray(y, float),
                    })
                # plot this kind across runs (your existing grouping-by-taub logic)
                for taub in sorted({s["taub"] for s in series_kind}):
                    group = [s for s in series_kind if s["taub"] == taub]
                    n_points = max(len(s["y"]) for s in group)
                    t_end_sec = taub * 13.513
                    t_grid_sec = np.linspace(0.0, t_end_sec, n_points)

                    for s in group:
                        y_raw = s["y"]
                        x_raw = np.linspace(0.0, t_end_sec, len(y_raw))
                        y_grid = np.interp(t_grid_sec, x_raw, y_raw, left=y_raw[0], right=y_raw[-1])

                        x = (t_grid_sec / 13.513) if x_axis_mode_multi == "Brownian time τ_B" else t_grid_sec

                        ax.plot(
                            x,
                            y_grid,
                            label = f"{s['tkb']:g} Tkb (TauB {taub:g}) — {kind}" if show_taub_in_legend else f"{s['tkb']:g} Tkb — {kind}"
                        )
                        # collect for post-analysis

                        # inside the loop where you have x and y_grid:
                        post_curves.append({
                            "run": s["run"],
                            "tkb": s["tkb"],
                            "taub": taub,
                            "kind": kind,  # for crystal type
                            "x": np.asarray(x, float),
                            "y": np.asarray(y_grid, float),
                        })

            # after plotting + after post_curves is populated
            st.session_state[f"post_curves_{metric_label}"] = post_curves
            st.session_state["multi_cached"] = True

            ax.set_title("Crystal metric")
            ax.set_ylabel(f"{', '.join(crystal_types_multi)} Crystal Fraction")
            if y_range_custom and y_range_min is not None and y_range_max is not None:
                ax.set_ylim(float(y_range_min), float(y_range_max))
            ax.set_xlabel("r (µm)" if metric_label == "Radial distribution function" else (
                "Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"
            ))
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig = plt.gcf()
            apply_global_styling(fig, legend_on=legend_on_multi, legend_fontsize=legend_fontsize_multi,
                                 axis_fontsize=axis_fontsize_multi, title_on=title_on_multi,
                                 title_fontsize=title_fontsize_multi)
            st.pyplot(fig)

            continue  # <-- IMPORTANT: skip the normal metric plotting below

        st.markdown(f"### Metric: {metric_label}")

        # ------------------------------------------------------------
        # NEW: multi-run plotting with
        #  - correct per-run TauB on x-axis
        #  - same-TauB runs resampled to same x-length
        #  - on-disk caching for faster reruns
        # ------------------------------------------------------------

        series = []  # will contain dicts with run/tkb/taub/y_len only; we regenerate x later
        missing = []

        if run:
            dataset_tag = os.path.basename(base_dir.rstrip("\\/"))  # e.g. NAF, ASF, ARF

            for folder_path, tkb_val, tau_val in selected_runs:
                run_name = os.path.basename(folder_path)
                tkb_val = float(tkb_val)
                tau_val = float(tau_val)

                npz_path = rundata_npz_path(base_dir, dataset_tag, tkb_val, tau_val, metric_label)
                #st.write(
                #    f"🔎 {metric_label} | {run_name} -> saved file exists? {npz_path.exists()} | export_data={export_data}")

                if (not export_data) and npz_path.exists():
                    # Load previously saved y (fast)
                    y = load_rundata_npz(npz_path)
                else:
                    # Recompute (only happens when export_data=True or file missing)
                    fx = os.path.join(folder_path, x_name)
                    fy = os.path.join(folder_path, y_name)
                    if not (os.path.isfile(fx) and os.path.isfile(fy)):
                        missing.append(run_name)
                        continue

                    kwargs_run = dict(skip=int(skip), normY=int(normY), TauB=float(tau_val))
                    if metric_label in ["Number of clusters", "Average cluster size","Area fraction","Bond orientational order", "Detect crystals", "Radius of gyration"]:
                        kwargs_run["eps"] = float(cluster_eps_multi)
                        kwargs_run["min_samples"] = int(cluster_min_samples_multi)
                        kwargs_run["min_cluster_size"] = int(cluster_min_cluster_size_multi)
                        kwargs_run["cluster_mode"] = area_fraction_mode
                    if metric_label == "Radial distribution function":
                        kwargs_run["r_max"] = float(rdf_r_max)
                        kwargs_run["dr"] = float(rdf_dr)
                    t, y = load_or_compute_metric_cached(
                        base_dir=base_dir,
                        run_folder=run_name,
                        fx=fx,
                        fy=fy,
                        metric_func=metric_func,
                        metric_kwargs=kwargs_run,
                        force_recompute=export_data,
                    )

                    # Save just y (no time)
                    try:
                        save_rundata_npz(npz_path, np.asarray(y, float))
                    except Exception as e:
                        st.warning(f"Could not save run data {npz_path.name}: {e}")

                    # Optional CSV export (no time)
                    if export_dir.strip():
                        try:
                            out = Path(export_dir.strip())
                            out.mkdir(parents=True, exist_ok=True)
                            csv_path = out / (npz_path.stem + ".csv")
                            export_rundata_csv(csv_path, np.asarray(y, float))
                            save_rundata_npz(npz_path, y)

                        except Exception as e:
                            st.warning(f"Could not export CSV for {run_name}: {e}")

                series.append({
                    "run": run_name,
                    "tkb": tkb_val,
                    "taub": tau_val,
                    "y": np.asarray(y, float),
                })



        if missing:
            st.warning("Missing data files for: " + ", ".join(missing))

        if not run:
            post_curves = st.session_state.get(f"post_curves_{metric_label}", [])
            if not post_curves:
                st.info(f"No cached curves for {metric_label}. Click Run once.")
                continue

            plt.figure()
            ax = plt.gca()
            for c in post_curves:
                ax.plot(
                    c["x"], c["y"],
                    label=f"{c['tkb']:g} Tkb (TauB {c['taub']:g})" if show_taub_in_legend else f"{c['tkb']:g} Tkb"
                )


            ax.relim()
            ax.autoscale_view()
            ax.set_title(default_ylabel)
            ylabel_plot = default_ylabel
            if y_unit_multi.strip():
                ylabel_plot = f"{y_unit_multi.strip()}"
            ax.set_ylabel(ylabel_plot)
            # log/linear scale
            if y_scale_multi == "Log":
                # guard: log needs y>0
                if np.any(np.asarray([v for c in post_curves for v in c["y"]], float) <= 0):
                    st.warning("Log scale requires y > 0. Using Linear.")
                    ax.set_yscale("linear")
                else:
                    ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            if y_range_custom and y_range_min is not None and y_range_max is not None:
                ax.set_ylim(float(y_range_min), float(y_range_max))

            ax.set_xlabel("r (µm)" if metric_label == "Radial distribution function" else (
                "Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"
            ))
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig = plt.gcf()
            apply_global_styling(
                fig,
                legend_on=legend_on_multi,
                legend_fontsize=legend_fontsize_multi,
                axis_fontsize=axis_fontsize_multi,
                title_on=title_on_multi,
                title_fontsize=title_fontsize_multi,
            )
            st.pyplot(fig)



            post_analysis_block(
                key=f"multi_{metric_label}",
                curves=st.session_state.get(f"post_curves_{metric_label}", []),
                ylabel=default_ylabel,
                x_axis_mode=x_axis_mode_multi,
                legend_on=legend_on_multi,
                legend_fontsize=legend_fontsize_multi,
                axis_fontsize=axis_fontsize_multi,
                title_on=title_on_multi,
                title_fontsize=title_fontsize_multi,
)
            continue  # <-- skip compute/export/plot path when using cached curves

        if export_data:
            if not export_dir.strip():
                st.warning("Export enabled but no export folder provided.")
            else:
                paths = export_series_to_csv(export_dir.strip(), dataset_tag, metric_label, series)
                st.info("Exported:\n" + "\n".join([f"- {p}" for p in paths]))

        # Keep series for calculator later (optional)
        st.session_state.setdefault("computed_metrics", {})
        st.session_state["computed_metrics"][metric_label] = series



        # ---- Plot ----
        plt.figure()
        ax = plt.gca()

        # Color handling (optional): gradient over TkB
        use_gradient = (color_mode_multi == "Gradient over TkB")
        if use_gradient:
            import matplotlib.colors as mcolors

            tkbs = np.array([s["tkb"] for s in series], dtype=float)
            tmin, tmax = float(np.min(tkbs)), float(np.max(tkbs))
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "tkb_grad", [color_start_multi, color_end_multi]
            )


        def color_for_tkb(tkb):
            if not use_gradient:
                return None
            if tmax <= tmin:
                return cmap(0.5)
            return cmap((tkb - tmin) / (tmax - tmin))


        post_curves = []  # define this once before the TauB grouping loop
        # Group by TauB so same-TauB runs share identical x-axis length
        if metric_label == "Radial distribution function":
            for s in series:
                x = np.linspace(0.0, rdf_r_max, len(s["y"]))
                ax.plot(
                    x, s["y"],
                    label=f"{s['tkb']:g} Tkb (TauB {s['taub']:g})" if show_taub_in_legend else f"{s['tkb']:g} Tkb",
                    color=color_for_tkb(s["tkb"]),
                )
                post_curves.append({
                    "run": s["run"],
                    "tkb": s["tkb"],
                    "taub": s["taub"],
                    "x": np.asarray(x, float),
                    "y": np.asarray(s["y"], float),
                })
            st.session_state[f"post_curves_{metric_label}"] = post_curves
            st.session_state["multi_cached"] = True
        else:
            for taub in sorted({s["taub"] for s in series}):
                group = [s for s in series if s["taub"] == taub]

                # same-TauB runs should have same x-length:
                n_points = max(len(s["y"]) for s in group)

                # Create x grid (seconds)
                t_end_sec = taub * 13.513
                t_grid_sec = np.linspace(0.0, t_end_sec, n_points)

                for s in group:
                    # resample y onto common length grid using index-based interpolation
                    y_raw = s["y"]
                    x_raw = np.linspace(0.0, t_end_sec, len(y_raw))
                    y_grid = np.interp(t_grid_sec, x_raw, y_raw, left=y_raw[0], right=y_raw[-1])

                    x = (t_grid_sec / 13.513) if x_axis_mode_multi == "Brownian time τ_B" else t_grid_sec
                    ax.plot(
                        x,
                        y_grid,
                        label = f"{s['tkb']:g} Tkb (TauB {taub:g})" if show_taub_in_legend else f"{s['tkb']:g} Tkb",
                        color=color_for_tkb(s["tkb"]),
                    )
                    # collect for post-analysis


                    # inside the loop where you have x and y_grid:
                    post_curves.append({
                        "run": s["run"],
                        "tkb": s["tkb"],
                        "taub": taub,
                        "x": np.asarray(x, float),
                        "y": np.asarray(y_grid, float),
                    })

        # after plotting + after post_curves is populated
        st.session_state[f"post_curves_{metric_label}"] = post_curves
        st.session_state["multi_cached"] = True
        ax.set_title(default_ylabel)
        ylabel_plot = default_ylabel
        if y_unit_multi.strip():
            ylabel_plot = f"{y_unit_multi.strip()}"
        # log/linear scale
        if y_scale_multi == "Log":
            # guard: log needs y>0
            if np.any(np.asarray([v for c in post_curves for v in c["y"]], float) <= 0):
                st.warning("Log scale requires y > 0. Using Linear.")
                ax.set_yscale("linear")
            else:
                ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
        ax.set_ylabel(ylabel_plot)
        if y_range_custom and y_range_min is not None and y_range_max is not None:
            ax.set_ylim(float(y_range_min), float(y_range_max))
        ax.set_xlabel("r (µm)" if metric_label == "Radial distribution function" else (
            "Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"
        ))
        ax.grid(True, alpha=0.3)
        ax.legend()


        fig = plt.gcf()
        apply_global_styling(
            fig,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )
        st.pyplot(fig)
        save_figure_if_requested(
            fig,
            base_name=f"multi_{metric_label}",
            save_enabled=save_plots,
            output_dir=output_dir,
        )
        post_analysis_block(
            key=f"multi_{metric_label}",
            curves=st.session_state.get(f"post_curves_{metric_label}", []),
            ylabel=default_ylabel,
            x_axis_mode=x_axis_mode_multi,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )

    st.success("Multiple-plot figure(s) done ✅")








##########################################################
# Function Mixer
##########################################################

elif plot_mode == "Function mixer":
    ui = sidebar_runs_common_ui(
        prefix="mix",
        source_root=source_root,
        metric_options=metric_options,
        include_metric_select=False,
        include_crystal_multiselect=False,
    )

    base_dir = ui["base_dir"]
    dataset_tag = ui["dataset_tag"]
    selected_runs = ui["selected_runs"]
    x_name = ui["x_name"]
    y_name = ui["y_name"]
    skip = ui["skip"]
    normY = ui["normY"]
    cluster_eps_multi = ui["cluster_eps"]
    cluster_min_samples_multi = ui["cluster_min_samples"]
    cluster_min_cluster_size_multi = ui["cluster_min_cluster_size"]
    legend_on_multi = ui["legend_on"]
    show_taub_in_legend = ui["show_taub_in_legend"]
    legend_fontsize_multi = ui["legend_fontsize"]
    axis_fontsize_multi = ui["axis_fontsize"]
    title_on_multi = ui["title_on"]
    title_fontsize_multi = ui["title_fontsize"]
    x_axis_mode_multi = ui["x_axis_mode"]
    y_scale_multi = ui.get("y_scale", "Linear")
    y_unit_multi = ui.get("y_unit", "")
    export_data = ui["export_data"]
    export_dir = ui["export_dir"]
    rdf_r_max = ui.get("rdf_r_max", 10.0)
    rdf_dr = ui.get("rdf_dr", 0.1)
    y_range_custom = ui.get("y_range_custom", False)
    y_range_min = ui.get("y_range_min", None)
    y_range_max = ui.get("y_range_max", None)
    area_fraction_mode = ui["area_fraction_mode"]

    st.markdown("---")
    st.subheader("🧮 Function Mixer")

    calc_metric_options = metric_options  # your existing list

    st.markdown("### Time window")
    c1, c2, c3 = st.columns([1, 2, 2])

    with c1:
        st.caption("Slice source data")

    with c2:
        mix_t_start_frac = st.slider(
            "Window start (fraction of run)",
            0.0, 0.9, 0.0, 0.01,
            key="mix_window_start",
        )

    with c3:
        mix_t_end_frac = st.slider(
            "Window end (fraction of run)",
            0.1, 1.0, 1.0, 0.01,
            key="mix_window_end",
        )

    mA_label = st.selectbox("Metric A", calc_metric_options, key="expr_calc_A")

    crystal_type_A = None
    if mA_label == "Detect crystals":
        with st.expander("Crystal options for A", expanded=True):
            crystal_type_A = st.selectbox(
                "Crystal type for A:",
                ["Hexagonal", "Square", "Triangular", "All", "None"],
                index=0,
                key="crystal_type_A",
            )

    mB_label = st.selectbox("Metric B", calc_metric_options, key="expr_calc_B")

    crystal_type_B = None
    if mB_label == "Detect crystals":
        with st.expander("Crystal options for B", expanded=True):
            crystal_type_B = st.selectbox(
                "Crystal type for B:",
                ["Hexagonal", "Square", "Triangular", "All", "None"],
                index=0,
                key="crystal_type_B",
            )

    expr = st.text_input(
        "Expression (use A and B):",
        value="A / B",
        key="expr_calc_expr",
        help="Examples: A/B, (A/B)*100, log(A)/sqrt(B), A/(B+1e-9), deriv(A,t), integ(A,t), fft_amp(A)"
    )

    calc_title = st.text_input(
        "Derived plot title",
        value=f"{expr}  (A={mA_label}, B={mB_label})",
        key="expr_calc_title",
    )

    plot_expr = st.button("Plot", key="expr_calc_plot")

    st.caption("Allowed functions: " + ", ".join(sorted(_ALLOWED_FUNCS.keys())))

    # ✅ If user clicked Run post-analysis (or any rerun) and we already have cached curves, redraw from cache
    if (not plot_expr) and st.session_state.get("post_curves_mixer"):
        curves = st.session_state["post_curves_mixer"]
        meta = st.session_state.get("mixer_meta", {})

        plt.close("all")
        plt.figure()
        ax = plt.gca()
        for c in curves:
            ax.plot(c["x"], c["y"], label=c["run"])

        ax.set_title(meta.get("title", "Derived expression"))
        title = meta.get("title", "Derived expression")
        ylabel_plot = title
        if y_unit_multi.strip():
            ylabel_plot = f"{title} [{y_unit_multi.strip()}]"
        ax.set_ylabel(ylabel_plot)

        if meta.get("plot_fft", False):
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xlabel("Time (τB)" if meta.get("x_axis_mode") == "Brownian time τ_B" else "Time (s)")

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig = plt.gcf()
        apply_global_styling(
            fig,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )
        st.pyplot(fig)

        post_analysis_block(
            key="mixer",
            curves=curves,
            ylabel=meta.get("title", "Derived expression"),
            x_axis_mode=meta.get("x_axis_mode", x_axis_mode_multi),
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )

        st.stop()

    if plot_expr:
        st.session_state["post_curves_mixer"] = []

        # ✅ define this AFTER expr exists and BEFORE plotting
        plot_fft = ("fft_amp" in expr) or ("fft_power" in expr)

        if not selected_runs:
            st.warning("Select at least one run (TkB/TauB) first.")
            st.stop()
        if not base_dir:
            st.warning("Select a base folder first.")
            st.stop()

        dataset_tag = os.path.basename(base_dir.rstrip("\\/"))

        funcA, _ = metric_map[mA_label]
        funcB, _ = metric_map[mB_label]

        extraA = {}
        if mA_label in ["Number of clusters", "Average cluster size","Bond orientational order", "Detect crystals", "Radius of gyration"]:
            extraA.update({
                "eps": float(cluster_eps_multi),
                "min_samples": int(cluster_min_samples_multi),
                "min_cluster_size": int(cluster_min_cluster_size_multi),
            })
        if mA_label == "Detect crystals" and crystal_type_A is not None:
            extraA["type"] = crystal_type_A

        if mA_label == "Area fraction":
            extraA["cluster_mode"] = area_fraction_mode



        extraB = {}
        if mB_label in ["Number of clusters", "Average cluster size","Bond orientational order", "Detect crystals", "Radius of gyration"]:
            extraB.update({
                "eps": float(cluster_eps_multi),
                "min_samples": int(cluster_min_samples_multi),
                "min_cluster_size": int(cluster_min_cluster_size_multi),
            })
        if mB_label == "Detect crystals" and crystal_type_B is not None:
            extraB["type"] = crystal_type_B

        if mB_label == "Area fraction":
            extraB["cluster_mode"] = area_fraction_mode

        metric_label_A = mA_label
        if mA_label == "Detect crystals" and crystal_type_A:
            metric_label_A = f"Detect crystals ({crystal_type_A})"

        metric_label_B = mB_label
        if mB_label == "Detect crystals" and crystal_type_B:
            metric_label_B = f"Detect crystals ({crystal_type_B})"

        with st.spinner("Loading / computing Metric A..."):
            A_series, missingA = get_series_for_metric(
                metric_label=metric_label_A,  # ✅ use the label with "(Square)" etc.
                metric_func=funcA,
                selected_runs=selected_runs,
                base_dir=base_dir,
                dataset_tag=dataset_tag,
                x_name=x_name,
                y_name=y_name,
                skip=int(skip),
                normY=int(normY),
                export_data=export_data,
                export_dir=export_dir,
                extra_kwargs=extraA,  # ✅ valid kwarg
            )

        with st.spinner("Loading / computing Metric B..."):
            B_series, missingB = get_series_for_metric(
                metric_label=metric_label_B,  # ✅
                metric_func=funcB,
                selected_runs=selected_runs,
                base_dir=base_dir,
                dataset_tag=dataset_tag,
                x_name=x_name,
                y_name=y_name,
                skip=int(skip),
                normY=int(normY),
                export_data=export_data,
                export_dir=export_dir,
                extra_kwargs=extraB,  # ✅
            )

        if missingA:
            st.warning("Missing files for Metric A runs: " + ", ".join(missingA))
        if missingB:
            st.warning("Missing files for Metric B runs: " + ", ".join(missingB))

        A_map = {s["run"]: s for s in A_series}
        B_map = {s["run"]: s for s in B_series}
        common_runs = sorted(set(A_map.keys()) & set(B_map.keys()))
        if not common_runs:
            st.warning("No overlapping runs between Metric A and Metric B.")
            st.stop()

        plt.close("all")
        plt.figure()
        ax = plt.gca()


        st.session_state.setdefault("post_curves_mixer", [])

        # reset curves for THIS derived expression run
        post_curves = []

        for run_name in common_runs:
            a_run = A_map[run_name]  # dict
            b_run = B_map[run_name]  # dict

            taub_a = float(a_run["taub"])
            taub_b = float(b_run["taub"])
            if abs(taub_a - taub_b) > 1e-9:
                st.warning(f"Skipping {run_name}: TauB mismatch between metrics ({taub_a} vs {taub_b})")
                continue

            t_grid_sec, yA, yB = align_y_by_taub_length(a_run["y"], b_run["y"], taub_a)

            # choose time axis for windowing
            if x_axis_mode_multi == "Brownian time τ_B":
                t_use = t_grid_sec / 13.513
            else:
                t_use = t_grid_sec

            t0, t1 = float(np.min(t_use)), float(np.max(t_use))
            win_a = t0 + (t1 - t0) * float(mix_t_start_frac)
            win_b = t0 + (t1 - t0) * float(mix_t_end_frac)
            sel = (t_use >= win_a) & (t_use <= win_b)

            t_grid_sec = t_grid_sec[sel]
            yA = yA[sel]
            yB = yB[sel]

            if t_grid_sec.size < 2:
                st.warning(f"Skipping {run_name}: time window too small.")
                continue

            env = {
                "A": yA,
                "B": yB,
                "t": t_grid_sec,
                "pi": float(np.pi),
                "e": float(np.e),
            }

            try:
                y_out = safe_eval_expr(expr, env)
                y_out = np.asarray(y_out, float)
            except Exception as e:
                st.error(f"Expression error: {e}")
                st.stop()

            plot_fft = ("fft_amp" in expr) or ("fft_power" in expr)
            if plot_fft:
                x = fft_freqs(t_grid_sec)  # Hz
            else:
                x = (t_grid_sec / 13.513) if x_axis_mode_multi == "Brownian time τ_B" else t_grid_sec

            ax.plot(x, y_out, label = f"{a_run['tkb']:g} Tkb (TauB {taub_a:g})" if show_taub_in_legend else f"{a_run['tkb']:g} Tkb")

            post_curves.append({
                "run": run_name,
                "tkb": float(a_run["tkb"]),
                "taub": float(taub_a),
                "x": np.asarray(x, float),
                "y": np.asarray(y_out, float),
            })


            # ✅ IMPORTANT: append INSIDE the loop (one entry per run)
            st.session_state["post_curves_mixer"].append({
                "run": run_name,
                "tkb": float(a_run["tkb"]),
                "taub": float(a_run["taub"]),
                "x": np.asarray(x, float),
                "y": np.asarray(y_out, float),
            })

        ax.set_title(calc_title)
        # y-label for mixer (expression)
        ylabel_plot = calc_title
        if y_unit_multi.strip():
            ylabel_plot = f"{y_unit_multi.strip()}"
        ax.set_ylabel(ylabel_plot)

        # ✅ overwrite cache with ALL curves
        st.session_state["post_curves_mixer"] = post_curves
        st.session_state["mixer_meta"] = {
            "title": calc_title,
            "plot_fft": plot_fft,
            "x_axis_mode": x_axis_mode_multi,
        }

        # ✅ label axis correctly
        if plot_fft:
            ax.set_xlabel("Frequency (Hz)")
        else:
            ax.set_xlabel("Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)")

        ax.grid(True, alpha=0.3)
        ax.legend()

        fig = plt.gcf()
        apply_global_styling(
            fig,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )
        st.pyplot(fig)


        post_analysis_block(
            key="mixer",
            curves=st.session_state.get("post_curves_mixer", []),
            ylabel=calc_title,
            x_axis_mode=x_axis_mode_multi,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
        )


# ---------------------------------------------------------------------
# PHASE DIAGRAM MODE  (folder-based, uses make_snapshot_phase_panel)
# ---------------------------------------------------------------------
elif plot_mode == "Phase diagram":
    st.sidebar.header("2. Data folder (for phase diagram)")

    folder_source = st.sidebar.radio(
        "Folder source:",
        ["HIWI root subfolder", "Custom path"],
        key="phase_folder_source",
    )

    hiwi_subdirs = []
    if os.path.isdir(source_root):
        try:
            hiwi_subdirs = [
                d for d in os.listdir(source_root)
                if os.path.isdir(os.path.join(source_root, d))
            ]
        except Exception as e:
            st.sidebar.error(f"Could not list subfolders under source root: {e}")

    if folder_source == "HIWI root subfolder":
        subchoice_phase = st.sidebar.selectbox(
            "Choose subfolder under source root:",
            hiwi_subdirs,
            key="phase_hiwi_sub",
        )
        base_dir = os.path.join(source_root, subchoice_phase) if subchoice_phase else ""
    else:
        base_dir = st.sidebar.text_input(
            "Custom base folder path:",
            value="",
            key="phase_folder_custom",
        )

    base_dir = base_dir.strip().strip('"').strip("'")

    st.sidebar.header("3. TkB rows")

    available_tkbs_phase = []
    if base_dir and os.path.isdir(base_dir):
        try:
            folder_re = re.compile(r"^([0-9]+(?:\.[0-9]+)?)Tkb_")
            for fn in os.listdir(base_dir):
                m = folder_re.match(fn)
                if m:
                    available_tkbs_phase.append(float(m.group(1)))
            available_tkbs_phase = sorted(set(available_tkbs_phase))
        except Exception as e:
            st.sidebar.error(f"Could not scan TkB folders in {base_dir}: {e}")

    if available_tkbs_phase:
        tkb_rows_selected = st.sidebar.multiselect(
            "Select TkB values for rows:",
            available_tkbs_phase,
            default=available_tkbs_phase,
            key="phase_tkbs",
        )
    else:
        tkb_rows_selected = []
        st.sidebar.warning(
            "No Tkb_* subfolders detected. You can still add custom TkB values below."
        )

    custom_tkb_rows_str = st.sidebar.text_input(
        "Add custom TkB rows (optional, comma-separated):",
        value="",
        key="phase_tkb_custom",
    )

    tkb_rows = list(tkb_rows_selected)
    if custom_tkb_rows_str.strip():
        try:
            extra_rows = [
                float(s.strip())
                for s in custom_tkb_rows_str.split(",")
                if s.strip()
            ]
            tkb_rows = sorted(set(tkb_rows + extra_rows))
        except ValueError:
            st.sidebar.error("Could not parse custom TkB row values.")

    # ---------- TauB (so we know total time for sliders) ----------
    st.sidebar.header("4. TauB (for mapping time→frame)")
    taub = st.sidebar.number_input(
        "TauB used in your simulations (e.g. 2):",
        value=2.0,
        step=0.5,
        key="phase_taub",
    )
    total_time_sec = float(taub) * 13.514

    # ---------- TIME SELECTION UI ----------
    st.sidebar.header("5. Time columns (seconds)")

    time_mode = st.sidebar.radio(
        "How to choose time columns?",
        ["Manual list", "Regular step (Δt)", "Fixed number of slices"],
        key="phase_time_mode",
    )

    column_times_s = []

    if time_mode == "Manual list":
        time_cols_str = st.sidebar.text_input(
            "Times for columns in seconds (comma-separated):",
            value="1,5,10,20,26",
            key="phase_times",
        )
        try:
            column_times_s = [
                float(s.strip()) for s in time_cols_str.split(",") if s.strip()
            ]
        except ValueError:
            column_times_s = []
            st.sidebar.error(
                "Could not parse time columns. Use e.g. 1, 5, 10, 20, 26."
            )

    elif time_mode == "Regular step (Δt)":
        t_start_step, t_end_step = st.sidebar.slider(
            "Time range (s)",
            min_value=0.0,
            max_value=float(total_time_sec) if total_time_sec > 0 else 10.0,
            value=(
                0.0,
                float(total_time_sec) if total_time_sec > 0 else 10.0,
            ),
            step=0.1,
            key="phase_range_step",
        )
        dt = st.sidebar.number_input(
            "Time step Δt (s)",
            value=1.0,
            step=0.5,
            min_value=0.0001,
            key="phase_dt_step",
        )

        if t_end_step <= t_start_step:
            st.sidebar.error("End time must be greater than start time.")
            column_times_s = []
        else:
            n_steps = int((t_end_step - t_start_step) / dt) + 1
            column_times_s = [
                t_start_step + i * dt for i in range(n_steps)
            ]

    elif time_mode == "Fixed number of slices":
        t_start_lin, t_end_lin = st.sidebar.slider(
            "Time range (s)",
            min_value=0.0,
            max_value=float(total_time_sec) if total_time_sec > 0 else 10.0,
            value=(
                0.0,
                float(total_time_sec) if total_time_sec > 0 else 10.0,
            ),
            step=0.1,
            key="phase_range_lin",
        )
        n_slices = st.sidebar.number_input(
            "Number of columns",
            value=5,
            step=1,
            min_value=1,
            key="phase_n_slices",
        )

        if t_end_lin <= t_start_lin:
            st.sidebar.error("End time must be greater than start time.")
            column_times_s = []
        else:
            column_times_s = list(
                np.linspace(float(t_start_lin), float(t_end_lin), int(n_slices))
            )

    st.sidebar.header("6. File names")
    x_name = st.sidebar.text_input(
        "X filename inside each Tkb_* folder:", value="datax.csv", key="phase_xname"
    )
    y_name = st.sidebar.text_input(
        "Y filename inside each Tkb_* folder:", value="datay.csv", key="phase_yname"
    )


    st.sidebar.header("7. Snapshot plot function")

    snapshot_label = st.sidebar.selectbox(
        "What to plot in each cell?",
        [
            "Bond orientational order snapshot (ψₙ)",
            "Crystal snapshot",
            "Local density snapshot",
            "Displacement vectors snapshot",
            "Alignment field snapshot",
            "Displacement vectors snapshot (Voronoi)",
        ],
        key="phase_snapshot",
    )

    if snapshot_label == "Bond orientational order snapshot (ψₙ)":
        snapshot_func = plot_bond_orientational_order_at_timestep
    elif snapshot_label == "Crystal snapshot":
        snapshot_func = plot_crystals_at_timestep
    elif snapshot_label == "Displacement vectors snapshot":
        snapshot_func = snapshot_displacement_vectors  # the per-particle one
    elif snapshot_label == "Displacement vectors snapshot (Voronoi)":
        snapshot_func = snapshot_displacement_vectors_voronoi
    elif snapshot_label == "Alignment field snapshot":
        snapshot_func = snapshot_alignment_field
    elif snapshot_label == "Local density snapshot":
        snapshot_func = snapshot_local_density

    # --- Styling options for phase diagram ---
    st.sidebar.header("8. Styling (phase diagram)")
    legend_on_phase = st.sidebar.checkbox(
        "Show legends in cells", value=True, key="legend_on_phase"
    )
    legend_fontsize_phase = st.sidebar.number_input(
        "Legend font size",
        value=10,
        min_value=1,
        max_value=30,
        key="legend_fs_phase",
    )
    axis_fontsize_phase = st.sidebar.number_input(
        "Axis label/tick font size",
        value=10,
        min_value=1,
        max_value=30,
        key="axis_fs_phase",
    )
    title_on_phase = st.sidebar.checkbox(
        "Show titles in cells", value=True, key="title_on_phase"
    )
    title_fontsize_phase = st.sidebar.number_input(
        "Title font size",
        value=12,
        min_value=1,
        max_value=40,
        key="title_fs_phase",
    )

    show_xy_labels = st.sidebar.checkbox(
        "Show X/Y labels in each cell", value=False, key="phase_xylabels"
    )

    st.sidebar.header("9. Plot kwargs for snapshot function")

    skip = st.sidebar.number_input(
        "Skip (frame stride):", value=1, step=1, min_value=0, key="phase_skip"
    )
    normY = st.sidebar.checkbox(
        "Normalize Y axis", value=True, key="phase_normY"
    )
    box_size_um = st.sidebar.number_input(
        "Box size (µm):", value=50.0, step=1.0, key="phase_boxsize"
    )
    particle_radius_um = st.sidebar.number_input(
        "Particle radius (µm):", value=1.0, step=0.1, key="phase_radius"
    )

    run = st.sidebar.button("▶ Run phase-diagram routine", key="run_phase")

    if not run:
        st.stop()

    if not base_dir:
        st.warning("Please select or enter a base folder path.")
        st.stop()

    if not tkb_rows:
        st.warning("No TkB rows selected. Please tick some or add custom values.")
        st.stop()

    if not column_times_s:
        st.warning("No valid time columns parsed.")
        st.stop()

    st.subheader("Phase diagram snapshot panel")
    st.write(f"Base folder: `{base_dir}`")
    st.write(f"TkB rows: `{tkb_rows}`")
    st.write(f"Time columns (s): `{[round(t, 3) for t in column_times_s]}`")
    st.write(f"Snapshot function: **{snapshot_label}**")

    plt.close("all")

    plot_kwargs = dict(
        skip=int(skip),
        normY=int(normY),
        box_size_um=float(box_size_um),
        particle_radius_um=float(particle_radius_um),
    )

    # Decide colorbar based on snapshot type
    if snapshot_label == "Local density snapshot":
        cb_cmap = "viridis"
        cb_label = "Normalized local density"
    elif snapshot_label == "Alignment field snapshot":
        cb_cmap = "plasma"
        cb_label = "|ψ| (local alignment)"
    else:
        cb_cmap = None
        cb_label = None

    fig, axes = make_snapshot_phase_panel(
        base_dir=base_dir,
        tkb_rows=tkb_rows,
        column_times_s=column_times_s,
        taub=float(taub),
        x_name=x_name,
        y_name=y_name,
        figure_size=(12, 14),
        dpi=300,
        plot_func=snapshot_func,
        plot_kwargs=plot_kwargs,
        show_xy_labels=show_xy_labels,
        colorbar_cmap=cb_cmap,
        colorbar_label=cb_label,
    )

    apply_global_styling(
        fig,
        legend_on=legend_on_phase,
        legend_fontsize=legend_fontsize_phase,
        axis_fontsize=axis_fontsize_phase,
        title_on=title_on_phase,
        title_fontsize=title_fontsize_phase,
    )
    st.pyplot(fig)
    save_figure_if_requested(
        fig,
        base_name=f"phase_{snapshot_label}",
        save_enabled=save_plots,
        output_dir=output_dir,
    )
    st.success("Phase-diagram snapshot panel done ✅")



















# ---------------------------------------------------------------------
# SUMMARY PLOT MODE  (folder-based)
# ---------------------------------------------------------------------

elif plot_mode == "Summary plots":


    st.sidebar.header("2. Data folder (summary plots)")

    st.sidebar.header("Time window (applies to all reductions)")
    window_time_units = st.sidebar.radio(
        "Window unit", ["seconds", "τB"], key="summary_window_units"
    )
    t_start_frac = st.sidebar.slider(
        "Window start (fraction of run)", 0.0, 0.9, 0.0, 0.01, key="summary_window_start"
    )
    t_end_frac = st.sidebar.slider(
        "Window end (fraction of run)", 0.1, 1.0, 1.0, 0.01, key="summary_window_end"
    )

    folder_source = st.sidebar.radio(
        "Folder source:",
        ["HIWI root subfolder", "Custom path"],
        key="summary_folder_source",
    )

    hiwi_subdirs = []
    if os.path.isdir(source_root):
        hiwi_subdirs = [
            d for d in os.listdir(source_root)
            if os.path.isdir(os.path.join(source_root, d))
        ]

    if folder_source == "HIWI root subfolder":
        subchoice = st.sidebar.selectbox("Choose subfolder:", hiwi_subdirs, key="summary_hiwi_sub")
        base_dir = os.path.join(source_root, subchoice) if subchoice else ""
    else:
        base_dir = st.sidebar.text_input("Custom base folder path:", value="", key="summary_folder_custom")

    base_dir = base_dir.strip().strip('"').strip("'")

    # --- detect runs like in Multiple plots ---
    available_runs = []
    if base_dir and os.path.isdir(base_dir):
        folder_re = re.compile(
            r"^([0-9]+(?:\.[0-9]+)?)Tkb[ _]*([0-9]+(?:\.[0-9]+)?)TauB$",
            re.IGNORECASE,
        )
        for fn in os.listdir(base_dir):
            m = folder_re.match(fn)
            if not m:
                continue
            tkb_val = float(m.group(1))
            tau_val = float(m.group(2))
            available_runs.append((os.path.join(base_dir, fn), tkb_val, tau_val))
    available_runs = sorted(available_runs, key=lambda r: (r[1], r[2]))

    # --- File names: collapsed right under folder chooser (and compact) ---
    with st.sidebar.expander("3. File names", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            x_name = st.text_input(
                "X",
                value="datax.csv",
                key=f"summary_xname",
                help="X filename inside each run folder",
            )
        with c2:
            y_name = st.text_input(
                "Y",
                value="datay.csv",
                key=f"summary_yname",
                help="Y filename inside each run folder",
            )
        c3, c4 = st.columns(2)
        with c3:
            skip = st.number_input(
                "Skip (frame stride)",
                value=1,
                step=1,
                min_value=0,
                key=f"summary_skip",
            )
        with c4:
            normY = st.checkbox(
                "Normalize Y axis",
                value=True,
                key=f"summary_normY",
            )


    def format_run_option(run) -> str:
        _, tkb, tau = run
        return f"{tkb:g} Tkb (TauB: {tau:g})"


    selected_runs = st.sidebar.multiselect(
        "Select runs:",
        options=available_runs,
        default=available_runs,
        key="summary_runs",
        format_func=format_run_option,
    )




    st.sidebar.header("4. Metric curve to summarize")
    metric_label = st.sidebar.selectbox("Metric:", metric_options, key="summary_metric_label")

    # ---- common params ----
    with st.sidebar.expander("5. Common parameters", expanded=False):

        # cluster params
        st.subheader("Cluster metrics parameters")
        cluster_eps_summary = st.number_input(
            "DBSCAN eps",
            value=3.5,
            step=0.1,
            key=f"summary_cluster_eps",
        )
        cluster_min_samples_summary = st.number_input(
            "DBSCAN min_samples",
            value=3,
            step=1,
            min_value=1,
            key=f"summary_cluster_min_samples",
        )
        cluster_min_cluster_size_summary = st.number_input(
            "Min cluster size",
            value=3,
            step=1,
            min_value=1,
            key=f"summary_cluster_min_cluster_size",
        )

    st.sidebar.header("6. Reduction (scalar from curve)")
    reducer = st.sidebar.selectbox(
        "Scalar to compute from y(t):",
        ["max", "min", "mean", "last", "slope (linear fit)"],
        key="summary_reducer",
    )

    st.sidebar.header("7. Plot against")
    x_choice = st.sidebar.selectbox("X-axis:", ["TkB", "TauB"], key="summary_x_choice")

    # caching behavior
    export_data = st.sidebar.checkbox("Recompute + overwrite saved run data", value=False, key="summary_recompute")
    export_dir = st.sidebar.text_input("Optional export folder (CSV of scalars)", value="", key="summary_export_dir")

    run_summary = st.sidebar.button("▶ Run summary plot", key="run_summary")

    have_cached = st.session_state.get("summary_cached", False)

    pa_key = f"summary_{metric_label}_{reducer}_{x_choice}_{window_time_units}_{t_start_frac:.3f}_{t_end_frac:.3f}"
    st.session_state.setdefault("summary_cache_by_key", {})
    cache_map = st.session_state["summary_cache_by_key"]
    have_cached = pa_key in cache_map


    cache_map = st.session_state.get("summary_cache_by_key", {})
    have_cached = pa_key in cache_map

    # ✅ If we are NOT re-running computation, but we already have cached summary output,
    # redraw from cache so post-analysis doesn't reset on reruns.
    if (not run_summary) and have_cached:
        cache = cache_map[pa_key]
        curves = cache["curves"]
        rows = cache.get("rows", None)  # we will store rows in cache (next step)
        x_choice = cache["x_choice"]
        xlabel_pa = cache["pa_xlabel"]

        if cache is None:
            st.info("No cached summary yet. Click Run summary plot once.")
            st.stop()

        label_mode = cache["label_mode"]
        label_fmt = (lambda ta: f"TauB {ta:g}") if label_mode == "TauB" else (lambda tk: f"TkB {tk:g}")


        plt.close("all")
        plt.figure()
        ax = plt.gca()

        xvals = np.asarray(cache["xvals"], float)
        yvals = np.asarray(cache["yvals"], float)
        group_key = np.asarray(cache["group_key"], float)

        for g in sorted(set(group_key)):
            mask = group_key == g
            ax.plot(xvals[mask], yvals[mask], marker="o", linestyle="-", label=label_fmt(g))

        ax.set_title(cache["title"])
        ax.set_xlabel(cache["xlabel"])
        ax.set_ylabel(cache["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig = plt.gcf()
        apply_global_styling(fig, legend_on=True, legend_fontsize=14, axis_fontsize=16, title_on=True,
                             title_fontsize=18)
        st.pyplot(fig)

        post_analysis_block_generic(
            key=pa_key,
            curves=curves,
            ylabel=f"{reducer}({metric_label})",
            xlabel=xlabel_pa,
            legend_on=True,
            legend_fontsize=14,
            axis_fontsize=16,
            title_on=True,
            title_fontsize=18,
            sweep_rows=rows,
            x_choice=x_choice,
        )

        st.stop()

    # nothing cached and button not pressed → stop
    if not run_summary:
        st.stop()

    if not base_dir or not os.path.isdir(base_dir):
        st.warning("Choose a valid base folder.")
        st.stop()
    if not selected_runs:
        st.warning("Select at least one run.")
        st.stop()

    dataset_tag = os.path.basename(base_dir.rstrip("\\/"))
    metric_func, default_ylabel = metric_map[metric_label]

    series, missing = get_series_for_metric(
        metric_label=metric_label,
        metric_func=metric_func,
        selected_runs=selected_runs,
        base_dir=base_dir,
        dataset_tag=dataset_tag,
        x_name=x_name,
        y_name=y_name,
        skip=int(skip),
        normY=int(normY),
        export_data=export_data,
        export_dir="",
        extra_kwargs=None,  # ✅ optional
    )

    if missing:
        st.warning("Missing data files for: " + ", ".join(missing))

    # if metric is cluster-based, force recompute branch params (same idea as calculator fix)
    # easiest: just overwrite series by recomputing with params if needed
    if metric_label in ["Number of clusters", "Average cluster size","Area fraction","Bond orientational order", "Detect crystals", "Radius of gyration"]:
        # recompute with params (and ignore old cached, per your earlier choice)
        series = []
        for folder_path, tkb_val, tau_val in selected_runs:
            run_name = os.path.basename(folder_path)
            fx = os.path.join(folder_path, x_name)
            fy = os.path.join(folder_path, y_name)
            if not (os.path.isfile(fx) and os.path.isfile(fy)):
                continue
            kwargs_run = dict(
                skip=int(skip),
                normY=int(normY),
                TauB=float(tau_val),
                eps=float(cluster_eps_summary),
                min_samples=int(cluster_min_samples_summary),
                min_cluster_size=int(cluster_min_cluster_size_summary),
            )
            t, y = load_or_compute_metric_cached(
                base_dir=base_dir,
                run_folder=run_name,
                fx=fx,
                fy=fy,
                metric_func=metric_func,
                metric_kwargs=kwargs_run,
                force_recompute=export_data,
            )
            series.append({"run": run_name, "tkb": float(tkb_val), "taub": float(tau_val), "y": np.asarray(y, float)})

    if not series:
        st.warning("No data series available for this metric.")
        st.stop()

    # --- compute scalar per run ---
    rows = []
    for s in series:
        y = np.asarray(s["y"], float)
        n = len(y)
        if n < 2:
            scalar = np.nan
            rows.append((s["run"], s["tkb"], s["taub"], scalar))
            continue

        taub = float(s["taub"])
        t_sec = np.linspace(0.0, taub * 13.513, n)

        # choose time axis unit for windowing
        if window_time_units == "τB":
            t_use = t_sec / 13.513
        else:
            t_use = t_sec

        # ✅ apply window to every reduction
        t_w, y_w = slice_by_time_window(t_use, y, t_start_frac, t_end_frac)

        if y_w.size == 0:
            scalar = np.nan
        else:
            if reducer == "max":
                scalar = float(np.nanmax(y_w))
            elif reducer == "min":
                scalar = float(np.nanmin(y_w))
            elif reducer == "mean":
                scalar = float(np.nanmean(y_w))
            elif reducer == "last":
                scalar = float(y_w[-1])
            elif reducer == "slope (linear fit)":
                # slope dy/dt (or dy/dτB depending on window unit)
                if t_w.size < 2:
                    scalar = np.nan
                else:
                    slope, intercept = np.polyfit(t_w, y_w, 1)
                    scalar = float(slope)
            else:
                scalar = np.nan

        rows.append((s["run"], s["tkb"], s["taub"], scalar))

    # ----------------------------
    # Read UI state FIRST (no widgets yet)
    # ----------------------------

    # --- choose x-axis + grouping ---
    if x_choice == "TkB":
        xvals = np.array([r[1] for r in rows], float)
        xlabel = "TkB"
        group_key = np.array([r[2] for r in rows], float)  # group by TauB
        label_fmt = lambda ta: f"TauB {ta:g}"
        xlabel_pa = "TkB"
        label_mode = "TauB"
    else:
        xvals = np.array([r[2] for r in rows], float)
        xlabel = "TauB"
        group_key = np.array([r[1] for r in rows], float)  # group by TkB
        label_fmt = lambda tk: f"TkB {tk:g}"
        xlabel_pa = "TauB"
        label_mode = "TkB"

    yvals = np.array([r[3] for r in rows], float)

    # build curves (needed for knee + post-analysis)
    curves = []
    for g in sorted(set(group_key)):
        mask = group_key == g
        curves.append({"run": label_fmt(g), "x": xvals[mask], "y": yvals[mask]})

    st.session_state.setdefault("summary_cache_by_key", {})
    st.session_state["summary_cache_by_key"][pa_key] = {
        "xvals": xvals,
        "yvals": yvals,
        "group_key": group_key,
        "xlabel": xlabel,
        "ylabel": f"{reducer}({metric_label})",
        "title": f"{reducer}({metric_label}) vs {xlabel}",
        "label_mode": "TauB" if x_choice == "TkB" else "TkB",
        "curves": curves,
        "rows": rows,  # ✅ ADD THIS
        "x_choice": x_choice,  # ✅ ADD THIS
        "pa_key": pa_key,
        "pa_xlabel": xlabel_pa,
        "pa_ylabel": f"{reducer}({metric_label})",
    }

    # ----------------------------
    # NOW build the figure using the CURRENT widget values
    # ----------------------------
    plt.close("all")
    plt.figure()
    ax = plt.gca()

    # measured curves
    for g in sorted(set(group_key)):
        mask = group_key == g
        ax.plot(xvals[mask], yvals[mask], marker="o", linestyle="-", label=label_fmt(g))



    # finalize
    ax.set_title(f"{reducer}({metric_label}) vs {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{reducer}({metric_label})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = plt.gcf()
    apply_global_styling(fig, legend_on=True, legend_fontsize=14, axis_fontsize=16, title_on=True, title_fontsize=18)

    # render plot at the TOP placeholder
    st.pyplot(fig)

    post_analysis_block_generic(
        key=pa_key,
        curves=curves,
        ylabel=f"{reducer}({metric_label})",
        xlabel=xlabel_pa,
        legend_on=True,
        legend_fontsize=14,
        axis_fontsize=16,
        title_on=True,
        title_fontsize=18,
        sweep_rows=rows,
        x_choice=x_choice,
    )

# TODO: add plotly toogle
# TODO: Fix  ########### LOCUS PLOT ######### in Phase-transition detection
# TODO: Error bars + statistics ( You compute: Mean curve, deviation, error band for runs of the same tkb, tauB)

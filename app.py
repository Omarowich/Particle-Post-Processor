# app.py

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from area_fraction_over_time import area_fraction_over_time
from bond_orientation_over_time import bond_orientational_order_over_time
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


from stats_detectors import (
    _overlay_feature_locus,
    _clamp_smooth_win,
    _overlay_prognosis_surface,
    _moving_average,
    apply_curve_detectors,
    slice_by_time_window,
    auc_and_mean,
    fit_relaxation_tau,
    curve_envelope_area,
)
from math_expr import (
    fft_freqs,
    safe_eval_expr,
    _ALLOWED_FUNCS,
)
from plot_utils import (
    save_uploaded_file,
    apply_global_styling,
    apply_grid,
    save_figure_if_requested,
    render_smoothing_preview,
)
from metric_caching import (
    rundata_npz_path,
    save_rundata_npz,
    load_rundata_npz,
    export_rundata_csv,
    get_series_for_metric,
    export_series_to_csv,
    align_y_by_taub_length,
    metric_cache_path,
    load_or_compute_metric_cached,
    run_tag,
    tag_suffix,
)
import pandas as pd
import regime_state_model as rsm


def _default_runs_root(root):
    """The merged Data folder holds ARF/NAF/AAF itself; older layouts kept Python runs in 'BD runs v2'."""
    if not root:
        return ""
    if any(os.path.isdir(os.path.join(root, r)) for r in ("ARF", "NAF", "AAF")):
        return root
    old = os.path.join(root, "BD runs v2")
    return old if os.path.isdir(old) else root


def clean_path(p):
    """Path typed or pasted by the user: drop spaces, any kind of quotes and trailing slashes."""
    p = (p or "").strip().strip('"\'\u201c\u201d\u2018\u2019').strip()
    return p.rstrip("\\/") if len(p) > 3 else p
import run_viewer as rv
import io
from pptx_export import build_pptx_from_images
from phase_panel import make_snapshot_phase_panel
from sidebar_widgets import render_cluster_params_widgets

# Sweep-folder 3rd-parameter units are parsed and stored lowercase (e.g. "f"
# for the "...F" acoustic-streaming-force folders) since that raw value is
# used as a grouping/cache key, filter widget value, and folder-matching
# literal throughout the app. This maps that raw value to what should be
# shown in plot legends only -- folder names, filter labels, and tables
# still show the raw unit as parsed.
UNIT_DISPLAY_NAMES = {"f": "N/m³"}


def display_unit(unit):
    if not unit:
        return unit
    return UNIT_DISPLAY_NAMES.get(unit.lower(), unit)


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

        run_opts = ["All"] + [c["run"] for c in curves]
        pick = st.selectbox("Analyze curve", run_opts, index=0, key=f"{key}_pick")
        smooth_win = st.slider("Smoothing window", 1, 51, 9, 2, key=f"{key}_smooth")

        preview_targets = curves if pick == "All" else [c for c in curves if c["run"] == pick]
        if preview_targets:
            preview_curves = []
            for c in preview_targets:
                x_p = np.asarray(c["x"], float)
                y_p = np.asarray(c["y"], float)
                preview_curves.append({
                    "label": c.get("run", "run"),
                    "x": x_p,
                    "y_raw": y_p,
                    "y_smoothed": _moving_average(y_p, smooth_win),
                })
            render_smoothing_preview(preview_curves, smooth_win)

        # ALWAYS rendered (state won’t reset)


        with st.form(key=f"post_form_{key}", clear_on_submit=False):
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

        result_cache_key = f"{key}_last_post_analysis"
        if not run_post:
            cached = st.session_state.get(result_cache_key)
            if cached is None:
                st.caption("Adjust settings, then click **Run post-analysis**.")
            else:
                import pandas as pd
                st.pyplot(cached["fig"])
                st.dataframe(pd.DataFrame(cached["results"]))
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

            row_extra, y_level = apply_curve_detectors(
                ax2, x, y,
                smooth_win=smooth_win,
                sustain_frac=sustain_frac,
                slope_eps=slope_eps,
                label_markers=label_markers,
                y_level=y_level,
                do_knee=do_knee,
                do_turn=do_turn,
                do_flat=do_flat,
                do_thr=do_thr,
                thr_mode=thr_mode,
                thr_val=thr_val,
                thr_dir=thr_dir,
                thr_sustain=thr_sustain,
                do_auc=do_auc,
                do_maxslope=do_maxslope,
                maxslope_smooth_win=sw,
                do_peaks=do_peaks,
                peak_min_prom=peak_min_prom,
                peak_min_dist=peak_min_dist,
                do_pwlin=do_pwlin,
                pw_min_seg=pw_min_seg,
                key_prefix="x_",
            )
            row.update(row_extra)

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
        apply_grid(ax2, show_grid)
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
        st.session_state[result_cache_key] = {"fig": fig2, "results": results}
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
    show_grid=False,
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

        run_opts = ["All"] + sorted({c.get("run", "run") for c in curves})
        pick = st.selectbox("Analyze run", run_opts, index=0, key=f"{key}_pick")
        smooth_win = st.slider("Smoothing window", 1, 51, 9, 2, key=f"{key}_smooth")

        preview_targets = curves if pick == "All" else [c for c in curves if c.get("run") == pick]
        if preview_targets:
            preview_curves = []
            for c in preview_targets:
                x_p = np.asarray(c["x"], float)
                y_p = np.asarray(c["y"], float)
                preview_curves.append({
                    "label": c.get("run", "run"),
                    "x": x_p,
                    "y_raw": y_p,
                    "y_smoothed": _moving_average(y_p, smooth_win),
                })
            render_smoothing_preview(preview_curves, smooth_win)

        with st.form(key=f"post_form_{key}", clear_on_submit=False):
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

        result_cache_key = f"{key}_last_post_analysis"
        if not run_post:
            cached = st.session_state.get(result_cache_key)
            if cached is None:
                st.caption("Adjust settings, then click **Run post-analysis**.")
            else:
                import pandas as pd
                st.pyplot(cached["fig"])
                st.dataframe(pd.DataFrame(cached["results"]))
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

            row_extra, y_level = apply_curve_detectors(
                ax2, x, y,
                smooth_win=smooth_win,
                sustain_frac=sustain_frac,
                slope_eps=slope_eps,
                label_markers=label_markers,
                y_level=y_level,
                do_knee=do_knee,
                do_turn=do_turn,
                do_flat=do_flat,
                do_thr=do_thr,
                thr_mode=thr_mode,
                thr_val=thr_val,
                thr_dir=thr_dir,
                thr_sustain=thr_sustain,
                do_auc=do_auc,
                do_maxslope=do_maxslope,
                do_peaks=do_peaks,
                peak_min_prom=peak_min_prom,
                peak_min_dist=peak_min_dist,
                do_pwlin=do_pwlin,
                pw_min_seg=pw_min_seg,
                key_prefix="t_",
            )
            row.update(row_extra)

            results.append(row)

        ax2.set_title(f"")
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel("Time (τB)" if x_axis_mode == "Brownian time τ_B" else "Time (s)")
        apply_grid(ax2, show_grid)
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
        st.session_state[result_cache_key] = {"fig": fig2, "results": results}
        st.pyplot(fig2)

        import pandas as pd
        st.dataframe(pd.DataFrame(results))


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
        key="multi_folder_source",
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
            key="multi_hiwi_sub",
        )
        base_dir = os.path.join(source_root, subchoice) if subchoice else ""
    else:
        base_dir = st.sidebar.text_input(
            "Custom base folder path:",
            value="",
            key="multi_folder_custom",
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
                key="multi_xname",
                help="X filename inside each run folder",
            )
        with c2:
            y_name = st.text_input(
                "Y",
                value="datay.csv",
                key="multi_yname",
                help="Y filename inside each run folder",
            )
        c3, c4 = st.columns(2)
        with c3:
            skip = st.number_input(
                "Skip (frame stride)",
                value=1,
                step=1,
                min_value=0,
                key="multi_skip",
            )
        with c4:
            normY = st.checkbox(
                "Normalize Y axis",
                value=True,
                key="multi_normY",
            )

    # ---- runs detection ----
    st.sidebar.header("3. Runs (TkB / TauB)")

    available_runs = []
    if base_dir and os.path.isdir(base_dir):
        try:
            folder_re = re.compile(
                r"^([0-9]+(?:\.[0-9]+)?)Tkb[ _]*([0-9]+(?:\.[0-9]+)?)TauB(?:[ _]*([0-9]+(?:\.[0-9]+)?)(nm|nn|dn|F))?((?:_[A-Za-z]+[0-9.]*)*)$",
                re.IGNORECASE,
            )
            for fn in os.listdir(base_dir):
                m = folder_re.match(fn)
                if not m:
                    continue
                tkb_val = float(m.group(1))
                tau_val = float(m.group(2))
                third_val = float(m.group(3)) if m.group(3) else None
                third_unit = m.group(4).lower() if m.group(4) else None
                if third_unit == "nn":   # typo in some folder names (e.g. 0Tkb_1TauB_3000nn)
                    third_unit = "nm"
                folder_path = os.path.join(base_dir, fn)
                available_runs.append((folder_path, tkb_val, tau_val, third_val, third_unit))
        except Exception as e:
            st.sidebar.error(f"Could not scan TkB/TauB folders in {base_dir}: {e}")

    available_runs = sorted(available_runs, key=lambda r: (r[1], r[2]))

    def format_run_option(run) -> str:
        _, tkb, tau = run[:3]
        third_val = run[3] if len(run) > 3 else None
        third_unit = run[4] if len(run) > 4 else None
        base = f"{tkb:g} Tkb (TauB: {tau:g})"
        if third_val is not None and third_unit:
            base += f" ({third_val:g} {third_unit})"
        tag = run_tag(run[0])
        if tag:
            base += f" [{tag}]"
        return base

    if available_runs:
        import run_filter as rf
        _keep = rf.widgets(rf.metas_from_paths([r[0] for r in available_runs]), key=f"multi__{dataset_tag}",
                           container=st.sidebar)
        filtered_runs = [r for r, k in zip(available_runs, _keep) if k]

        # Re-select every filtered run whenever the filter itself changes, instead of keeping whatever
        # subset of runs was previously (de)selected -- a filter change means "show me everything
        # matching this new filter", not "keep my old picks".
        runs_key = f"multi_runs__{dataset_tag}"
        filter_signature = tuple(_keep)
        filter_signature_key = f"multi_runs_filter_signature__{dataset_tag}"
        if st.session_state.get(filter_signature_key) != filter_signature:
            st.session_state[runs_key] = filtered_runs
            st.session_state[filter_signature_key] = filter_signature

        selected_runs = st.sidebar.multiselect(
            "Select runs (from filtered):",
            options=filtered_runs,
            key=runs_key,
            format_func=format_run_option,
        )
    else:
        selected_runs = []
        st.sidebar.warning("No Tkb_*TauB* subfolders detected. Please check your base folder.")


    has_third = any(len(r) > 3 and r[3] is not None for r in available_runs)
    group_by_options = ["TkB", "TauB"]
    if has_third:
        third_units = {r[4] for r in available_runs if len(r) > 4 and r[4]}
        third_label = "/".join(sorted(third_units)) if third_units else "3rd param"
        group_by_options.append(third_label)

    group_by = st.sidebar.selectbox(
        "Color/group curves by:",
        group_by_options,
        index=0,
        key="multi_group_by",
    )
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
            key="multi_metric_select",
        )


    # ---- Crystal metrics parameters (ONLY if Detect crystals selected) ----
    crystal_types = []
    if include_crystal_multiselect and metrics_selected and ("Detect crystals" in metrics_selected):
        st.sidebar.header("Crystal metrics parameters")
        crystal_types = st.sidebar.multiselect(
            "Crystal types (Detect crystals)",
            ["Hexagonal", "Square", "Triangular", "All", "None"],
            default=["Hexagonal"],
            key="multi_crystal_types",
        )

    # ---- common params ----
    with st.sidebar.expander("Common parameters", expanded=False):

        # cluster params
        cluster_params = render_cluster_params_widgets("multi")
        area_fraction_mode = cluster_params["mode"]
        cluster_eps = cluster_params["eps"]
        cluster_min_samples = cluster_params["min_samples"]
        cluster_min_cluster_size = cluster_params["min_cluster_size"]

        st.subheader("RDF parameters")
        rdf_r_max = st.number_input(
            "RDF r_max (µm)", value=10.0, step=0.5, key="multi_rdf_rmax"
        )
        rdf_dr = st.number_input(
            "RDF dr (bin width, µm)", value=0.1, step=0.05, format="%.3f", key="multi_rdf_dr"
        )


    # styling
    st.sidebar.header("Styling")
    c_left, c_right = st.sidebar.columns([1, 1])
    with c_left:
        legend_on = st.checkbox("Show legend", value=True, key="multi_legend_on")
        show_taub_in_legend = st.checkbox("Show TauB in legend", value=False, key="multi_show_taub")
        title_on = st.checkbox("Show title", value=False, key="multi_title_on")
        show_ylabel = st.checkbox("Show Y axis label", value=False, key="multi_show_ylabel")
        show_xlabel = st.checkbox("Show X axis label", value=False, key="multi_show_xlabel")
        show_grid = st.checkbox("Show grid", value=False, key="multi_show_grid")
    with c_right:
        legend_fontsize = st.number_input(
            "Legend font size", value=16, min_value=1, max_value=40, key="multi_legend_fs"
        )
        axis_fontsize = st.number_input(
            "Axis font size", value=18, min_value=1, max_value=40, key="multi_axis_fs"
        )
        title_fontsize = st.number_input(
            "Title font size", value=20, min_value=1, max_value=60, key="multi_title_fs"
        )

    # x-axis mode
    st.sidebar.subheader("X-Axis")
    c11, c12 = st.sidebar.columns(2)
    with c11:
        x_axis_mode = st.radio(
            "X-axis units",
            ["Brownian time τ_B", "Seconds"],
            key="multi_xaxis_mode",
        )
    with c12:
        TauB = st.number_input(
            "TauB (Brownian time)",
            value=2.0,
            step=0.5,
            key="multi_TauB",
        )
    st.sidebar.subheader("X Range Slicer")
    x_range_custom = st.sidebar.checkbox("Clip X range", value=False, key="multi_x_range_custom")
    if x_range_custom:
        xc1, xc2 = st.sidebar.columns(2)
        with xc1:
            x_range_min = st.sidebar.number_input("X min", value=0.0, step=0.01, format="%.3f", key="multi_x_range_min")
        with xc2:
            x_range_max = st.sidebar.number_input("X max", value=1.0, step=0.01, format="%.3f", key="multi_x_range_max")
    else:
        x_range_min = None
        x_range_max = None

    # y-axis mode
    st.sidebar.subheader("Y-Axis")
    c21, c22 = st.sidebar.columns(2)
    with c21:
        y_scale = st.selectbox(
            "Scale",
            ["Linear", "Log"],
            index=0,
            key="multi_y_scale",
        )
    with c22:
        y_unit = st.text_input(
            "Unit label",
            placeholder="μm, mm, 1/s, …",
            key="multi_y_unit",
            help="This is only the label shown on the plot (does not change the data).",
        )
    y_range_custom = st.sidebar.checkbox("Custom Y range", value=False, key="multi_y_range_custom")
    y_range_min = st.sidebar.number_input("Y min", value=0.0, step=1.0, key="multi_y_range_min") if y_range_custom else None
    y_range_max = st.sidebar.number_input("Y max", value=100.0, step=1.0, key="multi_y_range_max") if y_range_custom else None

    # export / recompute
    st.sidebar.header("Caching / export")
    export_data = st.sidebar.checkbox(
        "Export (recompute + overwrite saved run data)",
        value=False,
        key="multi_export_data",
    )
    export_dir = st.sidebar.text_input(
        "Optional CSV export folder (leave empty to skip CSV export):",
        value=DEFAULT_SAVE_DIR,
        key="multi_export_dir",
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
        "show_ylabel": show_ylabel,
        "show_xlabel": show_xlabel,
        "show_grid": show_grid,
        "group_by": group_by,
        "x_range_custom": x_range_custom,
        "x_range_min": x_range_min,
        "x_range_max": x_range_max,

    }
    return out


# Allowed functions (vectorized via numpy)


# ---------------------------------------------------------------------
# Default root directory (used as default value for the user-editable root)
# ---------------------------------------------------------------------
_NAS = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI"
_LOCAL = r"D:\Work\AIRPODS Paper 2627"
# local copy first (fast); the NAS only when the local folder is missing. AIRPODS_DATA overrides both.
HIWI_ROOT = os.environ.get("AIRPODS_DATA") or (os.path.join(_LOCAL, "Data") if os.path.isdir(os.path.join(_LOCAL, "Data"))
                                               else os.path.join(_NAS, "Raw Data"))
# app.py

DEFAULT_SAVE_DIR = (os.path.join(_LOCAL, "analysis results") if os.path.isdir(os.path.join(_LOCAL, "analysis results"))
                    else os.path.join(_NAS, "Saved Data for Streamlit"))


# ---------------------------------------------------------------------
# Helper: save uploaded files
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helper: global styling for any matplotlib figure
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helper: slugify for filenames + save figure if requested
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Helper: build PPTX from image folders
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Phase-diagram snapshot panel
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------


st.set_page_config(page_title="Particle Post Processor and Analyzer", layout="wide")
st.title("Acoustic Particles Analyzer")

# ========== STEP 0: Source root ==========
st.sidebar.header("0. Source root")
source_root = st.sidebar.text_input(
    "Base root folder (Data folder with the regime folders NAF / ARF / AAF / ASF = Regime I–IV):",
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
    ["Single plot", "Multiple plots", "Function mixer", "Summary plots", "Phase diagram", "Regime state model", "Run viewer", "3D viewer", "Paper figures", "Run simulation"],
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
        default_metrics=["Particle distance"],
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
    show_ylabel = ui.get("show_ylabel", False)
    show_xlabel = ui.get("show_xlabel", True)
    show_grid = ui.get("show_grid", True)
    x_range_custom = ui.get("x_range_custom", False)
    x_range_min = ui.get("x_range_min", None)
    x_range_max = ui.get("x_range_max", None)


    # ---- Color mode UI ----
    st.sidebar.header("Curve colors")
    color_mode_multi = st.sidebar.radio(
        "Color mode",
        ["Single color", "Gradient over TkB", "Group by 3rd param", "Group by 3rd param + gradient over TkB"],
        key="multi_color_mode"
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        color_start_multi = st.color_picker("Start color", "#D3D3D3", key="multi_color_start")
    with col2:
        color_end_multi = st.color_picker("End color", "#000000", key="multi_color_end")

    import matplotlib.colors as mcolors

    # collect unique values
    unique_tkbs = sorted({float(r[1]) for r in selected_runs})
    unique_taus = sorted({float(r[2]) for r in selected_runs})
    unique_thirds = sorted({r[3] for r in selected_runs if r[3] is not None})

    # assign a base color per group (3rd param) -- stable for the whole session
    # (assigned the first time a dn value is ever seen, kept forever after) so
    # deselecting one group doesn't visibly reassign every other group's color
    group_base_colors = {}
    if unique_thirds:
        group_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        dn_color_map = st.session_state.setdefault("dn_color_map", {})
        for v in unique_thirds:
            if v not in dn_color_map:
                dn_color_map[v] = group_palette[len(dn_color_map) % len(group_palette)]
            group_base_colors[v] = dn_color_map[v]

    cmap_default = mcolors.LinearSegmentedColormap.from_list(
        "default", [color_start_multi, color_end_multi]
    )

    def get_line_color(s):
        tkb = float(s["tkb"])
        third = s.get("third_val")

        if color_mode_multi == "Single color":
            return None

        elif color_mode_multi == "Gradient over TkB":
            if len(unique_tkbs) == 1:
                return cmap_default(0.5)
            idx = unique_tkbs.index(tkb)
            return mcolors.LinearSegmentedColormap.from_list(
                "g", [color_start_multi, color_end_multi]
            )(idx / max(len(unique_tkbs) - 1, 1))

        elif color_mode_multi == "Group by 3rd param":
            if third is not None and third in group_base_colors:
                return group_base_colors[third]
            return None

        elif color_mode_multi == "Group by 3rd param + gradient over TkB":
            if third is None or third not in group_base_colors:
                return None
            base_hex = group_base_colors[third]
            # make a light-to-dark gradient within the group color
            base_rgb = mcolors.to_rgb(base_hex)
            light = tuple(min(1.0, c * 1.0 + 0.5) for c in base_rgb)
            dark  = tuple(max(0.0, c * 0.4) for c in base_rgb)
            cmap_group = mcolors.LinearSegmentedColormap.from_list("grp", [dark, light])
            if len(unique_tkbs) == 1:
                return cmap_group(0.5)
            idx = unique_tkbs.index(tkb)
            return cmap_group(idx / max(len(unique_tkbs) - 1, 1))

        return None


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
    st.caption(f"Base folder: `{base_dir}` · Metrics: {', '.join(metrics_selected)}")

    with st.expander(f"Selected runs ({len(selected_runs)})", expanded=False):
        import pandas as pd
        runs_rows = []
        for r in selected_runs:
            tkb, tau = r[1], r[2]
            third_val = r[3] if len(r) > 3 else None
            third_unit = r[4] if len(r) > 4 else None
            row = {"TkB": tkb, "TauB": tau}
            if third_val is not None:
                row["3rd param"] = f"{third_val:g}{third_unit or ''}"
            row["tag"] = run_tag(r[0])
            runs_rows.append(row)
        st.dataframe(pd.DataFrame(runs_rows), use_container_width=True, hide_index=True)

    metric_kwargs = dict(TauB=TauB, skip=skip, normY=int(normY))


    def run_label(s, show_taub):
        base = f"{s['tkb']:g} Tkb"
        if show_taub:
            base += f" (TauB {s['taub']:g})"
        if s.get("third_val") is not None and s.get("third_unit"):
            base += f" {s['third_val']:g}{display_unit(s['third_unit'])}"
        return base + tag_suffix(s)


    # ----- RUN ALL SELECTED METRICS -----
    for metric_label in metrics_selected:
        plt.close("all")
        metric_func, default_ylabel = metric_map[metric_label]

        def render_curve_shape_analysis():
            # ---- Curve shape analysis ----
            with st.expander(f"📐 Curve shape analysis — {metric_label}", expanded=False):
                safe_key = metric_label.replace(" ", "_")
                curves_for_shape = st.session_state.get(f"post_curves_{metric_label}", [])
                if x_range_custom and x_range_min is not None and x_range_max is not None:
                    clipped_shape = []
                    for c in curves_for_shape:
                        x_c = np.asarray(c["x"], float)
                        y_c = np.asarray(c["y"], float)
                        mask = (x_c >= float(x_range_min)) & (x_c <= float(x_range_max))
                        if mask.sum() >= 2:
                            cc = dict(c)
                            cc["x"] = x_c[mask]
                            cc["y"] = y_c[mask]
                            clipped_shape.append(cc)
                    curves_for_shape = clipped_shape
                if not curves_for_shape:
                    st.info("Run the plot first to populate curves.")
                else:
                    from scipy.ndimage import uniform_filter1d
                    from scipy.stats import theilslopes
                    from collections import defaultdict
                    from matplotlib.lines import Line2D

                    def _robust_slope(x_seg, y_seg):
                        """Theil-Sen slope: the median of all pairwise slopes --
                        far less sensitive to one noisy point than a two-point
                        difference or an ordinary least-squares fit."""
                        if len(x_seg) < 2 or len(set(x_seg)) < 2:
                            return 0.0
                        return float(theilslopes(y_seg, x_seg)[0])

                    def _robust_spread(values):
                        """Median absolute deviation, scaled to be comparable to
                        std under normality -- doesn't get dragged around by a
                        single outlier the way std does."""
                        arr = np.asarray(values, float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size == 0:
                            return np.nan
                        return float(1.4826 * np.median(np.abs(arr - np.median(arr))))

                    unique_thirds_shape = sorted({c.get("third_val") for c in curves_for_shape if c.get("third_val") is not None})
                    third_units_shape = sorted({c.get("third_unit") for c in curves_for_shape if c.get("third_unit")})
                    third_axis_label = "/".join(third_units_shape) if third_units_shape else "3rd param"
                    shape_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
                    # Assign each dn value a color (and a line style) the first time
                    # it's ever seen this session, and keep it forever after --
                    # otherwise a color is only a *position* in the currently-selected
                    # set of dn values, so deselecting one group visibly reassigns
                    # everyone else's color. The line style rides along so two
                    # overlapping lines stay visually distinguishable even where
                    # their colors and y-values nearly coincide.
                    line_style_cycle = ["-", "--", "-.", ":"]
                    dn_color_map = st.session_state.setdefault("dn_color_map", {})
                    dn_style_map = st.session_state.setdefault("dn_style_map", {})
                    for v in unique_thirds_shape:
                        if v not in dn_color_map:
                            idx = len(dn_color_map)
                            dn_color_map[v] = shape_palette[idx % len(shape_palette)]
                            dn_style_map[v] = line_style_cycle[idx % len(line_style_cycle)]
                    third_color_map = {v: dn_color_map[v] for v in unique_thirds_shape}
                    third_style_map = {v: dn_style_map.get(v, "-") for v in unique_thirds_shape}

                    col_sa, col_sb, col_sc, col_sd, col_se = st.columns(5)
                    safe_key = metric_label.replace(" ", "_")
                    with col_sa:
                        smooth_win_shape = st.slider("Smoothing window", 1, 51, 9, 2, key=f"shape_{safe_key}_smooth")
                    with col_sb:
                        connect_dots_row1 = st.checkbox("Connect dots (row 1)", value=True, key=f"shape_{safe_key}_connect_r1")
                    with col_sc:
                        connect_dots_row2 = st.checkbox("Connect dots (row 2)", value=False, key=f"shape_{safe_key}_connect_r2")
                    with col_sd:
                        early_pct = st.slider("Early segment (%)", 5, 49, 20, 5, key=f"shape_{safe_key}_early")
                    with col_se:
                        late_pct = st.slider("Late segment (%)", 5, 49, 20, 5, key=f"shape_{safe_key}_late")

                    # --- live smoothing preview: raw vs. smoothed, one run or all at once ---
                    preview_run = st.selectbox(
                        "Preview smoothing on run:",
                        ["All"] + [c["run"] for c in curves_for_shape],
                        key=f"shape_{safe_key}_preview_run",
                    )
                    preview_targets = curves_for_shape if preview_run == "All" else [
                        c for c in curves_for_shape if c["run"] == preview_run
                    ]
                    preview_curves = []
                    for c in preview_targets:
                        x_p = np.asarray(c["x"], float)
                        y_p = np.asarray(c["y"], float)
                        preview_curves.append({
                            "label": c["run"],
                            "x": x_p,
                            "y_raw": y_p,
                            "y_smoothed": uniform_filter1d(y_p, size=smooth_win_shape),
                        })
                    render_smoothing_preview(preview_curves, smooth_win_shape)

                    # --- compute per-run metrics ---
                    results = []
                    group_curves_smoothed = defaultdict(list)
                    for c in curves_for_shape:
                        y = np.asarray(c["y"], float)
                        x = np.asarray(c["x"], float)
                        if len(y) < 10:
                            continue
                        y_sm = uniform_filter1d(y, size=smooth_win_shape)
                        n = len(y_sm)
                        early_seg = max(2, int(n * early_pct / 100))
                        late_seg  = max(2, int(n * late_pct  / 100))
                        # Theil-Sen slope over each whole segment, not just its
                        # two endpoints -- a two-point difference (or even an
                        # ordinary least-squares fit) is noise-sensitive once a
                        # segment is close to flat, which was causing slope_ratio
                        # to spike toward +inf on some runs
                        early_slope = _robust_slope(x[:early_seg + 1], y_sm[:early_seg + 1])
                        late_slope  = _robust_slope(x[-late_seg:], y_sm[-late_seg:])
                        late_scale = max(abs(late_slope), 0.05 * abs(early_slope))
                        slope_ratio = abs(early_slope) / (late_scale + 1e-12)
                        half_max = (np.nanmax(y_sm) + np.nanmin(y_sm)) / 2.0
                        is_decreasing = y_sm[-1] < y_sm[0]
                        if is_decreasing:
                            idx_half = np.where(y_sm <= half_max)[0]
                        else:
                            idx_half = np.where(y_sm >= half_max)[0]
                        t_half = float(x[idx_half[0]]) if len(idx_half) > 0 else float(x[-1])
                        tau = fit_relaxation_tau(x, y_sm)
                        _auc, auc_mean_val = auc_and_mean(x, y_sm)
                        results.append({
                            "run": c["run"],
                            "tkb": float(c["tkb"]),
                            "third_val": c.get("third_val"),
                            "third_unit": c.get("third_unit") or "",
                            "slope_ratio": slope_ratio,
                            "t_half": t_half,
                            "y_final": float(y_sm[-1]),
                            "tau": tau,
                            "auc_mean": auc_mean_val,
                        })
                        group_curves_smoothed[c.get("third_val")].append((x, y_sm))

                    # --- aggregate per nm group ---
                    group_data = defaultdict(lambda: {
                        "tkbs": [], "slope_ratios": [], "t_halfs": [], "y_finals": [],
                        "taus": [], "auc_means": [], "unit": "",
                    })
                    for r in results:
                        v = r["third_val"]
                        if v is not None:
                            group_data[v]["tkbs"].append(r["tkb"])
                            group_data[v]["slope_ratios"].append(r["slope_ratio"])
                            group_data[v]["t_halfs"].append(r["t_half"])
                            group_data[v]["y_finals"].append(r["y_final"])
                            group_data[v]["taus"].append(r["tau"])
                            group_data[v]["auc_means"].append(r["auc_mean"])
                            group_data[v]["unit"] = r["third_unit"]

                    def _sensitivity_slope(tkbs, values):
                        """Robust (Theil-Sen) slope of `values` against `tkbs` -- d(value)/d(TkB).
                        Resistant to a single outlier run dominating the fit."""
                        tkbs_arr = np.asarray(tkbs, float)
                        values_arr = np.asarray(values, float)
                        ok = np.isfinite(tkbs_arr) & np.isfinite(values_arr)
                        if ok.sum() < 2 or len(set(tkbs_arr[ok])) < 2:
                            return np.nan
                        return _robust_slope(tkbs_arr[ok], values_arr[ok])

                    group_summary = {}
                    for v, d in group_data.items():
                        group_summary[v] = {
                            "slope_ratio_mean": np.nanmedian(d["slope_ratios"]),
                            "slope_ratio_std":  _robust_spread(d["slope_ratios"]),
                            "t_half_mean":      np.nanmedian(d["t_halfs"]),
                            "t_half_std":       _robust_spread(d["t_halfs"]),
                            "spread":           np.nanstd(d["y_finals"]),
                            "tau_mean":         np.nanmedian(d["taus"]),
                            "tau_std":          _robust_spread(d["taus"]),
                            "auc_mean_mean":    np.nanmedian(d["auc_means"]),
                            "auc_mean_std":     _robust_spread(d["auc_means"]),
                            "envelope_area":    curve_envelope_area(group_curves_smoothed.get(v, [])),
                            "sens_slope_ratio": _sensitivity_slope(d["tkbs"], d["slope_ratios"]),
                            "sens_t_half":      _sensitivity_slope(d["tkbs"], d["t_halfs"]),
                            "sens_y_final":     _sensitivity_slope(d["tkbs"], d["y_finals"]),
                            "sens_tau":         _sensitivity_slope(d["tkbs"], d["taus"]),
                            "sens_auc_mean":    _sensitivity_slope(d["tkbs"], d["auc_means"]),
                            "unit":             d["unit"],
                        }

                    # --- plot 3 rows x 6 cols ---
                    fig_shape, axes = plt.subplots(3, 6, figsize=(24, 12), dpi=300)

                    metric_keys = [
                        ("slope_ratio",  "Early/late slope ratio",     "Slope ratio (exp→linear)"),
                        ("t_half",       "Time to half-max",           "Time to 50% of final value"),
                        ("y_final",      "Final value",                "Final value"),
                        ("tau",          "Relaxation time constant τ", "Growth time constant τ"),
                        ("auc_mean",     "Time-averaged value",        "Area under curve / duration"),
                    ]

                    # Row 1: per-run, x = TkB
                    for col, (mk, ylabel, title) in enumerate(metric_keys):
                        ax = axes[0][col]
                        # group by third_val for connecting
                        by_third = defaultdict(list)
                        for r in sorted(results, key=lambda r: r["tkb"]):
                            by_third[r["third_val"]].append(r)
                        for v, group in sorted(by_third.items()):
                            col_c = third_color_map.get(v, "gray")
                            style_c = third_style_map.get(v, "-")
                            group_sorted = sorted(group, key=lambda r: r["tkb"])
                            xs_r = [r["tkb"] for r in group_sorted]
                            ys_r = [r[mk] for r in group_sorted]
                            # raw per-run points: small and semi-transparent, since
                            # when several runs share a TkB (replicates) the trend
                            # line below connects through their *average*, which
                            # generally sits at a different y than any one of them
                            ax.scatter(xs_r, ys_r, color=col_c, s=35, zorder=2, alpha=0.55,
                                       edgecolors="white", linewidths=0.5)
                            if connect_dots_row1:
                                # connect through the per-TkB mean so replicate
                                # runs at the same TkB don't zig-zag the line
                                by_tkb = defaultdict(list)
                                for xv, yv in zip(xs_r, ys_r):
                                    by_tkb[xv].append(yv)
                                xs_line = sorted(by_tkb.keys())
                                ys_line = [np.nanmean(by_tkb[xv]) for xv in xs_line]
                                # the trend line gets its own bold, larger circle
                                # markers (distinct from the smaller/lighter raw-run
                                # dots above) so it's visually obvious these mark a
                                # per-TkB average, not a path through individual points
                                ax.plot(xs_line, ys_line, color=col_c, linewidth=1.3, linestyle=style_c,
                                        alpha=0.95, marker='o', markersize=7, zorder=4)
                        ax.set_xlabel("TkB")
                        ax.set_ylabel(ylabel)
                        ax.set_title(f"{title}\n(per run, colored by nm)")
                        apply_grid(ax, show_grid)
                    axes[0][5].axis("off")

                    # Row 2: per nm group, x = third_val, with error bars = std across TkB
                    agg_metrics = [
                        ("slope_ratio_mean", "slope_ratio_std", "Early/late slope ratio",    "Slope ratio (exp→linear)", True),
                        ("t_half_mean",      "t_half_std",      "Time to half-max",          "Time to 50% of final value", True),
                        ("spread",           None,              "Spread (std of final val)", "Spread across TkB values", False),
                        ("tau_mean",         "tau_std",         "Relaxation time constant τ","Growth time constant τ", True),
                        ("auc_mean_mean",    "auc_mean_std",    "Time-averaged value",       "Area under curve / duration", False),
                        ("envelope_area",    None,              "Envelope area",             "Area between highest & lowest curve", False),
                    ]
                    for col, (mk_mean, mk_std, ylabel, title, non_negative) in enumerate(agg_metrics):
                        ax = axes[1][col]
                        xs = sorted(group_summary.keys())
                        ys = [group_summary[v][mk_mean] for v in xs]
                        errs = [group_summary[v][mk_std] for v in xs] if mk_std else None
                        cols_g = [third_color_map.get(v, "gray") for v in xs]
                        units = [group_summary[v]["unit"] for v in xs]
                        xlabels = [f"{v:g}{u}" for v, u in zip(xs, units)]
                        for i, (xv, yv, col_c) in enumerate(zip(range(len(xs)), ys, cols_g)):
                            err = errs[i] if errs else None
                            yerr = err
                            # a symmetric error bar can dip below zero even with a
                            # robust median/MAD if the spread is large relative to
                            # the median -- clip it for quantities that can't
                            # physically be negative (a ratio, a time, tau)
                            if err is not None and non_negative and np.isfinite(yv) and np.isfinite(err):
                                yerr = [[min(err, yv)], [err]]
                            ax.errorbar(xv, yv, yerr=yerr, fmt='o', color=col_c, markersize=8,
                                        capsize=4, zorder=3)
                        if connect_dots_row2:
                            ax.plot(range(len(xs)), ys, color="gray", linewidth=1, alpha=0.5, zorder=2)
                        ax.set_xticks(range(len(xs)))
                        ax.set_xticklabels(xlabels, rotation=45, ha="right")
                        ax.set_xlabel(third_axis_label)
                        ax.set_ylabel(ylabel)
                        ax.set_title(f"{title}\n(per group, median ± MAD across TkB)")
                        apply_grid(ax, show_grid)

                    # Row 3: sensitivity d(metric)/d(TkB) per group -- one robust (Theil-Sen) slope per group
                    sens_metrics = [
                        ("sens_slope_ratio", "d(slope ratio)/d(TkB)", "Sensitivity: slope ratio vs TkB"),
                        ("sens_t_half",      "d(t_half)/d(TkB)",      "Sensitivity: time-to-half vs TkB"),
                        ("sens_y_final",     "d(final value)/d(TkB)", "Sensitivity: final value vs TkB"),
                        ("sens_tau",         "d(τ)/d(TkB)",           "Sensitivity: τ vs TkB"),
                        ("sens_auc_mean",    "d(time-avg)/d(TkB)",    "Sensitivity: time-averaged value vs TkB"),
                    ]
                    for col, (mk, ylabel, title) in enumerate(sens_metrics):
                        ax = axes[2][col]
                        xs = sorted(group_summary.keys())
                        ys = [group_summary[v][mk] for v in xs]
                        cols_g = [third_color_map.get(v, "gray") for v in xs]
                        units = [group_summary[v]["unit"] for v in xs]
                        xlabels = [f"{v:g}{u}" for v, u in zip(xs, units)]
                        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                        ax.bar(range(len(xs)), ys, color=cols_g)
                        ax.set_xticks(range(len(xs)))
                        ax.set_xticklabels(xlabels, rotation=45, ha="right")
                        ax.set_xlabel(third_axis_label)
                        ax.set_ylabel(ylabel)
                        ax.set_title(f"{title}\n(Theil-Sen slope per group)")
                        apply_grid(ax, show_grid)
                    axes[2][5].axis("off")

                    # shared legend
                    legend_els = [
                        Line2D([0], [0], marker='o', color=third_color_map[v],
                               linestyle=third_style_map.get(v, "-"),
                               markerfacecolor=third_color_map[v],
                               markersize=7, label=f"{v:g}{display_unit(group_summary[v]['unit'])}")
                        for v in sorted(group_summary.keys())
                    ]
                    axes[0][0].legend(handles=legend_els, fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig_shape)
                    save_figure_if_requested(
                        fig_shape,
                        base_name=f"multi_{metric_label}_curve_shape",
                        save_enabled=save_plots,
                        output_dir=output_dir,
                    )
                    plt.close(fig_shape)


        # --- Detect crystals: plot one curve per selected crystal type ---
        if metric_label == "Detect crystals":
            plt.figure()
            ax = plt.gca()

            post_curves = []  # define this once before the TauB grouping loop

            for kind in crystal_types_multi:
                # compute one series per kind (same cache system, because kwargs differ)
                series_kind = []
                missing = []

                for folder_path, tkb_val, tau_val, third_val, third_unit, *_ in [(r[0], r[1], r[2], r[3] if len(r)>3 else None, r[4] if len(r)>4 else None) for r in selected_runs]:
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
                        "third_val": third_val,
                        "third_unit": third_unit,
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

                        if x_range_custom and x_range_min is not None and x_range_max is not None:
                            mask = (x >= float(x_range_min)) & (x <= float(x_range_max))
                            x = x[mask]
                            y_grid = y_grid[mask]
                            if len(x) < 2:
                                continue

                        ax.plot(
                            x,
                            y_grid,
                            label = (f"{s['tkb']:g} Tkb (TauB {taub:g}) — {kind}" if show_taub_in_legend else f"{s['tkb']:g} Tkb — {kind}") + tag_suffix(s),
                            color = get_line_color(s),
                        )
                        # collect for post-analysis

                        # inside the loop where you have x and y_grid:
                        post_curves.append({
                            "run": s["run"],
                            "tkb": s["tkb"],
                            "taub": taub,
                            "kind": kind,
                            "third_val": s.get("third_val"),
                            "third_unit": s.get("third_unit"),
                            "x": np.asarray(x, float),
                            "y": np.asarray(y_grid, float),
                        })

            # after plotting + after post_curves is populated
            st.session_state[f"post_curves_{metric_label}"] = post_curves
            st.session_state["multi_cached"] = True

            ax.set_title("Crystal metric")
            ax.set_ylabel(f"{', '.join(crystal_types_multi)} Crystal Fraction" if show_ylabel else "")
            if y_range_custom and y_range_min is not None and y_range_max is not None:
                ax.set_ylim(float(y_range_min), float(y_range_max))
            ax.set_xlabel(("r (µm)" if metric_label == "Radial distribution function" else (
                "Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"
            ))if show_xlabel else "")
            apply_grid(ax, show_grid)
            if color_mode_multi in ("Group by 3rd param", "Group by 3rd param + gradient over TkB"):
                seen = {}
                third_groups = {}
                for r in selected_runs:
                    v = r[3] if len(r) > 3 else None
                    u = r[4] if len(r) > 4 else None
                    tkb = float(r[1])
                    if v is not None:
                        third_groups.setdefault(v, []).append({"tkb": tkb, "third_val": v, "third_unit": u})
                for v, group in third_groups.items():
                    group_sorted = sorted(group, key=lambda s: s["tkb"])
                    mid = group_sorted[len(group_sorted) // 2]
                    u = display_unit(mid.get("third_unit") or "")
                    seen[v] = (f"{v:g}{u}", get_line_color(mid))
                from matplotlib.lines import Line2D

                legend_handles = [
                    Line2D([0], [0], color=color, linewidth=2, label=label)
                    for v, (label, color) in sorted(seen.items())
                ]
                ax.legend(handles=legend_handles)
            else:
                ax.legend()

            fig = plt.gcf()
            apply_global_styling(fig, legend_on=legend_on_multi, legend_fontsize=legend_fontsize_multi,
                                 axis_fontsize=axis_fontsize_multi, title_on=title_on_multi,
                                 title_fontsize=title_fontsize_multi)
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
                ylabel=f"{', '.join(crystal_types_multi)} Crystal Fraction",
                x_axis_mode=x_axis_mode_multi,
                legend_on=legend_on_multi,
                legend_fontsize=legend_fontsize_multi,
                axis_fontsize=axis_fontsize_multi,
                title_on=title_on_multi,
                title_fontsize=title_fontsize_multi,
                show_grid=show_grid,
            )

            render_curve_shape_analysis()

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

            for folder_path, tkb_val, tau_val, third_val, third_unit, *_ in [(r[0], r[1], r[2], r[3] if len(r)>3 else None, r[4] if len(r)>4 else None) for r in selected_runs]:
                run_name = os.path.basename(folder_path)
                tkb_val = float(tkb_val)
                tau_val = float(tau_val)

                npz_path = rundata_npz_path(base_dir, dataset_tag, tkb_val, tau_val, metric_label, third_val, third_unit, run_tag(run_name))
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
                    "third_val": third_val,
                    "third_unit": third_unit,
                    "y": np.asarray(y, float),
                })


        if missing:
            st.warning("Missing data files for: " + ", ".join(missing))

        if not run:
            post_curves = st.session_state.get(f"post_curves_{metric_label}", [])
            if x_range_custom and x_range_min is not None and x_range_max is not None:
                clipped = []
                for c in post_curves:
                    mask = (np.asarray(c["x"], float) >= float(x_range_min)) & (np.asarray(c["x"], float) <= float(x_range_max))
                    cc = dict(c)
                    cc["x"] = np.asarray(c["x"], float)[mask]
                    cc["y"] = np.asarray(c["y"], float)[mask]
                    if len(cc["x"]) >= 2:
                        clipped.append(cc)
                post_curves = clipped
            if not post_curves:
                st.info(f"No cached curves for {metric_label}. Click Run once.")
                # still render shape expander below even if empty

            plt.figure()
            ax = plt.gca()
            for c in post_curves:
                ax.plot(
                    c["x"], c["y"],
                    label=run_label(c, show_taub_in_legend),
                    color=get_line_color(c),
                )


            ax.relim()
            ax.autoscale_view()
            ax.set_title(default_ylabel)
            ylabel_plot = default_ylabel
            if y_unit_multi.strip():
                ylabel_plot = f"{y_unit_multi.strip()}"
            ax.set_ylabel(ylabel_plot if show_ylabel else "")
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

            ax.set_xlabel(("r (µm)" if metric_label == "Radial distribution function" else (
                "Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"
            ))if show_xlabel else "")
            apply_grid(ax, show_grid)
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
                show_grid=show_grid,
)

            render_curve_shape_analysis()

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


        post_curves = []  # define this once before the TauB grouping loop
        # Group by TauB so same-TauB runs share identical x-axis length
        if metric_label == "Radial distribution function":
            taubs_sorted = sorted({float(s["taub"]) for s in series})
            for s in series:
                x = np.linspace(0.0, rdf_r_max, len(s["y"]))
                y_vals = np.asarray(s["y"], float)

                if x_range_custom and x_range_min is not None and x_range_max is not None:
                    mask = (x >= float(x_range_min)) & (x <= float(x_range_max))
                    x = x[mask]
                    y_vals = y_vals[mask]
                    if len(x) < 2:
                        continue

                col = get_line_color(s)
                ax.plot(
                    x, y_vals,
                    label=(f"{s['tkb']:g} Tkb (TauB {s['taub']:g})" if show_taub_in_legend else f"{s['tkb']:g} Tkb") + tag_suffix(s),
                    color=col,
                )
                post_curves.append({
                    "run": s["run"],
                    "tkb": s["tkb"],
                    "taub": s["taub"],
                    "third_val": s.get("third_val"),
                    "third_unit": s.get("third_unit"),
                    "x": np.asarray(x, float),
                    "y": np.asarray(y_vals, float),
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

                    if x_range_custom and x_range_min is not None and x_range_max is not None:
                        mask = (x >= float(x_range_min)) & (x <= float(x_range_max))
                        x = x[mask]
                        y_grid = y_grid[mask]
                        if len(x) < 2:
                            continue

                    ax.plot(
                        x,
                        y_grid,
                        label = run_label(s, show_taub_in_legend),
                        color=get_line_color(s),
                    )
                    # collect for post-analysis


                    # inside the loop where you have x and y_grid:
                    post_curves.append({
                        "run": s["run"],
                        "tkb": s["tkb"],
                        "taub": taub,
                        "third_val": s.get("third_val"),
                        "third_unit": s.get("third_unit"),
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
        ax.set_ylabel(ylabel_plot if show_ylabel else "")
        if y_range_custom and y_range_min is not None and y_range_max is not None:
            ax.set_ylim(float(y_range_min), float(y_range_max))
        ax.set_xlabel(
            ("r (µm)" if metric_label == "Radial distribution function" else
             ("Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)"))
            if show_xlabel else ""
        )
        apply_grid(ax, show_grid)
        if color_mode_multi in ("Group by 3rd param", "Group by 3rd param + gradient over TkB"):
            # one legend entry per third_val group
            seen = {}
            for s in sorted(series, key=lambda s: s["tkb"]):
                v = s.get("third_val")
                u = display_unit(s.get("third_unit") or "")
                key = f"{v:g}{u}" if v is not None else "unknown"
                # always overwrite so last (highest TkB = darkest) wins
                color = get_line_color(s)
                if v is not None:
                    seen[v] = (key, color)
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], color=color, linewidth=2, label=label)
                for v, (label, color) in sorted(seen.items())
            ]
            ax.legend(handles=legend_handles)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)


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
            show_grid=show_grid,
        )


        render_curve_shape_analysis()

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
    show_ylabel = ui.get("show_ylabel", False)
    show_xlabel = ui.get("show_xlabel", True)
    show_grid = ui.get("show_grid", True)
    x_range_custom = ui.get("x_range_custom", False)
    x_range_min = ui.get("x_range_min", None)
    x_range_max = ui.get("x_range_max", None)

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

    c1, c2 = st.columns(2)

    with c1:
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

    with c2:
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
        value=f"",
        key="expr_calc_title",
    )

    # ---- Color mode UI for Function Mixer ---

    color_mode_mix = st.sidebar.radio(
        "Curve colors",
        ["Matplotlib default", "Gradient over TkB"],
        key="mix_color_mode",
    )

    if color_mode_mix == "Gradient over TkB":
        st.sidebar.write("Gradient colors:")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            color_start_mix = st.color_picker(
                "Start",
                value="#D3D3D3",
                key="mix_color_start",
            )

        with col2:
            color_end_mix = st.color_picker(
                "End",
                value="#000000",
                key="mix_color_end",
            )
    else:
        color_start_mix = "#D3D3D3"
        color_end_mix = "#000000"

    use_gradient = color_mode_mix == "Gradient over TkB"
    tkb_to_color = {}

    if use_gradient:
        import matplotlib.colors as mcolors

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "tkb_grad_mix",
            [color_start_mix, color_end_mix],
        )

        tkbs_sorted = sorted({float(r[1]) for r in selected_runs})

        if len (tkbs_sorted) == 1:
            tkb_to_color[tkbs_sorted[0]] = cmap(0.5)
        else:
            for i, tkb in enumerate(tkbs_sorted):
                frac = i / (len(tkbs_sorted) - 1)
                tkb_to_color[tkb] = cmap(frac)


    def get_line_color(tkb):
        if not use_gradient:
            return None
        return tkb_to_color.get(float(tkb), None)

    mix_norm_mode = st.selectbox(
        "Normalize mixer output",
        ["None", "norm_series_max"],
        key="mix_norm_mode",
    )

    plot_expr = st.button("Plot", key="expr_calc_plot")

    st.caption("Allowed functions: " + ", ".join(sorted(_ALLOWED_FUNCS.keys())))

    # ✅ If user clicked Run post-analysis (or any rerun) and we already have cached curves, redraw from cache
    if (not plot_expr) and st.session_state.get("post_curves_mixer"):
        curves = [dict(c, y=np.asarray(c["y"], float).copy()) for c in st.session_state["post_curves_mixer"]]
        meta = st.session_state.get("mixer_meta", {})

        plt.close("all")
        plt.figure()
        ax = plt.gca()

        # normalize ALL mixer curves by one shared series max amplitude
        if mix_norm_mode == "norm_series_max" and curves:
            vals = []
            for c in curves:
                y_tmp = np.asarray(c["y"], float)
                if y_tmp.size > 0 and np.any(np.isfinite(y_tmp)):
                    amp = np.nanmax(y_tmp) - np.nanmin(y_tmp)
                    vals.append(amp)

            if vals:
                series_amp = float(np.nanmax(vals))
                if np.isfinite(series_amp) and series_amp != 0:
                    for c in curves:
                        c["y"] = np.asarray(c["y"], float) / series_amp

        # plot ALL curves, not only the last one
        for c in curves:
            ax.plot(
                c["x"],
                c["y"],
                label=(f"{c['tkb']:g} Tkb (TauB {c['taub']:g})" if show_taub_in_legend else f"{c['tkb']:g} Tkb") + tag_suffix(c),
                color=(c["tkb"]),
            )

        ax.set_title(meta.get("title", "Derived expression"))
        title = meta.get("title", "Derived expression")
        ylabel_plot = title
        if y_unit_multi.strip():
            ylabel_plot = f"{title} [{y_unit_multi.strip()}]"
        ax.set_ylabel(ylabel_plot if show_ylabel else "")

        if meta.get("plot_fft", False):
            ax.set_xlabel("Frequency (Hz)" if show_xlabel else "")
        else:
            ax.set_xlabel(("Time (τB)" if meta.get("x_axis_mode") == "Brownian time τ_B" else "Time (s)")if show_xlabel else "")

        apply_grid(ax, show_grid)
        ax.legend()

        fig = plt.gcf()
        apply_global_styling(
            fig,
            legend_on=legend_on_multi,
            legend_fontsize=legend_fontsize_multi,
            axis_fontsize=axis_fontsize_multi,
            title_on=title_on_multi,
            title_fontsize=title_fontsize_multi,
            show_grid=show_grid,
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
            show_grid=show_grid,
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

            post_curves.append({
                "run": run_name,
                "tkb": float(a_run["tkb"]),
                "taub": float(taub_a),
                "x": np.asarray(x, float),
                "y": np.asarray(y_out, float),
            })

            # # ✅ IMPORTANT: append INSIDE the loop (one entry per run)
            # st.session_state["post_curves_mixer"].append({
            #     "run": run_name,
            #     "tkb": float(a_run["tkb"]),
            #     "taub": float(a_run["taub"]),
            #     "x": np.asarray(x, float),
            #     "y": np.asarray(y_out, float),
            # })

        if mix_norm_mode == "norm_series_max" and post_curves:
            vals = []

            for c in post_curves:
                y_tmp = np.asarray(c["y"], float)
                if y_tmp.size > 0 and np.any(np.isfinite(y_tmp)):
                    vals.append(np.nanmax(y_tmp))

            if vals:
                series_max = np.nanmax(vals)

                if np.isfinite(series_max) and series_max != 0:
                    for c in post_curves:
                        c["y"] = np.asarray(c["y"], float) / series_max

        for c in post_curves:
            ax.plot(
                c["x"],
                c["y"],
                label=f"{c['tkb']:g} Tkb (TauB {c['taub']:g})"
                if show_taub_in_legend
                else f"{c['tkb']:g} Tkb",
                color=(c["tkb"]),
            )
        ax.set_title(calc_title)
        # y-label for mixer (expression)
        ylabel_plot = calc_title
        if y_unit_multi.strip():
            ylabel_plot = f"{y_unit_multi.strip()}"
        ax.set_ylabel(ylabel_plot if show_ylabel else "")

        # ✅ overwrite cache with ALL curves
        st.session_state["post_curves_mixer"] = post_curves
        st.session_state["mixer_meta"] = {
            "title": calc_title,
            "plot_fft": plot_fft,
            "x_axis_mode": x_axis_mode_multi,
        }

        # ✅ label axis correctly
        if plot_fft:
            ax.set_xlabel("Frequency (Hz)" if show_xlabel else "")
        else:
            ax.set_xlabel(("Time (τB)" if x_axis_mode_multi == "Brownian time τ_B" else "Time (s)")if show_xlabel else "")

        apply_grid(ax, show_grid)
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
            show_grid=show_grid,
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
    phase_folders = {}          # TkB -> the run folder shown in that row
    if base_dir and os.path.isdir(base_dir):
        try:
            import run_filter as rf
            _names = sorted(fn for fn in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, fn)))
            _metas = [rf.meta(fn, os.path.basename(base_dir.rstrip("\\/"))) for fn in _names]
            _keep = rf.widgets(_metas, key=f"phase__{base_dir}", container=st.sidebar,
                               title="Filter runs (one run per TkB row)")
            by_tkb = {}
            for fn, m, k in zip(_names, _metas, _keep):
                if k and m is not None:
                    by_tkb.setdefault(float(m["A"]), []).append(fn)
            available_tkbs_phase = sorted(by_tkb)
            for tkb, fns in by_tkb.items():
                if len(fns) > 1:
                    pick = st.sidebar.selectbox(f"Run for TkB = {tkb:g} ({len(fns)} pass the filter)", fns,
                                                key=f"phase_pick__{base_dir}__{tkb:g}")
                else:
                    pick = fns[0]
                phase_folders[tkb] = os.path.join(base_dir, pick)
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
        "Show legends in cells", value=False, key="legend_on_phase"
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
        folders=phase_folders,
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
            r"^([0-9]+(?:\.[0-9]+)?)Tkb[ _]*([0-9]+(?:\.[0-9]+)?)TauB(?:[ _]*[0-9]+(?:\.[0-9]+)?(?:nm|nn|dn|F))?(?:_[A-Za-z]+[0-9.]*)*$",
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


    import run_filter as rf
    _keep = rf.widgets(rf.metas_from_paths([r[0] for r in available_runs]), key=f"summary__{base_dir}",
                       container=st.sidebar)
    _summary_opts = [r for r, k in zip(available_runs, _keep) if k]
    if st.session_state.get("summary_runs_sig") != tuple(_keep):     # new filter: select everything it lets through
        st.session_state["summary_runs"] = _summary_opts
        st.session_state["summary_runs_sig"] = tuple(_keep)
    selected_runs = st.sidebar.multiselect(
        "Select runs:",
        options=_summary_opts,
        key="summary_runs",
        format_func=lambda r: f"{r[1]:g} Tkb (TauB: {r[2]:g})  {os.path.basename(r[0])}",
    )


    st.sidebar.header("4. Metric curve to summarize")
    metric_label = st.sidebar.selectbox("Metric:", metric_options, key="summary_metric_label")

    # ---- common params ----
    with st.sidebar.expander("5. Common parameters", expanded=False):

        cluster_params = render_cluster_params_widgets("summary")
        cluster_mode_summary = cluster_params["mode"]
        cluster_eps_summary = cluster_params["eps"]
        cluster_min_samples_summary = cluster_params["min_samples"]
        cluster_min_cluster_size_summary = cluster_params["min_cluster_size"]

    st.sidebar.header("6. Reduction (scalar from curve)")
    reducer = st.sidebar.selectbox(
        "Scalar to compute from y(t):",
        ["max", "min", "mean", "last", "slope (linear fit)"],
        key="summary_reducer",
    )

    st.sidebar.header("7. Plot against")
    x_choice = st.sidebar.selectbox("X-axis:", ["TkB", "TauB"], key="summary_x_choice")


    # ✅ ADD: Plot style and fitting options
    st.sidebar.subheader("Plot style & fitting")

    show_grid = st.sidebar.checkbox(
        "Show grid",
        value=False,
        key="summary_show_grid",
    )

    plot_style = st.sidebar.radio(
        "Plot style:",
        ["Connected (lines + dots)", "Scatter only"],
        key="summary_plot_style",
    )

    fit_function = st.sidebar.selectbox(
        "Fit curve (optional):",
        [
            "None",
            "Linear trend",
            "Polynomial degree n",
            "Robust linear trend",
            "Saturating exponential decay",
            "Inverse decay to plateau",
            "Logarithmic decay",
            "PCHIP smooth trend",
            "Smoothing spline",
        ],
        key="summary_fit_func",
    )
    fit_scope = st.sidebar.radio(
        "Fit scope:",
        ["Fit each group separately", "Fit all points together"],
        key="summary_fit_scope",
    )

    poly_degree = None
    if fit_function == "Polynomial degree n":
        poly_degree = st.sidebar.slider(
            "Polynomial degree n:",
            min_value=1,
            max_value=8,
            value=2,
            step=1,
            key="summary_poly_degree",
        )

    show_fit_quality = st.sidebar.checkbox(
        "Show fit quality (R²)",
        value=True,
        key="summary_show_fit_quality",
    )


    if fit_function != "None":
        fit_color = st.sidebar.color_picker(
            "Fit curve color:",
            value="#FF0000",
            key="summary_fit_color",
        )


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
        rows = cache.get("rows", None)
        x_choice = cache["x_choice"]
        xlabel_pa = cache["pa_xlabel"]
        plot_style = cache.get("plot_style", "Connected (lines + dots)")
        fit_function = cache.get("fit_function", "None")

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
            x_group = xvals[mask]
            y_group = yvals[mask]

            if plot_style == "Connected (lines + dots)":
                ax.plot(x_group, y_group, marker="o", linestyle="-", label=label_fmt(g), linewidth=2, markersize=8)
            else:
                ax.scatter(x_group, y_group, label=label_fmt(g), s=50, alpha=1)

        ax.set_title(cache["title"])
        ax.set_xlabel(cache["xlabel"])
        ax.set_ylabel(cache["ylabel"])
        apply_grid(ax, show_grid)
        ax.legend()

        fig = plt.gcf()
        apply_global_styling(fig, legend_on=True, legend_fontsize=14, axis_fontsize=16, title_on=True,
                             title_fontsize=18)
        st.pyplot(fig)
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
        for run_tuple in selected_runs:
            folder_path = run_tuple[0]
            tkb_val = float(run_tuple[1])
            tau_val = float(run_tuple[2])
            third_val = run_tuple[3] if len(run_tuple) > 3 else None
            third_unit = run_tuple[4] if len(run_tuple) > 4 else None
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
                cluster_mode=cluster_mode_summary,
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
        "rows": rows,
        "x_choice": x_choice,
        "pa_key": pa_key,
        "pa_xlabel": xlabel_pa,
        "pa_ylabel": f"{reducer}({metric_label})",
        "plot_style": plot_style,
        "fit_function": fit_function,
    }

    # ----------------------------
    # NOW build the figure using the CURRENT widget values
    # ----------------------------
    plt.close("all")
    plt.figure()
    ax = plt.gca()


    # ✅ Helper function for fitting
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline
    from scipy.stats import theilslopes
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.interpolate import PchipInterpolator, UnivariateSpline


    from scipy.optimize import curve_fit
    from scipy.interpolate import PchipInterpolator, UnivariateSpline
    from scipy.stats import theilslopes
    import numpy as np


    def _prepare_fit_xy(x_data, y_data):
        x = np.asarray(x_data, float)
        y = np.asarray(y_data, float)

        valid = np.isfinite(x) & np.isfinite(y)


        x = x[valid]
        y = y[valid]

        if len(x) < 2:
            return None, None

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Merge repeated x-values by averaging y-values
        unique_x = []
        unique_y = []

        for xv in np.unique(x):
            mask = x == xv
            unique_x.append(xv)
            unique_y.append(np.nanmean(y[mask]))

        return np.asarray(unique_x, float), np.asarray(unique_y, float)


    def _r2_score(y_true, y_pred):
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)

        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)

        if ss_tot == 0:
            return np.nan

        return 1.0 - ss_res / ss_tot


    def fit_and_plot(
            x_data,
            y_data,
            fit_type,
            color,
            ax,
            label_suffix="",
            poly_degree=None,
            show_fit_quality=True,
    ):
        if fit_type == "None":
            return

        x_clean, y_clean = _prepare_fit_xy(
            x_data,
            y_data,
        )

        if x_clean is None or len(x_clean) < 2:
            st.info(f"{fit_type} skipped {label_suffix}: not enough valid points.")
            return

        n_points = len(x_clean)

        if np.min(x_clean) == np.max(x_clean):
            st.info(f"{fit_type} skipped {label_suffix}: all x-values are identical.")
            return

        x_smooth = np.linspace(np.min(x_clean), np.max(x_clean), 300)

        try:
            if fit_type == "Linear trend":
                coeffs = np.polyfit(x_clean, y_clean, 1)
                y_fit = np.polyval(coeffs, x_smooth)
                y_pred = np.polyval(coeffs, x_clean)
                fit_name = "Linear"

            elif fit_type == "Polynomial degree n":
                degree = int(poly_degree or 2)

                max_safe_degree = n_points - 2

                if degree >= n_points:
                    st.warning(
                        f"Polynomial degree {degree} skipped {label_suffix}: "
                        f"degree must be smaller than number of points ({n_points})."
                    )
                    return

                if degree > max_safe_degree:
                    st.warning(
                        f"Polynomial degree {degree} may overfit {label_suffix}. "
                        f"You have {n_points} points; safer degree is ≤ {max_safe_degree}."
                    )

                coeffs = np.polyfit(x_clean, y_clean, degree)
                y_fit = np.polyval(coeffs, x_smooth)
                y_pred = np.polyval(coeffs, x_clean)
                fit_name = f"Poly n={degree}"

            elif fit_type == "Robust linear trend":
                if n_points < 3:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 3 points.")
                    return

                slope, intercept, _, _ = theilslopes(y_clean, x_clean)
                y_fit = intercept + slope * x_smooth
                y_pred = intercept + slope * x_clean
                fit_name = "Robust linear"

            elif fit_type == "Saturating exponential decay":
                if n_points < 3:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 3 points.")
                    return

                def model(x, y_inf, A, k):
                    return y_inf + A * np.exp(-k * x)

                y_inf0 = float(np.nanmin(y_clean))
                A0 = float(np.nanmax(y_clean) - np.nanmin(y_clean))
                k0 = 1.0

                popt, _ = curve_fit(
                    model,
                    x_clean,
                    y_clean,
                    p0=[y_inf0, A0, k0],
                    bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                    maxfev=50000,
                )

                y_fit = model(x_smooth, *popt)
                y_pred = model(x_clean, *popt)
                fit_name = "Saturating exp."

            elif fit_type == "Inverse decay to plateau":
                if n_points < 3:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 3 points.")
                    return

                def model(x, y_inf, A, x0):
                    return y_inf + A / (x + x0)

                y_inf0 = float(np.nanmin(y_clean))
                A0 = float((np.nanmax(y_clean) - np.nanmin(y_clean)) * (np.nanmax(x_clean) + 1.0))
                x0_0 = 1.0

                popt, _ = curve_fit(
                    model,
                    x_clean,
                    y_clean,
                    p0=[y_inf0, A0, x0_0],
                    bounds=([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf]),
                    maxfev=50000,
                )

                y_fit = model(x_smooth, *popt)
                y_pred = model(x_clean, *popt)
                fit_name = "Inverse decay"

            elif fit_type == "Logarithmic decay":
                if n_points < 3:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 3 points.")
                    return

                x_shift = x_clean - np.min(x_clean) + 1.0
                x_smooth_shift = x_smooth - np.min(x_clean) + 1.0

                coeffs = np.polyfit(np.log(x_shift), y_clean, 1)
                y_fit = coeffs[1] + coeffs[0] * np.log(x_smooth_shift)
                y_pred = coeffs[1] + coeffs[0] * np.log(x_shift)
                fit_name = "Log decay"

            elif fit_type == "PCHIP smooth trend":
                if n_points < 2:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 2 points.")
                    return

                interpolator = PchipInterpolator(x_clean, y_clean)
                y_fit = interpolator(x_smooth)
                y_pred = interpolator(x_clean)
                fit_name = "PCHIP"

            elif fit_type == "Smoothing spline":
                if n_points < 4:
                    st.info(f"{fit_type} skipped {label_suffix}: needs at least 4 points.")
                    return

                s_val = 0.2 * n_points * np.nanvar(y_clean)
                spline = UnivariateSpline(x_clean, y_clean, s=s_val)
                y_fit = spline(x_smooth)
                y_pred = spline(x_clean)
                fit_name = "Spline"

            else:
                return

            r2 = _r2_score(y_clean, y_pred)

            if show_fit_quality and np.isfinite(r2):
                label = f"{fit_name} {label_suffix}, R²={r2:.3f}"
            else:
                label = f"{fit_name} {label_suffix}"

            ax.plot(
                x_smooth,
                y_fit,
                color=color,
                linestyle="--",
                linewidth=2.5,
                label=label,
                alpha=0.9,
            )

        except Exception as e:
            st.warning(f"{fit_type} failed {label_suffix}: {e}")

    # measured curves
    for g in sorted(set(group_key)):
        mask = group_key == g
        x_group = xvals[mask]
        y_group = yvals[mask]
        label_group = label_fmt(g)

        order = np.argsort(x_group)
        x_group = x_group[order]
        y_group = y_group[order]

        if plot_style == "Connected (lines + dots)":
            ax.plot(
                x_group,
                y_group,
                marker="o",
                linestyle="-",
                label=label_group,
                linewidth=2,
                markersize=8,
            )
        else:
            ax.scatter(
                x_group,
                y_group,
                label=label_group,
                s=50,
                alpha=1,
            )

        # fit each TauB / TkB group separately
        if fit_function != "None" and fit_scope == "Fit each group separately":
            fit_and_plot(
                x_group,
                y_group,
                fit_function,
                fit_color,
                ax,
                label_suffix=f"({label_group})",
                poly_degree=poly_degree,
                show_fit_quality=show_fit_quality,
            )

    # ✅ ADD THIS HERE: after the group loop, before finalize
    if fit_function != "None" and fit_scope == "Fit all points together":
        fit_and_plot(
            xvals,
            yvals,
            fit_function,
            fit_color,
            ax,
            label_suffix="(all data)",
            poly_degree=poly_degree,
            show_fit_quality=show_fit_quality,
        )

    # finalize
    ax.set_title(f"{reducer}({metric_label}) vs {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{reducer}({metric_label})")
    apply_grid(ax, show_grid)
    ax.legend()


    fig = plt.gcf()
    apply_global_styling(fig, legend_on=True, legend_fontsize=14, axis_fontsize=16, title_on=True,
                         title_fontsize=18)

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

    st.stop()

# ---------------------------------------------------------------------
# REGIME STATE MODEL  (label runs, fit fluid/network boundary, suggest runs)
# ---------------------------------------------------------------------
elif plot_mode == "Regime state model":

    @st.cache_data(show_spinner=False)
    def _rsm_observables(root, regimes, rc, box, common_t, lag, times=(), stamp=()):
        bar = st.progress(0.0, text="Reading runs (only new runs and new times are computed; saved as it goes)…")
        out = rsm.compute_observables(root, regimes, rc=rc, box=box, common_time_tauB=common_t, lag_tauB=lag,
                                      map_times_tauB=times, progress=lambda f, s: bar.progress(min(f, 1.0), text=s))
        bar.empty()
        return out

    @st.cache_data(show_spinner=False)
    def _rsm_checks(root, regimes, stamp=()):
        return rsm.data_checks(root, regimes)

    def _rsm_stamp(root, regimes):
        """Run folders and their newest change, so new or re-run runs are picked up without a restart."""
        out = []
        for reg in regimes:
            d = os.path.join(root, reg)
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    f = os.path.join(d, n)
                    if os.path.isdir(f):
                        out.append((reg, n, int(os.path.getmtime(f))))
        return tuple(out)

    st.sidebar.header("2. Data folder")
    rsm_source = st.sidebar.radio(
        "Folder with the regime folders (NAF / ARF / AAF / ASF = Regime I–IV):",
        ["Source root", "Custom path"],
        key="rsm_folder_source",
    )
    if rsm_source == "Source root":
        rsm_root = source_root
    else:
        rsm_root = st.sidebar.text_input("Custom folder path:", value="", key="rsm_folder_custom")
    rsm_root = rsm_root.strip().strip('"').strip("'")

    rsm_found = [r for r in rsm.REGIMES if rsm_root and os.path.isdir(os.path.join(rsm_root, r))]
    rsm_regimes = st.sidebar.multiselect(
        "3. Regimes to include:",
        rsm_found,
        default=rsm_found,
        format_func=lambda r: rsm.regime_label(r, long=True),
        key=f"rsm_regimes__{rsm_root}",   # per folder, so a newly typed folder starts with all regimes
    )

    with st.sidebar.expander("4. Analysis settings", expanded=False):
        rsm_rc = st.number_input("Bond cutoff (µm)", 1.0, 10.0, 2.9, 0.1, key="rsm_rc",
                                 help="Two particles count as bonded when their centres are closer than this. "
                                      "Default: first minimum of g(r) after the 2.2 µm contact peak.")
        rsm_box = st.number_input("Box size (µm)", 10.0, 500.0, 50.0, 5.0, key="rsm_box")
        rsm_tc = st.number_input("Common evaluation time for the boundary model (τB)", 0.1, 50.0, 1.0, 0.25,
                                 key="rsm_tc",
                                 help="Runs of different length are compared at this time.")
        rsm_lag = st.number_input("Bond-persistence / MSD lag (τB)", 0.05, 5.0, 0.5, 0.05, key="rsm_lag")
        t1, t2 = st.columns(2)
        rsm_tstep = t1.number_input("Map time step (τB)", 0.05, 5.0, 0.25, 0.05, key="rsm_map_tstep",
                                    help="Every run is evaluated every this many τB (as far as it reaches), for the "
                                         "map's time slider and the animations. Smaller = smoother, first load slower.")
        rsm_tmax = t2.number_input("…up to (τB)", 1.0, 100.0, 10.0, 1.0, key="rsm_map_tmax")
        rsm_times_txt = st.text_input("Extra map times (τB)", "15, 20, 25, 30", key="rsm_map_times_extra",
                                      help="Added to the regular steps, e.g. for long runs.")
        rsm_times_txt = ", ".join(f"{v:g}" for v in np.round(np.arange(rsm_tstep, rsm_tmax + 1e-9, rsm_tstep), 4)) \
            + ", " + rsm_times_txt

    with st.sidebar.expander("5. State thresholds", expanded=True):
        rsm_th = rsm.Thresholds(
            z_fluid=st.slider("Fluid if mean coordination below", 0.5, 5.0, 2.5, 0.1, key="rsm_zf"),
            psi_order=st.slider("Ordered if ψ6 at least", 0.5, 1.0, 0.8, 0.01, key="rsm_psi"),
            smax_span=st.slider("Network if largest cluster fraction at least", 0.3, 1.0, 0.8, 0.05, key="rsm_smax"),
        )

    st.header("Regime state model")
    st.caption(
        "Labels every run with an operational state (fluid, ordered/disordered network or clusters), "
        "fits a Gaussian-process classifier for the fluid / network boundary, and suggests the "
        "simulations that would sharpen it. Thresholds are provisional - change them in the sidebar."
    )

    if not rsm_root or not os.path.isdir(rsm_root):
        st.info("Choose a folder that contains the regime folders NAF / ARF / AAF / ASF (Regime I–IV).")
        st.stop()
    if not rsm_regimes:
        st.warning("No regime folders selected (or none found in this folder).")
        st.stop()

    # new runs (e.g. from a running batch) are only loaded when asked, so the page does not reload every few minutes
    _sk = f"rsm_stamp__{rsm_root}__{'-'.join(rsm_regimes)}"
    _now = _rsm_stamp(rsm_root, tuple(rsm_regimes))
    if _sk not in st.session_state:
        st.session_state[_sk] = _now
    _n_new = len(set(_now) - set(st.session_state[_sk]))
    if _n_new and st.sidebar.button(f"Load {_n_new} new / changed run(s)", key="rsm_load_new", type="primary"):
        st.session_state[_sk] = _now
    if True:
        try:
            rsm_obs = _rsm_observables(rsm_root, tuple(rsm_regimes), float(rsm_rc), float(rsm_box),
                                       float(rsm_tc), float(rsm_lag), tuple(float(v) for v in re.findall(r"[0-9.]+", rsm_times_txt) if v.strip(".") ),
                                       stamp=st.session_state[_sk])
        except Exception as e:
            st.error(f"Could not compute observables: {e}")
            st.stop()
    if rsm_obs.empty:
        st.warning("No runs found. Folder names must look like 3Tkb_1TauB_2000nm, 50Tkb_0.75TauB_1000F or 5Tkb_2TauB.")
        st.stop()
    import run_filter as rf
    st.sidebar.caption("6. Runs to use (all runs, Python and COMSOL, to start with; narrow by any parameter):")
    rsm_obs = rf.filter_frame(rsm_obs, key=f"rsm2__{rsm_root}", container=st.sidebar)
    if rsm_obs.empty:
        st.warning("No runs pass the filter.")
        st.stop()
    rsm_df = rsm.drop_repeated_seeds(rsm.label_runs(rsm_obs, rsm_th))   # a 1 τB run = first τB of the 10 τB run

    tab_states, tab_map, tab_settle, tab_trans, tab_next, tab_checks, tab_oop = st.tabs(
        ["States", "Boundary map", "State settling", "Transitions along A", "Suggested runs", "Data checks",
         "Out-of-plane load"]
    )

    with tab_states:
        rsm_end = rsm_df[rsm_df["eval"] == "end"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Runs", len(rsm_end))
        c2.metric("Non-fluid", int((rsm_end.state != "fluid").sum()))
        c3.metric("Near a threshold", int(rsm_end.near_threshold.sum()))
        st.subheader("Runs per regime and state (at the end of each run)")
        st.dataframe(pd.crosstab(rsm_end.regime, rsm_end.state))
        show_cols = ["regime", "folder", "name", "potential", "q", "phi_p", "N", "box", "phi_area", "pd", "dim", "eval", "t_eval_tauB", "state", "z_mean", "Smax", "spans", "psi6",
                     "bond_persist", "Rg_ratio", "concentrated", "restructuring", "near_threshold"]
        st.subheader("All runs")
        st.dataframe(rsm_df[[c for c in show_cols if c in rsm_df]].round(3), hide_index=True)
        st.download_button("Download labelled runs (CSV)", rsm_df.to_csv(index=False).encode(),
                           "labelled_runs.csv", "text/csv", key="rsm_dl_states")

    rsm_model, rsm_plan = None, None
    with tab_map:
        map_regs = [r for r in rsm_regimes if r in set(rsm_df.regime) and rsm.map_axes(rsm_df, r)]
        if not map_regs:
            st.info("No selected regime has runs that vary in a parameter or in time.")
        else:
            m1, m2, m3 = st.columns([1.2, 1.2, 1])
            map_regime = m1.selectbox("Regime", map_regs, index=map_regs.index("ARF") if "ARF" in map_regs else 0,
                                      format_func=lambda r: rsm.regime_label(r, long=True), key="rsm_map_regime")
            axes_ = rsm.map_axes(rsm_df, map_regime)
            map_param = m2.selectbox("x axis", axes_, format_func=lambda p: rsm.axis_label(p).replace("$", "").replace("_", ""),
                                     key=f"rsm_map_x_{map_regime}")
            pots = sorted(set(rsm_df.loc[rsm_df.regime == map_regime, "potential"].fillna("LJ")))
            map_pot = m3.radio("Attraction", pots, index=pots.index("LJ") if "LJ" in pots else 0, horizontal=True,
                               key="rsm_map_pot",
                               help="One map per attraction; 'LJ vs AO boundaries' below shows both.") if len(pots) > 1 else pots[0]
            # AO runs at different polymer size ratios q are different physics: one q per map
            rsm_dfm = rsm_df
            _ao_q = sorted(set(rsm_df.loc[(rsm_df.regime == map_regime) & (rsm_df.potential == "AO"), "q"].dropna()))
            map_q = None
            if len(_ao_q) > 1:
                _cnt = rsm_df[(rsm_df.regime == map_regime) & (rsm_df.potential == "AO")].groupby("q").name.nunique()
                map_q = st.radio("AO size ratio q (one per map; LJ is unaffected)", _ao_q,
                                 index=_ao_q.index(_cnt.idxmax()), horizontal=True, key=f"rsm_map_q_{map_regime}",
                                 format_func=lambda v: f"q = {v:g} ({_cnt.get(v, 0)} runs)",
                                 help="AO attraction is matched to each A by B2*, but its range is 2·q·a, so maps "
                                      "at different q answer different questions. The experiment sits at q ≈ 0.03–0.08.")
                rsm_dfm = rsm_df[(rsm_df.potential != "AO") | np.isclose(rsm_df.q, map_q)]
            # particle number / box: one system per map (except on the area-fraction axis)
            if "N" in rsm_dfm and map_param not in ("N", "phi_area"):
                _sys = (rsm_dfm[rsm_dfm.regime == map_regime][["N", "box"]].dropna().drop_duplicates()
                        .sort_values(["N", "box"]).itertuples(index=False, name=None))
                _sys = [(int(n_), float(b_)) for n_, b_ in _sys]
                if len(_sys) > 1:
                    _d0 = _sys.index((144, 50.0)) if (144, 50.0) in _sys else 0
                    _cntN = rsm_dfm[rsm_dfm.regime == map_regime].groupby(["N", "box"]).name.nunique()
                    map_sys = st.radio("System (particle number, box)", _sys, index=_d0, horizontal=True,
                                       key=f"rsm_map_N_{map_regime}",
                                       format_func=lambda v: (f"N = {v[0]}, {v[1]:g} µm "
                                                              f"(area fraction {v[0] * np.pi / v[1] ** 2:.2f}; "
                                                              f"{_cntN.get(v, 0)} runs)"),
                                       help="Runs at other particle numbers or densities (folders like NAF_N576, "
                                            "NAF_density). Use the x axis 'area fraction' to see them together.")
                    rsm_dfm = rsm_dfm[(rsm_dfm.N == map_sys[0]) & np.isclose(rsm_dfm.box, map_sys[1])]
            times_ = rsm.map_times(rsm_dfm, map_regime, map_pot, map_param)
            if not times_:
                st.info(f"No {map_pot} runs in {rsm.regime_label(map_regime)} with the current filter.")
                times_ = [float(rsm_tc)]
            if map_param != "t_map":
                t0 = min(times_, key=lambda v: abs(v - float(rsm_tc))) if times_ else float(rsm_tc)
                map_t = st.select_slider("Time of the map (τ_B)", times_, value=t0, key=f"rsm_map_t_{map_regime}",
                                         help="Runs are evaluated at the times listed in the sidebar (4. Analysis "
                                              "settings); a run only counts at times it reaches.") if len(times_) > 1 else t0
            else:
                map_t = float(rsm_tc)
            views = ["Phase diagram (all states)", "States + boundary model", "Every run as its own dot", "Heat map: share not fluid",
                     "Heat map: measured value", "3D elevation", "LJ vs AO boundaries"]
            if map_param != "t_map":
                views.append("Play over time")
            v1, v2 = st.columns([2.2, 1.2])
            view = v1.radio("Show", views, horizontal=True, key="rsm_map_view")
            metric_opts = ["z_mean", "psi6", "Smax", "bond_persist", "Rg_ratio", "mobility"]
            metric = None
            if view in ("Heat map: measured value", "3D elevation", "Play over time"):
                choices = (["proba", "nonfluid"] if view == "3D elevation" else
                           ["nonfluid", "states"] if view == "Play over time" else []) + metric_opts
                metric = v2.selectbox("Measure", choices, format_func=lambda k_: {"states": "states (dots)"}.get(
                    k_, rsm.METRIC_LABELS.get(k_, k_)), key=f"rsm_map_metric_{view}")
            metric_cols = [c for c in ("z_mean", "psi6", "Smax", "bond_persist", "Rg_ratio", "mobility") if c in rsm_df]
            with st.expander("Boundary and look", expanded=False):
                b1, b2, b3 = st.columns(3)
                outcome = b1.selectbox("Boundary between", list(rsm.OUTCOMES), format_func=lambda k_: rsm.OUTCOMES[k_]["label"],
                                       key="rsm_map_outcome")
                o_metric = o_thr = None
                if outcome == "custom":
                    o_metric = b2.selectbox("measure", metric_cols, format_func=lambda k_: rsm.METRIC_LABELS.get(k_, k_),
                                            key="rsm_map_ometric")
                    o_thr = b3.number_input("value", value=float(np.nanmedian(rsm_dfm[o_metric])), key="rsm_map_othr")
                c_ = st.columns(6)
                sty = dict(show_uncertainty=c_[0].checkbox("uncertainty panel", True, key="rsm_sty_unc"),
                           show_plan=c_[1].checkbox("suggested runs", True, key="rsm_sty_plan"),
                           shade=c_[2].checkbox("shade regions", True, key="rsm_sty_shade"),
                           boundary_line=c_[3].checkbox("boundary line", True, key="rsm_sty_line"),
                           region_labels=c_[4].checkbox("region labels", True, key="rsm_sty_lab"),
                           counts=c_[5].checkbox("k/n counts", True, key="rsm_sty_counts"))
                d_ = st.columns([2, 1.2, 1.2, 1, 1, 1])
                sty["color_by"] = d_[0].selectbox("Colour the dots by", ["state"] + metric_cols,
                                                 format_func=lambda k_: rsm.METRIC_LABELS.get(k_, k_), key="rsm_sty_by")
                sty["cmap"] = d_[1].selectbox("colour scale (measures, heat maps)",
                                              ["viridis", "plasma", "magma", "cividis", "Blues", "RdYlBu_r", "coolwarm"],
                                              key="rsm_sty_cmap")
                sty["marker_size"] = d_[2].slider("dot size", 10, 200, 80, 5, key="rsm_sty_ms")
                sty["lo_color"] = d_[3].color_picker("below side", "#2a78d6", key="rsm_sty_lo")
                sty["hi_color"] = d_[4].color_picker("above side", "#eb6834", key="rsm_sty_hi")
                sty["line_color"] = d_[5].color_picker("line", "#555555", key="rsm_sty_lc")
                with st.popover("State colours"):
                    sty["state_colors"] = {k_: st.color_picker(k_, v_, key=f"rsm_sty_sc_{k_}")
                                           for k_, v_ in rsm.STATE_COLORS.items()}
            mdata = rsm.map_data(rsm_dfm, map_regime, map_param, float(map_t), map_pot)
            st.caption(f"{mdata.name.nunique()} {map_pot} runs" +
                       ("" if map_param == "t_map" else f" at t = {map_t:g} τ_B") +
                       ". Change which runs are used in the sidebar filter (6.).")
            title_ = f"{rsm.regime_label(map_regime)}, {map_pot}" + (f" (q = {map_q:g})" if map_q is not None and map_pot == "AO" else "") + ("" if map_param == "t_map" else f", t = {map_t:g} τB")
            try:
                if view == "Phase diagram (all states)":
                    p1, p2, p3, p4, p5, p6 = st.columns(6)
                    ph_runs = p1.checkbox("show runs", True, key="rsm_ph_runs")
                    ph_each = p2.checkbox("every run", True, key="rsm_ph_each", disabled=not ph_runs)
                    ph_lab = p3.checkbox("region names", True, key="rsm_ph_lab")
                    ph_borders = p4.checkbox("borders", True, key="rsm_ph_borders")
                    ph_conf = p5.checkbox("fade where unsure", False, key="rsm_ph_conf")
                    ph_even = p6.checkbox("tested values evenly spaced", True, key="rsm_ph_even",
                                          help="Off: true log scale of the parameter.")
                    ph_play = map_param != "t_map" and st.toggle(
                        "Play the phase diagram over time", False, key="rsm_ph_play",
                        help="Draws the diagram at every evaluation time on the same axes and plays it as an "
                             "animation. Each frame uses only the runs that reach that time.")
                    ph_kw = dict(show_runs=ph_runs, per_run=ph_each, region_labels=ph_lab, confidence=ph_conf,
                                 borders=ph_borders, even_spacing=ph_even, colors=sty.get("state_colors"))
                    ph_side = False
                    if map_param == "phi_area":
                        # grid start spacing box/sqrt(N) falls below the bond cutoff (2.9 um) at this area fraction:
                        # above it the 'network' label is reached without attraction
                        phi_net = float(np.pi * rsm.A_P_UM ** 2 / 2.9 ** 2)
                        pa1, pa2 = st.columns(2)
                        if pa1.checkbox(f"mark area fraction {phi_net:.2f} (network label not meaningful above)", True,
                                        key="rsm_ph_netnote"):
                            ph_kw["x_note"] = (phi_net, "network label\nnot meaningful →")
                        ph_side = len(pots) > 1 and pa2.checkbox("all attractions side by side", True, key="rsm_ph_side")
                    if ph_side and not ph_play:
                        _pp = [p_ for p_ in pots if p_ in set(rsm_dfm.loc[rsm_dfm.regime == map_regime, "potential"].fillna("LJ"))]
                        fig, axs = plt.subplots(1, len(_pp), figsize=(5.2 * len(_pp), 4.6), constrained_layout=True,
                                                sharey=True, squeeze=False)
                        for ax_, p_ in zip(axs[0], _pp):
                            md_ = rsm.map_data(rsm_dfm, map_regime, map_param, float(map_t), p_)
                            if md_.empty:
                                ax_.set_axis_off(); continue
                            rsm.fig_phase_map(rsm.fit_phase_map(md_, map_regime, map_param, p_), ax=ax_,
                                              title=f"{p_}" + (f" (q = {map_q:g})" if map_q is not None and p_ == "AO" else ""),
                                              **ph_kw)
                        st.pyplot(fig)
                        save_figure_if_requested(fig, f"phase_diagram_{map_regime}_{map_param}_all", save_plots, output_dir)
                        plt.close(fig)
                    elif not ph_play:
                        pm = rsm.fit_phase_map(mdata, map_regime, map_param, map_pot)
                        if len(pm.classes) == 1:
                            st.info(f"All {len(mdata)} runs here are '{pm.classes[0]}', so the whole map is one state.")
                        fig = rsm.fig_phase_map(pm, **ph_kw)
                        st.pyplot(fig)
                        save_figure_if_requested(fig, f"phase_diagram_{map_regime}_{map_pot}", save_plots, output_dir)
                        plt.close(fig)
                    else:
                        q1, q2, q3, q4 = st.columns([2, 1, 1, 1])
                        if len(times_) > 1:
                            lo_, hi_ = q1.select_slider("Time range (τ_B)", times_, value=(times_[0], min(times_[-1], 10.0)
                                                        if 10.0 in times_ else times_[-1]), key=f"rsm_ph_rng_{map_regime}")
                        else:
                            lo_ = hi_ = times_[0]
                        ph_step = q2.number_input("frame step (τ_B)", 0.05, 10.0, 0.25, 0.05, key="rsm_ph_step",
                                                  help="Uses the map times (sidebar 4.) that fall on this step. "
                                                       "For steps smaller than the sidebar's map time step, lower that first.")
                        ph_times = [t for t in times_ if lo_ - 1e-9 <= t <= hi_ + 1e-9
                                    and abs(t / ph_step - round(t / ph_step)) < 0.02]
                        ph_spf = q3.number_input("seconds per frame", 0.05, 5.0, 0.3, 0.05, key="rsm_ph_spf")
                        q3 = q4
                        st.caption(f"{len(ph_times)} frames: {', '.join(f'{t:g}' for t in ph_times[:12])}"
                                   + (" …" if len(ph_times) > 12 else ""))
                        ph_key = (map_regime, map_param, map_pot, tuple(ph_times), str(ph_kw),
                                  len(rsm_dfm), float(rsm_th.z_fluid), float(rsm_th.psi_order), float(rsm_th.smax_span),
                                  None if map_q is None else float(map_q))
                        if q3.button("Make animation", type="primary", key="rsm_ph_make") or \
                                st.session_state.get("rsm_ph_anim", {}).get("key") == ph_key:
                            if st.session_state.get("rsm_ph_anim", {}).get("key") != ph_key:
                                bar = st.progress(0.0, text="Drawing the frames…")
                                frames = rsm.phase_map_frames(rsm_dfm, map_regime, map_param, map_pot, sorted(ph_times),
                                                              progress=lambda f, t: bar.progress(f, text=f"t = {t:g} τB"),
                                                              **ph_kw)
                                bar.empty()
                                st.session_state["rsm_ph_anim"] = {"key": ph_key, "frames": frames}
                            frames = st.session_state["rsm_ph_anim"]["frames"]
                            gif = rsm.frames_to_gif(frames, ph_spf)
                            st.image(gif, use_container_width=True)
                            st.caption("Frames: " + ", ".join(f"{t:g} τB ({n} runs)" for t, _, n in frames)
                                       + ". Later frames contain only the runs that last that long.")
                            d1, d2 = st.columns([3, 1])
                            pick = d1.select_slider("Look at one frame (τ_B)", [t for t, _, _ in frames],
                                                    key="rsm_ph_pick")
                            d2.download_button("Download GIF", gif, file_name=f"phase_diagram_{map_regime}_{map_pot}_over_time.gif",
                                               mime="image/gif")
                            st.image(dict((t, p) for t, p, _ in frames)[pick], use_container_width=True)
                        else:
                            st.caption("Choose the times and press 'Make animation' (one model fit per time, a few "
                                       "seconds each).")
                    st.caption("One model learns all states at once and colours every point by its most likely state; "
                               "borders are where two states are equally likely. Change the state definitions with the "
                               "thresholds in the sidebar and the state colours under 'Boundary and look'.")
                if view in ("States + boundary model", "Every run as its own dot", "3D elevation"):
                    rsm_model = rsm.fit_boundary(rsm_dfm, map_regime, map_param, float(map_t), map_pot,
                                                 loo=(map_param != "t_map" and len(mdata) <= 400), data=mdata,
                                                 outcome=outcome, metric=o_metric, threshold=o_thr)
                if view in ("States + boundary model", "Every run as its own dot"):
                    rsm_plan = rsm.suggest_runs(rsm_model, rsm_df) if map_param not in ("t_map",) else None
                    bvals = [b for _, b in rsm_model.boundary]
                    unit = {"dn": "nm", "F": "N/m³", "t_map": "τB"}.get(map_param, "")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Fitted boundary (median over A)", f"{rsm.fmt_val(np.median(bvals))} {unit}")
                    c2.metric("Spread of boundary over A", f"{rsm.fmt_val(np.ptp(bvals))} {unit}")
                    c3.metric("Leave-one-out correct", f"{rsm_model.loo_hits} / {len(rsm_model.data)}"
                              if rsm_model.loo_hits else "skipped (many points)")
                    fig = rsm.fig_state_map(rsm_model, rsm_plan, per_run=(view == "Every run as its own dot"), style=sty)
                    st.pyplot(fig)
                    save_figure_if_requested(fig, f"regime_state_map_{map_regime}_{map_pot}", save_plots, output_dir)
                    plt.close(fig)
                    st.caption("The boundary sits where the model is 50/50. With one dot per point, the number "
                               "next to it is how many of the runs there share the shown state. "
                               f"Kernel: `{rsm_model.kernel}`.")
                elif view.startswith("Heat map"):
                    val = "nonfluid" if view == "Heat map: share not fluid" else metric
                    fig = rsm.fig_cell_heatmap(mdata, map_param, val, title=title_, cmap=sty["cmap"])
                    st.pyplot(fig)
                    save_figure_if_requested(fig, f"regime_heatmap_{map_regime}_{map_pot}_{val}", save_plots, output_dir)
                    plt.close(fig)
                elif view == "3D elevation":
                    st.plotly_chart(rsm.surface_figure(mdata, map_param, metric, rsm_model, title=title_),
                                    use_container_width=True)
                    st.caption("Drag to rotate, scroll to zoom. Dots are the runs (state colours); the surface is "
                               "the boundary model's probability or the measured value interpolated between points.")
                elif view == "LJ vs AO boundaries":
                    models = {}
                    for p_ in pots:
                        try:
                            models[p_] = rsm.fit_boundary(rsm_dfm, map_regime, map_param, float(map_t), p_, loo=False,
                                                          outcome=outcome, metric=o_metric, threshold=o_thr)
                        except ValueError as ex:
                            st.info(f"{p_}: {ex}")
                    if models:
                        fig = rsm.fig_boundary_compare(models)
                        st.pyplot(fig)
                        save_figure_if_requested(fig, f"regime_boundaries_{map_regime}", save_plots, output_dir)
                        plt.close(fig)
                        st.caption("Each line is where that attraction's boundary model is 50/50; AO is matched to "
                                   "LJ by B₂* at each A. Filled dots: not fluid, open dots: fluid.")
                elif view == "Play over time":
                    st.plotly_chart(rsm.map_animation(rsm_dfm, map_regime, map_param, map_pot, metric, times_,
                                                      title=f"{rsm.regime_label(map_regime)}, {map_pot}: map over time"),
                                    use_container_width=True)
                    st.caption("Press ▶ Play or drag the slider. Each frame uses the runs that reach that time.")
            except ValueError as e:
                st.warning(str(e))

    with tab_settle:
        st.caption("For every run: is its state at time t already the state it ends in? The curve is the share of "
                   "runs for which that holds. It shows how long runs must be before the state is final, per regime "
                   "and attraction. States are only known at the map times (sidebar 4.).")
        s1, s2, s3, s4 = st.columns([2, 1.3, 1, 1])
        se_by = s1.multiselect("One curve per", ["regime", "potential", "T_name", "q", "A", "dn", "F"],
                               default=["regime", "potential", "T_name"], key="rsm_se_by",
                               format_func=lambda k_: {"T_name": "run length", "potential": "attraction",
                                                       "dn": "d_n", "F": "F₀"}.get(k_, k_),
                               help="Keep 'run length': runs of 1, 10 or 20 τB end at different times, so their "
                                    "'final state' means something different.")
        se_rel = s2.radio("x axis", ["time (τB)", "fraction of run length"], key="rsm_se_rel") == "fraction of run length"
        se_min = s3.number_input("min runs per point", 1, 50, 3, key="rsm_se_min")
        se_log = s4.checkbox("log time axis", False, key="rsm_se_log", disabled=se_rel)
        se_regs = st.multiselect("Regimes", rsm_regimes, default=list(rsm_regimes), key="rsm_se_regs",
                                 format_func=lambda r_: rsm.regime_label(r_, long=True))
        cur = rsm.settling_curve(rsm_df[rsm_df.regime.isin(se_regs)], by=tuple(se_by) or ("regime",), relative=se_rel)
        if cur.empty:
            st.info("No runs with map-time evaluations for this selection.")
        else:
            fig = rsm.fig_settling(cur, relative=se_rel, min_runs=int(se_min), log_x=se_log)
            st.pyplot(fig)
            save_figure_if_requested(fig, "state_settling", save_plots, output_dir)
            plt.close(fig)
            t90 = (cur[cur.share >= 0.9].groupby([c for c in cur.columns if c not in ("x", "share", "n")]).x.min()
                   .rename("90 % settled at").reset_index())
            st.dataframe(t90, hide_index=True)

    with tab_trans:
        tr_regs = [r for r in ("AAF", "NAF", "ASF") if r in rsm_regimes]
        if not tr_regs:
            st.info("Select Regime I, III or IV to see how the state changes with A.")
        else:
            labels = {"psi6": "ψ6 (local hexagonal order)", "z_mean": "Mean coordination number",
                      "Smax": "Largest cluster fraction", "bond_persist": "Bond persistence",
                      "Rg_ratio": "Rg / Rg(t=0)", "mobility": "Mobility (MSD / free MSD)"}
            thr = {"psi6": rsm_th.psi_order, "z_mean": rsm_th.z_fluid, "Smax": rsm_th.smax_span}
            c1, c2 = st.columns(2)
            m1 = c1.selectbox("Metric", list(labels), format_func=labels.get, key="rsm_tr_metric")
            regs = c2.multiselect("Regimes", tr_regs, default=[r for r in tr_regs if r != "ASF"],
                                  key="rsm_tr_regs")
            fig = rsm.fig_transitions(rsm_df, m1, thr.get(m1), tuple(regs))
            st.pyplot(fig)
            save_figure_if_requested(fig, f"regime_transition_{m1}", save_plots, output_dir)
            plt.close(fig)

    with tab_next:
        if rsm_plan is None:
            st.info("Open the Boundary map tab with a regime that can be fitted first.")
        else:
            st.write("New runs either side of the fitted boundary, plus repeats with a **new random seed** "
                     "where a label could be noise.")
            st.dataframe(rsm_plan, hide_index=True)
            st.download_button("Download suggested runs (CSV)", rsm_plan.to_csv(index=False).encode(),
                               "suggested_runs.csv", "text/csv", key="rsm_dl_plan")

    with tab_checks:
        with st.spinner("Checking files..."):
            checks = _rsm_checks(rsm_root, tuple(rsm_regimes), st.session_state[_sk])
        if checks.empty:
            st.success("No duplicate files, run-length mismatches or shared random seeds found.")
        else:
            st.dataframe(checks, hide_index=True)

    with tab_oop:
        import out_of_plane as oop
        st.caption(
            "Where particles would try to leave the plane (the model itself is 2D). For every particle the pair forces "
            "from its neighbours are summed (+ pushed, − pulled) and averaged over the late frames in 2 µm bins: bonds "
            "held at contact average out, what remains is the squeeze the field or flow adds. A particle can lift where "
            "the squeeze beats buoyant weight + k_BT/a: "
            + ", ".join(f"{k} {v:.2f} k_BT/µm" for k, v in oop.THRESHOLDS.items()) + ". An estimate, not a 3D simulation."
        )
        o1, o2 = st.columns(2)
        oop_from = o1.slider("Average over the last … of each run", 0.05, 1.0, 0.2, 0.05, key="oop_last",
                             help="0.2 = last 20 % of the frames.")
        oop_metric = o2.radio("Squeeze measure", ["squeeze_centre", "squeeze_peak"], horizontal=True, key="oop_metric",
                              format_func=lambda m: {"squeeze_centre": "band centre (|y| < 2 µm)",
                                                     "squeeze_peak": "95th percentile of the map"}[m])
        oop_runs = rsm_df.drop_duplicates(["regime", "name"])
        st.write(f"{len(oop_runs)} run(s) from the sidebar filter (narrow it there, e.g. to Regime II and one potential).")

        @st.cache_data(show_spinner=False)
        def _oop_summary(root, names, t_from):
            d = oop_runs[oop_runs.apply(lambda r: (r["regime"], r["name"]) in set(names), axis=1)]
            bar = st.progress(0.0, text="Computing squeeze…")
            out = oop.squeeze_summary(d, root, t_from, progress=lambda f, s: bar.progress(min(f, 1.0), text=s))
            bar.empty()
            return out

        if st.button("Compute squeeze for these runs", key="oop_go", type="primary") or st.session_state.get("oop_done"):
            st.session_state["oop_done"] = True
            names = tuple(zip(oop_runs.regime, oop_runs.name))
            summ = _oop_summary(rsm_root, names, 1.0 - float(oop_from))
            if summ.empty:
                st.warning("No runs could be read.")
            else:
                fig = oop.fig_squeeze_vs_dn(summ, oop_metric)
                st.pyplot(fig)
                buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                st.download_button("Download plot (PNG)", buf.getvalue(), file_name="squeeze_vs_dn.png", key="oop_dl1")
                st.dataframe(summ, hide_index=True)
                st.download_button("Download table (CSV)", summ.to_csv(index=False).encode(), file_name="squeeze_by_run.csv",
                                   key="oop_dl2")
                st.subheader("Squeeze map of one run")
                pick = st.selectbox("Run", summ.regime + "/" + summ.name, key="oop_pick")
                rg, nm = pick.split("/", 1)
                p = oop_runs[(oop_runs.regime == rg) & (oop_runs.name == nm)].iloc[0].to_dict()
                X, Y, S, nf = oop.run_squeeze(os.path.join(rsm_root, rg, nm), p, 1.0 - float(oop_from))
                fig2 = oop.fig_squeeze_map(X, Y, S, title=f"{nm}  ({nf} frames)")
                st.pyplot(fig2)
                buf = io.BytesIO(); fig2.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                st.download_button("Download map (PNG)", buf.getvalue(), file_name=f"squeeze_map_{nm}.png", key="oop_dl3")

        q3 = os.path.join(rsm_root, "_quasi3D")
        if os.path.isdir(q3):
            st.subheader("Quasi-3D runs: side view (y–z)")
            st.caption("Runs of simulation/airpods_bd3d.py (floor, gravity, optional ceiling) in the _quasi3D folder of the "
                       "data folder.")
            q3_runs = sorted(d for d in os.listdir(q3) if os.path.isfile(os.path.join(q3, d, "dataz.csv")))
            q3_pick = st.multiselect("Runs", q3_runs, default=q3_runs[:4], key="oop_q3")
            if q3_pick:
                zt = st.number_input("Top of the plot (µm)", 4.0, 50.0, 10.0, 1.0, key="oop_q3_z")
                fig3 = oop.fig_side_profiles([(r, os.path.join(q3, r)) for r in q3_pick], 1.0 - float(oop_from), zt)
                st.pyplot(fig3)
                buf = io.BytesIO(); fig3.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                st.download_button("Download side views (PNG)", buf.getvalue(), file_name="side_views.png", key="oop_dl4")

    st.stop()

# ---------------------------------------------------------------------
# RUN VIEWER  (snapshot, animation, live particle inspector)
# ---------------------------------------------------------------------
elif plot_mode == "Paper figures":
    import paper_figures as pf
    import matplotlib.pyplot as plt

    @st.cache_data(show_spinner=False)
    def _pf_series(root, stamp):
        bar = st.progress(0.0, text="Computing observables (cached afterwards)…")
        df, pairs = pf.load_series(root, progress=lambda i, n: bar.progress(i / n, text=f"Observables {i}/{n} runs"))
        bar.empty()
        return df, pairs

    def _folder_stamp(root):
        out = []
        for reg in ("ARF", "NAF", "AAF", "ASF"):
            d = os.path.join(root, reg)
            if os.path.isdir(d):
                for n in sorted(os.listdir(d)):
                    f = os.path.join(d, n, "datax.csv")
                    if os.path.isfile(f):
                        out.append((reg, n, os.path.getmtime(f)))
        return tuple(out)

    st.sidebar.header("2. Data folder")
    pf_bd = st.sidebar.text_input("Data folder (with NAF / ARF / AAF / ASF; Python and COMSOL runs together):",
                                  value=_default_runs_root(source_root),
                                  key="pf_bd")
    pf_bd = clean_path(pf_bd)
    pf_co = pf_bd             # since the data merge, COMSOL runs (_CS) sit in the same regime folders
    st.sidebar.header("3. Figures")
    pf_keys = st.sidebar.multiselect("Fixed deck figures to draw:", list(pf.FIGURES), default=[],
                                     format_func=lambda k: f"{k.split('_')[0]}  {pf.FIGURES[k]}", key="pf_keys")
    pf_dpi = st.sidebar.number_input("Saved PNG dpi", 72, 600, 170, 10, key="pf_dpi")
    pf_save = st.sidebar.text_input("Also save PNGs to folder (optional):", value="", key="pf_save").strip().strip('"')

    st.header("Paper figures")
    st.caption("Recreates the figures of the proposal deck from the run folders (same code as "
               "`python paper_figures.py`). Figures 6 / 6a (two-particle B₂ and AO force check) come from "
               "`verify_ao.py` in the BD simulation folder. Snapshots (5b, 8, 8b) use the Run viewer code, "
               "so you can also make them there and change colours.")
    pf_found = [r for r in ("ARF", "NAF", "AAF", "ASF") if pf_bd and os.path.isdir(os.path.join(pf_bd, r))]
    if not pf_found:
        if not pf_bd:
            st.info("Set the Python BD runs folder (the one that contains the regime folders).")
        elif not os.path.isdir(pf_bd):
            st.error(f"This folder does not exist (as the app sees it): `{pf_bd}`")
        else:
            subs = sorted(d for d in os.listdir(pf_bd) if os.path.isdir(os.path.join(pf_bd, d)))[:20]
            st.error(f"`{pf_bd}` exists but has no regime folder (NAF / ARF / AAF / ASF) inside. Subfolders found: "
                     + (", ".join(subs) if subs else "none"))
        st.stop()
    df_pf, pairs_pf = _pf_series(pf_bd, _folder_stamp(pf_bd))
    st.write(f"{df_pf.name.nunique()} runs loaded.")

    tab_builder, tab_figs, tab_table = st.tabs(["Figure builder", "Deck figures (fixed)", "End-of-run numbers"])
    with tab_figs:
        if not pf_keys:
            st.info("Pick the fixed deck figures to draw in the sidebar (3. Figures). The editable versions of the "
                    "time and end-value figures are in the Figure builder tab.")
        for key in pf_keys:
            st.subheader(f"{key}  ·  {pf.FIGURES[key]}")
            try:
                fig = pf.make_figure(key, df_pf, pairs_pf, pf_bd, pf_co)
            except Exception as ex:
                st.warning(f"{key}: {ex}")
                continue
            st.pyplot(fig)
            buf = io.BytesIO()
            dpi = 150 if key.startswith(("5b", "8")) else pf_dpi
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
            st.download_button(f"Download {key}.png", buf.getvalue(), file_name=f"{key}.png",
                               mime="image/png", key=f"pf_dl_{key}")
            if pf_save:
                try:
                    os.makedirs(pf_save, exist_ok=True)
                    fig.savefig(os.path.join(pf_save, key + ".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
                except OSError as ex:
                    st.warning(f"could not save: {ex}")
            plt.close(fig)

    with tab_builder:
        import figure_builder as fb
        import figure_builder_ui as fbui

        @st.cache_data(show_spinner=False)
        def _fb_series(root, stamp):
            bar = st.progress(0.0, text="Computing observables incl. COMSOL runs (cached afterwards)…")
            d = fb.load_data(root, progress=lambda i, n: bar.progress(i / n, text=f"Observables {i}/{n} runs"))
            bar.empty()
            return d

        fbui.page(st, pf_bd, _fb_series(pf_bd, _folder_stamp(pf_bd)))

    with tab_table:
        t_from = st.number_input("Average over t > (τB)", 0.0, 100.0, 9.5, 0.5, key="pf_tf")
        tab = pf.summary_table(df_pf, t_from)
        st.dataframe(tab, use_container_width=True)
        st.download_button("Download CSV", tab.to_csv(index=False).encode(), file_name="summary_end_of_run.csv",
                           mime="text/csv")
    st.stop()

elif plot_mode == "Run simulation":
    import sim_gui as sg
    import matplotlib.pyplot as plt

    SS = st.session_state
    DEF = sg.sr.DEFAULTS

    def _d(key, val):
        if key not in SS:
            SS[key] = val

    # defaults (= the COMSOL model) for every widget
    for k, v in {"sim_pot": "LJ", "sim_A": 3.0, "sim_q": 0.2, "sim_phi": 0.633, "sim_match": True,
                 "sim_core": "coulomb", "sim_Z": 100, "sim_ljcut": 3.0, "sim_field": True, "sim_dn": 4000.0,
                 "sim_F0": float(DEF["arf_F0_ref"]), "sim_k": float(DEF["arf_k"]), "sim_start": "grid",
                 "sim_nside": 12, "sim_spacing": 4.0, "sim_nrand": 144, "sim_dmin": 2.2, "sim_runfolder": "",
                 "sim_runt": 1000.0, "sim_file": "", "sim_T": 1.0, "sim_steps_auto": True, "sim_steps": 4000,
                 "sim_out": 100, "sim_seeds": "1, 2, 3", "sim_stream": False, "sim_F": 3000.0, "sim_workers": max(1, min(3, (os.cpu_count() or 2) - 1)),
                 "sim_regime": "auto", "sim_tag": "", "sim_box": float(DEF["box_half"]), "sim_TK": float(DEF["T_K"]),
                 "sim_eta": float(DEF["eta"]) * 1e3, "sim_rad": float(DEF["radius_um"]),
                 "sim_cap": float(DEF["max_step_um"])}.items():
        _d(k, v)

    def _apply_config(c):
        """Fill the form from a saved sim_config.json."""
        m = {"potential": "sim_pot", "A": "sim_A", "q": "sim_q", "phi_p": "sim_phi", "core": "sim_core",
             "core_Z": "sim_Z", "lj_cutoff_sigma": "sim_ljcut", "arf_F0_ref": "sim_F0", "arf_k": "sim_k",
             "T_tauB": "sim_T", "steps_per_tauB": "sim_steps", "out_per_tauB": "sim_out", "regime": "sim_regime",
             "tag": "sim_tag", "box_half": "sim_box", "T_K": "sim_TK", "radius_um": "sim_rad", "max_step_um": "sim_cap"}
        for a, b in m.items():
            if a in c:
                SS[b] = type(SS[b])(c[a]) if not isinstance(SS[b], bool) else bool(c[a])
        SS.sim_eta = float(c.get("eta", DEF["eta"])) * 1e3
        SS.sim_field = float(c.get("dn_nm", 0)) > 0
        if SS.sim_field:
            SS.sim_dn = float(c["dn_nm"])
        SS.sim_match = False
        SS.sim_steps_auto = False
        stt = c.get("start", {})
        SS.sim_start = stt.get("kind", "grid")
        SS.sim_nside = int(stt.get("n_side", 12)); SS.sim_spacing = float(stt.get("spacing", 4.0))
        SS.sim_nrand = int(stt.get("n", 144)); SS.sim_dmin = float(stt.get("min_dist", 2.2))
        SS.sim_runfolder = stt.get("folder", ""); SS.sim_runt = float(stt.get("t_tauB", 1000.0))
        SS.sim_file = stt.get("path", "")
        SS.sim_seeds = ", ".join(str(s) for s in c.get("seeds", [c.get("run_seed", 1)]))

    st.header("Run simulation")
    st.caption("Python Brownian dynamics (`simulation/airpods_bd.py`), the same code as the BD runs. "
               "Runs go into `<output>/<regime folder>/<run name>/` (NAF = Regime I, ARF = II, AAF = III, ASF = IV) as datax.csv / datay.csv, so every other page "
               "reads them. Each run folder also gets `sim_config.json` with all settings, to repeat it exactly.")

    with st.sidebar:
        st.header("2. Output")
        sim_root = st.text_input("Output folder:", value=_default_runs_root(source_root) or "Data",
                                 key="sim_root")
        sim_root = clean_path(sim_root)
        with st.expander("Load settings from an existing run"):
            lf = clean_path(st.text_input("Run folder (with sim_config.json):", key="sim_loadfolder"))
            if st.button("Load settings", key="sim_load_btn"):
                try:
                    _apply_config(sg.load_config_from_run(lf))
                    st.success("Loaded.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Could not load: {ex}")
        if st.button("Reset all settings to the COMSOL model", key="sim_reset"):
            for k in [k for k in SS.keys() if str(k).startswith("sim_") and k != "sim_root"]:
                del SS[k]
            st.rerun()

    tab_set, tab_batch, tab_jobs = st.tabs(["Settings and start", "Batch", "Jobs"])
    with tab_set:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Interaction")
            pot = st.radio("Attraction", list(sg.POTENTIALS), format_func=sg.POTENTIALS.get, key="sim_pot")
            match_failed = False
            if pot == "LJ":
                A = st.number_input("A (k_BT)", 0.0, 50.0, step=0.25, key="sim_A")
                st.number_input("LJ cut-off (σ)", 1.5, 10.0, step=0.5, key="sim_ljcut")
            elif pot == "AO":
                st.number_input("q = δ/a (polymer / particle size)", 0.05, 1.0, step=0.05, key="sim_q",
                                help="AO range is 2δ; q = 0.2 → 0.4 µm. Smaller q needs a smaller time step.")
                match = st.checkbox("Match φ_p to an LJ depth by B₂*", key="sim_match")
                A = st.number_input("matched LJ A (k_BT)" if match else "A for the run name (k_BT)",
                                    0.0, 50.0, step=0.25, key="sim_A")
                match_failed = False
                if match:
                    phi, tgt = sg.match_phi({"q": SS.sim_q, "core": SS.sim_core, "core_Z": SS.sim_Z}, A) \
                        if A > 0 else (float("nan"), 1.0)
                    if np.isfinite(phi):
                        SS.sim_phi = round(float(phi), 3)
                        st.caption(f"φ_p = **{phi:.3f}** gives B₂* = {tgt:.2f} (= LJ A = {A:g}).")
                    else:
                        match_failed = True
                        st.warning("No φ_p matches this A. A = 0 means no attraction (choose 'no attraction'); "
                                   "otherwise untick the box and set φ_p yourself.")
                st.number_input("φ_p (polymer reservoir volume fraction)", 0.0, 20.0, step=0.01, format="%.3f",
                                key="sim_phi", disabled=match and not match_failed)
            else:
                A = 0.0
            st.selectbox("Repulsive core", list(sg.CORES), format_func=sg.CORES.get, key="sim_core")
            if SS.sim_core == "coulomb":
                st.number_input("Charge number Z", 1, 1000, step=10, key="sim_Z")
        with c2:
            st.subheader("Acoustic field")
            field = st.checkbox("Acoustic radiation force on", key="sim_field")
            if field:
                st.number_input("d_n (nm)", 1.0, 20000.0, step=250.0, key="sim_dn",
                                help="Inward wall displacement; the force scales with d_n².")
                if SS.sim_dn < 500:
                    st.warning(f"d_n = {SS.sim_dn:g} nm gives almost no force (F₀ scales with d_n²). Type the value "
                               "and press Enter or click outside the box so it is taken over.")
                with st.expander("Force law F_y = −F₀ (d_n/4000 nm)² sin(k y)"):
                    st.number_input("F₀ at 4000 nm (k_BT/µm)", 0.0, 1000.0, step=0.5, key="sim_F0")
                    st.number_input("k (1/µm)", 0.0, 2.0, step=0.001, format="%.3f", key="sim_k")
            stream = st.checkbox("Acoustic streaming flow (Regime IV)", key="sim_stream",
                                 help="The streaming flow of your COMSOL Regime IV model (stationary Laminar Flow with the body force F0), "
                                      "exported on a 201 × 201 grid; F is the body force density in N/m³.")
            if stream:
                st.number_input("Body force F0 (N/m³, as in COMSOL)", 0.0, 1e6, step=1000.0, key="sim_F")
                st.caption(("Flow exported from your COMSOL model (Laminar Flow, F0 = 1e5 N/m³), scaled by F / 1e5: "
                            f"fastest {78.8 * SS.sim_F / 1e5:.2f} µm/s. ") if sg.bd.ASF_SOURCE == "comsol" and
                           os.path.isfile(sg.bd.ASF_FIELD_FILE) else
                           f"Fitted two-vortex flow: fastest {2 * sg.bd.ASF_U_PER_1000F * SS.sim_F / 1000:.2f} µm/s. ")
            st.subheader("Start")
            sk = st.selectbox("Start positions", list(sg.STARTS), format_func=sg.STARTS.get, key="sim_start")
            if sk == "grid":
                st.number_input("Particles per side", 2, 40, key="sim_nside")
                st.number_input("Spacing (µm)", 2.0, 20.0, step=0.5, key="sim_spacing")
                start = {"kind": "grid", "n_side": int(SS.sim_nside), "spacing": float(SS.sim_spacing)}
            elif sk == "random":
                st.number_input("Number of particles", 2, 2000, key="sim_nrand")
                st.number_input("Minimum distance (µm)", 2.0, 10.0, step=0.1, key="sim_dmin")
                start = {"kind": "random", "n": int(SS.sim_nrand), "min_dist": float(SS.sim_dmin)}
            elif sk == "run":
                st.text_input("Run folder (with datax.csv):", key="sim_runfolder")
                st.number_input("Time of the frame (τ_B; large = last frame)", 0.0, 1e6, step=1.0, key="sim_runt")
                start = {"kind": "run", "folder": clean_path(SS.sim_runfolder), "t_tauB": float(SS.sim_runt)}
            elif sk == "file":
                st.text_input("CSV file (header line, then x, y in µm):", key="sim_file")
                start = {"kind": "file", "path": clean_path(SS.sim_file)}
            else:
                start = {"kind": "aaf"}
                st.caption("Frame 0 of your COMSOL Regime III runs (`simulation/aaf_start.csv`); use with the field off.")
        with c3:
            st.subheader("Time and seeds")
            st.number_input("Run length (τ_B)", 0.01, 1000.0, step=1.0, key="sim_T")
            auto = st.checkbox("Automatic time step", key="sim_steps_auto",
                               help="LJ τ_B/4000; AO q ≥ 0.2 τ_B/32000; q ≥ 0.1 τ_B/128000 (from the convergence checks).")
            cfg_now = {"potential": pot, "q": SS.sim_q}
            if auto:
                SS.sim_steps = sg.suggested_steps(cfg_now)
            st.number_input("Steps per τ_B", 100, 1_000_000, step=1000, key="sim_steps", disabled=auto)
            st.number_input("Saved frames per τ_B", 1, 1000, step=10, key="sim_out")
            st.text_input("Seeds (comma separated, one run each)", key="sim_seeds")
            st.number_input("Runs at the same time", 1, max(1, os.cpu_count() or 1), key="sim_workers",
                            help=f"This computer has {os.cpu_count()} CPU cores.")
            st.selectbox("Regime", ["auto", "NAF", "ARF", "AAF", "ASF"], key="sim_regime",
                         format_func=lambda r: "auto" if r == "auto" else rsm.regime_label(r, long=True),
                         help="auto: Regime IV (ASF) with streaming, Regime II (ARF) if the field is on, Regime III (AAF) if starting "
                              "from a band or a run, else Regime I (NAF).")
            st.text_input("Extra name tag (letters, then digits)", key="sim_tag")
            with st.expander("Physics constants (change with care)"):
                st.number_input("Box half-width (µm)", 5.0, 500.0, step=5.0, key="sim_box")
                st.number_input("Temperature (K)", 200.0, 400.0, step=1.0, key="sim_TK")
                st.number_input("Viscosity (mPa s)", 0.1, 100.0, step=0.1, key="sim_eta")
                st.number_input("Particle radius (µm)", 0.1, 10.0, step=0.1, key="sim_rad")
                st.number_input("Force cap per step (µm)", 0.001, 1.0, step=0.01, format="%.3f", key="sim_cap")
                st.caption("The analyser assumes radius 1 µm and τ_B = 13.5 s; other values change the "
                           "time unit of the runs.")

        try:
            seeds = [int(s) for s in SS.sim_seeds.replace(";", ",").split(",") if s.strip()]
        except ValueError:
            seeds = []
            st.error("Seeds must be whole numbers, e.g. 1, 2, 3")
        tag_ok = (not SS.sim_tag) or bool(re.fullmatch(r"[A-Za-z]+[0-9.]*", SS.sim_tag.strip().strip("_")))
        if not tag_ok:
            st.error("The tag must be letters followed by digits (e.g. test2), so the run picker can read it.")
        cfg = {"potential": pot, "A": float(A), "q": float(SS.sim_q), "phi_p": float(SS.sim_phi),
               "core": SS.sim_core, "core_Z": int(SS.sim_Z), "lj_cutoff_sigma": float(SS.sim_ljcut),
               "dn_nm": float(SS.sim_dn) if field else 0.0, "flow_F": float(SS.sim_F) if stream else 0.0, "arf_F0_ref": float(SS.sim_F0), "arf_k": float(SS.sim_k),
               "start": start, "T_tauB": float(SS.sim_T), "steps_per_tauB": int(SS.sim_steps),
               "out_per_tauB": int(SS.sim_out), "seeds": seeds, "workers": int(SS.sim_workers),
               "regime": SS.sim_regime, "tag": SS.sim_tag.strip().strip("_"), "out_root": sim_root,
               "box_half": float(SS.sim_box), "T_K": float(SS.sim_TK), "eta": float(SS.sim_eta) * 1e-3,
               "radius_um": float(SS.sim_rad), "max_step_um": float(SS.sim_cap)}

        # ---- sanity checks before anything runs ----
        try:
            sg._with(cfg)
            _r = np.array([1.999 * float(SS.sim_rad)])
            _ao = (float(SS.sim_q), float(SS.sim_phi)) if pot == "AO" else None
            _net = float(sg.bd.radial_force(_r, float(A) if pot == "LJ" else 0.0, _ao,
                                            SS.sim_core if SS.sim_core != "none" else "off")[0])
            if _net < 0:
                st.error(f"The attraction pulls harder at contact than the repulsive core pushes (net {_net:.0f} k_BT/µm "
                         "inward just inside contact): particles will pass through each other and B₂* is meaningless. "
                         "Lower φ_p (tick 'Match φ_p to an LJ depth by B₂*'), raise q, or use a harder core.")
        except Exception:
            pass
        if field and float(SS.sim_F0) * float(SS.sim_k) == 0:
            st.error("The acoustic force is zero because F₀ or k in 'Force law' is 0. COMSOL values: F₀ = 18.4 k_BT/µm, k = 0.124 /µm.")

        # ---- previews ----
        st.subheader("Preview")
        pc1, pc2, pc3 = st.columns(3)
        try:
            r, U, fr = sg.potential_curve(cfg)
            b2 = sg.b2_star(cfg)
            fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
            ax.plot(r, U, c="#2a78d6", lw=2)
            ax.axhline(0, c="#999", lw=0.8)
            lo = max(min(U.min() * 1.3, -1), -60)
            ax.set_ylim(lo, max(-lo * 0.8, 3)); ax.set_xlim(r[0], min(r[-1], 6 * float(SS.sim_rad)))
            ax.set_xlabel("r (µm)"); ax.set_ylabel("U(r) / k$_B$T")
            ax.text(0.98, 0.95, "repulsive core continues off the top", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="#888")
            ax.set_title(f"pair potential  ·  B$_2^*$ = {b2:.2f}  ·  min {U.min():.2f} k$_B$T", fontsize=9, loc="left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            pc1.pyplot(fig); plt.close(fig)
        except Exception as ex:
            pc1.warning(f"potential: {ex}")
        try:
            fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
            sg._with(cfg)
            yy = np.linspace(-cfg["box_half"], cfg["box_half"], 400)
            ax.plot(sg.bd.arf_force(yy, cfg["dn_nm"]), yy, c="#eb6834", lw=2)
            ax.axvline(0, c="#999", lw=0.8)
            ax.set_xlabel("F$_y$ (k$_B$T/µm)"); ax.set_ylabel("y (µm)")
            amp = float(cfg["arf_F0_ref"]) * (cfg["dn_nm"] / 4000.0) ** 2 if field else 0.0
            ax.set_title("acoustic force" + (f"  ·  d$_n$ = {cfg['dn_nm']:g} nm, amplitude {amp:.3g} k$_B$T/µm" if field else " (off)"),
                         fontsize=9, loc="left")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            pc2.pyplot(fig); plt.close(fig)
        except Exception as ex:
            pc2.warning(f"force: {ex}")
        n_part = 0
        try:
            P0 = sg.start_preview(cfg, seeds[0] if seeds else 1)
            n_part = len(P0)
            fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
            from matplotlib.patches import Circle, Rectangle
            from matplotlib.collections import PatchCollection
            b = cfg["box_half"]
            ax.add_patch(Rectangle((-b, -b), 2 * b, 2 * b, fill=False, ec="#c3c2b7"))
            ax.add_collection(PatchCollection([Circle(p, cfg["radius_um"]) for p in P0], fc="#2a78d6", ec="white", lw=0.3))
            ax.set_xlim(-b - 1, b + 1); ax.set_ylim(-b - 1, b + 1); ax.set_aspect("equal")
            ax.set_title(f"start: {n_part} particles", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
            pc3.pyplot(fig); plt.close(fig)
        except Exception as ex:
            pc3.warning(f"start positions: {ex}")

        # ---- runtime, names, start ----
        n_steps = int(round(cfg["T_tauB"] * cfg["steps_per_tauB"]))
        sps = sg.seconds_per_step(max(n_part, 2), pot) if n_part else 0.0
        waves = -(-len(seeds) // max(1, cfg["workers"])) if seeds else 0
        est = n_steps * sps * waves
        try:
            names = sg.run_names(cfg) if seeds else []
        except Exception as ex:
            names = []
            st.error(str(ex))
        st.markdown(f"**{len(seeds)} run(s)**, {n_steps:,} steps each · estimated "
                    f"**{est / 60:.1f} min** on this computer ({cfg['workers']} at a time)")
        if cfg["steps_per_tauB"] % cfg["out_per_tauB"]:
            st.error("Steps per τ_B must be a multiple of saved frames per τ_B.")
        if pot == "AO" and cfg["steps_per_tauB"] < sg.suggested_steps(cfg):
            st.warning(f"AO at q = {cfg['q']:g} needs about τ_B/{sg.suggested_steps(cfg)} to sample the well "
                       "correctly (the two-particle B₂ check); fewer steps make it less sticky than set.")
        exist = [os.path.isdir(os.path.join(sim_root, rg, nm)) for rg, nm in names]
        st.dataframe(pd.DataFrame({"folder": [f"{rg}/{nm}" for rg, nm in names],
                                   "exists already (skipped)": exist}), use_container_width=True, hide_index=True)
        problems = []
        if not seeds:
            problems.append("enter at least one seed")
        if not tag_ok:
            problems.append("fix the name tag")
        if n_part < 2:
            problems.append("the start positions could not be made (see the preview)")
        if cfg["steps_per_tauB"] % cfg["out_per_tauB"]:
            problems.append("steps per τ_B must be a multiple of saved frames per τ_B")
        if cfg["T_tauB"] * cfg["out_per_tauB"] < 2:
            problems.append(f"run length × saved frames per τ_B is {cfg['T_tauB'] * cfg['out_per_tauB']:g}: "
                            "at least 2 frames must be saved")
        if pot == "AO" and match_failed:
            problems.append("no φ_p matches the chosen A")
        if pot == "AO" and cfg["phi_p"] <= 0:
            problems.append("φ_p must be above 0 for AO")
        if not sim_root:
            problems.append("set the output folder")
        if problems:
            st.error("Cannot start yet: " + "; ".join(problems) + ".")
        ok = not problems
        if st.button("▶ Start", type="primary", disabled=not ok, key="sim_go"):
            os.makedirs(sim_root, exist_ok=True)
            jd = sg.launch(cfg)
            st.success(f"Started. Follow it in the Jobs tab (job {os.path.basename(jd)}).")

    with tab_batch:
        st.caption("Many parameter sets in one job. Physics, repulsive core, particle number (grid start), saved "
                   "frames and output folder come from **Settings and start**; AO is matched to each A by B₂*. "
                   "Runs that already exist are skipped, so a stopped batch can simply be started again.")
        bdir = os.path.join(sim_root, "_batches") if sim_root else ""
        saved = sorted(f[:-5] for f in os.listdir(bdir) if f.endswith(".json")) if bdir and os.path.isdir(bdir) else []
        if saved:
            lc1, lc2 = st.columns([3, 1])
            pick = lc1.selectbox("Load a saved batch", ["–"] + saved, key="bt_pick")
            if lc2.button("Load", key="bt_load") and pick != "–":
                with open(os.path.join(bdir, pick + ".json")) as f:
                    spec0 = json.load(f).get("batch_spec", {})
                SS.bt_sc = spec0.get("scenario", "II")
                SS.bt_A = ", ".join(f"{v:g}" for v in spec0.get("A", []))
                SS.bt_dn = ", ".join(f"{v:g}" for v in spec0.get("dn", []))
                SS.bt_F = ", ".join(f"{v:g}" for v in spec0.get("F", []))
                SS.bt_pots = spec0.get("potentials", ["LJ", "AO"])
                SS.bt_q = ", ".join(f"{v:g}" for v in spec0.get("q", [0.2]))
                SS.bt_T = float(spec0.get("T_tauB", 10)); SS.bt_seeds = ", ".join(map(str, spec0.get("seeds", [1, 2, 3])))
                SS.bt_loaded_name = pick
                st.rerun()

        def _nums(txt):
            out = []
            for t in str(txt).replace(";", ",").split(","):
                try:
                    out.append(float(t)) if t.strip() else None
                except ValueError:
                    st.error(f"'{t.strip()}' is not a number")
            return out

        sc = st.radio("Scenario", list(sg.SCENARIOS), format_func=sg.SCENARIOS.get, key="bt_sc", horizontal=False)
        b1, b2, b3 = st.columns(3)
        A_list = _nums(b1.text_input("A values (k_BT; AO at the same B₂*)", "1, 2, 3, 5", key="bt_A"))
        dn_list = _nums(b2.text_input("d_n values (nm)", "2000, 3000, 4000", key="bt_dn")) if sc == "II" else []
        F_list = _nums(b2.text_input("F₀ values (N/m³)", "300, 1000, 3000", key="bt_F")) if sc == "IV" else []
        pots = b3.multiselect("Attraction", ["LJ", "AO"], default=["LJ", "AO"], key="bt_pots")
        b4, b5, b6, b7 = st.columns(4)
        q_list = _nums(b4.text_input("AO q values", "0.2", key="bt_q")) if "AO" in pots else []
        T_b = b5.number_input("Run length (τ_B)", 0.01, 1000.0, 10.0, 1.0, key="bt_T")
        seeds_b = [int(v) for v in _nums(b6.text_input("Seeds", "1, 2, 3", key="bt_seeds"))]
        workers_b = b7.number_input("Runs at the same time", 1, max(1, os.cpu_count() or 1),
                                    min(4, max(1, (os.cpu_count() or 2) - 1)), key="bt_workers",
                                    help="Each one is a separate Python process (~200 MB). Stay below your number of "
                                         "cores and don't run two jobs at once, or the computer runs out of memory.")
        spec = {"scenario": sc, "A": A_list, "dn": dn_list, "F": F_list, "potentials": pots, "q": q_list,
                "T_tauB": float(T_b), "seeds": seeds_b, "auto_steps": True,
                "out_per_tauB": int(cfg["out_per_tauB"])}
        if "bt_match" not in SS:
            SS.bt_match = {}
        items, runs = [], []
        if A_list and pots and seeds_b and (sc not in ("II", "IV") or (dn_list if sc == "II" else F_list)):
            try:
                with st.spinner("matching AO to each A by B₂*…"):
                    items = sg.expand_batch(cfg, spec, SS.bt_match)
                runs = sg.batch_runs(cfg, items)
            except Exception as ex:
                st.error(f"batch: {ex}")
        if runs:
            todo = [r for r in runs if not r[2]]
            n_b = 144 if sc in ("III", "IV") else max(n_part, 2)
            sps = {p: sg.seconds_per_step(n_b, p) for p in {r[4] for r in todo}} if todo else {}
            est_b = sum(r[3] * sps[r[4]] for r in todo) / max(1, min(int(workers_b), len(todo) or 1))
            st.markdown(f"**{len(items)} parameter sets × seeds = {len(runs)} runs**, {len(runs) - len(todo)} exist "
                        f"already (skipped) · estimated **{est_b / 3600:.1f} h** with {int(workers_b)} at a time")
            if any(it["potential"] == "AO" for it in items):
                st.caption("AO φ_p per A: " + ", ".join(sorted({f"q={it['q']:g}, A={it['A']:g}: {it['phi_p']:.3f}"
                                                                 for it in items if it["potential"] == "AO"})))
            st.dataframe(pd.DataFrame({"folder": [f"{r[0]}/{r[1]}" for r in runs],
                                       "exists (skipped)": [r[2] for r in runs]}),
                         use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(runs)))
            s1, s2, s3 = st.columns([1, 2, 1])
            auto_name = (f"Regime{sc}_{'-'.join(pots)}_A{'-'.join(f'{a:g}' for a in A_list)}"
                         + (f"_dn{'-'.join(f'{d:g}' for d in dn_list)}" if sc == "II" else "")
                         + (f"_F{'-'.join(f'{f:g}' for f in F_list)}" if sc == "IV" else ""))
            bname = s2.text_input("Batch name (to save it)", value=SS.pop("bt_loaded_name", None) or auto_name)
            full = {**cfg, "batch": items, "workers": int(workers_b), "batch_spec": spec}
            if s3.button("Save batch", key="bt_save"):
                os.makedirs(bdir, exist_ok=True)
                with open(os.path.join(bdir, re.sub(r"[^A-Za-z0-9_.-]+", "_", bname) + ".json"), "w") as f:
                    json.dump(full, f, indent=2)
                st.success(f"saved in {bdir}. Command line: python simulation\\sim_runner.py \"{os.path.join(bdir, bname)}.json\"")
            if s1.button("▶ Start batch", type="primary", key="bt_go", disabled=not todo):
                os.makedirs(sim_root, exist_ok=True)
                jd = sg.launch(full)
                st.success(f"Started {len(todo)} runs. Follow them in the Jobs tab (job {os.path.basename(jd)}).")
        else:
            st.info("Enter at least one A value, an attraction, seeds and (for Regimes II / IV) d_n or F₀ values.")

    with tab_jobs:
        @st.fragment(run_every=3)
        def _jobs_view():
            jobs = sg.list_jobs(sim_root)
            if not jobs:
                st.info("No jobs in this output folder yet.")
                return
            for j in jobs[:15]:
                c = j["config"]
                pot_txt = (f"AO q={c['q']:g} φp={c['phi_p']:g}" if c["potential"] == "AO" else
                           f"LJ A={c['A']:g}" if c["potential"] == "LJ" else "no attraction")
                fld = f"d_n {c['dn_nm']:g} nm" if c["dn_nm"] > 0 else "field off"
                fr = np.mean([s.get("frac", 0) for s in j["status"]]) if j["status"] else 0
                icon = {"done": "✅", "running": "⏳", "failed": "❌", "stopped": "⏹"}[j["state"]]
                if c.get("batch"):
                    n_done = sum(s.get("state") in ("done", "exists") for s in j["status"])
                    head = (f"{icon} {j['id']} · batch: {len(c['batch'])} parameter sets, {len(j['status'])} runs · "
                            f"{n_done} done · {fr:.0%}")
                else:
                    head = (f"{icon} {j['id']} · {pot_txt}, {fld}, {c['T_tauB']:g} τ_B, seeds "
                            f"{', '.join(map(str, c['seeds']))} · {fr:.0%}")
                with st.expander(head, expanded=j["state"] == "running"):
                    shown = j["status"]
                    if c.get("batch"):
                        shown = [s for s in j["status"] if s.get("state") == "running"]
                        st.caption(", ".join(f"{k}: {v}" for k, v in pd.Series(
                            [s.get("state", "waiting") for s in j["status"]]).value_counts().items()))
                    for s in shown:
                        eta = s.get("eta")
                        txt = (f"{s.get('name', s['seed'])}: {s.get('state')}" if c.get("batch")
                               else f"seed {s['seed']}: {s.get('state')}")
                        if eta is not None and s.get("state") == "running":
                            txt += f" · about {eta / 60:.1f} min left"
                        if s.get("capped"):
                            txt += f" · force cap hit {s['capped']}×"
                        st.progress(float(s.get("frac", 0)), text=txt)
                    cols = st.columns(3)
                    if j["state"] == "running" and cols[0].button("Stop", key=f"stop_{j['id']}"):
                        sg.stop(j["dir"])
                    names_done = [s.get("folder") for s in j["status"] if s.get("state") in ("done", "exists")]
                    if names_done:
                        cols[1].caption("Open them in **Run viewer** or **Paper figures**: " +
                                        ", ".join(os.path.basename(n) for n in names_done))
                    log = sg.read_log(j["dir"])
                    if "Traceback" in log or j["state"] == "failed":
                        st.code(log[-3000:])
        _jobs_view()
    st.stop()

elif plot_mode == "3D viewer":
    import viewer3d as v3

    @st.cache_data(show_spinner=False)
    def _v3_load(folder, mtime):
        return v3.load_run3d(folder)

    st.title("3D viewer")
    st.caption("Runs with a z coordinate: simulation/bd_engine.py with dim = 3 (folders like ARF_3D, NAF_3D) and the "
               "quasi-3D runs (_quasi3D). Same pair forces as 2D; no hydrodynamics.")
    st.sidebar.header("2. Data folder")
    v3_root = st.sidebar.text_input("Data folder", value=source_root, key="v3_root").strip().strip('"').strip("'")
    v3_runs = v3.list_3d_runs(v3_root)
    if not v3_runs:
        st.info("No 3D runs (folders with dataz.csv) in this data folder.")
        st.stop()
    v3_folders = sorted(set(l.split("/")[0] for l, _ in v3_runs))
    v3_fold = st.sidebar.selectbox("Folder", v3_folders, key="v3_fold")
    v3_txt = st.sidebar.text_input("Name contains (e.g. H4, drho0, phi0.244)", "", key="v3_txt")
    v3_opts = [(l, p_) for l, p_ in v3_runs if l.split("/")[0] == v3_fold and v3_txt.lower() in l.lower()]
    if not v3_opts:
        st.info("No run matches."); st.stop()
    st.sidebar.header("3. Runs (up to 4, side by side)")
    v3_sel = st.sidebar.multiselect("Runs", [l for l, _ in v3_opts], default=[v3_opts[0][0]], max_selections=4,
                                    key="v3_runs", format_func=lambda l: l.split("/")[-1])
    if not v3_sel:
        st.info("Pick one or more runs in the sidebar."); st.stop()
    v3_paths = dict(v3_opts)
    v3_data = [(l, _v3_load(v3_paths[l], os.path.getmtime(os.path.join(v3_paths[l], "dataz.csv")))) for l in v3_sel]

    st.sidebar.header("4. View")
    v3_view_name = st.sidebar.radio("View", list(v3.VIEWS), key="v3_view",
                                    help="Top and side views draw particles to scale; the side views show every "
                                         "particle projected along the hidden axis (nearer ones on top).")
    v3_view = v3.VIEWS[v3_view_name]
    v3_elev, v3_azim = 22, -60
    if v3_view == "3d":
        cv1, cv2 = st.sidebar.columns(2)
        v3_elev = cv1.slider("Elevation (°)", -10, 90, 22, 2, key="v3_elev")
        v3_azim = cv2.slider("Azimuth (°)", -180, 180, -60, 5, key="v3_azim")
    st.sidebar.header("5. Colour and show")
    v3_by = st.sidebar.selectbox("Colour by", list(v3.COLOR_BY), format_func=lambda k_: v3.COLOR_BY[k_], key="v3_by",
                                 index=1, help="'resting on another particle': stacked particles in the particle colour, "
                                               "particles on the floor or beside others in grey.")
    cc1, cc2 = st.sidebar.columns(2)
    v3_plain = cc1.color_picker("Particle colour", "#e34948", key="v3_plain")
    v3_cmap = cc2.selectbox("Colour scale", ["viridis", "plasma", "magma", "cividis", "Blues", "Reds"], key="v3_cmap")
    cs1, cs2 = st.sidebar.columns(2)
    v3_show = dict(show_name=cs1.checkbox("Run name", True, key="v3_s_name"),
                   show_time=cs2.checkbox("Time stamp", True, key="v3_s_time"),
                   show_stats=cs1.checkbox("z / S_max / on top", True, key="v3_s_stats"),
                   show_box=cs2.checkbox("Chamber outline", True, key="v3_s_box"))
    with st.sidebar.expander("More display settings"):
        v3_scale = st.slider("Particle size (× real size)", 0.3, 1.5, 1.0, 0.05, key="v3_scale")
        v3_panel = st.slider("Panel size (inches)", 3.0, 8.0, 4.6, 0.2, key="v3_panel")
        v3_font = st.slider("Font size", 6, 16, 9, key="v3_font")
        v3_alpha = st.slider("Opacity", 0.1, 1.0, 1.0, 0.05, key="v3_alpha",
                             help="1 = solid particles; lower to see particles behind others.")
        v3_depth = st.checkbox("Fade far particles (3D depth shading)", False, key="v3_depth",
                               help="3D perspective only: particles further from the camera are drawn paler.")
        v3_edge = st.color_picker("Particle outline", "#ffffff", key="v3_edge")
    v3_stretch = st.sidebar.slider("Height stretch (×)", 1.0, 8.0, 1.0, 0.5, key="v3_stretch",
                                   help="Exaggerates the vertical axis in the 3D and side views so thin layers and "
                                        "stacking are easier to see (particles in the side views become ellipses).")
    st.sidebar.header("6. Acoustic field")
    v3_fmode = st.sidebar.selectbox("Overlay", list(v3.FIELD_OVERLAYS), format_func=lambda k_: v3.FIELD_OVERLAYS[k_],
                                    key="v3_fmode",
                                    help="The field the run used: its solved / COMSOL field file (sim_config arf3d_file, "
                                         "looked up in Data/_fields) or the fitted 1D standing wave. Top view: slice at the "
                                         "particles' median height; side views: slice through the middle; 3D view: the y–z "
                                         "slice on the back wall. Darker blue = higher potential (particles collect in the "
                                         "white valleys).")
    v3_falpha = st.sidebar.slider("Overlay strength", 0.1, 1.0, 0.45, 0.05, key="v3_falpha") if v3_fmode != "none" else 0.45

    @st.cache_resource(show_spinner=False)
    def _v3_field(cfg_json, root):
        import json as _json
        return v3.RunField(_json.loads(cfg_json), root)
    import json as _json3
    v3_fields = [_v3_field(_json3.dumps(d_[1][2], sort_keys=True, default=str), v3_root) for d_ in v3_data] \
        if v3_fmode != "none" else None
    if v3_fields:
        st.sidebar.caption("; ".join(f"{l.split('/')[-1][:28]}…: {f_.kind} field {f_.note}".strip()
                                     for l, f_ in zip(v3_sel, v3_fields)))
    v3_kw = dict(color_by=v3_by, plain=v3_plain, elev=v3_elev, scale=v3_scale, panel=v3_panel, font=v3_font,
                 cmap=v3_cmap, alpha=float(v3_alpha), depthshade=bool(v3_depth), edge=v3_edge, stretch=float(v3_stretch),
                 fields=v3_fields, field_mode=v3_fmode, field_alpha=float(v3_falpha), **v3_show)
    t3max = min(d_[1][1][-1] for d_ in v3_data) / v3.TAUB_S

    tab3_snap, tab3_anim, tab3_live, tab3_obs = st.tabs(["Snapshot", "Animation", "Interactive 3D", "Observables"])
    with tab3_snap:
        t3sel = st.slider("Time (τB)", 0.0, float(t3max), float(t3max), max(float(t3max) / 200, 0.01), key="v3_t",
                          help="Same time for all selected runs (limited by the shortest run).")
        ks3 = [int(np.clip(np.searchsorted(d_[1][1], t3sel * v3.TAUB_S - 1e-9), 0, len(d_[1][1]) - 1)) for d_ in v3_data]
        fig3 = v3.make_figure3d(v3_data, ks3, v3_view, azim=v3_azim, **v3_kw)
        st.pyplot(fig3)
        buf3 = io.BytesIO(); fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
        st.download_button("Download PNG (300 dpi)", buf3.getvalue(), f"3D_snapshot_{t3sel:.2f}tauB.png", "image/png",
                           key="v3_dl_png")
        save_figure_if_requested(fig3, f"3D_snapshot_{'_vs_'.join(l.split('/')[-1] for l in v3_sel)}_{t3sel:.2f}tauB",
                                 save_plots, output_dir)
        plt.close(fig3)
    with tab3_anim:
        a1_, a2_, a3_, a4_, a5_ = st.columns([2, 1, 1, 1, 1])
        ar0, ar1 = a1_.slider("Time range (τB)", 0.0, float(t3max), (0.0, float(t3max)), key="v3_arange")
        n3 = a2_.number_input("Frames", 5, 400, 60, 5, key="v3_nframes")
        fps3 = a3_.number_input("Frames per second", 1, 30, 12, 1, key="v3_fps")
        dpi3 = a4_.number_input("Resolution (dpi)", 50, 200, 90, 10, key="v3_dpi")
        orbit3 = a5_.number_input("Camera turn (°)", 0, 720, 0, 15, key="v3_orbit", disabled=v3_view != "3d",
                                  help="3D view only: the camera turns this far around the chamber during the animation.")
        st.caption(f"View: {v3_view_name}. Uses the colour and show settings from the sidebar; the colour scale is "
                   "fixed over the whole animation.")
        if st.button("Make animation", key="v3_make_gif", type="primary"):
            bar3 = st.progress(0.0, text="Drawing frames...")
            try:
                st.session_state["v3_gif"] = v3.make_gif3d(
                    v3_data, v3_view, t_start_s=ar0 * v3.TAUB_S, t_end_s=ar1 * v3.TAUB_S, n_frames=int(n3), fps=int(fps3),
                    dpi=int(dpi3), orbit=float(orbit3), azim=v3_azim,
                    progress=lambda p_: bar3.progress(p_, text=f"Drawing frames... {p_:.0%}"), **v3_kw)
            except Exception as e:
                st.error(f"Could not make the animation: {e}")
            bar3.empty()
        if st.session_state.get("v3_gif"):
            st.image(st.session_state["v3_gif"])
            st.download_button("Download GIF", st.session_state["v3_gif"], "3D_run_animation.gif", "image/gif",
                               key="v3_dl_gif")
            if save_plots and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "3D_run_animation.gif"), "wb") as fh:
                    fh.write(st.session_state["v3_gif"])
    with tab3_live:
        live3 = st.selectbox("Run", v3_sel, key="v3_live_run", format_func=lambda l: l.split("/")[-1]) \
            if len(v3_sel) > 1 else v3_sel[0]
        fr3, tt3, cf3 = dict(v3_data)[live3]
        st.caption("Drag to rotate, scroll to zoom; ▶ plays through up to 80 saved frames.")
        try:
            v3_lsize = st.slider("Marker size", 2, 14, 6, key="v3_lsize")
            st.plotly_chart(v3.fig_3d_animated(fr3, tt3, cf3, v3_by if v3_by != "coordination" else "height", v3_plain,
                                               title=live3.split("/")[-1], opacity=float(v3_alpha), size=v3_lsize),
                            use_container_width=True)
        except ImportError:
            st.error("The interactive view needs plotly: run `pip install plotly` and restart the app.")
    with tab3_obs:
        obs3 = st.selectbox("Run", v3_sel, key="v3_obs_run", format_func=lambda l: l.split("/")[-1]) \
            if len(v3_sel) > 1 else v3_sel[0]
        fr3, tt3, cf3 = dict(v3_data)[obs3]
        if cf3:
            st.caption(", ".join(f"{k} = {cf3[k]}" for k in ("potential", "A", "q", "phi_p", "eps", "dn_nm", "H", "drho",
                                                            "T_tauB", "steps_per_tauB", "run_seed") if k in cf3))
        figt, _ = v3.fig_stats_over_time(fr3, tt3, cf3)
        st.pyplot(figt)
        plt.close(figt)
    st.stop()

elif plot_mode == "Run viewer":

    @st.cache_data(show_spinner=False)
    def _rv_load(folder, mtime):
        return rv.load_run(folder)

    st.sidebar.header("2. Data folder")
    rv_source = st.sidebar.radio("Folder source:", ["Source root subfolder", "Custom path"], key="rv_source")
    if rv_source == "Source root subfolder":
        rv_subs = sorted(d for d in os.listdir(source_root)
                         if os.path.isdir(os.path.join(source_root, d))) if os.path.isdir(source_root) else []
        rv_sub = st.sidebar.selectbox("Subfolder:", rv_subs, key="rv_sub")
        rv_base = os.path.join(source_root, rv_sub) if rv_sub else ""
    else:
        rv_base = st.sidebar.text_input("Folder with the run folders:", value="", key="rv_custom")
    rv_base = rv_base.strip().strip('"').strip("'")
    rv_regime = os.path.basename(rv_base.rstrip("\\/")).upper()

    rv_all = rv.list_run_folders(rv_base)
    st.sidebar.header("3. Runs (up to 4, shown side by side)")
    import run_filter as rf
    _rv_keep = rf.widgets([rf.meta(n, rv_regime) for n in rv_all], key=f"rv__{rv_base}", container=st.sidebar,
                          expanded=False)
    rv_opts = [r for r, k in zip(rv_all, _rv_keep) if k] + [r for r in rv_all if rf.meta(r, rv_regime) is None]
    rv_sel = st.sidebar.multiselect("Runs:", rv_opts, default=rv_opts[:1], key="rv_runs")
    if len(rv_sel) > 4:
        st.sidebar.warning("Showing the first 4 selected runs.")
        rv_sel = rv_sel[:4]

    st.sidebar.header("4. Colour")
    rv_metric = st.sidebar.selectbox("Colour particles by:", list(rv.METRICS), key="rv_mode")
    rv_plain = st.sidebar.color_picker("Particle colour (Plain)", "#e34948", key="rv_plain")
    c1, c2 = st.sidebar.columns(2)
    rv_edge = c1.color_picker("Outline", "#ffffff", key="rv_edge")
    rv_bondc = c2.color_picker("Bonds", "#6b6a63", key="rv_bondc")
    rv_fix = st.sidebar.checkbox("Fixed colour range", value=False, key="rv_fix",
                                 help="Use the same min/max in every panel and frame.")
    rv_range = None
    if rv_fix:
        c1, c2 = st.sidebar.columns(2)
        rv_range = (c1.number_input("min", value=0.0, key="rv_rmin"), c2.number_input("max", value=1.0, key="rv_rmax"))

    st.sidebar.header("5. Show")
    c1, c2 = st.sidebar.columns(2)
    rv_o = dict(
        show_run_name=c1.checkbox("Run name", True, key="rv_s_name"),
        show_time=c2.checkbox("Time stamp", True, key="rv_s_time"),
        show_stats=c1.checkbox("Mean z / ψ6", True, key="rv_s_stats"),
        show_legend=c2.checkbox("Legend", True, key="rv_s_leg"),
        show_colorbar=c1.checkbox("Colour bar", True, key="rv_s_cbar"),
        show_axes=c2.checkbox("Axes", True, key="rv_s_axes"),
        show_box=c1.checkbox("Box outline", True, key="rv_s_box"),
        show_bonds=c2.checkbox("Bonds", False, key="rv_bonds"),
    )
    with st.sidebar.expander("More display settings", expanded=False):
        rv_rc = st.number_input("Bond cutoff (µm)", 1.0, 10.0, 2.9, 0.1, key="rv_rc")
        rv_trail = st.slider("Trail length (frames)", 0, 200, 0, 5, key="rv_trail")
        rv_trailc = st.color_picker("Trail colour", "#b8b6ad", key="rv_trailc")
        rv_scale = st.slider("Particle size (× real size)", 0.3, 1.5, 1.0, 0.05, key="rv_scale")
        rv_font = st.slider("Font size", 6, 18, 9, key="rv_font")
        rv_panel = st.slider("Panel size (inches)", 3.0, 8.0, 4.6, 0.2, key="rv_panel")
    rv_field = st.sidebar.selectbox("Acoustic field overlay (Regime II runs)", ["none", "potential", "force", "nodes"],
                                    key="rv_field", format_func=lambda k_: {"none": "none",
                                    "potential": "Gor'kov potential (shading) + nodes", "force": "|force| (shading) + nodes",
                                    "nodes": "pressure-node lines only"}[k_],
                                    help="The fitted 1D standing wave F = −F0 sin(k y) of the run's d_n; darker = higher "
                                         "potential / stronger force. Dashed: pressure nodes, where the band forms.")
    rv_falpha = st.sidebar.slider("Overlay strength", 0.05, 1.0, 0.35, 0.05, key="rv_falpha") if rv_field in ("potential", "force") else 0.35

    rv_opts_draw = rv.DrawOptions(metric=rv_metric, plain_color=rv_plain, edge_color=rv_edge, bond_color=rv_bondc,
                                  trail_color=rv_trailc, trail=int(rv_trail), rc=float(rv_rc),
                                  particle_scale=float(rv_scale), font_size=float(rv_font),
                                  fixed_range=rv_range, panel_size=float(rv_panel), field_overlay=rv_field,
                                  field_alpha=float(rv_falpha), **rv_o)

    st.header("Run viewer")
    if not rv_base or not os.path.isdir(rv_base):
        st.info("Choose a folder that contains run folders (each with datax.csv and datay.csv).")
        st.stop()
    if not rv_sel:
        st.info(f"{len(rv_all)} runs found. Pick one or more in the sidebar.")
        st.stop()

    rv_runs, rv_models = [], []
    for name in rv_sel:
        f = os.path.join(rv_base, name)
        try:
            rv_runs.append((name, _rv_load(f, os.path.getmtime(os.path.join(f, "datax.csv")))))
            _m = rv.RunModel.from_name(rv_regime, name)
            _m.box = rv.run_box(f, *rv_runs[-1][1][:2])          # N = 576 etc. use a larger box
            rv_models.append(_m)
        except Exception as e:
            st.error(f"Could not read {name}: {e}")
    if not rv_runs:
        st.stop()
    t_max = min(r[1][2][-1] for r in rv_runs) / rv.TAUB_S

    tab_snap, tab_anim, tab_live = st.tabs(["Snapshot", "Animation", "Live view"])

    with tab_snap:
        t_sel = st.slider("Time (τB)", 0.0, float(t_max), float(t_max), max(float(t_max) / 200, 0.01),
                          key="rv_time", help="Same time for all selected runs (limited by the shortest run).")
        fig = rv.make_figure(rv_runs, [rv.frame_at_time(r[1][2], t_sel * rv.TAUB_S) for r in rv_runs],
                             rv_opts_draw, rv_models)
        st.pyplot(fig)
        save_figure_if_requested(fig, f"snapshot_{'_vs_'.join(rv_sel)}_{t_sel:.2f}tauB", save_plots, output_dir)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("Download PNG (300 dpi)", buf.getvalue(), f"snapshot_{t_sel:.2f}tauB.png", "image/png",
                           key="rv_dl_png")
        plt.close(fig)

    with tab_anim:
        c1, c2, c3, c4 = st.columns(4)
        a0, a1 = c1.slider("Time range (τB)", 0.0, float(t_max), (0.0, float(t_max)), key="rv_arange")
        n_fr = c2.number_input("Frames", 5, 400, 60, 5, key="rv_nframes")
        fps = c3.number_input("Frames per second", 1, 30, 12, 1, key="rv_fps")
        gif_dpi = c4.number_input("Resolution (dpi)", 50, 200, 90, 10, key="rv_gifdpi")
        st.caption("Uses all the colour and show settings from the sidebar. "
                   "With a fixed colour range off, the colour scale is taken from the last frame.")
        if st.button("Make animation", key="rv_make_gif"):
            bar = st.progress(0.0, text="Drawing frames...")
            try:
                gif = rv.make_gif(rv_runs, rv_opts_draw, t_start_s=a0 * rv.TAUB_S, t_end_s=a1 * rv.TAUB_S,
                                  n_frames=int(n_fr), fps=int(fps), dpi=int(gif_dpi), models=rv_models,
                                  progress=lambda p: bar.progress(p, text=f"Drawing frames... {p:.0%}"))
                st.session_state["rv_gif"] = gif
            except Exception as e:
                st.error(f"Could not make the animation: {e}")
            bar.empty()
        if st.session_state.get("rv_gif"):
            st.image(st.session_state["rv_gif"])
            st.download_button("Download GIF", st.session_state["rv_gif"], "run_animation.gif", "image/gif",
                               key="rv_dl_gif")
            if save_plots and output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    pth = os.path.join(output_dir, f"animation_{'_vs_'.join(rv_sel)}.gif")
                    with open(pth, "wb") as fh:
                        fh.write(st.session_state["rv_gif"])
                    st.info(f"Saved animation to: `{pth}`")
                except Exception as e:
                    st.warning(f"Could not save the animation: {e}")

    with tab_live:
        try:
            import plotly  # noqa: F401
        except ImportError:
            st.error("The live view needs plotly: run `pip install plotly` and restart the app.")
            st.stop()
        live_name = st.selectbox("Run", [r[0] for r in rv_runs], key="rv_live_run")
        li = [r[0] for r in rv_runs].index(live_name)
        lx, ly, lt = rv_runs[li][1]
        lmodel = rv_models[li]
        n_frames = len(lt)
        SSv = st.session_state
        if "rv_live_k" not in SSv or SSv["rv_live_k"] >= n_frames:
            SSv["rv_live_k"] = n_frames - 1
        sel_key = f"rv_live_selset_{live_name}"
        SSv.setdefault(sel_key, [])

        # ---- playback and display settings (outside the auto-refreshing part)
        pc = st.columns([1.1, 1.3, 1.3, 1.4, 2.6])
        playing = SSv.get("rv_playing", False)
        if pc[0].button("⏸ Pause" if playing else "▶ Play", key="rv_play_btn", type="primary"):
            SSv["rv_playing"] = not playing
            if not playing and SSv["rv_live_k"] >= n_frames - 1:
                SSv["rv_live_k"] = 0
            st.rerun()
        play_step = pc[1].number_input("frames per step", 1, max(1, n_frames // 2), max(1, n_frames // 100),
                                       key="rv_play_step")
        play_dt = pc[2].select_slider("seconds per step", [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0], 0.5, key="rv_play_dt")
        play_loop = pc[3].checkbox("loop", True, key="rv_play_loop")
        pc[4].caption(f"Model for the forces: {lmodel.label()}." +
                      (" Note: " + "; ".join(lmodel.notes) if lmodel.notes else ""))
        with st.expander("Force arrows", expanded=True):
            fa = st.columns([3.2, 1.6, 1.4, 1.4])
            arrow_types = fa[0].multiselect("Show", list(rv.FORCE_TYPES), default=["attraction", "acoustic", "streaming", "net"],
                                            format_func=lambda k_: rv.FORCE_TYPES[k_][1], key="rv_arrow_types",
                                            help="Untick a force to hide its arrows. Colours: pair force violet, core grey, "
                                                 "acoustic green, streaming blue, net red.")
            arrows_for = fa[1].radio("For", ["selected", "all"], format_func={"selected": "selected particles",
                                                                             "all": "all particles"}.get,
                                     key="rv_arrows_for", horizontal=True)
            auto_len = fa[2].checkbox("automatic length", True, key="rv_arrow_auto")
            a_scale = None if auto_len else fa[2].number_input("µm per k_BT/µm", 0.001, 10.0, 0.1, format="%.3f",
                                                             key="rv_arrow_scale")
            a_max = fa[3].number_input("clip arrows at (µm)", 1.0, 50.0, 8.0, 1.0, key="rv_arrow_max",
                                       help="The core force at overlap can be huge; longer arrows are cut here "
                                            "(hover an arrow head for its real size).")

        def _rerun_part():
            """Redraw only the live panel when possible (a fragment rerun), else the page."""
            try:
                st.rerun(scope="fragment")
            except Exception:     # not inside a fragment rerun right now
                st.rerun()

        @st.fragment(run_every=play_dt if playing else None)
        def _live_panel():
            if SSv.get("rv_playing"):
                nk = SSv["rv_live_k"] + int(play_step)
                if nk >= n_frames:
                    if play_loop:
                        nk = 0
                    else:
                        nk = n_frames - 1
                        SSv["rv_playing"] = False
                SSv["rv_live_k"] = nk
            cA, cB, cC, cD, cE = st.columns([1, 1, 6, 1, 1])
            step = int(play_step)
            if cA.button("⏮", key="rv_first"):
                SSv["rv_live_k"] = 0
            if cB.button("◀", key="rv_prev"):
                SSv["rv_live_k"] = max(SSv["rv_live_k"] - step, 0)
            if cD.button("▶", key="rv_next"):
                SSv["rv_live_k"] = min(SSv["rv_live_k"] + step, n_frames - 1)
            if cE.button("⏭", key="rv_last"):
                SSv["rv_live_k"] = n_frames - 1
            k = cC.slider("Frame", 0, n_frames - 1, key="rv_live_k",
                          help=f"Saved frames; ◀ ▶ and Play move {step} frame(s) = "
                               f"{lt[min(step, n_frames - 1)] / rv.TAUB_S:.2f} τB.")
            sel_list = SSv[sel_key]
            col_plot, col_info = st.columns([3, 2])
            with col_plot:
                lfig = rv.live_figure(lx, ly, lt, k, rv_opts_draw, lmodel, sel_list, show_arrows=bool(arrow_types),
                                      arrow_types=tuple(arrow_types), arrows_for=arrows_for, arrow_scale=a_scale,
                                      arrow_max=float(a_max), primary=SSv.get(sel_key + "_primary"))
                event = None
                try:
                    event = st.plotly_chart(lfig, key=f"rv_live_chart_{live_name}", on_select="rerun",
                                            selection_mode="points", use_container_width=False)
                except TypeError:
                    st.plotly_chart(lfig)
                    st.caption("Your Streamlit version can't report clicks; add particle numbers on the right.")
                pts = []
                try:
                    pts = event.selection.points if event and event.selection else []
                except Exception:
                    pts = []
                clicked = None
                for p_ in pts:
                    cd = p_.get("customdata")
                    if cd is not None:
                        clicked = int(cd[0] if isinstance(cd, (list, tuple)) else cd)
                if clicked is not None and clicked != SSv.get(sel_key + "_lastclick"):
                    SSv[sel_key + "_lastclick"] = clicked
                    cur = list(SSv[sel_key])
                    if clicked in cur:          # click again = untick
                        cur.remove(clicked)
                    else:
                        cur.append(clicked)
                        SSv[sel_key + "_primary"] = clicked
                    SSv[sel_key] = cur
                    SSv[sel_key + "_ver"] = SSv.get(sel_key + "_ver", 0) + 1   # list widget shows the click
                    _rerun_part()
            with col_info:
                st.caption("Click a particle to mark it, click it again to untick it. Several can be marked; "
                           "the force arrows are drawn for all marked particles.")
                ver = SSv.get(sel_key + "_ver", 0)
                cur = st.multiselect("Marked particles", list(range(lx.shape[1])), default=SSv[sel_key],
                                     key=f"rv_live_ms_{live_name}_{ver}")
                if cur != SSv[sel_key]:
                    SSv[sel_key] = cur          # edited in the list: keep the list widget as it is
                    _rerun_part()
                if st.button("Clear all", key="rv_live_clear"):
                    SSv[sel_key] = []
                    SSv[sel_key + "_lastclick"] = None
                    SSv[sel_key + "_ver"] = ver + 1
                    _rerun_part()
                if SSv[sel_key]:
                    shown = st.selectbox("Details for", SSv[sel_key], index=len(SSv[sel_key]) - 1,
                                         key=f"rv_live_detail_{live_name}")
                    SSv[sel_key + "_primary"] = shown
                    summary, rows = rv.particle_report(lx, ly, lt, k, int(shown), lmodel, float(rv_rc))
                    parts = rv.force_parts(np.c_[lx[k], ly[k]], lmodel, [int(shown)])
                    for t_, (c_, lab_) in rv.FORCE_TYPES.items():
                        fx, fy = parts[t_][0]
                        summary[f"{lab_} Fx, Fy (kBT/µm)"] = f"{fx:.2f}, {fy:.2f}  (|F| = {np.hypot(fx, fy):.2f})"
                    st.table(pd.DataFrame({"value": [str(v) for v in summary.values()]}, index=list(summary.keys())))
                    if np.linalg.norm(parts["net"][0]) < 1e-3:
                        Pk = np.c_[lx[k], ly[k]]
                        rr = np.hypot(*(Pk - Pk[int(shown)]).T); rr[int(shown)] = np.inf
                        rng_ = 2 * (1 + lmodel.ao[0]) if lmodel.ao else (6.0 if lmodel.A > 0 else 2.0)
                        why = []
                        if not lmodel.ao and lmodel.A <= 0:
                            why.append("this run has no attraction (A = 0), so the only pair force is the repulsive "
                                       f"core, which acts only below 2 µm (nearest neighbour {rr.min():.2f} µm)")
                        elif not (rr < rng_).any():
                            why.append(f"no other particle within the {'AO' if lmodel.ao else 'LJ' if lmodel.A > 0 else 'core'} "
                                       f"range ({rng_:.1f} µm centre to centre; nearest is {rr.min():.2f} µm)")
                        else:
                            why.append("the pulls from its neighbours cancel (symmetric surroundings, e.g. the start grid)")
                        if not lmodel.dn and not lmodel.F:
                            why.append("no acoustic field or flow in this run")
                        st.info("Zero model force: " + "; ".join(why) + ". Only Brownian motion moves it now.")
                    if not SSv.get("rv_playing"):
                        st.markdown("**Neighbours within 6.5 µm** (bonded = closer than the bond cutoff; "
                                    "forces are the model forces from each neighbour on this particle)")
                        st.dataframe(pd.DataFrame(rows), hide_index=True)
                else:
                    st.info("No particle marked yet.")

        _live_panel()

    st.stop()

# TODO: add plotly toogle
# TODO: Fix  ########### LOCUS PLOT ######### in Phase-transition detection
# TODO: Error bars + statistics ( You compute: Mean curve, deviation, error band for runs of the same tkb, tauB)

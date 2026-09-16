# app.py

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
    apply_curve_detectors,
    slice_by_time_window,
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
)
from pptx_export import build_pptx_from_images
from phase_panel import make_snapshot_phase_panel
from sidebar_widgets import render_cluster_params_widgets


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
                r"^([0-9]+(?:\.[0-9]+)?)Tkb[ _]*([0-9]+(?:\.[0-9]+)?)TauB(?:[ _]*([0-9]+(?:\.[0-9]+)?)(nm|dn))?$",
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
        return base

    if available_runs:
        all_tkbs = sorted({r[1] for r in available_runs})
        all_taus = sorted({r[2] for r in available_runs})
        all_thirds = sorted({r[3] for r in available_runs if r[3] is not None})
        all_units = sorted({r[4] for r in available_runs if r[4] is not None})

        st.sidebar.markdown("**Filter runs:**")
        filter_tkbs = st.sidebar.multiselect(
            "TkB values", all_tkbs, default=all_tkbs, key="multi_filter_tkb"
        )
        filter_taus = st.sidebar.multiselect(
            "TauB values", all_taus, default=all_taus, key="multi_filter_tau"
        )
        if all_thirds:
            unit_label = "/".join(all_units) if all_units else "3rd param"
            filter_thirds = st.sidebar.multiselect(
                f"{unit_label} values", all_thirds, default=all_thirds,
                key="multi_filter_third"
            )
        else:
            filter_thirds = []

        def run_passes_filter(r):
            if r[1] not in filter_tkbs:
                return False
            if r[2] not in filter_taus:
                return False
            if all_thirds and r[3] not in filter_thirds:
                return False
            return True

        filtered_runs = [r for r in available_runs if run_passes_filter(r)]

        selected_runs = st.sidebar.multiselect(
            "Select runs (from filtered):",
            options=filtered_runs,
            default=filtered_runs,
            key="multi_runs",
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
HIWI_ROOT = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data"
# app.py

DEFAULT_SAVE_DIR = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Saved Data for Streamlit"


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

    # assign a base color per group (3rd param)
    group_base_colors = {}
    if unique_thirds:
        group_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        for i, v in enumerate(unique_thirds):
            group_base_colors[v] = group_palette[i % len(group_palette)]

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
    st.write(f"Base folder: `{base_dir}`")
    st.write(f"Metrics: **{metrics_selected}**")
    st.write("Selected runs (TkB, TauB):")


    readable_runs = [
        f"{tkb:g} Tkb  {tau:g} TauB"
        for _, tkb, tau, *_ in selected_runs
    ]

    st.write("Selected runs:")
    for txt in readable_runs:
        st.write(f"- {txt}")

    metric_kwargs = dict(TauB=TauB, skip=skip, normY=int(normY))


    def run_label(s, show_taub):
        base = f"{s['tkb']:g} Tkb"
        if show_taub:
            base += f" (TauB {s['taub']:g})"
        if s.get("third_val") is not None and s.get("third_unit"):
            base += f" {s['third_val']:g}{s['third_unit']}"
        return base


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

                        ax.plot(
                            x,
                            y_grid,
                            label = f"{s['tkb']:g} Tkb (TauB {taub:g}) — {kind}" if show_taub_in_legend else f"{s['tkb']:g} Tkb — {kind}",
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
                    u = mid.get("third_unit") or ""
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

                npz_path = rundata_npz_path(base_dir, dataset_tag, tkb_val, tau_val, metric_label, third_val, third_unit)
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

                if use_gradient and len({float(r["tkb"]) for r in series}) == 1:
                    # same TkB, differentiate by TauB index
                    idx = taubs_sorted.index(float(s["taub"]))
                    frac = idx / max(len(taubs_sorted) - 1, 1)
                    col = cmap(frac) if cmap else None
                else:
                    col = get_line_color(s)
                ax.plot(
                    x, y_vals,
                    label=f"{s['tkb']:g} Tkb (TauB {s['taub']:g})" if show_taub_in_legend else f"{s['tkb']:g} Tkb",
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
                u = s.get("third_unit") or ""
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
            from collections import defaultdict
            from matplotlib.lines import Line2D

            unique_thirds_shape = sorted({c.get("third_val") for c in curves_for_shape if c.get("third_val") is not None})
            shape_palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
            third_color_map = {v: shape_palette[i % len(shape_palette)] for i, v in enumerate(unique_thirds_shape)}

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

            # --- compute per-run metrics ---
            results = []
            for c in curves_for_shape:
                y = np.asarray(c["y"], float)
                x = np.asarray(c["x"], float)
                if len(y) < 10:
                    continue
                y_sm = uniform_filter1d(y, size=smooth_win_shape)
                n = len(y_sm)
                early_seg = max(1, int(n * early_pct / 100))
                late_seg  = max(1, int(n * late_pct  / 100))
                early_slope = (y_sm[early_seg] - y_sm[0]) / (x[early_seg] - x[0] + 1e-12)
                late_slope  = (y_sm[-1] - y_sm[-late_seg]) / (x[-1] - x[-late_seg] + 1e-12)
                slope_ratio = abs(early_slope) / (abs(late_slope) + 1e-12)
                half_max = (np.nanmax(y_sm) + np.nanmin(y_sm)) / 2.0
                is_decreasing = y_sm[-1] < y_sm[0]
                if is_decreasing:
                    idx_half = np.where(y_sm <= half_max)[0]
                else:
                    idx_half = np.where(y_sm >= half_max)[0]
                t_half = float(x[idx_half[0]]) if len(idx_half) > 0 else float(x[-1])
                results.append({
                    "run": c["run"],
                    "tkb": float(c["tkb"]),
                    "third_val": c.get("third_val"),
                    "third_unit": c.get("third_unit") or "",
                    "slope_ratio": slope_ratio,
                    "t_half": t_half,
                    "y_final": float(y_sm[-1]),
                })

            # --- aggregate per nm group ---
            group_data = defaultdict(lambda: {"slope_ratios": [], "t_halfs": [], "y_finals": [], "unit": ""})
            for r in results:
                v = r["third_val"]
                if v is not None:
                    group_data[v]["slope_ratios"].append(r["slope_ratio"])
                    group_data[v]["t_halfs"].append(r["t_half"])
                    group_data[v]["y_finals"].append(r["y_final"])
                    group_data[v]["unit"] = r["third_unit"]

            group_summary = {}
            for v, d in group_data.items():
                group_summary[v] = {
                    "slope_ratio_mean": np.mean(d["slope_ratios"]),
                    "slope_ratio_std":  np.std(d["slope_ratios"]),
                    "t_half_mean":      np.mean(d["t_halfs"]),
                    "t_half_std":       np.std(d["t_halfs"]),
                    "spread":           np.std(d["y_finals"]),
                    "unit":             d["unit"],
                }

            # --- plot 2 rows x 3 cols ---
            fig_shape, axes = plt.subplots(2, 3, figsize=(15, 8))

            metric_keys = [
                ("slope_ratio",  "Early/late slope ratio",     "Slope ratio (exp→linear)"),
                ("t_half",       "Time to half-max",           "Time to 50% of final value"),
                ("y_final",      "Final value",                 "Final value"),
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
                    xs_r = [r["tkb"] for r in group]
                    ys_r = [r[mk] for r in group]
                    ax.scatter(xs_r, ys_r, color=col_c, s=60, zorder=3)
                    if connect_dots_row1:
                        ax.plot(xs_r, ys_r, color=col_c, linewidth=1, alpha=0.6)
                ax.set_xlabel("TkB")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{title}\n(per run, colored by nm)")
                apply_grid(ax, show_grid)

            # Row 2: per nm group, x = third_val, with error bars = std across TkB
            agg_metrics = [
                ("slope_ratio_mean", "slope_ratio_std", "Early/late slope ratio",  "Slope ratio (exp→linear)"),
                ("t_half_mean",      "t_half_std",      "Time to half-max",        "Time to 50% of final value"),
                ("spread",           None,              "Spread (std of final val)","Spread across TkB values"),
            ]
            for col, (mk_mean, mk_std, ylabel, title) in enumerate(agg_metrics):
                ax = axes[1][col]
                xs = sorted(group_summary.keys())
                ys = [group_summary[v][mk_mean] for v in xs]
                errs = [group_summary[v][mk_std] for v in xs] if mk_std else None
                cols_g = [third_color_map.get(v, "gray") for v in xs]
                units = [group_summary[v]["unit"] for v in xs]
                xlabels = [f"{v:g}{u}" for v, u in zip(xs, units)]
                for i, (xv, yv, col_c) in enumerate(zip(range(len(xs)), ys, cols_g)):
                    err = errs[i] if errs else None
                    ax.errorbar(xv, yv, yerr=err, fmt='o', color=col_c, markersize=8,
                                capsize=4, zorder=3)
                if connect_dots_row2:
                    ax.plot(range(len(xs)), ys, color="gray", linewidth=1, alpha=0.5, zorder=2)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xlabels, rotation=45, ha="right")
                ax.set_xlabel("dn value")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{title}\n(per group ± std across TkB)")
                apply_grid(ax, show_grid)

            # shared legend
            legend_els = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=third_color_map[v],
                       markersize=8, label=f"{v:g}{group_summary[v]['unit']}")
                for v in sorted(group_summary.keys())
            ]
            axes[0][0].legend(handles=legend_els, fontsize=9)

            plt.tight_layout()
            st.pyplot(fig_shape)
            plt.close(fig_shape)
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
                label=f"{c['tkb']:g} Tkb (TauB {c['taub']:g})" if show_taub_in_legend else f"{c['tkb']:g} Tkb",
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

# TODO: add plotly toogle
# TODO: Fix  ########### LOCUS PLOT ######### in Phase-transition detection
# TODO: Error bars + statistics ( You compute: Mean curve, deviation, error band for runs of the same tkb, tauB)

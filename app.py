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
from particle_distance_over_time import particle_distance_over_time
from plot_bond_orientation_at_time_step import (
    plot_bond_orientational_order_at_timestep,
)
from plot_crystal_at_timestep import plot_crystals_at_timestep
from plot_fourier_analysis import fourier_of_metric
from plot_together_tkb_func import plots_together
from rg_over_time import average_rg_over_time
from track_cluster_movement import track_cluster_centroids
from track_crystal_formation_over_time import detect_crystals_over_time
from phase_snapshot_fields import (
    snapshot_local_density,
    snapshot_displacement_vectors,
    snapshot_alignment_field,
    snapshot_displacement_vectors_voronoi
)


# ---------------------------------------------------------------------
# Default root directory (used as default value for the user-editable root)
# ---------------------------------------------------------------------
HIWI_ROOT = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI"


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
st.sidebar.header("0.5 Output saving")
save_plots = st.sidebar.checkbox(
    "Save all generated plots to folder", value=False, key="save_plots"
)
output_dir = st.sidebar.text_input(
    "Output folder path (for saving plots):",
    value="",
    key="output_dir",
)

# ========== EXTRA: PowerPoint export (collapsible, runs before st.stop) ==========
with st.sidebar.expander("📊 PowerPoint export (from existing PNGs)", expanded=False):
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
    ["Single plot", "Multiple plots", "Phase diagram"],
    key="plot_mode",
)

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
        "Show title", value=True, key="title_on_single"
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
    st.sidebar.header("2. Data folder (for multiple plots)")

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

    st.sidebar.header("3. Metric for curves")

    metric_options = [
        "Radius of gyration",
        "Area fraction",
        "Bond orientational order",
        "Detect crystals",
        "Particle distance",
        "Median total path distance",
        "Particle displacement (over time)",
        "Particle displacement (from initial)",
    ]

    metric_map = {
        "Radius of gyration": (average_rg_over_time, "Radius of gyration"),
        "Area fraction": (area_fraction_over_time, "Area fraction"),
        "Bond orientational order": (
            bond_orientational_order_over_time,
            "Bond orientational order",
        ),
        "Detect crystals": (detect_crystals_over_time, "Crystal metric"),
        "Particle distance": (particle_distance_over_time, "Particle distance"),
        "Median total path distance": (
            median_total_path_distance_over_time,
            "Median path distance",
        ),
        "Particle displacement (over time)": (
            particle_displacement_over_time,
            "Displacement",
        ),
        "Particle displacement (from initial)": (
            particle_displacement_from_inintal_position_over_time,
            "Displacement from initial",
        ),
    }

    metrics_selected = st.sidebar.multiselect(
        "Select metric function(s) (y-axis):",
        metric_options,
        default=[metric_options[0]],
        key="multi_metric",
    )

    if not metrics_selected:
        st.sidebar.warning("Select at least one metric.")
        st.stop()

    # ---- TkB selection: detect & tick, plus optional custom ----
    st.sidebar.header("4. TkB values")

    available_tkbs = []
    if base_dir and os.path.isdir(base_dir):
        try:
            folder_re = re.compile(r"^([0-9]+(?:\.[0-9]+)?)Tkb_")
            for fn in os.listdir(base_dir):
                m = folder_re.match(fn)
                if m:
                    available_tkbs.append(float(m.group(1)))
            available_tkbs = sorted(set(available_tkbs))
        except Exception as e:
            st.sidebar.error(f"Could not scan TkB folders in {base_dir}: {e}")

    if available_tkbs:
        TkBs_selected = st.sidebar.multiselect(
            "Select TkB values (from folders):",
            available_tkbs,
            default=available_tkbs,
            key="multi_tkbs",
        )
    else:
        TkBs_selected = []
        st.sidebar.warning(
            "No Tkb_* subfolders detected. You can still add custom TkB values below."
        )

    custom_tkb_str = st.sidebar.text_input(
        "Add custom TkB values (optional, comma-separated):",
        value="",
        key="multi_tkb_custom",
    )
    TkBs = list(TkBs_selected)
    if custom_tkb_str.strip():
        try:
            extra = [float(s.strip()) for s in custom_tkb_str.split(",") if s.strip()]
            TkBs = sorted(set(TkBs + extra))
        except ValueError:
            st.sidebar.error("Could not parse custom TkB values.")

    st.sidebar.header("5. File names")
    x_name = st.sidebar.text_input(
        "X filename inside each Tkb_* folder:", value="datax.csv", key="multi_xname"
    )
    y_name = st.sidebar.text_input(
        "Y filename inside each Tkb_* folder:", value="datay.csv", key="multi_yname"
    )

    st.sidebar.header("6. Common parameters passed to metric_func")
    TauB = st.sidebar.number_input(
        "TauB (Brownian time)", value=2.0, step=0.5, key="TauB_multi"
    )
    skip = st.sidebar.number_input(
        "Skip (frame stride)", value=1, step=1, min_value=0, key="skip_multi"
    )
    normY = st.sidebar.checkbox(
        "Normalize Y axis", value=True, key="normY_multi"
    )

    # --- Styling options for multiple plots ---
    st.sidebar.header("7. Styling (multiple plots)")
    legend_on_multi = st.sidebar.checkbox(
        "Show legend", value=True, key="legend_on_multi"
    )
    legend_fontsize_multi = st.sidebar.number_input(
        "Legend font size",
        value=16,
        min_value=1,
        max_value=40,
        key="legend_fs_multi",
    )
    axis_fontsize_multi = st.sidebar.number_input(
        "Axis label/tick font size",
        value=18,
        min_value=1,
        max_value=40,
        key="axis_fs_multi",
    )
    title_on_multi = st.sidebar.checkbox(
        "Show title", value=True, key="title_on_multi"
    )
    title_fontsize_multi = st.sidebar.number_input(
        "Title font size",
        value=20,
        min_value=1,
        max_value=60,
        key="title_fs_multi",
    )

    color_mode_multi = st.sidebar.radio(
        "Curve colors",
        ["Matplotlib default", "Gradient over TkB"],
        key="color_mode_multi",
    )

    if color_mode_multi == "Gradient over TkB":
        st.sidebar.write("Gradient colors:")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            color_start_multi = st.color_picker(
                "Start",
                value="#000000",
                key="color_start_multi",
            )
        with col2:
            color_end_multi = st.color_picker(
                "End",
                value="#CCCCCC",
                key="color_end_multi",
            )
    else:
        color_start_multi = "#000000"
        color_end_multi = "#CCCCCC"

    x_axis_mode_multi = st.sidebar.radio(
        "X-axis units",
        ["Seconds", "Brownian time T_kB"],
        key="x_axis_mode_multi",
    )

    run = st.sidebar.button("▶ Run multiple-plot routine", key="run_multi")

    if not run:
        st.stop()

    if not base_dir:
        st.warning("Please select or enter a base folder path.")
        st.stop()

    if not TkBs:
        st.warning("No TkB values selected. Please tick some or add custom values.")
        st.stop()

    st.subheader("Multiple plots over TkB")
    st.write(f"Base folder: `{base_dir}`")
    st.write(f"Metrics: **{metrics_selected}**")
    st.write(f"TkB values: `{TkBs}`")

    metric_kwargs = dict(TauB=TauB, skip=skip, normY=int(normY))

    # ----- RUN ALL SELECTED METRICS -----
    for metric_label in metrics_selected:
        plt.close("all")
        metric_func, default_ylabel = metric_map[metric_label]

        st.markdown(f"### Metric: {metric_label}")

        plots_together(
            base_dir=base_dir,
            metric_func=metric_func,
            metric_kwargs=metric_kwargs,
            plot_title=default_ylabel,
            x_name=x_name,
            y_name=y_name,
            TkBs=TkBs,
            color_mode="gradient"
            if color_mode_multi == "Gradient over TkB"
            else "default",
            color_start=color_start_multi,
            color_end=color_end_multi,
            x_axis_mode="tkb"
            if x_axis_mode_multi == "Brownian time T_kB"
            else "seconds",
        )

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

    st.success("Multiple-plot figure(s) done ✅")


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

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
from rg_over_time import average_rg_over_time
from track_cluster_movement import track_cluster_centroids
from track_crystal_formation_over_time import detect_crystals_over_time
from phase_snapshot_fields import (
    snapshot_local_density,
    snapshot_displacement_vectors,
    snapshot_alignment_field,
    snapshot_displacement_vectors_voronoi
)

import json, hashlib

import csv
from datetime import datetime


import json, hashlib
from pathlib import Path
from datetime import datetime
import ast
import operator as op
import numpy as np


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
            if metric_label in ["Number of clusters", "Average cluster size"]:
                kwargs_run["eps"] = float(cluster_eps_multi)
                kwargs_run["min_samples"] = int(cluster_min_samples_multi)
                kwargs_run["min_cluster_size"] = int(cluster_min_cluster_size_multi)
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

def load_or_compute_metric_cached(base_dir: str, run_folder: str, fx: str, fy: str, metric_func, metric_kwargs: dict):
    p = metric_cache_path(base_dir, run_folder, metric_func.__name__, metric_kwargs)
    if p.exists():
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
    ["Single plot", "Multiple plots", "Phase diagram", "Summary plots"],
    key="plot_mode",
)

metric_options = [
    "Radius of gyration",
    "Area fraction",
    "Bond orientational order",
    "Detect crystals",
    "Particle distance",
    "Median total path distance",
    "Particle displacement (over time)",
    "Particle displacement (from initial)",
    "Number of clusters",
    "Average cluster size",
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
    "Number of clusters": (num_clusters_over_time, "Number of clusters"),
    "Average cluster size": (avg_cluster_size_over_time, "Average cluster size"),

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
        dataset_tag = os.path.basename(base_dir.rstrip("\\/"))  # e.g. "NAF"

    else:
        base_dir = st.sidebar.text_input(
            "Custom base folder path:",
            value="",
            key="multi_folder_custom",
        )

    base_dir = base_dir.strip().strip('"').strip("'")

    st.sidebar.header("3. Metric for curves")


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

    # Each run is a single simulation folder (TkB, TauB)
    # Folder names like: "5Tkb_2TauB", "50Tkb 10TauB", etc.
    available_runs = []  # list of (folder_path, tkb, taub)

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

    available_runs = sorted(available_runs, key=lambda r: (r[1], r[2]))  # sort by TkB, TauB

    def format_run_option(run) -> str:
        _, tkb, tau = run
        return f"{tkb:g} Tkb (TauB: {tau:g})"

    if available_runs:
        selected_runs = st.sidebar.multiselect(
            "Select TkB / TauB combinations (from folders):",
            options=available_runs,
            default=available_runs,
            key="multi_runs",
            format_func=format_run_option,
        )
    else:
        selected_runs = []
        st.sidebar.warning(
            "No Tkb_*TauB* subfolders detected. Please check your base folder."
        )

    if not selected_runs:
        st.warning("No TkB/TauB runs selected.")
        st.stop()



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

    cluster_eps_multi = st.sidebar.number_input("DBSCAN eps", value=3.5, step=0.1, key="cluster_eps_multi")

    cluster_min_samples_multi = st.sidebar.number_input("DBSCAN min_samples", value=3, step=1, min_value=1,
                                                        key="cluster_min_samples_multi")

    cluster_min_cluster_size_multi = st.sidebar.number_input("Min cluster size", value=3, step=1, min_value=1,
                                                             key="cluster_min_cluster_size_multi")

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
        ["Seconds", "Brownian time τ_B"],
        key="x_axis_mode_multi",
    )



    st.sidebar.checkbox(
        "Prefer saved datasets (skip recompute if available)",
        value=True,
        key="prefer_saved_dataset_multi",
    )
    prefer_saved_dataset = st.session_state["prefer_saved_dataset_multi"]

    # If checked => recompute + overwrite saved arrays (and optionally export CSV)
    export_data = st.sidebar.checkbox(
        "Export (recompute + overwrite saved run data)",
        value=False,
        key="export_data_multi",
    )
    export_dir = st.sidebar.text_input(
        "Optional CSV export folder (leave empty to skip CSV export):",
        value=DEFAULT_SAVE_DIR,
        key="export_dir_multi",
    )

    st.markdown("---")
    st.subheader("🧮 Calculator (typed expression)")

    calc_metric_options = metric_options  # your existing list

    mA_label = st.selectbox("Metric A", calc_metric_options, key="expr_calc_A")
    mB_label = st.selectbox("Metric B", calc_metric_options, key="expr_calc_B")

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

    plot_expr = st.button("Plot derived (auto-load/compute)", key="expr_calc_plot")

    st.caption("Allowed functions: " + ", ".join(sorted(_ALLOWED_FUNCS.keys())))

    if plot_expr:
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

        with st.spinner("Loading / computing Metric A..."):
            A_series, missingA = get_series_for_metric(
                metric_label=mA_label,
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
            )

        with st.spinner("Loading / computing Metric B..."):
            B_series, missingB = get_series_for_metric(
                metric_label=mB_label,
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

        for run_name in common_runs:
            a = A_map[run_name]
            b = B_map[run_name]

            taub_a = float(a["taub"])
            taub_b = float(b["taub"])
            if abs(taub_a - taub_b) > 1e-9:
                st.warning(f"Skipping {run_name}: TauB mismatch between metrics ({taub_a} vs {taub_b})")
                continue

            t_grid_sec, yA, yB = align_y_by_taub_length(a["y"], b["y"], taub_a)

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

            # ✅ choose x-axis correctly
            if plot_fft:
                x = fft_freqs(t_grid_sec)  # Hz
            else:
                x = (t_grid_sec / 13.513) if x_axis_mode_multi == "Brownian time τ_B" else t_grid_sec

            ax.plot(x, y_out, label=run_name)

        ax.set_title(calc_title)

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

    ########## RUN ####################
    run = st.sidebar.button("▶ Run multiple-plot routine", key="run_multi")



    if not run:
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

        st.markdown(f"### Metric: {metric_label}")

        # ------------------------------------------------------------
        # NEW: multi-run plotting with
        #  - correct per-run TauB on x-axis
        #  - same-TauB runs resampled to same x-length
        #  - on-disk caching for faster reruns
        # ------------------------------------------------------------

        dataset_tag = os.path.basename(base_dir.rstrip("\\/"))  # e.g. NAF, ASF, ARF

        series = []  # will contain dicts with run/tkb/taub/y_len only; we regenerate x later

        missing = []
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
                if metric_label in ["Number of clusters", "Average cluster size"]:
                    kwargs_run["eps"] = float(cluster_eps_multi)
                    kwargs_run["min_samples"] = int(cluster_min_samples_multi)
                    kwargs_run["min_cluster_size"] = int(cluster_min_cluster_size_multi)
                t, y = load_or_compute_metric_cached(
                    base_dir=base_dir,
                    run_folder=run_name,
                    fx=fx,
                    fy=fy,
                    metric_func=metric_func,
                    metric_kwargs=kwargs_run,
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

        if not series:
            st.warning("No valid runs found.")
            st.stop()


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


        # Group by TauB so same-TauB runs share identical x-axis length
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
                ax.plot(x, y_grid, label=f"{s['tkb']:g} Tkb (TauB {taub:g})")

        ax.set_title(default_ylabel)
        ax.set_ylabel(default_ylabel)
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
        "Window start (fraction of run)", 0.0, 0.9, 0.0, 0.05, key="summary_window_start"
    )
    t_end_frac = st.sidebar.slider(
        "Window end (fraction of run)", 0.1, 1.0, 1.0, 0.05, key="summary_window_end"
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

    st.sidebar.header("3. Files")
    x_name = st.sidebar.text_input("X filename:", value="datax.csv", key="summary_xname")
    y_name = st.sidebar.text_input("Y filename:", value="datay.csv", key="summary_yname")

    st.sidebar.header("4. Metric curve to summarize")
    metric_label = st.sidebar.selectbox("Metric:", metric_options, key="summary_metric_label")

    st.sidebar.header("5. Common parameters")
    skip = st.sidebar.number_input("Skip (frame stride)", value=1, step=1, min_value=0, key="summary_skip")
    normY = st.sidebar.checkbox("Normalize Y axis (only affects plotting)", value=True, key="summary_normY")

    # (optional) share your cluster params if you want summary for cluster metrics
    st.sidebar.header("Cluster metrics parameters")
    cluster_eps_multi = st.sidebar.number_input("DBSCAN eps", value=3.5, step=0.1, key="summary_cluster_eps")
    cluster_min_samples_multi = st.sidebar.number_input("DBSCAN min_samples", value=3, step=1, min_value=1, key="summary_cluster_mins")
    cluster_min_cluster_size_multi = st.sidebar.number_input("Min cluster size", value=3, step=1, min_value=1, key="summary_cluster_minsz")

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

    # --- load/compute series using your existing helper ---
    # IMPORTANT: add cluster kwargs when needed
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
        export_dir="",  # we export scalars separately below
    )

    if missing:
        st.warning("Missing data files for: " + ", ".join(missing))

    # if metric is cluster-based, force recompute branch params (same idea as calculator fix)
    # easiest: just overwrite series by recomputing with params if needed
    if metric_label in ["Number of clusters", "Average cluster size"]:
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
                eps=float(cluster_eps_multi),
                min_samples=int(cluster_min_samples_multi),
                min_cluster_size=int(cluster_min_cluster_size_multi),
            )
            t, y = load_or_compute_metric_cached(
                base_dir=base_dir,
                run_folder=run_name,
                fx=fx,
                fy=fy,
                metric_func=metric_func,
                metric_kwargs=kwargs_run,
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


    # --- plot ---
    plt.close("all")
    plt.figure()
    ax = plt.gca()

    if x_choice == "TkB":
        xvals = np.array([r[1] for r in rows], float)
        xlabel = "TkB"
        group_key = np.array([r[2] for r in rows], float)  # group by TauB in legend
        label_fmt = lambda ta: f"TauB {ta:g}"
    else:
        xvals = np.array([r[2] for r in rows], float)
        xlabel = "TauB"
        group_key = np.array([r[1] for r in rows], float)  # group by TkB in legend
        label_fmt = lambda tk: f"TkB {tk:g}"

    yvals = np.array([r[3] for r in rows], float)

    for g in sorted(set(group_key)):
        mask = group_key == g
        ax.plot(xvals[mask], yvals[mask], marker="o", linestyle="-", label=label_fmt(g))

    ax.set_title(f"{reducer}({metric_label}) vs {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{reducer}({metric_label})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = plt.gcf()
    apply_global_styling(fig, legend_on=True, legend_fontsize=14, axis_fontsize=16, title_on=True, title_fontsize=18)
    st.pyplot(fig)

    # --- export scalars (optional) ---
    if export_dir.strip():
        out = Path(export_dir.strip())
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / slugify(f"{dataset_tag}_{metric_label}_{reducer}_vs_{xlabel}")  # no extension yet
        csv_path = csv_path.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            f.write("run,tkb,taub,scalar\n")
            for run_name, tkb, taub, scalar in rows:
                f.write(f"{run_name},{tkb},{taub},{scalar}\n")
        st.info(f"Exported scalars to: `{csv_path}`")

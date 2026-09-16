"""On-disk caching for computed metric curves and dataset series (NPZ-backed),
plus CSV export helpers and small array alignment/resampling utilities.

Extracted from app.py (Phase 1 de-spaghetti pass).

Note: app.py previously defined _stable_hash twice (identical implementations,
once for the dataset cache and once for the per-run metric cache); only one
copy is kept here and reused by both.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st

from plot_utils import slugify


RUN_DATA_DIRNAME = ".run_data"

def rundata_dir(base_dir: str) -> Path:
    d = Path(base_dir) / RUN_DATA_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d

def make_rundata_stem(dataset_tag: str, tkb: float, taub: float, metric_label: str, third_val=None, third_unit=None) -> str:
    third_part = f"_{third_val:g}{third_unit}" if third_val is not None and third_unit else ""
    return slugify(f"{dataset_tag}_{tkb:g}tkb_{taub:g}taub{third_part}_{metric_label}")

def rundata_npz_path(base_dir: str, dataset_tag: str, tkb: float, taub: float, metric_label: str, third_val=None, third_unit=None) -> Path:
    return rundata_dir(base_dir) / (make_rundata_stem(dataset_tag, tkb, taub, metric_label, third_val, third_unit) + ".npz")

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

DATASET_CACHE_DIRNAME = ".dataset_cache"

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

    for run_tuple in selected_runs:
        folder_path = run_tuple[0]
        tkb_val = float(run_tuple[1])
        tau_val = float(run_tuple[2])
        third_val = run_tuple[3] if len(run_tuple) > 3 else None
        third_unit = run_tuple[4] if len(run_tuple) > 4 else None
        run_name = os.path.basename(folder_path)

        npz_path = rundata_npz_path(base_dir, dataset_tag, tkb_val, tau_val, metric_label, third_val, third_unit)

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

        series.append({"run": run_name, "tkb": tkb_val, "taub": tau_val, "third_val": third_val, "third_unit": third_unit, "y": np.asarray(y, float)})

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

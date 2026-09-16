"""Curve-analysis helpers: knee/turning/flattening/threshold/peak detectors,
overlay-locus builders, and simple time-window/slope utilities.

Extracted from app.py (Phase 1 de-spaghetti pass). Pure numpy/matplotlib-axes
logic only -- no Streamlit widgets live here.

Note: app.py previously defined _overlay_feature_locus twice (a 4-positional-arg
version and a keyword-only version); the first was dead code, silently shadowed
by the second. Only the live (keyword-only) version is kept here.
"""

import re

import numpy as np


_num_re = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

def _try_float_from_label(s: str):
    m = _num_re.search(str(s))
    return float(m.group(1)) if m else None

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


def apply_curve_detectors(
    ax,
    x,
    y,
    *,
    smooth_win,
    sustain_frac,
    slope_eps,
    label_markers,
    y_level,
    do_knee=True,
    do_turn=True,
    do_flat=True,
    do_thr=False,
    thr_mode="% of max (per curve)",
    thr_val=0.6,
    thr_dir="rising",
    thr_sustain=3,
    do_auc=False,
    do_maxslope=False,
    maxslope_smooth_win=None,
    do_peaks=False,
    peak_min_prom=0.0,
    peak_min_dist=5,
    do_pwlin=False,
    pw_min_seg=0.1,
    key_prefix="t_",
):
    """
    Run the shared post-analysis detector suite (knee / turning point /
    flattening / threshold crossing / AUC+mean / max-slope / peaks /
    piecewise-linear breakpoint) against one already-cleaned (x, y) curve,
    drawing vertical marker lines (+ optional text labels) on `ax`.

    Returns (row, y_level): `row` is a dict of result fields (prefixed with
    `key_prefix` for the per-feature x-locations, unprefixed for shared
    scalar fields like "thr"/"auc"/"mean"/"n_peaks"/"pw_sse") and `y_level`
    is the updated marker-label stack offset to pass into the next curve.

    `maxslope_smooth_win` lets a caller pass a (possibly clamped) smoothing
    window specifically for the max-|slope| marker, independent of the
    window used by the other detectors; it defaults to `smooth_win`.
    """
    row = {}

    if do_knee:
        xk, _ = detect_knee_point_triangle(x, y, smooth_win=smooth_win)
        row[f"{key_prefix}knee"] = xk
        if np.isfinite(xk):
            ax.axvline(xk, linestyle="--", linewidth=1.5)
            if label_markers:
                _label_vline(ax, xk, "knee", y_level=y_level); y_level += 1

    if do_turn:
        xt, _ = detect_turning_point_curvature(x, y, smooth_win=smooth_win)
        row[f"{key_prefix}turn"] = xt
        if np.isfinite(xt):
            ax.axvline(xt, linestyle=":", linewidth=1.5)
            if label_markers:
                _label_vline(ax, xt, "turn", y_level=y_level); y_level += 1

    if do_flat:
        xf, _ = detect_flattening_point(x, y, smooth_win=smooth_win, slope_eps=slope_eps, sustain_frac=sustain_frac)
        row[f"{key_prefix}flat"] = xf
        if np.isfinite(xf):
            ax.axvline(xf, linestyle="-.", linewidth=1.5)
            if label_markers:
                _label_vline(ax, xf, "flat", y_level=y_level); y_level += 1

    if do_thr:
        if "max" in thr_mode:
            thr = float(thr_val) * float(np.nanmax(y))
        else:
            thr = float(thr_val)

        xthr, _ = first_sustained_crossing(x, y, thr, direction=thr_dir, sustain=int(thr_sustain))
        row["thr"] = thr
        row[f"{key_prefix}thr"] = xthr
        if np.isfinite(xthr):
            ax.axvline(xthr, linestyle="-", linewidth=2.0)
            if label_markers:
                _label_vline(ax, xthr, "thr", y_level=y_level); y_level += 1

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
        sw = maxslope_smooth_win if maxslope_smooth_win is not None else smooth_win
        dy = np.gradient(_moving_average(y, sw), x)
        i = int(np.nanargmax(np.abs(dy)))
        row[f"{key_prefix}maxabs_slope"] = float(x[i])
        row["maxabs_slope"] = float(dy[i])
        ax.axvline(x[i], linestyle="--", linewidth=1.2)
        if label_markers:
            _label_vline(ax, x[i], "max|slope|", y_level=y_level); y_level += 1

    if do_peaks:
        idxs = simple_peaks(y, min_prom=float(peak_min_prom), min_dist=int(peak_min_dist))
        row["n_peaks"] = len(idxs)
        for j, i in enumerate(idxs[:10]):
            ax.axvline(x[i], linestyle=":", linewidth=1.0)
            if label_markers and j == 0:
                _label_vline(ax, x[i], "peaks", y_level=y_level); y_level += 1

    if do_pwlin:
        xb, _, sse = piecewise_linear_breakpoint(x, y, min_seg_frac=float(pw_min_seg))
        row[f"{key_prefix}break"] = xb
        row["pw_sse"] = sse
        if np.isfinite(xb):
            ax.axvline(xb, linestyle="-", linewidth=2.5)
            if label_markers:
                _label_vline(ax, xb, "break", y_level=y_level); y_level += 1

    return row, y_level


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

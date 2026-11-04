"""viewer3d.py - 3D run viewer for the app (runs of simulation/bd_engine.py with dim = 3 and simulation/airpods_bd3d.py).

A 3D run folder has datax/datay/dataz.csv (COMSOL export layout: position row / time row pairs) and usually a
sim_config.json. Functions return plotly / matplotlib figures and per-frame observables:
  list_3d_runs(root)        all run folders with a dataz.csv below the data folder (ARF_3D, NAF_3D, _quasi3D, ...)
  load_run3d(folder)        positions (frames, N, 3), times (s), settings
  frame_stats(P, ...)       3D coordination, largest cluster, spanning, stacked fraction, crystalline fraction
  fig_3d(P, ...)            plotly 3D scatter (one frame) with the chamber outline
  fig_views(P, ...)         matplotlib top (x-y) and side (y-z) views
  fig_stats_over_time(...)  observables against time
"""
from __future__ import annotations

import json
import os

import numpy as np

TAUB_S = 13.51288926834258
A_UM = 1.0


def list_3d_runs(root: str) -> list[tuple[str, str]]:
    """(label 'folder/run', path) of every run folder with dataz.csv, one or two levels below root."""
    out = []
    if not os.path.isdir(root):
        return out
    for f in sorted(os.listdir(root)):
        fd = os.path.join(root, f)
        if not os.path.isdir(fd) or f.startswith("."):
            continue
        if os.path.isfile(os.path.join(fd, "dataz.csv")):
            out.append((f, fd))
            continue
        for r in sorted(os.listdir(fd)):
            rd = os.path.join(fd, r)
            if os.path.isfile(os.path.join(rd, "dataz.csv")):
                out.append((f"{f}/{r}", rd))
    return out


def load_run3d(folder: str):
    arr = []
    for nm in ("datax.csv", "datay.csv", "dataz.csv"):
        a = np.loadtxt(os.path.join(folder, nm), delimiter=",", skiprows=1)
        arr.append(a[0::2])
        t = a[1::2, 0]
    cfg = {}
    f = os.path.join(folder, "sim_config.json")
    if os.path.isfile(f):
        try:
            with open(f) as fh:
                cfg = json.load(fh)
        except Exception:
            cfg = {}
    return np.stack(arr, axis=-1), t, cfg


def _pairs(P, rc):
    try:
        from scipy.spatial import cKDTree
        b = cKDTree(P).query_pairs(rc, output_type="ndarray")
        return b[:, 0], b[:, 1]
    except ImportError:
        d = P[:, None] - P[None]
        r = np.sqrt((d ** 2).sum(-1))
        i, j = np.nonzero(np.triu(r < rc, 1))
        return i, j


def frame_stats(P, rc=2.9, box=50.0, a=A_UM):
    """3D observables of one frame (same definitions as Data/_tools/obs3d.py)."""
    n = len(P)
    i, j = _pairs(P, rc)
    z = np.bincount(np.r_[i, j], minlength=n)
    lab = np.arange(n)

    def root(u):
        while lab[u] != u:
            lab[u] = lab[lab[u]]
            u = lab[u]
        return u
    for u, v in zip(i, j):
        ru, rv = root(u), root(v)
        if ru != rv:
            lab[ru] = rv
    roots = np.array([root(u) for u in range(n)])
    sizes = np.bincount(roots)
    m = roots == np.argmax(sizes)
    spans = max(np.ptp(P[m, 0]), np.ptp(P[m, 1])) > 0.8 * box
    ii, jj = _pairs(P, 2.2 * a + 2.0 * a)            # candidates for "resting on"
    dz = P[ii, 2] - P[jj, 2]
    lat = np.hypot(P[ii, 0] - P[jj, 0], P[ii, 1] - P[jj, 1])
    top = np.zeros(n, bool)
    sel = lat < 2.2 * a
    top[ii[sel & (dz > a)]] = True
    top[jj[sel & (dz < -a)]] = True
    return dict(z3=float(z.mean()), Smax=float(sizes.max() / n), spans=bool(spans),
                f_on_top=float(top.mean()), f_cryst=float((z >= 10).mean()),
                h_mean=float(P[:, 2].mean()), h_max=float(P[:, 2].max())), z


def _box_geometry(cfg, P):
    half = float(cfg.get("box_half", 25.0))
    H = cfg.get("H")
    top = float(H) if H not in (None, "", 0) else max(float(np.max(P[..., 2])) + A_UM, 4.0)
    return half, top


def fig_3d(P, cfg=None, color="height", z=None, title=""):
    """Plotly 3D scatter of one frame. color: 'height' or 'coordination' (pass z)."""
    import plotly.graph_objects as go
    cfg = cfg or {}
    half, top = _box_geometry(cfg, P)
    c = P[:, 2] if color == "height" or z is None else z
    cl = "height z (µm)" if color == "height" or z is None else "3D coordination"
    fig = go.Figure(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                 marker=dict(size=4 if len(P) > 500 else 6, color=c, colorscale="Blues",
                                             cmin=float(np.min(c)), cmax=float(np.max(c)),
                                             colorbar=dict(title=cl, thickness=12), line=dict(width=0)),
                                 hovertemplate="x %{x:.2f} µm<br>y %{y:.2f} µm<br>z %{z:.2f} µm<extra></extra>"))
    # chamber outline
    xs, ys = [-half, half, half, -half, -half], [-half, -half, half, half, -half]
    for zz in (0.0, top):
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[zz] * 5, mode="lines", line=dict(color="#999", width=2),
                                   hoverinfo="skip", showlegend=False))
    for x0, y0 in zip(xs[:4], ys[:4]):
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[0, top], mode="lines", line=dict(color="#999", width=2),
                                   hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title, height=620, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                      scene=dict(xaxis_title="x (µm)", yaxis_title="y (µm) – field axis", zaxis_title="z (µm)",
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=max(top / (2 * half), 0.15))))
    return fig


def fig_views(P, cfg=None, title=""):
    """Top view (x-y, colour = height) and side view (y-z, along the field axis)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    cfg = cfg or {}
    half, top = _box_geometry(cfg, P)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.4]}, constrained_layout=True)
    order = np.argsort(P[:, 2])
    cmap = plt.get_cmap("Blues")
    zmin, zmax = float(P[:, 2].min()), max(float(P[:, 2].max()), float(P[:, 2].min()) + 1e-6)
    for k in order:
        ax[0].add_patch(Circle((P[k, 0], P[k, 1]), A_UM, color=cmap(0.25 + 0.75 * (P[k, 2] - zmin) / (zmax - zmin)),
                               ec="#333", lw=0.3))
    ax[0].set_xlim(-half, half); ax[0].set_ylim(-half, half); ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)"); ax[0].set_title("Top view (darker = higher)", loc="left", fontsize=9)
    for k in np.argsort(P[:, 0]):
        ax[1].add_patch(Circle((P[k, 1], P[k, 2]), A_UM, color="#2a78d6", alpha=0.35, ec="#184f95", lw=0.3))
    ax[1].axhline(0, color="#555", lw=1)
    if cfg.get("H") not in (None, "", 0):
        ax[1].axhline(float(cfg["H"]), color="#555", lw=1, ls="--")
    ax[1].set_xlim(-half, half); ax[1].set_ylim(-0.5, top + 0.5); ax[1].set_aspect("equal")
    ax[1].set_xlabel("y (µm), field axis"); ax[1].set_ylabel("z (µm)")
    ax[1].set_title("Side view along x (all particles)", loc="left", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig


def fig_stats_over_time(frames, t, cfg=None, every=None):
    """Observables against time (subsampled to <= 60 frames)."""
    import matplotlib.pyplot as plt
    cfg = cfg or {}
    half, _ = _box_geometry(cfg, frames[0])
    idx = np.unique(np.linspace(0, len(t) - 1, min(60, len(t))).astype(int)) if every is None else np.arange(0, len(t), every)
    S = [frame_stats(frames[k], box=2 * half)[0] for k in idx]
    tt = t[idx] / TAUB_S
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), constrained_layout=True)
    ax[0].plot(tt, [s["z3"] for s in S], color="#2a78d6"); ax[0].set_ylabel("3D coordination z")
    ax[1].plot(tt, [s["Smax"] for s in S], color="#2a78d6", label="largest cluster")
    ax[1].plot(tt, [s["f_on_top"] for s in S], color="#eb6834", label="resting on another")
    ax[1].plot(tt, [s["f_cryst"] for s in S], color="#1baf7a", label="≥ 10 neighbours")
    ax[1].set_ylabel("fraction"); ax[1].legend(frameon=False, fontsize=7)
    ax[2].plot(tt, [s["h_mean"] for s in S], color="#2a78d6", label="mean"); ax[2].plot(tt, [s["h_max"] for s in S], color="#86b6ef", label="max")
    ax[2].set_ylabel("height (µm)"); ax[2].legend(frameon=False, fontsize=7)
    for a in ax:
        a.set_xlabel("t (τ$_B$)")
        a.spines[["top", "right"]].set_visible(False)
    return fig, S

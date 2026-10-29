"""Run viewer: draw a run's particles at any saved time, side by side for several
runs, and turn a run into an animated GIF. Used by the "Run viewer" section of app.py.

Works on the COMSOL export layout and on the Python BD runs (same layout):
datax.csv / datay.csv, header row, then alternating position rows (µm) and time rows (s).
"""
from __future__ import annotations

import io
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TAUB_S = 13.514
COLOR_MODES = ["Plain", "Coordination number", "Local order ψ6", "Clusters", "Speed"]

_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_PLAIN = "#2a78d6"
_LARGEST = "#eb6834"
_OTHER_CLUSTER = "#3987e5"
_SINGLE = "#b8b6ad"


def list_run_folders(base_dir: str) -> list[str]:
    if not base_dir or not os.path.isdir(base_dir):
        return []
    return sorted(d for d in os.listdir(base_dir)
                  if os.path.isfile(os.path.join(base_dir, d, "datax.csv"))
                  and os.path.isfile(os.path.join(base_dir, d, "datay.csv")))


def load_run(folder: str):
    """x, y: frames x particles (µm); t: seconds."""
    rx = np.loadtxt(os.path.join(folder, "datax.csv"), delimiter=",", skiprows=1)
    ry = np.loadtxt(os.path.join(folder, "datay.csv"), delimiter=",", skiprows=1)
    t = rx[1::2, 0]
    x, y = rx[0::2], ry[0::2]
    n = min(len(x), len(y), len(t))
    return x[:n], y[:n], t[:n]


def frame_fields(P: np.ndarray, rc: float = 2.9):
    """Per-particle coordination number, |psi6| and cluster label, plus the bond list."""
    n = len(P)
    b = cKDTree(P).query_pairs(rc, output_type="ndarray")
    z = np.bincount(b.ravel(), minlength=n)
    _, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n)), directed=False)
    psi = np.zeros(n, complex)
    if len(b):
        d = P[b[:, 1]] - P[b[:, 0]]
        e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))
        np.add.at(psi, b[:, 0], e)
        np.add.at(psi, b[:, 1], e)
    psi6 = np.where(z > 0, np.abs(psi) / np.maximum(z, 1), 0.0)
    return z, psi6, lab, b


def _colors(mode, P, z, psi6, lab, speed):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", _BLUE_RAMP)
    if mode == "Coordination number":
        return cmap(np.clip(z / 6.0, 0, 1)), (cmap, 0, 6, "neighbours")
    if mode == "Local order ψ6":
        return cmap(np.clip(psi6, 0, 1)), (cmap, 0, 1, "|ψ6|")
    if mode == "Speed" and speed is not None:
        vmax = max(np.percentile(speed, 95), 1e-9)
        return cmap(np.clip(speed / vmax, 0, 1)), (cmap, 0, vmax, "speed (µm/s)")
    if mode == "Clusters":
        sizes = np.bincount(lab)
        big = np.argmax(sizes)
        cols = np.array([_SINGLE] * len(P), dtype=object)
        cols[sizes[lab] >= 3] = _OTHER_CLUSTER
        cols[lab == big] = _LARGEST
        return list(cols), None
    return [_PLAIN] * len(P), None


def draw_frame(ax, x, y, t, k, *, mode="Plain", show_bonds=False, trail=0, radius=1.0,
               box=25.0, rc=2.9, title=None, show_colorbar=None):
    """Draw frame k of a run on ax. Particles are drawn to scale (radius in µm)."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Circle, Rectangle
    P = np.c_[x[k], y[k]]
    z, psi6, lab, b = frame_fields(P, rc)
    speed = None
    if mode == "Speed" and k > 0:
        dt = t[k] - t[k - 1]
        speed = np.linalg.norm(P - np.c_[x[k - 1], y[k - 1]], axis=1) / max(dt, 1e-12)
    cols, cbar = _colors(mode, P, z, psi6, lab, speed)

    ax.clear()
    ax.add_patch(Rectangle((-box, -box), 2 * box, 2 * box, fill=False, ec="#c3c2b7", lw=1))
    if trail > 0:
        k0 = max(0, k - trail)
        segs = [np.c_[x[k0:k + 1, i], y[k0:k + 1, i]] for i in range(P.shape[0])]
        ax.add_collection(LineCollection(segs, colors="#b8b6ad", linewidths=0.6, alpha=0.6, zorder=1))
    if show_bonds and len(b):
        ax.add_collection(LineCollection(np.stack([P[b[:, 0]], P[b[:, 1]]], 1),
                                         colors="#6b6a63", linewidths=0.6, zorder=2))
    ax.add_collection(PatchCollection([Circle(p, radius) for p in P], facecolors=cols,
                                      edgecolors="white", linewidths=0.4, zorder=3))
    ax.set_xlim(-box - 1, box + 1)
    ax.set_ylim(-box - 1, box + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    txt = f"t = {t[k] / TAUB_S:.2f} τ$_B$   mean z = {z.mean():.2f}   mean ψ6 = {psi6[z >= 2].mean() if (z >= 2).any() else 0:.2f}"
    ax.set_title((title + "\n" if title else "") + txt, loc="left", fontsize=9)
    return cbar


def make_figure(runs, k_per_run, **kw):
    """runs: list of (label, (x, y, t)); k_per_run: frame index for each."""
    import matplotlib.pyplot as plt
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.9), constrained_layout=True, squeeze=False)
    cbar = None
    for ax, (label, (x, y, t)), k in zip(axes[0], runs, k_per_run):
        cbar = draw_frame(ax, x, y, t, k, title=label, **kw) or cbar
    if cbar is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cmap, lo, hi, lab = cbar
        fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), ax=axes[0].tolist(), label=lab, shrink=0.8)
    if kw.get("mode") == "Clusters":
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", mfc=c, mec="white", ms=9, label=l)
             for c, l in ((_LARGEST, "largest cluster"), (_OTHER_CLUSTER, "other clusters (≥3)"), (_SINGLE, "single / pair"))]
        axes[0][0].legend(handles=h, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    return fig


def frame_at_time(t, t_s):
    return int(np.clip(np.searchsorted(t, t_s - 1e-9), 0, len(t) - 1))


def make_gif(runs, *, t_start_s, t_end_s, n_frames=60, fps=12, dpi=90, progress=None, **kw):
    """Animated GIF (bytes) of one or several runs, sampled at the same times."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    times = np.linspace(t_start_s, t_end_s, n_frames)
    images = []
    for i, ts in enumerate(times):
        fig = make_figure(runs, [frame_at_time(r[1][2], ts) for r in runs], **kw)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))
        if progress:
            progress((i + 1) / n_frames)
    # all frames must share a size
    h = min(im.shape[0] for im in images)
    w = min(im.shape[1] for im in images)
    images = [im[:h, :w] for im in images]
    out = io.BytesIO()
    imageio.mimsave(out, images, format="GIF", duration=1.0 / fps, loop=0)
    return out.getvalue()

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
    top = float(H) if H not in (None, "", 0) else max(float(np.max(np.asarray(P)[..., 2])) + A_UM, 4.0)
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


# --------------------------------------------------------------------------- views, snapshots and animations
VIEWS = {"3D perspective": "3d", "Top (x–y)": "xy", "Side along x (y–z, field axis)": "yz",
         "Side along y (x–z)": "xz", "Top + side": "xy+yz"}
COLOR_BY = {"plain": "plain colour", "height": "height z (µm)", "coordination": "3D coordination",
            "clusters": "largest cluster"}


def particle_colours(P, by="plain", plain="#e34948", rc=2.9, rng=None, cmap="viridis"):
    """RGBA per particle and the (lo, hi) value range (None for plain / clusters)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    n = len(P)
    if by == "plain":
        return np.tile(to_rgba(plain), (n, 1)), None
    if by == "clusters":
        i, j = _pairs(P, rc)
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
        big = roots == np.argmax(np.bincount(roots))
        return np.array([to_rgba(plain) if b else to_rgba("#b8b6ad") for b in big]), None
    if by == "height":
        v = P[:, 2]
    else:
        i, j = _pairs(P, rc)
        v = np.bincount(np.r_[i, j], minlength=n).astype(float)
    lo, hi = rng if rng is not None else (float(v.min()), float(max(v.max(), v.min() + 1e-6)))
    return plt.get_cmap(cmap)(np.clip((v - lo) / (hi - lo), 0, 1)), (lo, hi)


def _draw_2d(ax, P, cols, plane, half, top, scale=1.0, show_box=True, edge="#ffffff", alpha=1.0):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle
    a_, b_, depth = {"xy": (0, 1, 2), "yz": (1, 2, 0), "xz": (0, 2, 1)}[plane]
    order = np.argsort(P[:, depth])                     # far particles first
    ax.add_collection(PatchCollection([Circle((P[k, a_], P[k, b_]), A_UM * scale) for k in order],
                                      facecolors=cols[order], edgecolors=edge, linewidths=0.3, alpha=alpha, zorder=3))
    lim = {0: (-half, half), 1: (-half, half), 2: (0.0, top)}
    if show_box:
        (x0, x1), (y0, y1) = lim[a_], lim[b_]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#c3c2b7", lw=1, zorder=1))
    pad = 1.0
    ax.set_xlim(lim[a_][0] - pad, lim[a_][1] + pad); ax.set_ylim(lim[b_][0] - pad, lim[b_][1] + pad)
    ax.set_aspect("equal")
    names = "xyz"
    ax.set_xlabel(f"{names[a_]} (µm)" + (", field axis" if a_ == 1 else ""))
    ax.set_ylabel(f"{names[b_]} (µm)" + (", field axis" if b_ == 1 else ""))


def _draw_3d(ax, P, cols, half, top, elev=22, azim=-60, scale=1.0, show_box=True, panel_in=4.6, alpha=1.0,
             depthshade=False, edge="#ffffff"):
    span = max(2 * half, top)
    pts_per_um = panel_in * 72 * 0.62 / span                 # rough: axes fill ~62 % of the panel
    s = (2 * A_UM * scale * pts_per_um) ** 2
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=cols, edgecolors=edge, linewidths=0.3, alpha=alpha,
               depthshade=depthshade)
    if show_box:
        xs = [-half, half, half, -half, -half]; ys = [-half, -half, half, half, -half]
        for zz in (0.0, top):
            ax.plot(xs, ys, [zz] * 5, color="#c3c2b7", lw=0.8)
        for x0, y0 in zip(xs[:4], ys[:4]):
            ax.plot([x0, x0], [y0, y0], [0, top], color="#c3c2b7", lw=0.8)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, top)
    try:
        ax.set_box_aspect((2 * half, 2 * half, max(top, 0.15 * 2 * half)))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm), field"); ax.set_zlabel("z (µm)")


def make_figure3d(runs, k_per_run, view="3d", *, color_by="plain", plain="#e34948", elev=22, azim=-60,
                  scale=1.0, show_box=True, show_time=True, show_stats=True, show_name=True, panel=4.6,
                  font=9, fixed_range=None, cmap="viridis", dpi=None, alpha=1.0, depthshade=False, edge="#ffffff"):
    """One figure, one column per run. runs: list of (label, (frames, t, cfg)). view: a value of VIEWS."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    n = len(runs)
    rows = 2 if view == "xy+yz" else 1
    # side views are as tall as the chamber (plus room for labels)
    asp = max(max(_box_geometry(c_, f_)[1] / (2 * _box_geometry(c_, f_)[0]) for _, (f_, _t, c_) in runs), 0.08)
    side_h = panel * asp + 0.9
    h = {"xy+yz": panel + side_h, "yz": side_h + 0.4, "xz": side_h + 0.4}.get(view, panel)
    fig = plt.figure(figsize=(panel * n + (0.9 if color_by in ("height", "coordination") else 0), h + 0.4), dpi=dpi,
                     constrained_layout=True)
    gs = fig.add_gridspec(rows, n, height_ratios=[panel, side_h] if rows == 2 else [1])
    rng_all, axes = None, []
    for c, ((label, (frames, t, cfg)), k) in enumerate(zip(runs, k_per_run)):
        P = frames[k]
        half, top = _box_geometry(cfg, frames)
        cols, rng = particle_colours(P, color_by, plain, rng=fixed_range, cmap=cmap)
        rng_all = rng or rng_all
        planes = ["xy", "yz"] if view == "xy+yz" else [view]
        for r, pl in enumerate(planes):
            if pl == "3d":
                ax = fig.add_subplot(gs[r, c], projection="3d")
                _draw_3d(ax, P, cols, half, top, elev, azim, scale, show_box, panel, alpha, depthshade, edge)
            else:
                ax = fig.add_subplot(gs[r, c])
                _draw_2d(ax, P, cols, pl, half, top, scale, show_box, edge, alpha)
                ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=font - 1)
            axes.append(ax)
            if r == 0:
                lines = []
                if show_name:
                    lines.append(label.split("/")[-1])
                info = []
                if show_time:
                    info.append(f"t = {t[k] / TAUB_S:.2f} τ$_B$")
                if show_stats:
                    s_, _ = frame_stats(P, box=2 * half)
                    info.append(f"z = {s_['z3']:.2f}  S$_{{max}}$ = {s_['Smax']:.2f}  on top = {s_['f_on_top']:.0%}")
                if info:
                    lines.append("   ".join(info))
                if lines:
                    ax.set_title("\n".join(lines), loc="left", fontsize=font)
    if color_by in ("height", "coordination") and rng_all is not None:
        cb = fig.colorbar(ScalarMappable(Normalize(*rng_all), plt.get_cmap(cmap)), ax=axes, shrink=0.75)
        cb.set_label(COLOR_BY[color_by], fontsize=font)
    return fig


def colour_range(runs, color_by, t_end_s=None):
    """A fixed colour range for animations: over the first and last frame of every run."""
    if color_by not in ("height", "coordination"):
        return None
    lo, hi = np.inf, -np.inf
    for _, (frames, t, cfg) in runs:
        ks = [0, len(t) - 1 if t_end_s is None else int(np.clip(np.searchsorted(t, t_end_s), 0, len(t) - 1))]
        for k in ks:
            _, r = particle_colours(frames[k], color_by)
            lo, hi = min(lo, r[0]), max(hi, r[1])
    return lo, hi


def make_gif3d(runs, view="3d", *, t_start_s, t_end_s, n_frames=60, fps=12, dpi=90, orbit=0.0, progress=None,
               azim=-60, **kw):
    """Animated GIF (bytes) of the runs side by side. orbit: degrees the camera turns over the animation (3D view)."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    times = np.linspace(t_start_s, t_end_s, int(n_frames))
    if kw.get("fixed_range") is None:
        kw["fixed_range"] = colour_range(runs, kw.get("color_by", "plain"), t_end_s)
    frames, size = [], None
    for i, ts in enumerate(times):
        ks = [int(np.clip(np.searchsorted(r[1][1], ts - 1e-9), 0, len(r[1][1]) - 1)) for r in runs]
        fig = make_figure3d(runs, ks, view, azim=azim + orbit * i / max(len(times) - 1, 1), **kw)
        if size is None:
            size = fig.get_size_inches()
        fig.set_size_inches(size)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        if progress:
            progress((i + 1) / len(times))
    w = min(f.width for f in frames); h = min(f.height for f in frames)
    frames = [f.crop((0, 0, w, h)).convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / max(fps, 1))), loop=0, disposal=2)
    return out.getvalue()


def fig_3d_animated(frames, t, cfg=None, color_by="plain", plain="#e34948", n_max=80, title="", opacity=1.0,
                    size=None):
    """Plotly 3D figure with a play button and time slider (rotate / zoom while it plays)."""
    import plotly.graph_objects as go
    from matplotlib.colors import to_hex
    cfg = cfg or {}
    half, top = _box_geometry(cfg, frames)
    idx = np.unique(np.linspace(0, len(t) - 1, min(n_max, len(t))).astype(int))
    rng = colour_range([("", (frames, t, cfg))], color_by)

    def trace(k):
        cols, _ = particle_colours(frames[k], color_by, plain, rng=rng)
        return go.Scatter3d(x=frames[k][:, 0], y=frames[k][:, 1], z=frames[k][:, 2], mode="markers",
                            marker=dict(size=size or (4 if frames.shape[1] > 500 else 6), color=[to_hex(c) for c in cols],
                                        opacity=opacity, line=dict(width=0)), hoverinfo="skip")
    xs, ys = [-half, half, half, -half, -half], [-half, -half, half, half, -half]
    box = [go.Scatter3d(x=xs, y=ys, z=[zz] * 5, mode="lines", line=dict(color="#999", width=2), hoverinfo="skip")
           for zz in (0.0, top)]
    box += [go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[0, top], mode="lines", line=dict(color="#999", width=2),
                         hoverinfo="skip") for x0, y0 in zip(xs[:4], ys[:4])]
    fig = go.Figure(data=[trace(idx[0])] + box,
                    frames=[go.Frame(data=[trace(k)], traces=[0], name=f"{t[k] / TAUB_S:.2f}") for k in idx])
    steps = [dict(method="animate", label=f"{t[k] / TAUB_S:.2f}",
                  args=[[f"{t[k] / TAUB_S:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for k in idx]
    fig.update_layout(title=title, height=640, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                      scene=dict(xaxis_title="x (µm)", yaxis_title="y (µm) – field axis", zaxis_title="z (µm)",
                                 xaxis=dict(range=[-half, half]), yaxis=dict(range=[-half, half]),
                                 zaxis=dict(range=[0, top]), aspectmode="manual",
                                 aspectratio=dict(x=1, y=1, z=max(top / (2 * half), 0.15))),
                      updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                                        buttons=[dict(label="▶ play", method="animate",
                                                      args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                                                 dict(label="⏸", method="animate",
                                                      args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
                      sliders=[dict(steps=steps, currentvalue=dict(prefix="t (τB) = "), x=0.15, len=0.8)])
    return fig

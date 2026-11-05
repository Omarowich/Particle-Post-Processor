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
    """3D observables of one frame. z: neighbours within rc = 2.9 um (as obs3d.py and the 2D analysis);
    S_max and spanning: clusters of bonded particles, centres within OTHER_ALPHA = 0.3     # opacity of particles outside the largest cluster ("largest cluster" colouring)
CLUSTER_RC = 2.2 um (1.1 sigma). NOTE: the 2D
    analysis and Data/_tools/obs3d.py use 2.9 um for clusters too, so S_max here is stricter (smaller)."""
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
    # clusters by bond distance (1.1 sigma), as the "largest cluster" colouring; z keeps the 2.9 um shell
    lab = np.arange(n)
    bi, bj = _pairs(P, CLUSTER_RC)
    for u, v in zip(bi, bj):
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


# --------------------------------------------------------------------------- acoustic field of a run
class RunField:
    """Radiation force of a 3D run on a grid (kBT/um) and its Gor'kov potential (kBT). From the run's arf3d_file
    (solved / COMSOL field, .npz with xs, ys, zs in m and F; scaled like the run: to the 2D fit at mid-height unless
    arf3d_match is false) or, without one, the fitted 1D standing wave F_y = -F0 sin(k y), uniform in z."""

    def __init__(self, cfg, data_root=None, frames=None):
        self.kind, self.note = "1D", ""
        half, top = _box_geometry(cfg, frames if frames is not None else np.zeros((1, 3)))
        dn = float(cfg.get("dn_nm", 0) or 0)
        f = cfg.get("arf3d_file")
        fa = cfg.get("field_auto") or {}
        self.flow = None                                  # (ys, zs, U, W) in um, um/s: computed streaming in the y-z plane
        if fa and data_root:
            p0 = os.path.join(data_root, "_fields", f"auto_{fa.get('drive', 'side')}{float(fa.get('freq_MHz', 20)):g}"
                                                    f"_H{float(cfg.get('H') or top):g}_box{2 * float(cfg.get('box_half', half)):g}"
                                                    f"_{fa.get('lid', 'glass')}.npz")
            sf = (dn / 4000.0) ** 2 * float(cfg.get("field_auto_streaming") if cfg.get("field_auto_streaming") is not None else 1.0)
            if os.path.isfile(p0) and sf:
                d0 = np.load(p0)
                self.flow = (d0["ys"] * 1e6, d0["zs"] * 1e6, d0["U"] * sf, d0["W"] * sf)
        if cfg.get("radiation_off"):                       # streaming-only run: no radiation force acted
            self.xs = np.linspace(-half, half, 21); self.ys = np.linspace(-half, half, 21); self.zs = np.linspace(0, top, 5)
            self.F = np.zeros((21, 21, 5, 3)); self.kind = "none"
            self.note = "streaming only: the radiation force was off in this run"
            self.U = np.zeros(self.F.shape[:3]); self.empty = self.flow is None
            return
        if fa and data_root:                               # computed chamber field (sim_runner field_auto cache)
            H_ = float(cfg.get("H") or top); bh = float(cfg.get("box_half", half))
            p_ = os.path.join(data_root, "_fields", f"auto_{fa.get('drive', 'side')}{float(fa.get('freq_MHz', 20)):g}"
                                                    f"_H{H_:g}_box{2 * bh:g}_{fa.get('lid', 'glass')}.npz")
            if os.path.isfile(p_):
                d = np.load(p_); s_ = (dn / 4000.0) ** 2
                self.ys, self.zs = d["ys"] * 1e6, d["zs"] * 1e6; self.xs = np.linspace(-half, half, 21)
                F = np.zeros((len(self.xs), len(self.ys), len(self.zs), 3))
                F[..., 1] = (d["Fy"] * s_)[None]; F[..., 2] = (d["Fz"] * s_)[None]
                self.F, self.kind, self.note = F, "3D", os.path.basename(p_)
                self.U = self._potential(); self.empty = not np.any(self.F)
                return
            self.note = f"computed field {os.path.basename(p_)} not found: showing the 1D standing wave"
        if f:
            path = f if os.path.isfile(f) else None
            if path is None and data_root:                 # Windows path from another machine: look in Data/_fields
                cand = os.path.join(data_root, "_fields", os.path.basename(str(f).replace("\\", "/")))
                path = cand if os.path.isfile(cand) else None
            if path and path.endswith(".npz"):
                d = np.load(path)
                xs, ys, zs = d["xs"] * 1e6, d["ys"] * 1e6, d["zs"] * 1e6
                F = np.nan_to_num(d["F"])
                if cfg.get("arf3d_match", True):
                    ix, iz = len(xs) // 2, len(zs) // 2
                    peak = np.abs(F[ix, :, iz, 1]).max()
                    F = F * (float(cfg.get("arf_F0_ref", 18.4)) * (dn / 4000.0) ** 2 / peak if peak > 0 else 1.0)
                self.xs, self.ys, self.zs, self.F, self.kind = xs, ys, zs, F, "3D"
                self.note = os.path.basename(path)
            else:
                self.note = f"field file {os.path.basename(str(f))} not found: showing the 1D standing wave"
        if self.kind == "1D":
            F0 = float(cfg.get("arf_F0_ref", 18.4)) * (dn / 4000.0) ** 2
            k = float(cfg.get("arf_k", 0.124))
            self.xs = np.linspace(-half, half, 21); self.ys = np.linspace(-half, half, 201)
            self.zs = np.linspace(0, top, 21)
            Fy = -F0 * np.sin(k * self.ys)
            self.F = np.zeros((len(self.xs), len(self.ys), len(self.zs), 3))
            self.F[:, :, :, 1] = Fy[None, :, None]
        self.U = self._potential()
        self.empty = not np.any(self.F)

    def _potential(self):
        """U = -integral of F (path: along y at each x, z; plus along z at y = ys[0]), minimum set to 0."""
        def cum(v, x, ax):
            dx = np.diff(x)
            sl = lambda a, b: tuple(slice(a, b) if i == ax else slice(None) for i in range(v.ndim))
            inc = 0.5 * (v[sl(1, None)] + v[sl(None, -1)]) * dx.reshape([-1 if i == ax else 1 for i in range(v.ndim)])
            return np.concatenate([np.zeros_like(v[sl(0, 1)]), np.cumsum(inc, axis=ax)], axis=ax)
        Uy = -cum(self.F[..., 1], self.ys, 1)
        Uz0 = -cum(self.F[:, :1, :, 2], self.zs, 2)
        U = Uy + Uz0
        return U - U.min()

    def plane(self, which, at):
        """2D slice (a-axis values, b-axis values, U, Fa, Fb) of plane 'xy' (at z), 'yz' (at x), 'xz' (at y)."""
        if which == "xy":
            k = int(np.argmin(np.abs(self.zs - at)))
            return self.xs, self.ys, self.U[:, :, k].T, self.F[:, :, k, 0].T, self.F[:, :, k, 1].T
        if which == "yz":
            k = int(np.argmin(np.abs(self.xs - at)))
            return self.ys, self.zs, self.U[k].T, self.F[k, :, :, 1].T, self.F[k, :, :, 2].T
        k = int(np.argmin(np.abs(self.ys - at)))
        return self.xs, self.zs, self.U[:, k].T, self.F[:, k, :, 0].T, self.F[:, k, :, 2].T


def _overlay_2d(ax, field, plane, at, mode, alpha=0.45):
    if field is None or mode == "none":
        return
    if mode == "streaming":
        if getattr(field, "flow", None) is None or plane != "yz":
            return
        ys, zs, U, W = field.flow
        sp = np.hypot(U, W)
        ax.pcolormesh(ys, zs, sp.T, cmap="Blues", alpha=alpha, shading="auto", zorder=0, rasterized=True)
        ax.streamplot(ys, zs, U.T, W.T, color="#33415c", density=(1.4, 0.8), linewidth=0.5, arrowsize=0.6, zorder=1)
        return
    if field.empty:
        return
    a, b, U, Fa, Fb = field.plane(plane, at)
    if "potential" in mode or mode == "force":
        img = U if "potential" in mode else np.hypot(Fa, Fb)
        ax.imshow(img, extent=(a[0], a[-1], b[0], b[-1]), origin="lower", aspect="auto", alpha=alpha, zorder=0,
                  cmap="Blues" if "potential" in mode else "Oranges", interpolation="bilinear")
    if "arrows" in mode:
        sa = max(1, len(a) // 14); sb = max(1, len(b) // (10 if plane == "xy" else 6))
        A_, B_ = np.meshgrid(a[::sa], b[::sb])
        ax.quiver(A_, B_, Fa[::sb, ::sa], Fb[::sb, ::sa], color="#2a78d6", alpha=0.8, zorder=1, width=0.004,
                  scale=None, pivot="mid")


# --------------------------------------------------------------------------- views, snapshots and animations
VIEWS = {"3D perspective": "3d", "Top (x–y)": "xy", "Side along x (y–z, field axis)": "yz",
         "Side along y (x–z)": "xz", "Top + side": "xy+yz"}
OTHER_ALPHA = 0.3     # opacity of particles outside the largest cluster ("largest cluster" colouring)
CLUSTER_RC = 2.2      # um: two particles count as bonded (same cluster) within 1.1 sigma
FIELD_VERSION = 3     # bump when RunField changes, so the app's cached fields are rebuilt
COLOR_BY = {"plain": "plain colour", "stacked": "resting on another particle", "height": "height z (µm)",
            "coordination": "3D coordination", "clusters": "largest cluster"}
FIELD_OVERLAYS = {"none": "none", "potential": "Gor'kov potential (shading)", "force": "|force| (shading)",
                  "arrows": "force arrows", "potential+arrows": "potential + arrows",
                  "streaming": "streaming flow (side views and 3D back wall)"}


def _with_alpha(cols, alpha):
    """RGBA colours with the overall opacity multiplied into each particle's own alpha."""
    c = np.array(cols, float, copy=True)
    if c.ndim == 2 and c.shape[1] == 4:
        c[:, 3] *= float(alpha)
    return c


def particle_colours(P, by="plain", plain="#e34948", rc=2.9, rng=None, cmap="viridis"):
    """RGBA per particle and the (lo, hi) value range (None for plain / clusters)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    n = len(P)
    if by == "plain":
        return np.tile(to_rgba(plain), (n, 1)), None
    if by == "stacked":
        ii, jj = _pairs(P, 4.2 * A_UM)
        dz = P[ii, 2] - P[jj, 2]
        lat = np.hypot(P[ii, 0] - P[jj, 0], P[ii, 1] - P[jj, 1])
        top = np.zeros(n, bool)
        sel = lat < 2.2 * A_UM
        top[ii[sel & (dz > A_UM)]] = True
        top[jj[sel & (dz < -A_UM)]] = True
        return np.array([to_rgba(plain) if t_ else to_rgba("#c9c7bd") for t_ in top]), None
    if by == "clusters":
        i, j = _pairs(P, CLUSTER_RC)        # bonded = centres within 1.1 sigma (Lindemann), not the 2.9 um shell of z
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
        cnt = np.bincount(roots)
        big = (roots == np.argmax(cnt)) & (cnt.max() >= 3)      # no "largest cluster" while everything is single
        return np.array([to_rgba(plain) if b else to_rgba("#b8b6ad", OTHER_ALPHA) for b in big]), None
    if by == "height":
        v = P[:, 2]
    else:
        i, j = _pairs(P, rc)
        v = np.bincount(np.r_[i, j], minlength=n).astype(float)
    lo, hi = rng if rng is not None else (float(v.min()), float(max(v.max(), v.min() + 1e-6)))
    return plt.get_cmap(cmap)(np.clip((v - lo) / (hi - lo), 0, 1)), (lo, hi)


def _draw_2d(ax, P, cols, plane, half, top, scale=1.0, show_box=True, edge="#ffffff", alpha=1.0, field=None,
             field_mode="none", field_alpha=0.45, stretch=1.0):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Rectangle
    a_, b_, depth = {"xy": (0, 1, 2), "yz": (1, 2, 0), "xz": (0, 2, 1)}[plane]
    at = {"xy": float(np.median(P[:, 2])), "yz": 0.0, "xz": 0.0}[plane]
    _overlay_2d(ax, field, plane, at, field_mode, field_alpha)
    order = np.argsort(P[:, depth])                     # far particles first
    ax.add_collection(PatchCollection([Circle((P[k, a_], P[k, b_]), A_UM * scale) for k in order],
                                      facecolors=_with_alpha(cols[order], alpha), edgecolors=edge, linewidths=0.3, zorder=3))
    lim = {0: (-half, half), 1: (-half, half), 2: (0.0, top)}
    if show_box:
        (x0, x1), (y0, y1) = lim[a_], lim[b_]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#c3c2b7", lw=1, zorder=1))
    pad = 1.0
    ax.set_xlim(lim[a_][0] - pad, lim[a_][1] + pad); ax.set_ylim(lim[b_][0] - pad, lim[b_][1] + pad)
    ax.set_aspect(stretch if plane in ("yz", "xz") else "equal")
    names = "xyz"
    ax.set_xlabel(f"{names[a_]} (µm)" + (", field axis" if a_ == 1 else ""))
    ax.set_ylabel(f"{names[b_]} (µm)" + (", field axis" if b_ == 1 else ""))


def _draw_3d(ax, P, cols, half, top, elev=22, azim=-60, scale=1.0, show_box=True, panel_in=4.6, alpha=1.0,
             depthshade=False, edge="#ffffff", stretch=1.0, field=None, field_mode="none", field_alpha=0.45):
    span = max(2 * half, top)
    pts_per_um = panel_in * 72 * 0.62 / span                 # rough: axes fill ~62 % of the panel
    s = (2 * A_UM * scale * pts_per_um) ** 2
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=_with_alpha(cols, alpha), edgecolors=edge, linewidths=0.3,
               depthshade=depthshade)
    if show_box:
        xs = [-half, half, half, -half, -half]; ys = [-half, -half, half, half, -half]
        for zz in (0.0, top):
            ax.plot(xs, ys, [zz] * 5, color="#c3c2b7", lw=0.8)
        for x0, y0 in zip(xs[:4], ys[:4]):
            ax.plot([x0, x0], [y0, y0], [0, top], color="#c3c2b7", lw=0.8)
    if field is not None and field_mode == "streaming" and getattr(field, "flow", None) is not None:
        ys_, zs_, U_, W_ = field.flow
        sa, sb = max(1, len(ys_) // 22), max(1, len(zs_) // 8)
        Y_, Z_ = np.meshgrid(ys_[::sa], zs_[::sb], indexing="ij")
        ax.quiver(np.full_like(Y_, -half), Y_, Z_ * stretch, np.zeros_like(Y_), U_[::sa, ::sb], W_[::sa, ::sb] * stretch,
                  length=1.5, normalize=False, color="#2a5aa8", alpha=0.8, linewidth=0.6)
    elif field is not None and not field.empty and field_mode not in ("none", "streaming"):
        a, b, U, Fa, Fb = field.plane("yz", 0.0)
        img = U if "potential" in field_mode or "arrows" in field_mode else np.hypot(Fa, Fb)
        import matplotlib.pyplot as plt
        Y_, Z_ = np.meshgrid(a, b)
        cm = plt.get_cmap("Blues" if "potential" in field_mode or "arrows" in field_mode else "Oranges")
        ax.plot_surface(np.full_like(Y_, -half), Y_, Z_, facecolors=cm((img - img.min()) / max(np.ptp(img), 1e-12)),
                        shade=False, alpha=field_alpha, rstride=1, cstride=max(1, len(a) // 60), linewidth=0, zorder=0)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_zlim(0, top)
    try:
        ax.set_box_aspect((2 * half, 2 * half, max(top * stretch, 0.15 * 2 * half)))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm), field"); ax.set_zlabel("z (µm)")



_B2_CACHE = {}


def _b2_of(cfg):
    """Reduced B2* of the run's pair potential (sim_gui.b2_star), cached; None if it cannot be computed."""
    keys = ("potential", "A", "q", "phi_p", "eps", "n", "kappa", "width", "core", "core_Z", "lj_cutoff_sigma")
    k = tuple(str(cfg.get(x)) for x in keys)
    if k not in _B2_CACHE:
        try:
            import sim_gui as _sg
            _B2_CACHE[k] = float(_sg.b2_star(cfg))
        except Exception:
            _B2_CACHE[k] = None
    return _B2_CACHE[k]


def run_info_lines(cfg, name=""):
    """Compact, complete description of a run for panel titles (from its sim_config.json):
    line 1 pair potential and B2*, line 2 chamber and system, line 3 field / flow, start, length, seed."""
    import re
    c = cfg or {}
    if not c:
        return [str(name).split("/")[-1]]
    pot = c.get("potential", "?")
    if pot == "AO":
        p1 = f"AO q = {c.get('q', 0):g}, φ$_p$ = {c.get('phi_p', 0):g}"
    elif pot == "LJ":
        p1 = f"LJ 12-6 ε = {c.get('A', 0):g} k$_B$T"
    elif pot == "LJ2n":
        p1 = f"2n–n LJ n = {c.get('n', 0):g}, ε = {c.get('eps', 0):g}"
    elif pot == "SW":
        p1 = f"square well w = {1000 * float(c.get('width', 0)):.0f} nm, ε = {c.get('eps', 0):g}"
    elif pot == "YUK":
        p1 = f"Yukawa κ = {c.get('kappa', 0):g}/µm, ε = {c.get('eps', 0):g}"
    else:
        p1 = "no attraction"
    b2 = _b2_of(c) if pot not in ("none", None) else None
    p1 += f" · A = {c.get('A', 0):g}" + (f" (B$_2^*$ = {b2:.2f})" if b2 is not None else "")
    if c.get("polydispersity"):
        p1 += f" · pd {100 * float(c['polydispersity']):g} %"
    half = float(c.get("box_half", 25.0))
    st_ = c.get("start") or {}
    n_ = (int(st_.get("n_side", 0)) ** 2 * int(st_.get("layers", 1))) if st_.get("kind") in ("grid", "grid3d") else None
    m_ = re.search(r"_N(\d+)", str(name))
    n_ = int(m_.group(1)) if m_ else n_
    H = c.get("H")
    if n_:
        a_ = float(c.get("radius_um", 1.0))
        if int(c.get("dim", 2)) == 3 and H:
            frac = f" (φ$_V$ {n_ * 4 / 3 * np.pi * a_ ** 3 / ((2 * half) ** 2 * float(H)):.3f})"
        else:
            frac = f" (φ$_A$ {n_ * np.pi * a_ ** 2 / (2 * half) ** 2:.3f})"
    p2 = (f"H {float(H):g} µm · " if H else ("open top · " if int(c.get("dim", 2)) == 3 else "2D · ")) + \
         f"box {2 * half:g}" + (f" · N {n_}{frac}" if n_ else "") + \
         (f" · Δρ {float(c.get('drho', 0)):g}" if int(c.get("dim", 2)) == 3 else "")
    fa = c.get("field_auto") or {}
    dn = float(c.get("dn_nm", 0) or 0)
    if fa:   # compact: drive+MHz, lid, d_n, streaming
        drv = {"side": "wall", "idt": "IDT", "idt_travel": "IDT-trav"}.get(fa.get("drive"), str(fa.get("drive")))
        lid = {"glass": "glass", "pdms": "PDMS", "absorb": "abs"}.get(fa.get("lid"), str(fa.get("lid", "")))
        fld = f"{drv}{float(fa.get('freq_MHz', 20)):g} {lid}"
        fld += " str-only" if c.get("radiation_off") else (f" d$_n${dn:g}" + (" +str" if c.get("field_auto_streaming", 0) else ""))
    elif c.get("arf3d_file"):
        fld = f"file d$_n${dn:g}"
    elif dn > 0:
        fld = f"1D d$_n${dn:g}"
    else:
        fld = "no field"
    if float(c.get("flow_F", 0) or 0) > 0:
        fld += f" +F{float(c['flow_F']):g}"
    if float(c.get("rayleigh_A", 0) or 0) > 0:
        fld += f" +roll{float(c['rayleigh_A']):g}"
    seed = re.search(r"seed(\d+)", str(name))
    p3 = f"{fld} · {float(c.get('T_tauB', 0)):g}τ$_B$" + (f" · s{seed.group(1)}" if seed else "")
    return [p1, p2, p3]


def _safe(s):
    import re
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", str(s)).strip("-")


def compact_name(name):
    """Shorter, file-safe version of a run folder name (drops the parts every run shares)."""
    import re
    n = str(name).split("/")[-1].split("\\")[-1]
    for pat in (r"_dt\d+", r"_core", r"_BD", r"_drho52", r"Tkb", r"TauB"):
        n = re.sub(pat, {"Tkb": "kT", "TauB": "tB"}.get(pat, ""), n)
    return _safe(n.replace("_", "-"))


def run_file_tag(cfg, name=""):
    """File-name tag of a 3D run from its settings: model, A, chamber, box, field, length, seed."""
    import re
    c = cfg or {}
    if not c:
        return compact_name(name)
    pot = c.get("potential", "none")
    p = {"AO": f"AO-q{c.get('q', 0):g}", "LJ": "LJ", "LJ2n": f"LJ2n-n{c.get('n', 0):g}",
         "SW": f"SW-w{1000 * float(c.get('width', 0)):.0f}nm", "YUK": f"YUK-k{c.get('kappa', 0):g}"}.get(pot, "noatt")
    parts = [f"{p}-A{c.get('A', 0):g}"]
    if c.get("H"):
        parts.append(f"H{float(c['H']):g}")
    parts.append(f"box{2 * float(c.get('box_half', 25)):g}")
    fa = c.get("field_auto") or {}
    dn = float(c.get("dn_nm", 0) or 0)
    if fa:
        f_ = f"{ {'side': 'wall', 'idt': 'IDT', 'idt_travel': 'IDTtrav'}.get(fa.get('drive'), fa.get('drive'))}{float(fa.get('freq_MHz', 20)):g}{fa.get('lid', '')}"
        f_ += "-stronly" if c.get("radiation_off") else f"-dn{dn:g}"
    elif dn > 0:
        f_ = f"dn{dn:g}"
    else:
        f_ = "nofield"
    if float(c.get("flow_F", 0) or 0) > 0:
        f_ += f"-F{float(c['flow_F']):g}" + ("hs" if c.get("flow_hele_shaw") else "")
    parts.append(f_)
    seed = re.search(r"seed(\d+)", str(name))
    parts.append(f"{float(c.get('T_tauB', 0)):g}tB" + (f"-s{seed.group(1)}" if seed else ""))
    return _safe("_".join(parts))


def file_name(prefix, tags, suffix, ext, max_len=180):
    """prefix_tag1_vs_tag2_suffix.ext, shortened (with a count) if very long."""
    body = "_vs_".join(tags)
    if len(body) > max_len:
        body = tags[0][:max_len - 12] + f"_and{len(tags) - 1}more"
    return f"{prefix}_{body}_{suffix}.{ext}"

def make_figure3d(runs, k_per_run, view="3d", *, color_by="plain", plain="#e34948", elev=22, azim=-60,
                  scale=1.0, show_box=True, show_time=True, show_stats=True, show_name=True, panel=4.6,
                  font=9, fixed_range=None, cmap="viridis", dpi=None, alpha=1.0, depthshade=False, edge="#ffffff",
                  stretch=1.0, fields=None, field_mode="none", field_alpha=0.45):
    """One figure, one column per run. runs: list of (label, (frames, t, cfg)). view: a value of VIEWS."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    n = len(runs)
    rows = 2 if view == "xy+yz" else 1
    # side views are as tall as the chamber (plus room for labels)
    asp = max(max(_box_geometry(c_, f_)[1] / (2 * _box_geometry(c_, f_)[0]) for _, (f_, _t, c_) in runs), 0.08) * stretch
    side_h = panel * asp + 0.9
    h = {"xy+yz": panel + side_h, "yz": side_h + 0.4, "xz": side_h + 0.4}.get(view, panel)
    fig = plt.figure(figsize=(panel * n + (0.9 if color_by in ("height", "coordination") else 0), h + 0.4), dpi=dpi,
                     constrained_layout=True)
    gs = fig.add_gridspec(rows, n, height_ratios=[panel, side_h] if rows == 2 else [1])
    rng_all, axes = None, []
    fields = fields or [None] * n
    for c, ((label, (frames, t, cfg)), k, fld) in enumerate(zip(runs, k_per_run, fields)):
        P = frames[k]
        half, top = _box_geometry(cfg, frames)
        cols, rng = particle_colours(P, color_by, plain, rng=fixed_range, cmap=cmap)
        rng_all = rng or rng_all
        planes = ["xy", "yz"] if view == "xy+yz" else [view]
        for r, pl in enumerate(planes):
            if pl == "3d":
                ax = fig.add_subplot(gs[r, c], projection="3d")
                _draw_3d(ax, P, cols, half, top, elev, azim, scale, show_box, panel, alpha, depthshade, edge, stretch,
                         fld, field_mode, field_alpha)
            else:
                ax = fig.add_subplot(gs[r, c])
                _draw_2d(ax, P, cols, pl, half, top, scale, show_box, edge, alpha, fld, field_mode, field_alpha, stretch)
                ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=font - 1)
            axes.append(ax)
            if r == 0:
                lines = []
                if show_name:
                    lines.extend(run_info_lines(cfg, label))
                info = []
                if show_time:
                    info.append(f"t = {t[k] / TAUB_S:.2f} τ$_B$")
                if show_stats:
                    s_, _ = frame_stats(P, box=2 * half)
                    info.append(f"z = {s_['z3']:.2f}  S$_{{max}}$ = {s_['Smax']:.2f}  on top = {s_['f_on_top']:.0%}")
                if info:
                    lines.append("   ".join(info))
                if lines:
                    import textwrap
                    width = max(20, int(panel * 72 / (0.62 * font)))       # characters that fit the panel
                    ax.set_title("\n".join(textwrap.fill(l_, width) for l_ in lines), loc="left", fontsize=font)
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

"""Run viewer: draw a run's particles at any saved time, side by side for several runs,
turn runs into an animated GIF, and inspect single particles in a live (clickable) view.
Used by the "Run viewer" section of app.py.

Works on the COMSOL export layout and on the Python BD runs (same layout):
datax.csv / datay.csv, header row, then alternating position rows (µm) and time rows (s).

Model forces (live view) are recomputed from the run's folder name (A, d_n, AO q/phi) with
the same force laws as the COMSOL model / airpods_bd.py: Lennard-Jones (sigma = 2a) or AO
attraction, the Coulomb core of "Particle-Particle Interaction 2" (Z = 100, only r < 2a) and the
acoustic radiation force fitted to the COMSOL ARF runs (standing wave along y, node at y = 0).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TAUB_S = 13.514

# --------------------------------------------------------------------------- physics (kBT, µm, s)
KB, T_K, ETA, A_P = 1.380649e-23, 293.0, 2.9e-3, 1.0
D_UM2 = KB * T_K / (6 * np.pi * ETA * A_P * 1e-6) * 1e12        # 0.0740 µm²/s
KT_PER_UM_IN_PN = KB * T_K / 1e-6 * 1e12                         # 1 kBT/µm = 0.00405 pN
SIGMA = 2.0 * A_P
CORE_K = (100 * 1.602176634e-19) ** 2 / (4 * np.pi * 8.8541878128e-12) / (KB * T_K) * 1e6
ARF_F0_REF, ARF_K, ARF_DN_REF = 18.4, 0.124, 4000.0


@dataclass
class RunModel:
    """Interaction parameters of a run, read from its folder name."""
    regime: str = "ARF"
    A: float = 0.0
    dn: float = 0.0
    ao: tuple | None = None
    notes: list = field(default_factory=list)

    @classmethod
    def from_name(cls, regime: str, name: str):
        try:
            import regime_state_model as rsm
            p = rsm.parse_run_name(regime, name) or {}
        except Exception:
            p = {}
        m = cls(regime=regime)
        m.A = float(p.get("A", 0.0) or 0.0)
        dn = p.get("dn", np.nan)
        m.dn = float(dn) if dn is not None and np.isfinite(dn) else 0.0
        if p.get("potential") == "AO" and np.isfinite(p.get("q", np.nan)):
            m.ao = (float(p["q"]), float(p["phi_p"]))
        if regime != "ARF":
            m.dn = 0.0
            if regime == "ASF":
                m.notes.append("Regime IV: acoustic streaming is not modelled here, no field force shown")
            if regime == "AAF":
                m.notes.append("Regime III: field off")
        if not p:
            m.notes.append("could not read parameters from the folder name")
        return m

    def label(self):
        pot = f"AO q={self.ao[0]:g}, φp={self.ao[1]:g}" if self.ao else f"LJ A={self.A:g} kBT"
        return f"{pot}; d_n = {self.dn:g} nm" + ("" if self.dn else " (no acoustic force)")


def radial_force(r, model: RunModel):
    """Pair force along the line of centres, kBT/µm, + = repulsive."""
    r = np.asarray(r, float)
    fr = np.zeros_like(r)
    m = r < 2 * A_P
    fr[m] += CORE_K / r[m] ** 2
    if model.ao is None:
        if model.A > 0:
            m = r < 3 * SIGMA
            s6 = (SIGMA / r[m]) ** 6
            fr[m] += 24 * model.A / r[m] * (2 * s6 ** 2 - s6)
    else:
        q, phi = model.ao
        d = q * A_P
        n_p = phi / (4 * np.pi / 3 * d ** 3)
        m = r < 2 * (A_P + d)
        rc = np.maximum(r[m], 2 * A_P)
        fr[m] -= n_p * np.pi * ((A_P + d) ** 2 - rc ** 2 / 4)
    return fr


def arf_force_y(y, model: RunModel):
    if not model.dn:
        return np.zeros_like(np.asarray(y, float))
    return -ARF_F0_REF * (model.dn / ARF_DN_REF) ** 2 * np.sin(ARF_K * np.asarray(y, float))


def forces_on(P, i, model: RunModel, r_max=6.5):
    """Model forces on particle i: total pair force, per-neighbour contributions, ARF (kBT/µm)."""
    d = P[i] - P                      # vector from j to i
    r = np.linalg.norm(d, axis=1)
    r[i] = np.inf
    js = np.where(r < r_max)[0]
    fr = radial_force(r[js], model)
    contrib = (fr / r[js])[:, None] * d[js]
    f_pair = contrib.sum(0) if len(js) else np.zeros(2)
    f_arf = np.array([0.0, float(arf_force_y(P[i, 1], model))])
    return f_pair, f_arf, js, r[js], fr, contrib


def all_forces(P, model: RunModel):
    dx = P[:, None, :] - P[None, :, :]
    r = np.sqrt((dx ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    fr = radial_force(r, model)
    f_pair = ((fr / r)[:, :, None] * dx).sum(1)
    f_arf = np.c_[np.zeros(len(P)), arf_force_y(P[:, 1], model)]
    return f_pair, f_arf


# --------------------------------------------------------------------------- data
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


def frame_at_time(t, t_s):
    return int(np.clip(np.searchsorted(t, t_s - 1e-9), 0, len(t) - 1))


# --------------------------------------------------------------------------- per-particle fields
def frame_fields(P: np.ndarray, rc: float = 2.9):
    """Coordination number, complex psi6, cluster label and bond list for one frame."""
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
    psi = np.where(z > 0, psi / np.maximum(z, 1), 0)
    return z, psi, lab, b


def hexagonal_mask(P, eps=3.043, thr=0.65):
    """Analyser 'Detect crystals' rule: >= 3 neighbours within eps and |psi6| > thr."""
    nb = cKDTree(P).query_ball_point(P, eps)
    out = np.zeros(len(P), bool)
    for i, lst in enumerate(nb):
        lst = [j for j in lst if j != i]
        if len(lst) >= 3:
            d = P[lst] - P[i]
            out[i] = abs(np.exp(6j * np.arctan2(d[:, 1], d[:, 0])).mean()) > thr
    return out


# name -> (kind, colorbar label). kind: seq (0..max), div (signed), cyc (angle), cat (categories)
METRICS = {
    "Plain": ("plain", ""),
    "Coordination number": ("seq", "neighbours within the bond cutoff"),
    "Local order |ψ6|": ("seq", "|ψ6|"),
    "Crystal orientation (arg ψ6)": ("cyc", "lattice orientation (°)"),
    "Hexagonal (analyser rule)": ("cat", ""),
    "Clusters": ("cat", ""),
    "Cluster size": ("seq", "particles in its cluster"),
    "Speed": ("seq", "speed (µm/s)"),
    "Displacement from start": ("seq", "distance from start position (µm)"),
    "Distance from node |y|": ("seq", "|y| (µm)"),
    "Local density": ("seq", "particles within 5 µm"),
    "Pair force |F|": ("seq", "pair force (kBT/µm)"),
    "Acoustic force F_y": ("div", "acoustic force F_y (kBT/µm)"),
    "Net model force |F|": ("seq", "|pair + acoustic| (kBT/µm)"),
    "Particle ID": ("cyc", "particle number"),
}
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_DIVERGING = ["#2a78d6", "#86b6ef", "#e8e7e1", "#f2a37f", "#eb6834"]
CAT_COLORS = {"largest cluster": "#eb6834", "other clusters (≥3)": "#3987e5", "single / pair": "#b8b6ad",
              "hexagonal": "#1baf7a", "not hexagonal": "#b8b6ad"}


def metric_values(name, x, y, t, k, model=None, rc=2.9, cache=None):
    """Values (and for categorical metrics, category names) of a metric for frame k."""
    P = np.c_[x[k], y[k]]
    z, psi, lab, b = frame_fields(P, rc) if cache is None else cache
    if name == "Coordination number":
        return z.astype(float)
    if name == "Local order |ψ6|":
        return np.abs(psi)
    if name == "Crystal orientation (arg ψ6)":
        return (np.degrees(np.angle(psi)) / 6.0) % 60.0
    if name == "Hexagonal (analyser rule)":
        return np.where(hexagonal_mask(P), "hexagonal", "not hexagonal")
    if name == "Clusters":
        sizes = np.bincount(lab)
        cat = np.where(sizes[lab] >= 3, "other clusters (≥3)", "single / pair").astype(object)
        cat[lab == np.argmax(sizes)] = "largest cluster"
        return cat
    if name == "Cluster size":
        return np.bincount(lab)[lab].astype(float)
    if name == "Speed":
        k0, k1 = max(k - 1, 0), min(k + 1, len(t) - 1)
        if k1 == k0:
            return np.zeros(len(P))
        return np.linalg.norm(np.c_[x[k1] - x[k0], y[k1] - y[k0]], axis=1) / (t[k1] - t[k0])
    if name == "Displacement from start":
        return np.hypot(x[k] - x[0], y[k] - y[0])
    if name == "Distance from node |y|":
        return np.abs(y[k])
    if name == "Local density":
        return np.array([len(l) - 1 for l in cKDTree(P).query_ball_point(P, 5.0)], float)
    if name in ("Pair force |F|", "Acoustic force F_y", "Net model force |F|"):
        model = model or RunModel()
        fp, fa = all_forces(P, model)
        if name == "Pair force |F|":
            return np.linalg.norm(fp, axis=1)
        if name == "Acoustic force F_y":
            return fa[:, 1]
        return np.linalg.norm(fp + fa, axis=1)
    if name == "Particle ID":
        return np.arange(len(P), dtype=float)
    return np.zeros(len(P))


def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib
    return {"seq": LinearSegmentedColormap.from_list("seq", _BLUE_RAMP),
            "div": LinearSegmentedColormap.from_list("div", _DIVERGING),
            "cyc": matplotlib.colormaps["twilight"]}


def value_range(name, vals):
    kind = METRICS[name][0]
    if kind == "cyc":
        return (0, 60) if "orientation" in name else (0, max(len(vals) - 1, 1))
    if kind == "div":
        m = max(np.abs(vals).max(), 1e-9)
        return -m, m
    if name == "Local order |ψ6|":
        return 0, 1
    if name == "Coordination number":
        return 0, 6
    hi = np.percentile(vals, 98) if len(vals) else 1
    return 0, max(hi, 1e-9)


# --------------------------------------------------------------------------- drawing (matplotlib)
@dataclass
class DrawOptions:
    metric: str = "Plain"
    plain_color: str = "#2a78d6"
    edge_color: str = "#ffffff"
    bond_color: str = "#6b6a63"
    trail_color: str = "#b8b6ad"
    show_bonds: bool = False
    trail: int = 0
    rc: float = 2.9
    particle_scale: float = 1.0
    show_run_name: bool = True
    show_time: bool = True
    show_stats: bool = True
    show_legend: bool = True
    show_colorbar: bool = True
    show_axes: bool = True
    show_box: bool = True
    font_size: float = 9.0
    fixed_range: tuple | None = None     # (lo, hi) for the colour scale; None = automatic
    panel_size: float = 4.6


def draw_frame(ax, x, y, t, k, opts: DrawOptions, *, title=None, model=None, box=25.0):
    """Draw frame k of a run on ax (particles to scale). Returns (values, range) for the colour bar."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Circle, Rectangle
    P = np.c_[x[k], y[k]]
    cache = frame_fields(P, opts.rc)
    z, psi, lab, b = cache
    kind = METRICS[opts.metric][0]
    vals, rng = None, None
    if kind == "plain":
        cols = [opts.plain_color] * len(P)
    else:
        vals = metric_values(opts.metric, x, y, t, k, model, opts.rc, cache)
        if kind == "cat":
            cols = [CAT_COLORS[v] for v in vals]
        else:
            rng = opts.fixed_range or value_range(opts.metric, vals)
            cmap = _cmaps()[kind]
            cols = cmap(np.clip((vals - rng[0]) / max(rng[1] - rng[0], 1e-12), 0, 1))

    ax.clear()
    if opts.show_box:
        ax.add_patch(Rectangle((-box, -box), 2 * box, 2 * box, fill=False, ec="#c3c2b7", lw=1))
    if opts.trail > 0:
        k0 = max(0, k - opts.trail)
        segs = [np.c_[x[k0:k + 1, i], y[k0:k + 1, i]] for i in range(P.shape[0])]
        ax.add_collection(LineCollection(segs, colors=opts.trail_color, linewidths=0.6, alpha=0.6, zorder=1))
    if opts.show_bonds and len(b):
        ax.add_collection(LineCollection(np.stack([P[b[:, 0]], P[b[:, 1]]], 1),
                                         colors=opts.bond_color, linewidths=0.6, zorder=2))
    ax.add_collection(PatchCollection([Circle(p, A_P * opts.particle_scale) for p in P], facecolors=cols,
                                      edgecolors=opts.edge_color, linewidths=0.4, zorder=3))
    ax.set_xlim(-box - 1, box + 1)
    ax.set_ylim(-box - 1, box + 1)
    ax.set_aspect("equal")
    if opts.show_axes:
        ax.set_xlabel("x (µm)", fontsize=opts.font_size)
        ax.set_ylabel("y (µm)", fontsize=opts.font_size)
        ax.tick_params(labelsize=opts.font_size * 0.9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    else:
        ax.set_axis_off()
    lines = []
    if opts.show_run_name and title:
        lines.append(title)
    info = []
    if opts.show_time:
        info.append(f"t = {t[k] / TAUB_S:.2f} τ$_B$")
    if opts.show_stats:
        info.append(f"mean z = {z.mean():.2f}   mean ψ6 = {np.abs(psi)[z >= 2].mean() if (z >= 2).any() else 0:.2f}")
    if info:
        lines.append("   ".join(info))
    if lines:
        ax.set_title("\n".join(lines), loc="left", fontsize=opts.font_size)
    return vals, rng


def make_figure(runs, k_per_run, opts: DrawOptions, models=None, dpi=None):
    """runs: list of (label, (x, y, t)); k_per_run: frame index for each run."""
    import matplotlib.pyplot as plt
    n = len(runs)
    extra = 0.9 if (opts.show_colorbar and METRICS[opts.metric][0] not in ("plain", "cat")) else 0
    fig, axes = plt.subplots(1, n, figsize=(opts.panel_size * n + extra, opts.panel_size + 0.3),
                             constrained_layout=True, squeeze=False, dpi=dpi)
    rng = None
    models = models or [None] * n
    for ax, (label, (x, y, t)), k, mdl in zip(axes[0], runs, k_per_run, models):
        _, r = draw_frame(ax, x, y, t, k, opts, title=label, model=mdl)
        rng = r or rng
    kind = METRICS[opts.metric][0]
    if opts.show_colorbar and rng is not None and kind in ("seq", "div", "cyc"):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cb = fig.colorbar(ScalarMappable(Normalize(*rng), _cmaps()[kind]), ax=axes[0].tolist(), shrink=0.8)
        cb.set_label(METRICS[opts.metric][1], fontsize=opts.font_size)
    if opts.show_legend and kind == "cat":
        from matplotlib.lines import Line2D
        names = (["largest cluster", "other clusters (≥3)", "single / pair"] if opts.metric == "Clusters"
                 else ["hexagonal", "not hexagonal"])
        h = [Line2D([], [], marker="o", ls="", mfc=CAT_COLORS[c], mec="white", ms=9, label=c) for c in names]
        fig.legend(handles=h, frameon=False, fontsize=opts.font_size, loc="outside lower left", ncol=len(h))
    return fig


def make_gif(runs, opts: DrawOptions, *, t_start_s, t_end_s, n_frames=60, fps=12, dpi=90, progress=None,
             models=None, fix_colour_range=True):
    """Animated GIF (bytes). Uses Pillow (installed with matplotlib)."""
    import matplotlib.pyplot as plt
    from PIL import Image
    times = np.linspace(t_start_s, t_end_s, int(n_frames))
    if fix_colour_range and opts.fixed_range is None and METRICS[opts.metric][0] in ("seq", "div"):
        # same colour scale in every frame, taken from the last frame
        lo, hi = 0.0, 0.0
        for label, (x, y, t) in runs:
            k = frame_at_time(t, t_end_s)
            v = metric_values(opts.metric, x, y, t, k, (models or [None])[0], opts.rc)
            a, b = value_range(opts.metric, v)
            lo, hi = min(lo, a), max(hi, b)
        opts = DrawOptions(**{**opts.__dict__, "fixed_range": (lo, hi)})
    frames = []
    size = None
    for i, ts in enumerate(times):
        fig = make_figure(runs, [frame_at_time(r[1][2], ts) for r in runs], opts, models)
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
    w = min(f.width for f in frames)
    h = min(f.height for f in frames)
    frames = [f.crop((0, 0, w, h)).convert("P", palette=Image.ADAPTIVE, colors=255) for f in frames]
    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   duration=int(round(1000 / max(fps, 1))), loop=0, optimize=False, disposal=2)
    return out.getvalue()


# --------------------------------------------------------------------------- live view (plotly)
def live_figure(x, y, t, k, opts: DrawOptions, model=None, selected=None, height=720, show_arrows=True):
    """Plotly figure of one frame; particles carry their index as customdata so a click selects them."""
    import plotly.graph_objects as go
    P = np.c_[x[k], y[k]]
    cache = frame_fields(P, opts.rc)
    z, psi, lab, b = cache
    kind = METRICS[opts.metric][0]
    fig = go.Figure()
    box = 25.0
    if opts.show_box:
        fig.add_shape(type="rect", x0=-box, y0=-box, x1=box, y1=box, line=dict(color="#c3c2b7", width=1))
    if opts.show_bonds and len(b):
        xs = np.c_[P[b[:, 0], 0], P[b[:, 1], 0], np.full(len(b), np.nan)].ravel()
        ys = np.c_[P[b[:, 0], 1], P[b[:, 1], 1], np.full(len(b), np.nan)].ravel()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=opts.bond_color, width=1),
                                 hoverinfo="skip", showlegend=False))
    # Particles are drawn as circles in data units (true size at any zoom or window width);
    # an almost invisible marker on each centre makes them clickable and carries the hover text.
    R = A_P * opts.particle_scale
    hover = [f"particle {i}<br>z = {z[i]}<br>|ψ6| = {abs(psi[i]):.2f}" for i in range(len(P))]
    cbar_trace = None
    if kind == "plain":
        fills = [opts.plain_color] * len(P)
    elif kind == "cat":
        vals = metric_values(opts.metric, x, y, t, k, model, opts.rc, cache)
        fills = [CAT_COLORS[v] for v in vals]
        hover = [h + f"<br>{v}" for h, v in zip(hover, vals)]
    else:
        from matplotlib.colors import to_hex
        vals = metric_values(opts.metric, x, y, t, k, model, opts.rc, cache)
        lo, hi = opts.fixed_range or value_range(opts.metric, vals)
        cmap = _cmaps()[kind]
        fills = [to_hex(c) for c in cmap(np.clip((vals - lo) / max(hi - lo, 1e-12), 0, 1))]
        hover = [h + f"<br>{opts.metric}: {v:.3g}" for h, v in zip(hover, vals)]
        if opts.show_colorbar:
            scale = [[f, to_hex(cmap(f))] for f in np.linspace(0, 1, 11)]
            cbar_trace = go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                    marker=dict(colorscale=scale, cmin=lo, cmax=hi, color=[lo], showscale=True,
                                                colorbar=dict(title=METRICS[opts.metric][1], thickness=12)))
    shapes = list(fig.layout.shapes)
    for (px_, py_), fc in zip(P, fills):
        shapes.append(dict(type="circle", xref="x", yref="y", x0=px_ - R, y0=py_ - R, x1=px_ + R, y1=py_ + R,
                           fillcolor=fc, line=dict(color=opts.edge_color, width=0.6), layer="below"))
    sel_ring = []
    if selected is not None and 0 <= selected < len(P):
        f_pair, f_arf, js, rj, fr, contrib = forces_on(P, selected, model or RunModel())
        bonded = js[rj < opts.rc]
        for j_ in bonded:
            sel_ring.append(dict(type="circle", xref="x", yref="y", x0=P[j_, 0] - R, y0=P[j_, 1] - R,
                                 x1=P[j_, 0] + R, y1=P[j_, 1] + R, line=dict(color="#eda100", width=3)))
        sel_ring.append(dict(type="circle", xref="x", yref="y", x0=P[selected, 0] - 1.15 * R,
                             y0=P[selected, 1] - 1.15 * R, x1=P[selected, 0] + 1.15 * R,
                             y1=P[selected, 1] + 1.15 * R, line=dict(color="#e34948", width=4)))
    fig.update_layout(shapes=shapes + sel_ring)
    if cbar_trace is not None:
        fig.add_trace(cbar_trace)
    fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", customdata=np.arange(len(P)),
                             marker=dict(size=14, color="rgba(0,0,0,0.01)"),
                             hovertext=hover, hoverinfo="text", showlegend=False, name="particles"))
    if selected is not None and 0 <= selected < len(P):
        if show_arrows:
            fmax = max(np.linalg.norm(f_pair), np.linalg.norm(f_arf), np.linalg.norm(f_pair + f_arf), 1e-9)
            L = 6.0
            for vec, col, name in ((f_pair, "#6250d6", "pair"), (f_arf, "#1baf7a", "acoustic"),
                                   (f_pair + f_arf, "#e34948", "net")):
                if np.linalg.norm(vec) < 1e-9:
                    continue
                e = P[selected] + L * vec / fmax
                fig.add_annotation(x=e[0], y=e[1], ax=P[selected, 0], ay=P[selected, 1], xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2.5,
                                   arrowcolor=col, text="")
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=3),
                                         name=f"{name} force"))
    title = []
    if opts.show_time:
        title.append(f"t = {t[k] / TAUB_S:.2f} τB (frame {k})")
    if opts.show_stats:
        title.append(f"mean z = {z.mean():.2f}, mean |ψ6| = {np.abs(psi)[z >= 2].mean() if (z >= 2).any() else 0:.2f}")
    fig.update_layout(height=height, width=height + (90 if cbar_trace is not None else 0),
                      margin=dict(l=50, r=20, t=40, b=40), title="   ".join(title) or None,
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#222"),
                      xaxis=dict(range=[-box - 1, box + 1], title="x (µm)" if opts.show_axes else None,
                                 showgrid=False, zeroline=False, visible=opts.show_axes),
                      yaxis=dict(range=[-box - 1, box + 1], title="y (µm)" if opts.show_axes else None,
                                 showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
                                 visible=opts.show_axes),
                      legend=dict(orientation="h", y=-0.08), clickmode="event+select", dragmode="pan")
    return fig


def particle_report(x, y, t, k, i, model: RunModel, rc=2.9):
    """Everything about particle i at frame k, as dicts/tables for the live view."""
    P = np.c_[x[k], y[k]]
    z, psi, lab, b = frame_fields(P, rc)
    f_pair, f_arf, js, rj, fr, contrib = forces_on(P, i, model)
    k0, k1 = max(k - 1, 0), min(k + 1, len(t) - 1)
    v = (np.array([x[k1, i] - x[k0, i], y[k1, i] - y[k0, i]]) / (t[k1] - t[k0])) if k1 > k0 else np.zeros(2)
    sizes = np.bincount(lab)
    f_net = f_pair + f_arf
    summary = {
        "particle": i,
        "position x, y (µm)": f"{P[i, 0]:.3f}, {P[i, 1]:.3f}",
        "velocity vx, vy (µm/s)": f"{v[0]:.3f}, {v[1]:.3f}",
        "speed (µm/s)": f"{np.linalg.norm(v):.3f}",
        "displacement from start (µm)": f"{np.hypot(x[k, i] - x[0, i], y[k, i] - y[0, i]):.3f}",
        "bonded neighbours (r < cutoff)": int(z[i]),
        "|ψ6|": f"{abs(psi[i]):.3f}",
        "lattice orientation (°)": f"{(np.degrees(np.angle(psi[i])) / 6) % 60:.1f}",
        "hexagonal (analyser rule)": bool(hexagonal_mask(P)[i]),
        "cluster size": int(sizes[lab[i]]),
        "pair force Fx, Fy (kBT/µm)": f"{f_pair[0]:.3f}, {f_pair[1]:.3f}",
        "acoustic force Fy (kBT/µm)": f"{f_arf[1]:.3f}",
        "net model force |F| (kBT/µm)": f"{np.linalg.norm(f_net):.3f}  (= {np.linalg.norm(f_net) * KT_PER_UM_IN_PN:.4f} pN)",
        "drift from model force D·F (µm/s)": f"{D_UM2 * f_net[0]:.3f}, {D_UM2 * f_net[1]:.3f}",
        "Brownian speed scale √(2D/Δt) (µm/s)": f"{np.sqrt(2 * D_UM2 / max(t[k1] - t[k0], 1e-12)):.3f}",
    }
    order = np.argsort(rj)
    rows = [{"neighbour": int(js[o]), "distance (µm)": round(float(rj[o]), 3),
             "bonded": bool(rj[o] < rc),
             "radial force (kBT/µm, + repulsive)": round(float(fr[o]), 3),
             "Fx on this particle": round(float(contrib[o, 0]), 3),
             "Fy on this particle": round(float(contrib[o, 1]), 3)} for o in order]
    return summary, rows

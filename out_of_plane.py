"""Out-of-plane (buckling) load: how hard the field squeezes each particle against its neighbours.

The model is 2D, so particles cannot leave the plane. This module estimates where they would try to:
for every particle and frame it sums the pair forces from its neighbours (signed: + pushing, - pulling),
and these are averaged over frames in 2 µm bins. Bonds that are just held at contact average to about zero
(core push and attraction cancel, thermal kicks average out); what is left is the net in-plane squeeze
that the field (or flow) adds. Negative = the layer is under tension. A squeezed particle that sits slightly higher than its neighbours is pushed further up,
so a particle can pop out of the layer where the squeeze beats what holds it in the plane.

Rough criterion (an estimate, not a 3D simulation): a particle can lift when its squeeze exceeds
    buoyant weight + k_BT / a
(the k_BT / a term: lifting by about one radius must also beat thermal motion). Both are in k_BT/µm,
the force unit of the simulation (1 k_BT/µm ≈ 4 fN).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation"))
import airpods_bd as bd          # noqa: E402  (single source of the pair forces)
import regime_state_model as rsm  # noqa: E402

G = 9.81
KBT_PER_UM_N = bd.KB * bd.T_K / 1e-6        # 1 k_BT/µm in newtons
# density difference particle - liquid (kg/m^3)
MATERIALS = {
    "polystyrene in water": 1050.0 - 998.0,
    "density-matched PMMA in decalin": 0.0,
}


def lift_threshold(drho, a_um=bd.A_P):
    """Squeeze (k_BT/µm) above which a particle can leave the plane: buoyant weight + k_BT/a."""
    weight = 4.0 / 3.0 * np.pi * (a_um * 1e-6) ** 3 * drho * G / KBT_PER_UM_N
    return float(weight + 1.0 / a_um)


THRESHOLDS = {k: lift_threshold(v) for k, v in MATERIALS.items()}


def run_core(p):
    """Core used by a run (stiff-core runs carry 'phs' / 'wca' in their name)."""
    tags = f"{p.get('extra', '')}_{p.get('name', '')}".lower()
    return "phs" if "phs" in tags else ("wca" if "wca" in tags else "coulomb")


def frame_squeeze(P, A, ao=None, core="coulomb"):
    """Per particle: sum of the pair-force magnitudes from its neighbours, signed (+ push, - pull), k_BT/µm."""
    dx = P[:, None, :] - P[None, :, :]
    r = np.sqrt((dx ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    fr = bd.radial_force(r, A, ao, core)
    return fr.sum(1)


def run_squeeze(run_dir, p, t_from=0.8, max_frames=150):
    """Squeeze samples of one run over its late frames.
    t_from: start of the window, as a fraction of the run (0.8 = last 20 %) or, if > 1, in τB.
    Returns x, y, s (flattened over frames and particles) and the number of frames used."""
    x, y, t = rsm.read_run(run_dir)
    T = t[-1]
    t0 = t_from * T if t_from <= 1 else t_from * bd.TAUB
    idx = np.where(t >= t0 - 1e-9)[0]
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    A = float(p.get("A", 0) or 0)
    ao = None
    if str(p.get("potential", "LJ")) == "AO":
        ao = (float(p["q"]), float(p["phi_p"]))
        A = 0.0
    core = run_core(p)
    X, Y, S = [], [], []
    for k in idx:
        P = np.c_[x[k], y[k]]
        s = frame_squeeze(P, A, ao, core)
        X.append(P[:, 0]); Y.append(P[:, 1]); S.append(s)
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), len(idx)


def squeeze_summary(df, root, t_from=0.8, progress=None):
    """One row per run: peak of the time-averaged squeeze map, squeeze at the band centre (|y| < 2 µm) and,
    per material, the share of occupied 2 µm bins above the lift threshold."""
    runs = df.drop_duplicates(["regime", "name"])
    rows = []
    for i, (_, r) in enumerate(runs.iterrows()):
        if progress:
            progress(i / max(len(runs), 1), f"{r['regime']}/{r['name']}")
        rd = os.path.join(root, r["regime"], r["name"])
        try:
            X, Y, S, nf = run_squeeze(rd, r.to_dict(), t_from)
        except Exception:
            continue
        _, m = squeeze_map(X, Y, S)
        occ = m[np.isfinite(m)]
        row = {k: r.get(k) for k in ("regime", "name", "potential", "q", "phi_p", "A", "dn", "F", "seed", "source")}
        row.update(squeeze_peak=float(np.percentile(occ, 95)) if occ.size else np.nan,
                   squeeze_centre=float(S[np.abs(Y) < 2.0].mean()) if (np.abs(Y) < 2.0).any() else np.nan,
                   squeeze_mean=float(S.mean()), frames=nf)
        for name, thr in THRESHOLDS.items():
            row[f"share_above__{name}"] = float((occ > thr).mean()) if occ.size else np.nan
        rows.append(row)
    if progress:
        progress(1.0, "done")
    return pd.DataFrame(rows)


def squeeze_map(X, Y, S, box=50.0, bin_um=2.0, min_samples=10):
    """Mean squeeze per 2 µm bin (NaN where no particle came)."""
    edges = np.arange(-box / 2, box / 2 + 1e-9, bin_um)
    tot, _, _ = np.histogram2d(X, Y, [edges, edges], weights=S)
    n, _, _ = np.histogram2d(X, Y, [edges, edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = tot / n
    m[n < min_samples] = np.nan
    return edges, m.T


def fig_squeeze_map(X, Y, S, title="", box=50.0, vmax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    edges, m = squeeze_map(X, Y, S, box)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    hi = vmax or max(10.0, float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else 10.0)
    im = ax.pcolormesh(edges, edges, m, cmap="RdBu_r", norm=SymLogNorm(linthresh=0.5, vmin=-hi, vmax=hi))
    cb = fig.colorbar(im, ax=ax, label="net in-plane squeeze (k$_B$T/µm); + pushed, − pulled")
    styles = ["--", ":"]
    for (name, thr), ls in zip(THRESHOLDS.items(), styles):
        cb.ax.axhline(thr, color="#2a78d6", ls=ls, lw=2)
        if np.nanmax(np.nan_to_num(m)) > thr:
            ax.contour(0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] + edges[:-1]), np.nan_to_num(m),
                       levels=[thr], colors="#2a78d6", linestyles=ls, linewidths=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)  (pressure node at y = 0)")
    ax.set_title(title, fontsize=10, loc="left")
    lines = [plt.Line2D([], [], color="#2a78d6", ls=ls, lw=2) for ls in styles]
    ax.legend(lines, [f"lift threshold, {n} ({t:.2f})" for n, t in THRESHOLDS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_squeeze_vs_dn(summary, metric="squeeze_centre", by=("potential", "q")):
    """Squeeze at the band centre (or metric="squeeze_peak": 95th percentile of the map) against d_n, one line
    per group (default: potential and q; markers = mean, bars = range over A and seeds; add "A" to `by` for one
    line per attraction). Runs without a field (Regime I) are drawn as a grey reference band."""
    import matplotlib.pyplot as plt
    ref = summary[(summary.dn.astype(float).fillna(0) <= 0)]
    d = summary[summary.dn.astype(float).fillna(0) > 0].copy()
    def _lab(r):
        s_ = str(r.potential) + (f", q = {r.q:g}" if r.potential == "AO" and "q" in by else "")
        return s_ + (f", A = {r.A:g}" if "A" in by else "")
    d["label"] = d.apply(_lab, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for lab, g in d.groupby("label"):
        g = g.groupby("dn")[metric].agg(["mean", "min", "max"]).reset_index()
        ax.errorbar(g.dn, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], marker="o", ms=4,
                    capsize=2, label=lab)
    for (name, thr), ls in zip(THRESHOLDS.items(), ["--", ":"]):
        ax.axhline(thr, color="#555", ls=ls, lw=1.2, label=f"lift threshold, {name}")
    if len(ref):
        lo, hi = np.nanpercentile(ref[metric], [10, 90])
        ax.axhspan(lo, hi, color="#bbb", alpha=0.4, label="no field (Regime I), 10–90 %")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("inward displacement d$_n$ (nm)  (field strength grows as d$_n^2$)")
    ax.set_ylabel({"squeeze_centre": "net squeeze at the band centre", "squeeze_peak": "net squeeze, 95th percentile of the map"}.get(metric, metric) + " (k$_B$T/µm)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_title("Where particles would try to leave the plane", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quasi-3D runs (simulation/airpods_bd3d.py): side view of the band
# ---------------------------------------------------------------------------
LIFT_Z_UM = 2.0     # a particle counts as lifted when its centre is more than one radius above the floor layer


def read_run3d(run_dir):
    """x, y, z (frames x N, µm) and t (s) of a quasi-3D run (datax / datay / dataz)."""
    x, y, t = rsm.read_run(run_dir)
    raw = np.loadtxt(os.path.join(run_dir, "dataz.csv"), delimiter=",", skiprows=1)
    return x, y, raw[0::2], t


def lifted_share(run_dir, t_from=0.8):
    x, y, z, t = read_run3d(run_dir)
    k = t >= t_from * t[-1]
    zz, yy = z[k], y[k]
    centre = np.abs(yy) < 4.0
    return float((zz > LIFT_Z_UM).mean()), float((zz[centre] > LIFT_Z_UM).mean()) if centre.any() else np.nan, float(zz.max())


def fig_side_profiles(runs, t_from=0.8, zmax=None):
    """Side view (y–z plane, all x together) of each run's late frames: particle density, one panel per run.
    runs: list of (label, folder)."""
    import matplotlib.pyplot as plt
    n = len(runs)
    fig, axs = plt.subplots(n, 1, figsize=(7.5, 1.35 * n + 0.8), sharex=True, squeeze=False)
    for ax, (lab, rd) in zip(axs[:, 0], runs):
        x, y, z, t = read_run3d(rd)
        k = t >= t_from * t[-1]
        Y, Z = y[k].ravel(), z[k].ravel()
        top = zmax or max(6.0, float(np.percentile(Z, 99.9)) + 1)
        h, ye, ze = np.histogram2d(Y, Z, [np.arange(-25, 25.01, 0.5), np.arange(0, top + 0.01, 0.25)])
        ax.pcolormesh(ye, ze, (h / k.sum()).T, cmap="Greys", vmin=0, vmax=np.percentile(h / k.sum(), 99.5))
        ax.axhline(bd.A_P, color="#2a78d6", lw=0.8, ls=":")
        ax.text(0.99, 0.9, f"{lab}   mean height {Z.mean():.1f} µm, band width (2 SD) {2 * Y.std():.1f} µm",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_ylim(0, top)
        ax.set_ylabel("z (µm)", fontsize=8)
    axs[-1, 0].set_xlabel("y (µm)  (pressure node at y = 0; floor layer dotted)")
    fig.suptitle("Side view of the band (last 20 % of each run, all x)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig

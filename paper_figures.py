"""paper_figures.py - recreates every figure of the "Python model / AO vs LJ" proposal deck.

Needs only the run folders (datax.csv / datay.csv), so anyone with the data can rebuild
the figures exactly.

    BD runs     <bd_root>/ARF/<run>, <bd_root>/NAF/<run>      (airpods_bd.py output, "BD runs v2")
    COMSOL runs <comsol_root>/ARF/<run>                       (only for figures 0 and 2)

Command line (makes all figures into --out):
    python paper_figures.py --bd "D:/.../BD runs v2" --comsol "D:/.../raw csv data" --out figures

In the app: sidebar -> "Paper figures".

Observables are computed once per run (every 10th saved frame) and cached in
<bd_root>/_paper_figures_cache.pkl; a run is recomputed only if its CSV files change.

Figure list (file names as in the deck):
    0_seeds                      COMSOL "Generate unique" kicks vs Python seeds
    1_timestep_convergence       LJ at tau_B/2000, /4000, /8000
    2_python_vs_comsol           same start, LJ
    3_AO_vs_LJ                   z, psi6, S_max, NN distance over time
    3b_AO_vs_LJ_pair_distances   g(r) at the end
    4_state_map_1_and_10_tauB    state of every run
    5_confinement_vs_attraction  NAF vs band only vs band + attraction
    5b_snapshots_10tauB          snapshots coloured by psi6
    6b_ao_timestep_full_system   AO with all 144 particles at tau_B/16000, /32000, /64000 (plan ao_dt)
    7_hexagonal_fraction         analyser rule and strict rule
    8_orientation                lattice orientation (grains), LJ vs AO
    8b_hexmask                   hexagonal / not hexagonal, LJ vs AO
Figures 6 / 6a (two-particle B2 and AO force check) come from verify_ao.py in the BD folder.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import regime_state_model as rsm
import run_viewer as rv

TAUB = 13.514           # s
RC = 2.9                # bond cutoff (µm)
HEX_EPS, HEX_THR = 3.043, 0.65   # analyser "Detect crystals" rule
EVERY = 10              # use every 10th saved frame (= 0.1 tau_B)
CACHE_NAME = "_paper_figures_cache.pkl"
CACHE_VERSION = 1

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8a8980"
LAB = {"z": "mean coordination number", "psi6": "local order ψ$_6$", "Smax": "largest cluster fraction",
       "y_spread": "band width, std(y) (µm)", "nn6": "mean 6-neighbour distance (µm)",
       "nn1": "nearest-neighbour distance (µm)", "hex": "hexagonal fraction (analyser rule)",
       "hex_strict": "bulk hexagonal fraction (z = 6, ψ$_6$ > 0.8)"}
OBS = ["z", "psi6", "Smax", "y_spread", "nn6", "nn1", "hex", "hex_strict"]

FIGURES = {
    "0_seeds": "Random seeds: COMSOL vs Python",
    "1_timestep_convergence": "Time-step convergence",
    "2_python_vs_comsol": "Python vs COMSOL",
    "6b_ao_timestep_full_system": "AO time step, all 144 particles",
    "9_b2_explained": "What B2* is (formula and plots)",
    "10a_aaf_time_LJ": "Regime III over time, LJ",
    "10b_aaf_time_AO": "Regime III over time, AO",
    "10c_aaf_end_state": "Regime III end state vs attraction",
    "11a_asf_breakup": "Regime IV break-up, LJ vs AO",
    "11b_asf_cores": "Regime IV re-clustering in the vortex cores",
    "11c_asf_snapshots": "Regime IV end snapshots",
    "12_asf_flow": "Regime IV streaming flow (COMSOL export)",
    "13a_naf_time": "Regime I cluster formation, LJ vs AO",
    "13b_naf_bonds_snapshots": "Regime I bond lifetime and end state",
    "3_AO_vs_LJ": "AO vs LJ over time",
    "3b_AO_vs_LJ_pair_distances": "AO vs LJ pair distances g(r)",
    "4_state_map_1_and_10_tauB": "State map at 1 and 10 τB",
    "5_confinement_vs_attraction": "Confinement vs attraction",
    "5b_snapshots_10tauB": "Snapshots at 10 τB (ψ6)",
    "7_hexagonal_fraction": "Hexagonal fraction",
    "8_orientation": "Crystal orientation (grains)",
    "8b_hexmask": "Hexagonal particles",
}
NEEDS_COMSOL = {"0_seeds", "2_python_vs_comsol"}


def _style():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9b9a92", "xtick.color": "#555", "ytick.color": "#555",
                         "axes.titlesize": 10, "legend.frameon": False})
    return plt


# --------------------------------------------------------------------------- observables
def frame_obs(P):
    """z, psi6, S_max, std(y), mean 6-NN distance, NN distance, hexagonal fraction (analyser / strict)."""
    n = len(P)
    tree = cKDTree(P)
    b = tree.query_pairs(RC, output_type="ndarray")
    z = np.bincount(b.ravel(), minlength=n)
    _, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n)), directed=False)
    psi = np.zeros(n, complex)
    if len(b):
        d = P[b[:, 1]] - P[b[:, 0]]
        e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))
        np.add.at(psi, b[:, 0], e)
        np.add.at(psi, b[:, 1], e)
    ok = z >= 2
    psi6 = np.abs(psi[ok] / z[ok]).mean() if ok.sum() > 5 else np.nan
    dd, _ = tree.query(P, k=7)
    # hexagonal fraction, same rule as the analyser (neighbours within 3.043 µm)
    bh = tree.query_pairs(HEX_EPS, output_type="ndarray")
    zh = np.bincount(bh.ravel(), minlength=n)
    ph = np.zeros(n, complex)
    if len(bh):
        d = P[bh[:, 1]] - P[bh[:, 0]]
        e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))
        np.add.at(ph, bh[:, 0], e)
        np.add.at(ph, bh[:, 1], e)
    aph = np.abs(ph / np.maximum(zh, 1))
    hexf = ((zh >= 3) & (aph > HEX_THR)).mean()
    strict = ((zh == 6) & (aph > 0.8)).mean()
    return (z.mean(), psi6, np.bincount(lab).max() / n, P[:, 1].std(), dd[:, 1:].mean(), dd[:, 1].mean(),
            hexf, strict)


def _run_series(args):
    reg, name, rd, every = args
    x, y, t = rv.load_run(rd)
    rows = [(reg, name, t[k] / TAUB, *frame_obs(np.c_[x[k], y[k]])) for k in range(0, len(t), every)]
    P = np.c_[x[-1], y[-1]]
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))[np.triu_indices(len(P), 1)]
    return (reg, name), rows, d.astype(np.float32)


def _stamp(rd):
    return tuple((os.path.getmtime(os.path.join(rd, f)), os.path.getsize(os.path.join(rd, f)))
                 for f in ("datax.csv", "datay.csv"))


def load_series(bd_root, every=EVERY, workers=None, progress=None, use_cache=True):
    """DataFrame of observables over time for every BD run, plus end-frame pair distances.

    Returns (df, pairs) with pairs[(regime, name)] = all pair distances at the last frame.
    """
    # only the Python runs: the folder may also hold COMSOL runs (_CS) since the data merge
    runs = [(q, rd) for q, rd in rsm.discover_runs(bd_root, regimes=("ARF", "NAF", "AAF", "ASF")) if q.get("source") == "BD"]
    cache_file = os.path.join(bd_root, CACHE_NAME)
    cache = {}
    if use_cache and os.path.isfile(cache_file):
        try:
            with open(cache_file, "rb") as f:
                c = pickle.load(f)
            if c.get("version") == CACHE_VERSION and c.get("every") == every:
                cache = c["runs"]
        except Exception:
            cache = {}
    todo, out = [], {}
    for p, rd in runs:
        key = (p["regime"], p["name"])
        st = _stamp(rd)
        if key in cache and cache[key][0] == st:
            out[key] = cache[key]
        else:
            todo.append((p["regime"], p["name"], rd, every, st))
    if todo:
        args = [a[:4] for a in todo]
        stamps = {(a[0], a[1]): a[4] for a in todo}
        results = []
        try:
            if workers == 1:
                raise RuntimeError
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, r in enumerate(ex.map(_run_series, args)):
                    results.append(r)
                    if progress:
                        progress(i + 1, len(args))
        except Exception:
            results = []
            for i, a in enumerate(args):
                results.append(_run_series(a))
                if progress:
                    progress(i + 1, len(args))
        for key, rows, d in results:
            out[key] = (stamps[key], rows, d)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"version": CACHE_VERSION, "every": every, "runs": out}, f)
        except OSError:
            pass
    rows = [r for (_, rr, _) in out.values() for r in rr]
    df = pd.DataFrame(rows, columns=["regime", "name", "t", *OBS])
    meta = pd.DataFrame([{k: p[k] for k in ("regime", "name", "A", "T_name", "dn", "F", "tags", "seed",
                                             "potential", "q", "phi_p")} for p, _ in runs])
    df = df.merge(meta, on=["regime", "name"]).sort_values(["regime", "name", "t"]).reset_index(drop=True)
    df["base_tags"] = df.tags.str.replace(r"_?seed\d+", "", regex=True)
    pairs = {k: v[2] for k, v in out.items()}
    return df, pairs


# --------------------------------------------------------------------------- helpers
def sel(df, potential="LJ", A=None, dn=None, T=None, tags=None, regime="ARF", q=None):
    d = df[(df.regime == regime) & (df.potential == potential)]
    if A is not None:
        d = d[np.isclose(d.A, A)]
    if dn is not None:
        d = d[np.isclose(d.dn, dn)]
    if T is not None:
        d = d[np.isclose(d.T_name, T)]
    if tags is not None:
        d = d[d.base_tags == tags]
    if q is not None:
        d = d[np.isclose(d.q, q)]
    return d


def band(ax, d, m, col, label, ls="-"):
    """Mean over seeds (line) and min-max over seeds (shaded)."""
    if d.empty:
        return
    piv = d.pivot_table(index="t", columns="seed", values=m)
    t = piv.index.values
    ax.plot(t, piv.mean(axis=1).values, ls, c=col, lw=2, label=f"{label} (n={piv.shape[1]})")
    if piv.shape[1] > 1:
        ax.fill_between(t, piv.min(axis=1).values, piv.max(axis=1).values, color=col, alpha=0.18, lw=0)


def comsol_dir(root, regime, base):
    """COMSOL run folder: <base>_CS (merged Data folder) or <base> (old raw csv data folder)."""
    for n in (base + "_CS", base):
        d = os.path.join(root, regime, n)
        if os.path.isfile(os.path.join(d, "datax.csv")):
            return d
    raise FileNotFoundError(f"COMSOL run {regime}/{base}(_CS) not found in {root}")


def comsol_series(run_dir, every=5):
    x, y, t = rv.load_run(run_dir)
    co = np.array([frame_obs(np.c_[x[k], y[k]]) for k in range(0, len(t), every)])
    return t[::every] / TAUB, co


def first_step(run_dir):
    x, _, _ = rv.load_run(run_dir)
    return x[1] - x[0]


def state_of(r):
    if r.z < 2.5:
        return "fluid"
    conn = "network" if r.Smax >= 0.8 else "clusters"
    return ("ordered " if r.psi6 >= 0.8 else "disordered ") + conn


def _run_dir(bd_root, regime, name):
    return os.path.join(bd_root, regime, name)


# --------------------------------------------------------------------------- figures
def fig_seeds(df, bd_root, comsol_root,
              comsol_runs=("0Tkb_1TauB_2000nm", "5Tkb_1TauB_4000nm"),
              bd_runs=("2.5Tkb_10TauB_2750nm_BD_seed1", "2.5Tkb_10TauB_2750nm_BD_seed2"),
              A=2.5, dn=2750):
    plt = _style()
    a = first_step(comsol_dir(comsol_root, "ARF", comsol_runs[0]))
    b = first_step(comsol_dir(comsol_root, "ARF", comsol_runs[1]))
    c = first_step(_run_dir(bd_root, "ARF", bd_runs[0]))
    d = first_step(_run_dir(bd_root, "ARF", bd_runs[1]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
    lim = max(np.abs(np.r_[a, b, c, d]).max(), 0.1) * 1.1
    for axx, (u, v, ttl, col) in zip(ax[:2], (
            (a, b, "COMSOL: two different runs\n(0 k$_B$T, 2000 nm vs 5 k$_B$T, 4000 nm)", ORANGE),
            (c, d, f"Python: same run, seed 1 vs seed 2\n({A:g} k$_B$T, {dn:g} nm)", BLUE))):
        axx.plot([-lim, lim], [-lim, lim], c="#ccc", lw=1, zorder=0)
        axx.scatter(u, v, s=14, c=col)
        axx.set_title(ttl, loc="left")
        axx.set_xlim(-lim, lim)
        axx.set_ylim(-lim, lim)
        axx.set_aspect("equal")
        axx.set_xlabel("first-step x displacement, run 1 (µm)")
        axx.set_ylabel("same particle, run 2 (µm)")
        axx.text(0.04, 0.93, f"correlation r = {np.corrcoef(u, v)[0, 1]:.3f}", transform=axx.transAxes, fontsize=10)
    d3 = sel(df, A=A, dn=dn, T=10, tags="BD")
    d3 = d3[d3.t <= 3.0001]
    labels = []
    for s, col in zip(sorted(d3.seed.unique()), (BLUE, AQUA, VIOLET)):
        e = d3[d3.seed == s]
        ax[2].plot(e.t, e.z, c=col, lw=1.8, label=f"seed {s}")
        ax[2].plot(1.0, np.interp(1.0, e.t, e.z), "o", c=col, ms=7)
        r1 = e.iloc[(e.t - 1.0).abs().argmin()]
        labels.append(state_of(r1).replace("disordered ", "").replace("ordered ", ""))
    ax[2].axhline(2.5, ls="--", c="#888", lw=1)
    ax[2].text(2.95, 2.55, "fluid threshold z = 2.5", ha="right", fontsize=9, color="#555")
    ax[2].axvline(1.0, c="#ddd", lw=1, zorder=0)
    ax[2].set_xlabel("t (τ$_B$)")
    ax[2].set_ylabel("mean coordination number z")
    ax[2].set_title(f"Same parameters, three seeds: different label at 1 τ$_B$\n"
                    f"({A:g} k$_B$T, {dn:g} nm: {', '.join(labels)})", loc="left")
    ax[2].legend(loc="upper left")
    return fig


def fig_timestep(df, A=3, dn=4000):
    """Top: the three time steps on the 3-seed band. Bottom: difference to the 3-seed mean,
    so the lines that sit on top of each other in the top row can be told apart."""
    plt = _style()
    mets = ["z", "psi6", "y_spread", "nn1"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True, sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1]))
    ref = sel(df, A=A, dn=dn, T=10, tags="BD")
    ref = ref[ref.t <= 2.0001]
    runs = (("BD_dt2000", YELLOW, "τ$_B$/2000", "-", 2.2), ("BD", BLUE, "τ$_B$/4000", "--", 1.8),
            ("BD_dt8000", ORANGE, "τ$_B$/8000", ":", 2.2))
    for j, m in enumerate(mets):
        piv = ref.pivot_table(index="t", columns="seed", values=m)
        mu = piv.mean(axis=1)
        band(ax[0, j], ref, m, GRAY, "τ$_B$/4000, 3 seeds")
        ax[1, j].fill_between(mu.index, piv.min(axis=1) - mu, piv.max(axis=1) - mu, color=GRAY, alpha=0.25, lw=0,
                              label="spread of 3 seeds")
        ax[1, j].axhline(0, c="#999", lw=0.8)
        for tag, col, lab, ls, lw in runs:
            d = sel(df, A=A, dn=dn, T=2, tags=tag)
            if d.empty:
                continue
            ax[0, j].plot(d.t, d[m], ls, c=col, lw=lw, label=f"{lab} (1 run)")
            ax[1, j].plot(d.t, d[m].values - np.interp(d.t, mu.index, mu.values), ls, c=col, lw=lw, label=lab)
        ax[0, j].set_title(LAB[m], loc="left")
        ax[1, j].set_xlabel("t (τ$_B$)")
    ax[1, 0].set_ylabel("difference to the\n3-seed mean")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"1  Time-step convergence  (LJ, A = {A:g} k$_B$T, d$_n$ = {dn:g} nm). Top: values; bottom: "
                 "difference to the 3-seed mean (grey = seed-to-seed spread)", x=0.01, ha="left", fontsize=11)
    return fig



def fig_ao_timestep(df, dn=4000, q=0.2, ref_steps=32000, steps=(16000, 64000)):
    """Full 144-particle AO system at three time steps, 3 seeds each (plan "ao_dt" + plan "ao")."""
    plt = _style()
    mets = ["z", "psi6", "Smax", "nn1"]
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.2), constrained_layout=True, sharex=True,
                           gridspec_kw=dict(height_ratios=[1.3, 1]))
    ao = df[(df.regime == "ARF") & (df.potential == "AO") & np.isclose(df.dn, dn) & np.isclose(df.q, q)]
    ref = ao[np.isclose(ao.T_name, 10) & ao.base_tags.str.endswith(f"_dt{ref_steps}")]
    tmax = ao[np.isclose(ao.T_name, 2)].t.max() if (np.isclose(ao.T_name, 2)).any() else 2.0
    ref = ref[ref.t <= tmax + 1e-4]
    cols = {steps[0]: YELLOW, ref_steps: BLUE, steps[-1]: ORANGE}
    for j, m in enumerate(mets):
        mu = ref.pivot_table(index="t", columns="seed", values=m).mean(axis=1)
        for st in (steps[0], ref_steps, steps[-1]):
            d = ref if st == ref_steps else ao[np.isclose(ao.T_name, 2) & ao.base_tags.str.endswith(f"_dt{st}")]
            if d.empty:
                continue
            band(ax[0, j], d, m, cols[st], f"τ$_B$/{st}")
            piv = d.pivot_table(index="t", columns="seed", values=m)
            dm = piv.sub(np.interp(piv.index, mu.index, mu.values), axis=0)
            ax[1, j].plot(dm.index, dm.mean(axis=1), c=cols[st], lw=1.8)
            if piv.shape[1] > 1:
                ax[1, j].fill_between(dm.index, dm.min(axis=1), dm.max(axis=1), color=cols[st], alpha=0.15, lw=0)
        ax[1, j].axhline(0, c="#999", lw=0.8)
        ax[0, j].set_title(LAB[m], loc="left")
        ax[1, j].set_xlabel("t (τ$_B$)")
    ax[1, 0].set_ylabel(f"difference to the\nτ$_B$/{ref_steps} mean")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"6b  AO time-step check with all 144 particles (AO q = {q:g}, φ$_p$ = 0.633, d$_n$ = {dn:g} nm; "
                 "band = min–max over 3 seeds). Bottom: difference to the τ$_B$/32000 mean", x=0.01, ha="left", fontsize=11)
    return fig


def fig_b2_explainer(A=3.0, ao=(0.2, 0.633)):
    """What B2* is: the formula with its parts numbered, and the same parts in the plots,
    computed from the potentials in simulation/airpods_bd.py (core included)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation"))
    import airpods_bd as bd
    plt = _style()
    r = np.linspace(1.0, 8.0, 700001)
    trap = getattr(np, "trapezoid", None) or np.trapz

    def pot(args):
        fr = bd.radial_force(r, *args)
        U = np.zeros_like(r)
        U[:-1] = np.cumsum((0.5 * (fr[1:] + fr[:-1]) * np.diff(r))[::-1])[::-1]
        return U
    cases = [(f"LJ, A = {A:g} k$_B$T", BLUE, pot((A, None))),
             (f"AO, q = {ao[0]:g}, φ$_p$ = {ao[1]:g}", ORANGE, pot((0.0, ao)))]
    s2 = bd.SIGMA ** 2
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.30, 1], hspace=0.12, wspace=0.28,
                          left=0.05, right=0.99, top=0.98, bottom=0.11)
    axf = fig.add_subplot(gs[0, :]); axf.set_axis_off()
    ax = [fig.add_subplot(gs[1, j]) for j in range(3)]

    # formula, built from pieces so each part can be numbered
    parts = [(r"$B_2^* \,=\, \dfrac{B_2}{B_2^{\rm hard\ disc}} \,=\, \dfrac{2}{\sigma^2}\,$", "k", None),
             (r"$\int_0^\infty$", "#6250d6", "③"),
             (r"$\left[\,1 - \exp\left(-\right.\right.$", "#c0397a", "②"),
             (r"$U(r)$", "#333333", "①"),
             (r"$\left.\left./\,k_BT\,\right)\right]\, r$", "#c0397a", None),
             (r"$\;dr$", "#6250d6", None),
             (r"$\qquad B_2^{\rm hard\ disc} = \pi\sigma^2/2,\;\; \sigma = 2\,\mu m$", "#555555", None)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x = 0.01
    for txt, col, tag in parts:
        t = axf.text(x, 0.38, txt, fontsize=19, color=col, va="center", transform=axf.transAxes)
        bb = t.get_window_extent(rend).transformed(axf.transAxes.inverted())
        if tag:
            axf.text(0.5 * (bb.x0 + bb.x1), 0.84, tag, fontsize=15, color=col, ha="center", va="center",
                     transform=axf.transAxes, fontweight="bold")
        x = bb.x1
    axf.text(x + 0.01, 0.55, "", transform=axf.transAxes)

    # ① potential
    a = ax[0]
    a.axvspan(1.9, 2.0, color="#eeeeee", zorder=0)
    a.text(1.93, -6.2, "core", ha="center", fontsize=9, color="#777")
    for lab, col, U in cases:
        a.plot(r, U, c=col, lw=2, label=lab)
    a.axhline(0, c="#999", lw=0.8)
    a.set_xlim(1.9, 4.0); a.set_ylim(-6.5, 5.5)
    a.set_xlabel("centre distance r (µm)"); a.set_ylabel("U(r) / k$_B$T")
    a.set_title("①  the pair potential", loc="left", color="#333333", fontweight="bold")
    a.annotate("AO: −5.4 at contact,\nnarrow (gone by 2.4 µm)", (2.0, -5.38), (2.35, -5.6), fontsize=9, color=ORANGE,
               arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    a.annotate("LJ: −3.0 at 2.24 µm,\nwide", (2.245, -2.98), (2.75, -3.7), fontsize=9, color=BLUE,
               arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    a.legend(fontsize=9, loc="upper right")

    # ② integrand, area = B2*
    a = ax[1]
    out = []
    for lab, col, U in cases:
        h = (2 / s2) * (1 - np.exp(-U)) * r
        a.plot(r, h, c=col, lw=1.8)
        a.fill_between(r, 0, h, where=U < 0, color=col, alpha=0.22, lw=0)
        rep = trap(np.where(U > 0, h, 0), r) + (2 / s2) * r[0] ** 2 / 2
        att = trap(np.where(U < 0, h, 0), r)
        out.append((lab, col, rep, att))
    a.fill_between(r, 0, np.where(r < 2.0, (2 / s2) * r, 0), color="#bbbbbb", alpha=0.6, lw=0)
    a.set_yscale("symlog", linthresh=1.0)
    a.axhline(0, c="#999", lw=0.8)
    a.set_xlim(1.0, 4.0); a.set_ylim(-400, 5)
    a.set_xlabel("centre distance r (µm)"); a.set_ylabel("(2/σ²) [1 − e$^{-U/k_BT}$] r   (µm$^{-1}$)")
    a.set_title("②  what is integrated  (area = B$_2^*$)", loc="left", color="#c0397a", fontweight="bold")
    a.text(1.5, 1.6, "core: particles cannot\noverlap → area ≈ +1", ha="center", fontsize=9, color="#555")
    a.text(2.75, -60, "attraction: pairs\nmore likely together\n→ negative area", ha="left", fontsize=9, color="#555")

    # ③ running integral -> B2*
    a = ax[2]
    for (lab, col, U), (_, _, rep, att) in zip(cases, out):
        h = (2 / s2) * (1 - np.exp(-U)) * r
        run = np.concatenate([[0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(r))]) + (2 / s2) * r[0] ** 2 / 2
        a.plot(r, run, c=col, lw=2, label=f"{lab}:  +{rep:.1f} core − {-att:.1f} attraction = {rep + att:.1f}")
    a.axhline(1, ls="--", c="#999", lw=0.8)
    a.text(5.95, 1.4, "hard discs: B$_2^*$ = 1", ha="right", fontsize=9, color="#777")
    a.set_xlim(1.0, 6.0); a.set_ylim(-12, 2.5)
    a.set_xlabel("integrated up to r (µm)"); a.set_ylabel("running integral  →  B$_2^*$")
    a.set_title("③  the integral: both end at B$_2^*$ ≈ −10", loc="left", color="#6250d6", fontweight="bold")
    a.legend(fontsize=8.5, loc="lower left")
    return fig


# --------------------------------------------------------------------------- regime III (AAF)
AAF_METRICS = ("y_spread", "z", "Smax", "hex")


def _aaf_groups(df):
    a = df[df.regime == "AAF"]
    return a[a.potential == "LJ"], a[a.potential == "AO"]


def _comsol_aaf(comsol_root, every=5):
    """End-state and time series of the COMSOL AAF runs (one run per A, same start band)."""
    out = {}
    base = os.path.join(comsol_root or "", "AAF")
    if not os.path.isdir(base):
        return out
    for n in os.listdir(base):
        p = rsm.parse_run_name("AAF", n)
        if p is None or p.get("source") != "COMSOL" or not os.path.isfile(os.path.join(base, n, "datax.csv")):
            continue
        t, co = comsol_series(os.path.join(base, n), every)
        out[p["A"]] = (t, co)
    return out


def fig_aaf_time(df, comsol_root=None, potential="LJ"):
    """Regime III: each metric over time after the field is switched off, one line per A."""
    plt = _style()
    import matplotlib as mpl
    lj, ao = _aaf_groups(df)
    d = lj if potential == "LJ" else ao
    As = sorted(d.A.unique())
    cmap = mpl.colormaps["viridis"]
    fig, ax = plt.subplots(1, len(AAF_METRICS), figsize=(16, 3.9), constrained_layout=True)
    for k, A in enumerate(As):
        col = cmap(k / max(len(As) - 1, 1))
        for j, m in enumerate(AAF_METRICS):
            band(ax[j], d[np.isclose(d.A, A)], m, col, f"A = {A:g}" if potential == "LJ" else f"matched to A = {A:g}")
    for j, m in enumerate(AAF_METRICS):
        ax[j].set_title(LAB[m], loc="left")
        ax[j].set_xlabel("t after field off (τ$_B$)")
    ax[-1].legend(fontsize=7.5, loc="best")
    what = "LJ" if potential == "LJ" else "AO (q = 0.2, φ$_p$ matched by B$_2$*)"
    fig.suptitle(f"Regime III  After the acoustic field is switched off: {what}, same start band, 3 seeds per line",
                 x=0.01, ha="left", fontsize=11)
    return fig


def fig_aaf_end(df, comsol_root=None, t_from=18.0):
    """Regime III: end state (t > t_from) against attraction, LJ vs AO vs COMSOL."""
    plt = _style()
    lj, ao = _aaf_groups(df)
    co = _comsol_aaf(comsol_root)
    idx = {"z": 0, "psi6": 1, "Smax": 2, "y_spread": 3, "nn1": 5, "hex": 6}
    fig, ax = plt.subplots(1, len(AAF_METRICS), figsize=(16, 3.9), constrained_layout=True)
    for j, m in enumerate(AAF_METRICS):
        for d, col, lab, mk in ((lj, BLUE, "Python LJ", "o"), (ao, ORANGE, "Python AO (matched B$_2$*)", "s")):
            if d.empty:
                continue
            e = d[d.t > t_from].groupby(["A", "seed"])[m].mean().reset_index()
            g = e.groupby("A")[m]
            mu, lo, hi = g.mean(), g.min(), g.max()
            ax[j].plot(mu.index, mu.values, "-", c=col, lw=1.6, marker=mk, ms=5, label=lab)
            ax[j].fill_between(mu.index, lo.values, hi.values, color=col, alpha=0.18, lw=0)
        if co:
            As = sorted(co)
            vals = [co[A][1][co[A][0] > t_from, idx[m]].mean() for A in As]
            ax[j].plot(As, vals, "x", c="#555", ms=7, mew=1.6, label="COMSOL (1 run, identical kicks)")
        t0 = df[(df.regime == "AAF") & (df.t < 1e-6)][m].mean()
        if np.isfinite(t0):
            ax[j].axhline(t0, ls="--", c="#aaa", lw=1)
            ax[j].text(5.0, t0, " start", va="bottom", ha="right", fontsize=8, color="#888")
        ax[j].set_title(LAB[m], loc="left")
        ax[j].set_xlabel("attraction A (k$_B$T; AO matched by B$_2$*)")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Regime III  End state after the field is switched off (mean over t > {t_from:g} τ$_B$; band = min–max over 3 seeds; "
                 "dashed = start band)", x=0.01, ha="left", fontsize=11)
    return fig

def fig_vs_comsol(df, comsol_root, cases=((0, 2000), (0, 4000), (5, 2000), (5, 4000))):
    plt = _style()
    fig, ax = plt.subplots(2, len(cases), figsize=(15, 6), constrained_layout=True, sharex=True, squeeze=False)
    for j, (A, dn) in enumerate(cases):
        tc, co = comsol_series(comsol_dir(comsol_root, "ARF", f"{A:g}Tkb_1TauB_{dn:g}nm"))
        d = sel(df, A=A, dn=dn, T=1, tags="BD")
        for i, (m, ci) in enumerate((("z", 0), ("y_spread", 3))):
            band(ax[i, j], d, m, BLUE, "Python BD")
            ax[i, j].plot(tc, co[:, ci], c=ORANGE, lw=1.6, label="COMSOL")
            ax[i, j].set_ylabel(LAB[m] if j == 0 else "")
        ax[0, j].set_title(f"A = {A:g} k$_B$T, d$_n$ = {dn:g} nm", loc="left")
        ax[1, j].set_xlabel("t (τ$_B$)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("2  Python Brownian dynamics vs COMSOL (LJ, same start; band = 3 seeds)", x=0.01, ha="left", fontsize=11)
    return fig


def fig_ao_vs_lj(df, dns=(2000, 2750, 4000), A=3):
    plt = _style()
    mets = ["z", "psi6", "Smax", "nn1"]
    fig, ax = plt.subplots(4, len(dns), figsize=(14, 11), constrained_layout=True, sharex=True, squeeze=False)
    for j, dn in enumerate(dns):
        for i, m in enumerate(mets):
            band(ax[i, j], sel(df, A=A, dn=dn, T=10, tags="BD"), m, BLUE, f"LJ, A = {A:g} k$_B$T")
            band(ax[i, j], sel(df, "AO", dn=dn, T=10, q=0.2), m, ORANGE, "AO, q = 0.2, φ$_p$ = 0.633")
            if dn == 2750:
                band(ax[i, j], sel(df, "AO", dn=dn, T=2, q=0.1), m, AQUA, "AO, q = 0.1, φ$_p$ = 0.368")
            if j == 0:
                ax[i, j].set_ylabel(LAB[m])
        ax[0, j].set_title(f"d$_n$ = {dn:g} nm", loc="left")
        ax[-1, j].set_xlabel("t (τ$_B$)")
    ax[0, min(1, len(dns) - 1)].legend(fontsize=8, loc="lower right")
    fig.suptitle("3  AO vs LJ, matched by B$_2$* (Coulomb core as in COMSOL); band = min–max over 3 seeds",
                 x=0.01, ha="left", fontsize=11)
    return fig


def fig_pair_distances(df, pairs, dns=(2000, 2750, 4000), A=3):
    plt = _style()
    fig, ax = plt.subplots(1, len(dns), figsize=(14, 3.6), constrained_layout=True, squeeze=False)
    ax = ax[0]
    edges = np.arange(1.5, 8, 0.05)
    c = 0.5 * (edges[1:] + edges[:-1])
    for j, dn in enumerate(dns):
        for pot, col, lab in (("LJ", BLUE, "LJ"), ("AO", ORANGE, "AO q = 0.2")):
            d = sel(df, pot, A=A, dn=dn, T=10, tags=None if pot == "AO" else "BD")
            names = d[d.q.isna() | np.isclose(d.q.fillna(0), 0.2)].name.unique()
            H = [np.histogram(pairs[("ARF", n)], edges)[0] for n in names]
            if H:
                g = np.mean(H, 0) / (c * 0.05)
                ax[j].plot(c, g / g[c > 6].mean(), c=col, lw=1.6, label=lab)
        ax[j].set_title(f"d$_n$ = {dn:g} nm, t = 10 τ$_B$", loc="left")
        ax[j].set_xlabel("r (µm)")
    ax[0].set_ylabel("pair distance distribution\n(normalised at r > 6 µm)")
    ax[0].legend()
    return fig


def fig_state_map(df, times=(1.0, 10.0), dns=(0, 2000, 2250, 2750, 3000, 4000, 5000)):
    plt = _style()
    from matplotlib.lines import Line2D
    COL = {"fluid": BLUE, "disordered network": ORANGE, "ordered network": AQUA,
           "disordered clusters": YELLOW, "ordered clusters": VIOLET}
    MK = {"fluid": "o", "disordered network": "s", "ordered network": "D",
          "disordered clusters": "^", "ordered clusters": "v"}
    arf = df[(df.regime.isin(["ARF", "NAF"])) & ~df.tags.str.contains("dt2000|dt8000")].copy()
    arf.loc[arf.regime == "NAF", "dn"] = 0.0          # Regime I = no field = d_n 0
    fig, ax = plt.subplots(1, len(times), figsize=(13, 4.6), constrained_layout=True, sharey=True, squeeze=False)
    ax = ax[0]
    pos = {d: i for i, d in enumerate(dns)}
    for a, tt in zip(ax, times):
        snap = arf.loc[(arf.t - tt).abs() < 0.051].copy()
        snap = snap[snap.T_name >= tt - 0.01]
        snap["state"] = snap.apply(state_of, axis=1)
        for _, r in snap.iterrows():
            if r.dn not in pos:
                continue
            ao = r.potential == "AO"
            xx = pos[r.dn] + (r.seed - 2) * 0.13
            yy = r.A + (0.35 if ao else 0) + (0.18 if ao and r.q == 0.1 else 0)
            a.scatter(xx, yy, marker=MK[r.state], s=70, c="white" if ao else COL[r.state], edgecolors=COL[r.state],
                      linewidths=1.8 if ao else 0.8, zorder=3)
        a.set_xticks(range(len(dns)))
        a.set_xticklabels([f"{d:g}" if d else "0\n(no field)" for d in dns])
        if 0 in pos:
            a.axvline(pos[0] + 0.5, color="#bbb", lw=0.8, ls=":")
        a.set_xlim(-0.6, len(dns) - 0.4)
        a.set_xlabel("inward displacement d$_n$ (nm)")
        a.set_title(f"t = {tt:g} τ$_B$", loc="left")
        a.grid(axis="x", color="#eee", zorder=0)
    ax[0].set_ylabel("attraction A (k$_B$T)")
    ax[0].set_yticks([0, 1, 2.5, 3, 5])
    fig.text(0.01, -0.02, "filled = LJ, hollow (drawn just above A = 3) = AO matched to A = 3; "
             "three markers side by side = three seeds", fontsize=8, color="#555", ha="left", va="top")
    h = [Line2D([], [], marker=MK[k], ls="", mfc=COL[k], mec=COL[k], ms=9, label=k)
         for k in ("fluid", "disordered clusters", "ordered clusters", "disordered network", "ordered network")]
    h += [Line2D([], [], marker="o", ls="", mfc="white", mec="#555", mew=1.8, ms=9, label="hollow = AO")]
    ax[-1].legend(handles=h, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("4  State of every Python run at " + " and ".join(f"{t:g} τ$_B$" for t in times) +
                 " (criteria: z < 2.5 fluid; largest cluster ≥ 0.8 network; ψ$_6$ ≥ 0.8 ordered)",
                 x=0.01, ha="left", fontsize=11)
    return fig


def _confinement_cases(df, palette=(None, None, None)):
    """No field / field without attraction / field + the strongest attraction (AO), all at d_n = 4000 nm."""
    return [("Regime I: no field, no attraction", sel(df, regime="NAF", A=0, T=10), palette[0]),
            ("field (4000 nm), no attraction", sel(df, A=0, dn=4000, T=10, tags="BD"), palette[1]),
            ("field (4000 nm) + AO attraction", sel(df, "AO", dn=4000, T=10, q=0.2), palette[2])]


def fig_confinement(df):
    plt = _style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    for lab, d, col in _confinement_cases(df, (GRAY, VIOLET, ORANGE)):
        for j, m in enumerate(("psi6", "z", "nn1")):
            band(ax[j], d, m, col, lab)
    for j, m in enumerate(("psi6", "z", "nn1")):
        ax[j].set_title(LAB[m], loc="left")
        ax[j].set_xlabel("t (τ$_B$)")
    ax[1].legend(fontsize=8, loc="center right")
    fig.suptitle("5  Order from the field, not from attraction (d$_n$ = 4000 nm; band = 3 seeds)", x=0.01, ha="left", fontsize=11)
    return fig


def fig_hexfrac(df, dns=(2000, 2750, 4000), A=3):
    plt = _style()
    fig, ax = plt.subplots(2, len(dns) + 1, figsize=(17, 7), constrained_layout=True, sharey="row")
    for i, m in enumerate(("hex", "hex_strict")):
        for j, dn in enumerate(dns):
            band(ax[i, j], sel(df, A=A, dn=dn, T=10, tags="BD"), m, BLUE, f"LJ, A = {A:g} k$_B$T")
            band(ax[i, j], sel(df, "AO", dn=dn, T=10, q=0.2), m, ORANGE, "AO, q = 0.2")
            if dn == 2750:
                band(ax[i, j], sel(df, "AO", dn=dn, T=2, q=0.1), m, AQUA, "AO, q = 0.1")
            if i == 0:
                ax[i, j].set_title(f"AO vs LJ, d$_n$ = {dn:g} nm", loc="left")
        for lab, d, col in _confinement_cases(df, (GRAY, VIOLET, ORANGE)):
            band(ax[i, -1], d, m, col, lab)
        if i == 0:
            ax[i, -1].set_title("field vs attraction (all at 4000 nm)", loc="left")
        ax[i, 0].set_ylabel(LAB[m])
    for a in ax[1]:
        a.set_xlabel("t (τ$_B$)")
    ax[0, 1].legend(fontsize=8, loc="lower right")
    ax[0, -1].legend(fontsize=8, loc="center right")
    fig.suptitle("7  Hexagonal fraction over time (band = min–max over 3 seeds). Top: same rule as the analyser’s "
                 "“Detect crystals” (≥ 3 neighbours within 3.04 µm, |ψ$_6$| > 0.65). Bottom: strict, interior "
                 "particles only.", x=0.01, ha="left", fontsize=10)
    return fig


SNAPSHOT_RUNS = (("Regime I (no field), A = 0", "NAF", "0Tkb_10TauB_core_BD_seed1"),
                 ("band only, 5000 nm, A = 0", "ARF", "0Tkb_10TauB_5000nm_BD_seed1"),
                 ("LJ, 4000 nm, A = 3", "ARF", "3Tkb_10TauB_4000nm_BD_seed1"),
                 ("AO, 4000 nm, matched to A = 3", "ARF", "3Tkb_10TauB_4000nm_BD_AO_q0.2_phi0.633_dt32000_seed1"))
GRAIN_RUNS = (("LJ, A = 3 k$_B$T, 4000 nm", "ARF", "3Tkb_10TauB_4000nm_BD_seed1"),
              ("AO matched to A = 3, 4000 nm", "ARF", "3Tkb_10TauB_4000nm_BD_AO_q0.2_phi0.633_dt32000_seed1"))


def fig_snapshots(bd_root, runs=SNAPSHOT_RUNS, metric="Local order |ψ6|", title=None, **opt):
    """Last frame of each run, drawn with the Run viewer (same code as the app's Run viewer)."""
    _style()
    data = [(lab, rv.load_run(_run_dir(bd_root, reg, name))) for lab, reg, name in runs]
    models = [rv.RunModel.from_name(reg, name) for _, reg, name in runs]
    opts = rv.DrawOptions(metric=metric, **opt)
    f = rv.make_figure(data, [len(d[2]) - 1 for _, d in data], opts, models=models)
    if title:
        f.suptitle(title, x=0.01, ha="left", fontsize=11)
    return f


FIG_FUNCS = {
    "0_seeds": lambda df, pairs, bd, co: fig_seeds(df, bd, co),
    "1_timestep_convergence": lambda df, pairs, bd, co: fig_timestep(df),
    "2_python_vs_comsol": lambda df, pairs, bd, co: fig_vs_comsol(df, co),
    "6b_ao_timestep_full_system": lambda df, pairs, bd, co: fig_ao_timestep(df),
    "9_b2_explained": lambda df, pairs, bd, co: fig_b2_explainer(),
    "10a_aaf_time_LJ": lambda df, pairs, bd, co: fig_aaf_time(df, co, "LJ"),
    "10b_aaf_time_AO": lambda df, pairs, bd, co: fig_aaf_time(df, co, "AO"),
    "10c_aaf_end_state": lambda df, pairs, bd, co: fig_aaf_end(df, co),
    "11a_asf_breakup": lambda df, pairs, bd, co: fig_asf_breakup(df),
    "11b_asf_cores": lambda df, pairs, bd, co: fig_asf_cores(df, bd),
    "11c_asf_snapshots": lambda df, pairs, bd, co: fig_asf_snapshots(df, bd),
    "12_asf_flow": lambda df, pairs, bd, co: fig_asf_flow(),
    "13a_naf_time": lambda df, pairs, bd, co: fig_naf_time(df),
    "13b_naf_bonds_snapshots": lambda df, pairs, bd, co: fig_naf_bonds(df, bd),
    "3_AO_vs_LJ": lambda df, pairs, bd, co: fig_ao_vs_lj(df),
    "3b_AO_vs_LJ_pair_distances": lambda df, pairs, bd, co: fig_pair_distances(df, pairs),
    "4_state_map_1_and_10_tauB": lambda df, pairs, bd, co: fig_state_map(df),
    "5_confinement_vs_attraction": lambda df, pairs, bd, co: fig_confinement(df),
    "5b_snapshots_10tauB": lambda df, pairs, bd, co: fig_snapshots(
        bd, show_bonds=True, title="5b  Snapshots at 10 τ$_B$, coloured by local order ψ$_6$"),
    "7_hexagonal_fraction": lambda df, pairs, bd, co: fig_hexfrac(df),
    "8_orientation": lambda df, pairs, bd, co: fig_snapshots(bd, GRAIN_RUNS, "Crystal orientation (arg ψ6)"),
    "8b_hexmask": lambda df, pairs, bd, co: fig_snapshots(bd, GRAIN_RUNS, "Hexagonal (analyser rule)",
                                                          show_time=False, show_stats=False),
}


# --------------------------------------------------------------------------- Regime I (NAF), LJ vs AO
def _naf_pair(df, A=3.0):
    """Regime I runs of LJ at A and AO at the same B2*; the longest run length that has both."""
    d = df[(df.regime == "NAF") & np.isclose(df.A, A)]
    lens = [T for T in sorted(d.T_name.unique(), reverse=True)
            if {"LJ", "AO"} <= set(d[np.isclose(d.T_name, T)].potential)]
    if not lens:
        raise ValueError(f"no Regime I runs with both LJ and AO at A = {A:g} (plan naf_ref or naf_study)")
    return d[np.isclose(d.T_name, lens[0])], lens[0]


def fig_naf_time(df, A=3.0):
    """Regime I: who clusters first without field? z, largest cluster and psi6 over time."""
    plt = _style()
    d, T = _naf_pair(df, A)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for pot, col, lab in (("LJ", BLUE, f"LJ, A = {A:g} k$_B$T"), ("AO", ORANGE, "AO, same B$_2$*")):
        for a, m in zip(ax, ("z", "Smax", "psi6")):
            band(a, d[d.potential == pot], m, col, lab)
    ax[0].axhline(2.5, ls="--", c="#999", lw=0.8); ax[2].axhline(0.8, ls="--", c="#999", lw=0.8)
    ax[0].text(0.01, 0.97, "dashed: fluid threshold z = 2.5", transform=ax[0].transAxes, fontsize=8, color="#777", va="top")
    ax[2].text(0.99, 0.03, "dashed: ordered threshold ψ$_6$ = 0.8", transform=ax[2].transAxes, fontsize=8, color="#777", ha="right")
    for a, m in zip(ax, ("z", "Smax", "psi6")):
        a.set_title(LAB[m] + (" (particles with z ≥ 2)" if m == "psi6" else ""), loc="left"); a.set_xlabel("t (τ$_B$)")
    ax[0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"Regime I  No field: cluster formation, LJ vs AO at the same B$_2$* ({T:g} τ$_B$; band = min–max over seeds)",
                 x=0.01, ha="left", fontsize=11)
    return fig


def bond_persistence(x, y, lags, t_from_frac=0.5, stride=20, rc=RC):
    """Fraction of the bonds present at a frame that are still present `lag` frames later
    (averaged over start frames in the second half of the run)."""
    B = [set(map(tuple, cKDTree(np.c_[x[k], y[k]]).query_pairs(rc, output_type="ndarray"))) for k in range(len(x))]
    out = []
    for lag in lags:
        v = [len(B[k] & B[k + lag]) / len(B[k]) for k in range(int(len(x) * t_from_frac), len(x) - lag, stride) if B[k]]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


def fig_naf_bonds(df, bd_root, A=3.0, max_lag_tauB=2.0):
    """Regime I: bond lifetime (fraction of bonds still intact after a delay) and the end state of seed 1."""
    plt = _style()
    import matplotlib.colors as mc
    d, T = _naf_pair(df, A)
    fig = plt.figure(figsize=(15, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])
    a0 = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("Blues")
    for j, (pot, col, lab) in enumerate((("LJ", BLUE, f"LJ, A = {A:g} k$_B$T"), ("AO", ORANGE, "AO, same B$_2$*"))):
        names = sorted(d[d.potential == pot].name.unique())
        curves, tl = [], None
        for name in names:
            x, y, t = rv.load_run(_run_dir(bd_root, "NAF", name))
            dt = (t[1] - t[0]) / TAUB
            lags = np.arange(0, int(round(max_lag_tauB / dt)) + 1, max(1, int(round(0.1 / dt))))
            curves.append(bond_persistence(x, y, lags)); tl = lags * dt
            if name.endswith("seed1") or name == names[0] and not any(n.endswith("seed1") for n in names):
                snap = (np.c_[x[-1], y[-1]], t[-1] / TAUB)
        C = np.array(curves)
        a0.fill_between(tl, C.min(0), C.max(0), color=col, alpha=0.18, lw=0)
        a0.plot(tl, C.mean(0), c=col, lw=2, label=f"{lab} (n={len(C)})")
        b = fig.add_subplot(gs[j + 1]); P, tt = snap
        bonds = cKDTree(P).query_pairs(RC, output_type="ndarray")
        z = np.bincount(bonds.ravel(), minlength=len(P)); psi = np.zeros(len(P), complex)
        if len(bonds):
            dd = P[bonds[:, 1]] - P[bonds[:, 0]]; e = np.exp(6j * np.arctan2(dd[:, 1], dd[:, 0]))
            np.add.at(psi, bonds[:, 0], e); np.add.at(psi, bonds[:, 1], e)
        ap = np.abs(psi) / np.maximum(z, 1)
        for (px, py), v, zz in zip(P, ap, z):
            b.add_patch(plt.Circle((px, py), 1.0, fc=cmap(0.15 + 0.85 * v) if zz >= 2 else "#d8d8d8", ec="white", lw=0.3))
        b.set_xlim(-25, 25); b.set_ylim(-25, 25); b.set_aspect("equal"); b.set_xlabel("x (µm)"); b.set_ylabel("y (µm)")
        ok = z >= 2
        b.set_title(f"{lab} (seed 1)\nt = {tt:.1f} τ$_B$   mean z = {z.mean():.2f}   mean ψ$_6$ = {ap[ok].mean():.2f}", loc="left")
    sm = plt.cm.ScalarMappable(cmap=mc.LinearSegmentedColormap.from_list("c", [cmap(0.15), cmap(1.0)]), norm=mc.Normalize(0, 1))
    fig.colorbar(sm, ax=b, shrink=0.8, label="|ψ$_6$| (grey: fewer than 2 neighbours)")
    a0.set_ylim(0, 1.02); a0.set_xlabel("time later Δt (τ$_B$)"); a0.legend(fontsize=8.5, loc="lower left")
    a0.set_title(f"bonds still intact after Δt (bonds present in the second half of the run)", loc="left")
    fig.suptitle("Regime I  Bond lifetime and end state without field", x=0.01, ha="left", fontsize=11)
    return fig


# --------------------------------------------------------------------------- Regime IV (ASF)
def fig_asf_flow(flow_csv=None, start_csv=None):
    """The streaming flow exported from COMSOL (stationary laminar flow, F0 = 1e5 N/m^3) with the start band."""
    plt = _style()
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation")
    a = np.loadtxt(flow_csv or os.path.join(here, "asf_flow_F100000.csv"), delimiter=",", skiprows=1)
    x, y, u, v = a.T; o = np.lexsort((x, y)); n = int(round(np.sqrt(len(x))))
    X, Y, U, V = (q[o].reshape(n, n) for q in (x, y, u, v))
    xs, ys = np.linspace(X[0, 0], X[0, -1], n), np.linspace(Y[0, 0], Y[-1, 0], n)
    P = np.loadtxt(start_csv or os.path.join(here, "aaf_start.csv"), delimiter=",", comments="#")
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(np.hypot(U, V), origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="Blues", vmin=0)
    ax.streamplot(xs, ys, U, V, color="#1F2A44", density=1.1, linewidth=0.7, arrowsize=0.9)
    for px, py in P[:, :2]:
        ax.add_patch(plt.Circle((px, py), 1.0, fc=ORANGE, ec="white", lw=0.4, zorder=3))
    ax.set_xlim(-25, 25); ax.set_ylim(-25, 25); ax.set_aspect("equal"); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="flow speed (µm/s) at F$_0$ = 10$^5$ N/m³")
    ax.set_title("Streaming flow exported from COMSOL (stationary laminar flow)\nwith the start band of every Regime IV run", loc="left")
    return fig



ASF_CORES = ((-12.5, 0.0), (12.5, 0.0))   # centres of the left / right streaming vortex
ASF_CORE_R = 8.0                          # µm


def core_stats(P, rc=RC):
    """Particles in the two vortex cores: fraction of all, their mean coordination and the
    largest-cluster fraction inside each core (averaged over the two cores)."""
    out_n, out_z, out_s = 0, [], []
    for cx, cy in ASF_CORES:
        m = np.hypot(P[:, 0] - cx, P[:, 1] - cy) < ASF_CORE_R
        Q = P[m]
        out_n += m.sum()
        if len(Q) >= 3:
            b = cKDTree(Q).query_pairs(rc, output_type="ndarray")
            z = np.bincount(b.ravel(), minlength=len(Q))
            _, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(len(Q),) * 2),
                                          directed=False)
            out_z.append(z.mean())
            out_s.append(np.bincount(lab).max() / len(Q))
    return out_n / len(P), (np.mean(out_z) if out_z else np.nan), (np.mean(out_s) if out_s else np.nan)


def _asf_core_series(bd_root, df, every=10):
    rows = []
    for name in df[df.regime == "ASF"].name.unique():
        x, y, t = rv.load_run(_run_dir(bd_root, "ASF", name))
        for k in range(0, len(t), every):
            rows.append((name, t[k] / TAUB, *core_stats(np.c_[x[k], y[k]])))
    c = pd.DataFrame(rows, columns=["name", "t", "core_frac", "core_z", "core_Smax"])
    meta = df[df.regime == "ASF"].drop_duplicates("name")[["name", "A", "F", "potential", "seed"]]
    return c.merge(meta, on="name")


def breakup_time(d, thr=0.5):
    """Per seed: first time the largest cluster holds less than thr of all particles (NaN: never)."""
    out = {}
    for s, e in d.sort_values("t").groupby("seed"):
        below = e[e.Smax < thr]
        out[s] = below.t.iloc[0] if len(below) else np.nan
    return out


def fig_asf_breakup(df):
    """Regime IV, question 1: do AO or LJ bands break faster under streaming?"""
    plt = _style()
    a = df[df.regime == "ASF"]
    combos = sorted({(A, F) for A, F in zip(a.A, a.F)})
    fig, ax = plt.subplots(2, len(combos), figsize=(4.2 * len(combos), 6.4), constrained_layout=True,
                           sharex=True, squeeze=False)
    for j, (A, F) in enumerate(combos):
        for pot, col, lab in (("LJ", BLUE, f"LJ, A = {A:g}"), ("AO", ORANGE, "AO (matched B$_2$*)")):
            d = a[np.isclose(a.A, A) & np.isclose(a.F, F) & (a.potential == pot)]
            if d.empty:
                continue
            tb = breakup_time(d)
            vals = [v for v in tb.values() if np.isfinite(v)]
            txt = (f"{np.mean(vals):.2f} τ$_B$" + (f" ({len(vals)}/{len(tb)})" if len(vals) < len(tb) else "")) if vals else "no breakup"
            band(ax[0, j], d, "Smax", col, f"{lab}: breakup {txt}")
            band(ax[1, j], d, "hex", col, lab)
        ax[0, j].axhline(0.5, ls="--", c="#999", lw=0.8)
        ax[0, j].set_title(f"A = {A:g}, streaming F = {F:g}", loc="left")
        ax[0, j].legend(fontsize=7.5, loc="upper right")
        ax[1, j].set_xlabel("t in the streaming flow (τ$_B$)")
    ax[0, 0].set_ylabel(LAB["Smax"]); ax[1, 0].set_ylabel(LAB["hex"])
    fig.suptitle("Regime IV  Break-up under acoustic streaming, LJ vs AO at the same B$_2$* (band = min–max over 3 seeds; "
                 "breakup = largest cluster < 50 %)", x=0.01, ha="left", fontsize=11)
    return fig


def fig_asf_cores(df, bd_root):
    """Regime IV, question 2: do the broken-up particles re-cluster in the two vortex cores?"""
    plt = _style()
    c = _asf_core_series(bd_root, df)
    combos = sorted({(A, F) for A, F in zip(c.A, c.F)})
    fig, ax = plt.subplots(3, len(combos), figsize=(4.2 * len(combos), 8.6), constrained_layout=True,
                           sharex=True, squeeze=False)
    for j, (A, F) in enumerate(combos):
        for pot, col, lab in (("LJ", BLUE, f"LJ, A = {A:g}"), ("AO", ORANGE, "AO (matched B$_2$*)")):
            d = c[np.isclose(c.A, A) & np.isclose(c.F, F) & (c.potential == pot)]
            for i, m in enumerate(("core_frac", "core_z", "core_Smax")):
                band(ax[i, j], d, m, col, lab)
        ax[0, j].set_title(f"A = {A:g}, streaming F = {F:g}", loc="left")
        ax[-1, j].set_xlabel("t in the streaming flow (τ$_B$)")
    ax[0, 0].set_ylabel("fraction of particles\nin the vortex cores")
    ax[1, 0].set_ylabel("mean coordination number\ninside the cores")
    ax[2, 0].set_ylabel("largest cluster fraction\ninside a core")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Regime IV  Re-clustering in the two vortex cores (circles of {ASF_CORE_R:g} µm around x = ±12.5 µm, "
                 "y = 0; band = 3 seeds)", x=0.01, ha="left", fontsize=11)
    return fig


def fig_asf_snapshots(df, bd_root, F=None, A=None):
    """End frames, LJ vs AO, coloured by cluster, for one streaming strength and attraction."""
    a = df[(df.regime == "ASF") & (df.seed == 1)]
    F = F if F is not None else sorted(a.F.unique())[0]
    A = A if A is not None else sorted(a.A.unique())[0]
    runs = []
    for pot, lab in (("LJ", f"LJ, A = {A:g}"), ("AO", f"AO matched to A = {A:g}")):
        n = a[np.isclose(a.A, A) & np.isclose(a.F, F) & (a.potential == pot)].name.unique()
        if len(n):
            runs.append((f"{lab}, F = {F:g}", "ASF", n[0]))
    return fig_snapshots(bd_root, runs, "Clusters", title=f"Regime IV  End of the streaming runs (seed 1)")


def make_figure(key, df, pairs, bd_root, comsol_root=None):
    if key in NEEDS_COMSOL and not (comsol_root and os.path.isdir(comsol_root)):
        raise FileNotFoundError(f"{key} needs the COMSOL runs folder (with ARF/ inside)")
    if key.startswith("10"):
        lj, ao = _aaf_groups(df)
        need = {"10a_aaf_time_LJ": lj, "10b_aaf_time_AO": ao, "10c_aaf_end_state": pd.concat([lj, ao])}[key]
        if need.empty:
            raise FileNotFoundError("no Regime III runs (AAF folder) for this figure in the BD runs folder yet")
    if key.startswith("11") and not (df.regime == "ASF").any():
        raise FileNotFoundError("no Regime IV runs (ASF folder) yet: airpods_bd.py --plan asf")
    if key.startswith("6b") and not ((df.potential == "AO") & np.isclose(df.T_name, 2) & np.isclose(df.dn, 4000)).any():
        raise FileNotFoundError("needs the AO time-step runs (airpods_bd.py --plan ao_dt)")
    return FIG_FUNCS[key](df, pairs, bd_root, comsol_root)


def summary_table(df, t_from=9.5):
    """End-of-run averages (t > t_from) per parameter set, the numbers quoted on the slides."""
    e = df[df.t > t_from]
    g = e.groupby(["regime", "potential", "A", "dn", "q", "base_tags"], dropna=False)
    return g[OBS].mean().round(3).reset_index().assign(n_seeds=g.seed.nunique().values)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bd", required=True, help="Data folder with the regime folders (NAF, ARF, AAF)")
    ap.add_argument("--comsol", default=None, help="COMSOL csv folder with the regime folders (figures 0, 2 and 10)")
    ap.add_argument("--out", default="paper_figures_out")
    ap.add_argument("--only", nargs="*", default=None, help="figure keys, default all")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, pairs = load_series(a.bd, workers=a.workers,
                            progress=lambda i, n: print(f"\r  observables {i}/{n}", end="", flush=True))
    print(f"\n{df.name.nunique()} runs")
    import matplotlib.pyplot as plt
    for key in a.only or FIGURES:
        try:
            fig = make_figure(key, df, pairs, a.bd, a.comsol)
            fig.savefig(os.path.join(a.out, key + ".png"), dpi=150 if key.startswith(("5b", "8")) else a.dpi,
                        bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print("  ✓", key)
        except Exception as ex:
            print("  ✗", key, "-", ex)
    summary_table(df).to_csv(os.path.join(a.out, "summary_end_of_run.csv"), index=False)
    print("  ✓ summary_end_of_run.csv")


if __name__ == "__main__":
    main()

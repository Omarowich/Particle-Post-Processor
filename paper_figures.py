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
    runs = rsm.discover_runs(bd_root, regimes=("ARF", "NAF"))
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
    meta = pd.DataFrame([{k: p[k] for k in ("regime", "name", "A", "T_name", "dn", "tags", "seed",
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
    a = first_step(os.path.join(comsol_root, "ARF", comsol_runs[0]))
    b = first_step(os.path.join(comsol_root, "ARF", comsol_runs[1]))
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

def fig_vs_comsol(df, comsol_root, cases=((0, 2000), (0, 4000), (5, 2000), (5, 4000))):
    plt = _style()
    fig, ax = plt.subplots(2, len(cases), figsize=(15, 6), constrained_layout=True, sharex=True, squeeze=False)
    for j, (A, dn) in enumerate(cases):
        tc, co = comsol_series(os.path.join(comsol_root, "ARF", f"{A:g}Tkb_1TauB_{dn:g}nm"))
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


def fig_state_map(df, times=(1.0, 10.0), dns=(2000, 2250, 2750, 3000, 4000, 5000)):
    plt = _style()
    from matplotlib.lines import Line2D
    COL = {"fluid": BLUE, "disordered network": ORANGE, "ordered network": AQUA,
           "disordered clusters": YELLOW, "ordered clusters": VIOLET}
    MK = {"fluid": "o", "disordered network": "s", "ordered network": "D",
          "disordered clusters": "^", "ordered clusters": "v"}
    arf = df[(df.regime == "ARF") & ~df.tags.str.contains("dt2000|dt8000")]
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
        a.set_xticklabels([f"{d:g}" for d in dns])
        a.set_xlim(-0.6, len(dns) - 0.4)
        a.set_xlabel("inward displacement d$_n$ (nm)")
        a.set_title(f"t = {tt:g} τ$_B$", loc="left")
        a.grid(axis="x", color="#eee", zorder=0)
    ax[0].set_ylabel("attraction A (k$_B$T)")
    ax[0].set_yticks([0, 1, 2.5, 3, 5])
    ax[0].text(0.01, -0.2, "filled = LJ, hollow (drawn just above A = 3) = AO matched to A = 3; "
               "three markers side by side = three seeds", transform=ax[0].transAxes, fontsize=8, color="#555")
    h = [Line2D([], [], marker=MK[k], ls="", mfc=COL[k], mec=COL[k], ms=9, label=k)
         for k in ("fluid", "disordered clusters", "disordered network", "ordered network")]
    h += [Line2D([], [], marker="o", ls="", mfc="white", mec="#555", mew=1.8, ms=9, label="hollow = AO")]
    ax[-1].legend(handles=h, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("4  State of every Python run at " + " and ".join(f"{t:g} τ$_B$" for t in times) +
                 " (criteria: z < 2.5 fluid; largest cluster ≥ 0.8 network; ψ$_6$ ≥ 0.8 ordered)",
                 x=0.01, ha="left", fontsize=11)
    return fig


def _confinement_cases(df, palette):
    return [("NAF control: no field, A = 0", sel(df, regime="NAF", A=0, T=10), palette[0]),
            ("band only: d$_n$ = 5000 nm, A = 0", sel(df, A=0, dn=5000, T=10, tags="BD"), palette[1]),
            ("band + weak attraction: 5000 nm, A = 1", sel(df, A=1, dn=5000, T=10, tags="BD"), palette[2]),
            ("band + attraction: 4000 nm, A = 3 (LJ)", sel(df, A=3, dn=4000, T=10, tags="BD"), palette[3])]


def fig_confinement(df):
    plt = _style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    for lab, d, col in _confinement_cases(df, (GRAY, BLUE, AQUA, ORANGE)):
        for j, m in enumerate(("psi6", "z", "nn1")):
            band(ax[j], d, m, col, lab)
    for j, m in enumerate(("psi6", "z", "nn1")):
        ax[j].set_title(LAB[m], loc="left")
        ax[j].set_xlabel("t (τ$_B$)")
    ax[0].legend(fontsize=8)
    fig.suptitle("5  Order from confinement alone vs with attraction (band = 3 seeds)", x=0.01, ha="left", fontsize=11)
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
        cases = _confinement_cases(df, (GRAY, VIOLET, YELLOW, BLUE))
        for (lab, d, col), short in zip(cases, ("NAF: no field, A = 0", "band only: 5000 nm, A = 0",
                                                "band + A = 1: 5000 nm", "band + A = 3: 4000 nm (LJ)")):
            band(ax[i, -1], d, m, col, short)
        if i == 0:
            ax[i, -1].set_title("confinement vs attraction", loc="left")
        ax[i, 0].set_ylabel(LAB[m])
    for a in ax[1]:
        a.set_xlabel("t (τ$_B$)")
    ax[0, 1].legend(fontsize=8, loc="lower right")
    ax[0, -1].legend(fontsize=8, loc="center right")
    fig.suptitle("7  Hexagonal fraction over time (band = min–max over 3 seeds). Top: same rule as the analyser’s "
                 "“Detect crystals” (≥ 3 neighbours within 3.04 µm, |ψ$_6$| > 0.65). Bottom: strict, interior "
                 "particles only.", x=0.01, ha="left", fontsize=10)
    return fig


SNAPSHOT_RUNS = (("NAF control, A = 0", "NAF", "0Tkb_10TauB_core_BD_seed1"),
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


def make_figure(key, df, pairs, bd_root, comsol_root=None):
    if key in NEEDS_COMSOL and not (comsol_root and os.path.isdir(comsol_root)):
        raise FileNotFoundError(f"{key} needs the COMSOL runs folder (with ARF/ inside)")
    return FIG_FUNCS[key](df, pairs, bd_root, comsol_root)


def summary_table(df, t_from=9.5):
    """End-of-run averages (t > t_from) per parameter set, the numbers quoted on the slides."""
    e = df[df.t > t_from]
    g = e.groupby(["regime", "potential", "A", "dn", "q", "base_tags"], dropna=False)
    return g[OBS].mean().round(3).reset_index().assign(n_seeds=g.seed.nunique().values)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bd", required=True, help='folder with ARF/ and NAF/ BD runs ("BD runs v2")')
    ap.add_argument("--comsol", default=None, help="COMSOL csv folder with ARF/ (figures 0 and 2)")
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

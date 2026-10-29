"""Regime state model: label every run with an operational state and map the
fluid / network boundary with a Gaussian-process classifier.

Used by the "Regime state model" section of app.py, but works on its own too:

    from regime_state_model import compute_observables, label_runs, fit_boundary
    obs = compute_observables(r"D:\\...\\raw csv data")
    df = label_runs(obs)
    model = fit_boundary(df)

Data layout expected (same as the rest of the repo):
    <root>/<REGIME>/<A>Tkb_<T>TauB[_<d_n>nm | _<F>F]/datax.csv, datay.csv
Each CSV: header row of particle ids, then alternating rows of positions (µm)
and the time column (s).

State definitions (all thresholds are arguments, defaults are provisional):
    fluid                  mean coordination number z < z_fluid
    network / clusters     largest cluster spans the box (or holds >= smax_span
                           of the particles) / it does not
    ordered / disordered   mean local |psi6| (particles with z >= 2) >= psi_order / below
"""
from __future__ import annotations

import hashlib
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TAUB_S = 13.514          # Brownian time in seconds (same constant as the rest of the app)
REGIMES = ("AAF", "ARF", "ASF", "NAF")
REGIME_NAMES = {
    "NAF": "no acoustic force",
    "AAF": "after acoustic force",
    "ARF": "acoustic radiation force",
    "ASF": "acoustic streaming flow",
}
_NAME_RE = re.compile(r"^([\d.]+)Tkb_([\d.]+)TauB(?:_(\d+(?:\.\d+)?)(nm|nn|F))?((?:_[A-Za-z]+[\d.]*)*)$",
                      re.IGNORECASE)   # optional tags: _core, _dt50, _seed2, _AO_q0.2_phi0.664, ...

# colours: categorical slots of the app's chart palette (identity), blue ramp for magnitude
STATE_COLORS = {
    "fluid": "#2a78d6",
    "disordered network": "#eb6834",
    "ordered network": "#1baf7a",
    "disordered clusters": "#eda100",
    "ordered clusters": "#6250d6",
}
STATE_MARKERS = {
    "fluid": "o",
    "disordered network": "s",
    "ordered network": "D",
    "disordered clusters": "^",
    "ordered clusters": "v",
}
_BLUE_RAMP = ["#ffffff", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
              "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def parse_run_name(regime: str, name: str) -> dict | None:
    m = _NAME_RE.match(name)
    if not m:
        return None
    unit = (m.group(4) or "").lower()
    val = float(m.group(3)) if m.group(3) else np.nan
    return {
        "regime": regime,
        "name": name,
        "A": float(m.group(1)),
        "T_name": float(m.group(2)),
        "dn": val if unit in ("nm", "nn") else np.nan,
        "F": val if unit == "f" else np.nan,
        # source: _CS = COMSOL, _BD = Python Brownian dynamics; untagged names are COMSOL (old folders).
        # The _CS tag itself is not kept in "tags", so COMSOL runs behave as before (untagged).
        "tags": "_".join(t for t in (m.group(5) or "").lstrip("_").split("_") if t and t.upper() != "CS"),
        "source": "BD" if re.search(r"(^|_)BD(_|$)", m.group(5) or "") else "COMSOL",
        "seed": _tag_value(m.group(5), "seed", int, 0),
        "potential": "AO" if re.search(r"(^|_)AO(_|$)", m.group(5) or "") else "LJ",
        "q": _tag_value(m.group(5), "q", float, np.nan),
        "phi_p": _tag_value(m.group(5), "phi", float, np.nan),
    }


def _tag_value(tags, key, cast, default):
    """Value of a tag like _q0.2 or _seed2 in the folder-name suffix."""
    m = re.search(rf"_{key}([\d.]+)(?=_|$)", tags or "")
    return cast(m.group(1)) if m else default


def discover_runs(root: str, regimes=REGIMES) -> list[tuple[dict, str]]:
    out = []
    for reg in regimes:
        base = os.path.join(root, reg)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            rd = os.path.join(base, name)
            if not os.path.isfile(os.path.join(rd, "datax.csv")):
                continue
            p = parse_run_name(reg, name)
            if p is not None:
                out.append((p, rd))
    return out


def read_run(run_dir: str):
    """Return x, y (frames x particles, µm) and t (s)."""
    raw_x = np.loadtxt(os.path.join(run_dir, "datax.csv"), delimiter=",", skiprows=1)
    raw_y = np.loadtxt(os.path.join(run_dir, "datay.csv"), delimiter=",", skiprows=1)
    return raw_x[0::2], raw_y[0::2], raw_x[1::2, 0]


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def frame_observables(P: np.ndarray, rc: float, box: float):
    n = len(P)
    b = cKDTree(P).query_pairs(rc, output_type="ndarray")
    z = np.bincount(b.ravel(), minlength=n)
    g = coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    m = lab == np.argmax(sizes)
    spans = max(np.ptp(P[m, 0]), np.ptp(P[m, 1])) > 0.8 * box
    psi = np.zeros(n, complex)
    if len(b):
        d = P[b[:, 1]] - P[b[:, 0]]
        e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))   # 6-fold: angle and angle+pi give the same value
        np.add.at(psi, b[:, 0], e)
        np.add.at(psi, b[:, 1], e)
    ok = z >= 2
    psi6 = float(np.abs(psi[ok] / z[ok]).mean()) if ok.sum() > 5 else np.nan
    rg = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()))
    obs = dict(
        f_bonded=float((z > 0).mean()), z_mean=float(z.mean()), Smax=float(sizes.max() / n),
        n_clusters=int((sizes >= 3).sum()), spans=bool(spans), psi6=psi6, Rg=rg,
    )
    return obs, set(map(tuple, b))


def _evaluate(x, y, t, t_eval_s, rc, box, lag_s):
    k = min(int(np.searchsorted(t, t_eval_s - 1e-9)), len(t) - 1)
    o, B1 = frame_observables(np.c_[x[k], y[k]], rc, box)
    kh = int(np.searchsorted(t, t[k] - lag_s))
    _, B0 = frame_observables(np.c_[x[kh], y[kh]], rc, box)
    o["bond_persist"] = len(B0 & B1) / len(B0) if len(B0) >= 10 else np.nan
    o["msd_lag"] = float(np.mean((x[k] - x[kh]) ** 2 + (y[k] - y[kh]) ** 2))
    o["lag_s"] = float(t[k] - t[kh])
    P0 = np.c_[x[0], y[0]]
    o["Rg_ratio"] = o["Rg"] / np.sqrt(((P0 - P0.mean(0)) ** 2).sum(1).mean())
    o["t_eval_tauB"] = float(t[k] / TAUB_S)
    return o


def compute_observables(root: str, regimes=REGIMES, *, rc: float = 2.9, box: float = 50.0,
                        common_time_tauB: float = 1.0, lag_tauB: float = 0.5,
                        progress=None) -> pd.DataFrame:
    """One row per run at its last frame ('end'); ARF runs longer than
    common_time_tauB also get a row at that time ('common'), so runs of
    different length can be compared."""
    runs = discover_runs(root, regimes)
    rows = []
    for i, (p, rd) in enumerate(runs):
        if progress:
            progress(i / max(len(runs), 1), p["regime"] + "/" + p["name"])
        x, y, t = read_run(rd)
        T = t[-1]
        evals = [("end", T)]
        if p["regime"] == "ARF" and T > 1.1 * common_time_tauB * TAUB_S:
            evals.append(("common", common_time_tauB * TAUB_S))
        for tag, te in evals:
            r = dict(p)
            r["eval"] = tag
            r["T_actual_tauB"] = float(T / TAUB_S)
            r.update(_evaluate(x, y, t, te, rc, box, lag_tauB * TAUB_S))
            rows.append(r)
    if progress:
        progress(1.0, "done")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ref = df[(df.regime == "NAF") & (df.A == 0)]
    D0 = float((ref.msd_lag / (4 * ref.lag_s)).mean()) if len(ref) else np.nan
    df["mobility"] = df.msd_lag / (4 * D0 * df.lag_s) if np.isfinite(D0) else np.nan
    return df


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
@dataclass
class Thresholds:
    z_fluid: float = 2.5
    psi_order: float = 0.8
    smax_span: float = 0.8
    rg_concentrated: float = 0.85
    persist_low: float = 0.5


def label_runs(df: pd.DataFrame, th: Thresholds | None = None) -> pd.DataFrame:
    th = th or Thresholds()
    df = df.copy()

    def _state(r):
        if r.z_mean < th.z_fluid:
            return "fluid"
        conn = "network" if (r.spans or r.Smax >= th.smax_span) else "clusters"
        return ("ordered " if r.psi6 >= th.psi_order else "disordered ") + conn

    df["state"] = df.apply(_state, axis=1)
    df["concentrated"] = df.Rg_ratio < th.rg_concentrated
    df["restructuring"] = (df.state != "fluid") & (df.bond_persist < th.persist_low)
    df["near_threshold"] = (np.abs(df.z_mean - th.z_fluid) < 0.3) | (
        (df.state != "fluid") & (np.abs(df.psi6 - th.psi_order) < 0.04))
    return df


# ---------------------------------------------------------------------------
# Boundary model (Gaussian-process classifier, fluid vs not fluid)
# ---------------------------------------------------------------------------
@dataclass
class BoundaryModel:
    regime: str
    param: str
    data: pd.DataFrame
    gpc: object
    mu: np.ndarray
    sd: np.ndarray
    loo_hits: int
    boundary: list = field(default_factory=list)   # [(A, param*), ...]
    kernel: str = ""

    def proba(self, param_vals, A_vals):
        X = np.c_[np.log10(param_vals), A_vals]
        return self.gpc.predict_proba((X - self.mu) / self.sd)[:, 1]


def boundary_data(df: pd.DataFrame, regime="ARF", param="dn", common_time_tauB=1.0):
    d = df[(df.regime == regime) & np.isfinite(df[param])].copy()
    if "tags" in d:   # time-step test runs use a different step size: keep them out of the fit
        d = d[~d.tags.fillna("").str.contains("dt")]
    if "potential" in d:   # the boundary is fitted on the LJ runs; AO runs are compared against it
        d = d[d.potential.fillna("LJ") == "LJ"]
    d = d[np.isclose(d.t_eval_tauB, common_time_tauB, atol=0.05)]
    has_common = set(d.loc[d["eval"] == "common", "name"])
    return d[~((d["eval"] == "end") & d["name"].isin(has_common))]


def fit_boundary(df, regime="ARF", param="dn", common_time_tauB=1.0) -> BoundaryModel:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    d = boundary_data(df, regime, param, common_time_tauB)
    y = (d.state != "fluid").astype(int).values
    if len(d) < 6 or y.min() == y.max():
        raise ValueError(f"Need runs of both outcomes in {regime} at t = {common_time_tauB} τB "
                         f"(found {len(d)} runs, {int(y.sum())} non-fluid).")
    X = np.c_[np.log10(d[param].values), d.A.values]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    def _gpc(restarts):
        return GaussianProcessClassifier(C(1.0, (0.1, 10)) * RBF([1.0, 1.0], (0.3, 100)),
                                         n_restarts_optimizer=restarts, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpc = _gpc(5).fit(Xs, y)
        hits = 0
        for i in range(len(y)):
            tr = np.arange(len(y)) != i
            if y[tr].min() == y[tr].max():
                continue
            hits += int(_gpc(0).fit(Xs[tr], y[tr]).predict(Xs[[i]])[0] == y[i])
    m = BoundaryModel(regime, param, d, gpc, mu, sd, hits, kernel=str(gpc.kernel_))
    grid = np.logspace(np.log10(d[param].min()), np.log10(d[param].max()), 300)
    for a in sorted(d.A.unique()):
        p = m.proba(grid, np.full_like(grid, a))
        m.boundary.append((float(a), float(grid[np.argmin(np.abs(p - 0.5))])))
    return m


def suggest_runs(model: BoundaryModel, df: pd.DataFrame, *, step=250.0, n_repeat=2) -> pd.DataFrame:
    """New runs either side of the fitted boundary at low / mid / high A, repeats
    (new random seed) of the existing runs closest to it, and repeats of runs
    whose order label sits on the psi6 threshold."""
    unit = "nm" if model.param == "dn" else "F"
    As = sorted(model.data.A.unique())
    pick_A = sorted({As[0], float(np.round(np.median(As) * 2) / 2), As[-1]})
    bstar = float(np.median([b for _, b in model.boundary]))
    tested = np.sort(model.data[model.param].unique())
    below = tested[tested < bstar].max() if (tested < bstar).any() else bstar * 0.8
    above = tested[tested > bstar].min() if (tested > bstar).any() else bstar * 1.2
    cand = sorted({round((below + (above - below) * f) / step) * step for f in (0.25, 0.75)})
    plan = [dict(A_kT=a, value=v, unit=unit, kind="new",
                 purpose="locate the fluid/network boundary and test its A-dependence")
            for a in pick_A for v in cand if not ((model.data.A == a) & (model.data[model.param] == v)).any()]
    d = model.data.assign(dist=np.abs(np.log10(model.data[model.param]) - np.log10(bstar)))
    for _, r in d[d.state != "fluid"].nsmallest(n_repeat, "dist").iterrows():
        plan.append(dict(A_kT=r.A, value=r[model.param], unit=unit, kind="repeat",
                         purpose="same parameters, new random seed: scatter at the boundary"))
    nt = d[(d.state != "fluid") & d.near_threshold]
    for _, r in nt.nsmallest(1, "psi6").iterrows():
        plan.append(dict(A_kT=r.A, value=r[model.param], unit=unit, kind="repeat",
                         purpose=f"new seed: psi6 = {r.psi6:.2f} sits on the order threshold"))
    out = pd.DataFrame(plan).drop_duplicates(subset=["A_kT", "value", "kind"])
    return out.rename(columns={"value": f"{model.param} ({unit})"})


# ---------------------------------------------------------------------------
# Data checks
# ---------------------------------------------------------------------------
def data_checks(root: str, regimes=REGIMES) -> pd.DataFrame:
    rows, hashes, first = [], {}, {}
    for p, rd in discover_runs(root, regimes):
        key = f"{p['regime']}/{p['name']}"
        with open(os.path.join(rd, "datax.csv"), "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        if h in hashes:
            rows.append(dict(run=key, issue=f"identical datax.csv to {hashes[h]}"))
        hashes.setdefault(h, key)
        x, y, t = read_run(rd)
        T = t[-1] / TAUB_S
        if abs(T - p["T_name"]) > 0.1 * p["T_name"] + 0.05:
            rows.append(dict(run=key, issue=f"name says {p['T_name']:g} τB, data runs {T:.2f} τB"))
        if len(x) > 1:
            first.setdefault((p["regime"], len(x[0])), []).append((key, x[1] - x[0]))
    for (reg, _), lst in first.items():
        same = [a for (a, da), (b, db) in zip(lst, lst[1:]) if np.allclose(da, db, atol=1e-6)]
        if len(same) >= 2:
            rows.append(dict(run=f"{reg} ({len(lst)} runs)",
                             issue="first step identical across runs: same random seed, no independent repeats"))
    return pd.DataFrame(rows, columns=["run", "issue"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "xtick.color": "#555",
                         "ytick.color": "#555", "axes.labelcolor": "#222"})


def fig_state_map(model: BoundaryModel, plan: pd.DataFrame | None = None, figsize=(12, 5)):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    _style(plt)
    d, prm = model.data, model.param
    TK = np.sort(d[prm].unique())
    LTK = np.log10(TK)
    pos = lambda v: np.interp(np.log10(v), LTK, np.arange(len(TK)))  # tested values evenly spaced
    U = np.linspace(-0.4, len(TK) - 0.6, 400)
    A_lo, A_hi = d.A.min(), d.A.max()
    pad = 0.22 * (A_hi - A_lo or 1)
    UA = np.linspace(A_lo - pad, A_hi + pad, 160)
    GU, GA = np.meshgrid(U, UA)
    GL = np.interp(GU, np.arange(len(TK)), LTK)
    P = model.gpc.predict_proba((np.c_[GL.ravel(), GA.ravel()] - model.mu) / model.sd)
    pn = P[:, 1].reshape(GU.shape)
    H = (-(P * np.log(P + 1e-12)).sum(1) / np.log(2)).reshape(GU.shape)
    bstar = float(np.median([b for _, b in model.boundary]))
    unit = "nm" if prm == "dn" else "F"
    xlabel = ("inward displacement d$_n$ (nm" if prm == "dn" else "streaming force F (") + "; tested values evenly spaced)"

    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax[0].contourf(GU, GA, pn, levels=[0, 0.5, 1], colors=["#2a78d6", "#eb6834"], alpha=0.08)
    ax[0].contour(GU, GA, pn, levels=[0.5], colors="#555", linewidths=1.2, linestyles="--")
    ax[0].text(pos(bstar) + 0.08, A_lo - 0.75 * pad, f"fitted boundary ≈ {bstar:,.0f} {unit}",
               ha="left", va="center", fontsize=8, color="#555")
    ax[0].text(pos(bstar) / 2 - 0.2, A_lo - 0.75 * pad, "FLUID", ha="center", va="center",
               fontsize=9, color="#2a78d6", weight="bold")
    ax[0].text((pos(bstar) + len(TK) - 1) / 2, A_hi + 0.75 * pad, "BAND / NETWORK", ha="center",
               va="center", fontsize=9, color="#eb6834", weight="bold")
    for c in sorted(d.state.unique()):
        s = d[d.state == c]
        ax[0].scatter(pos(s[prm]), s.A, c=STATE_COLORS.get(c, "#888"), marker=STATE_MARKERS.get(c, "o"),
                      s=80, edgecolors="white", linewidths=1.2, label=c, zorder=3)
    ax[0].set_title(f"State after {d.t_eval_tauB.round(1).iloc[0]:g} τ$_B$ ({model.regime})",
                    loc="left", fontsize=11, pad=30)
    cs = ax[1].contourf(GU, GA, H, levels=np.linspace(0, 1, 11), cmap=ListedColormap(_BLUE_RAMP))
    ax[1].scatter(pos(d[prm]), d.A, c="#555", s=16, zorder=3, label="existing run")
    if plan is not None and len(plan):
        col = [c for c in plan.columns if c.startswith(prm)][0]
        new, rep = plan[plan.kind == "new"], plan[plan.kind == "repeat"]
        ax[1].scatter(pos(new[col]), new.A_kT, marker="*", s=220, c="#eb6834", edgecolors="white",
                      linewidths=1, zorder=4, label="suggested new run")
        ax[1].scatter(pos(rep[col]), rep.A_kT, marker="o", s=170, facecolors="none",
                      edgecolors="#eb6834", linewidths=2, zorder=4, label="repeat with new seed")
    fig.colorbar(cs, ax=ax[1], label="model uncertainty (0 = sure, 1 = coin flip)")
    ax[1].set_title("Where the model is unsure, and what to run next", loc="left", fontsize=11, pad=30)
    for a in ax:
        a.set_xticks(np.arange(len(TK)))
        a.set_xticklabels([f"{v:g}" for v in TK])
        a.set_xlim(-0.4, len(TK) - 0.6)
        a.set_ylim(A_lo - pad, A_hi + pad)
        a.set_yticks(np.unique(np.round(d.A.values, 3)))
        a.set_xlabel(xlabel)
        a.set_ylabel("attraction A (k$_B$T)")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                 borderaxespad=0.2, handletextpad=0.3, columnspacing=1.0)
    return fig


def fig_transitions(df: pd.DataFrame, metric="psi6", threshold=None, regimes=("AAF", "NAF"),
                    figsize=(7, 4.2)):
    """Metric vs A; one line per regime and run length (max 4 lines, palette order)."""
    import matplotlib.pyplot as plt
    _style(plt)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    labels = {"psi6": "Local hexagonal order ψ$_6$", "z_mean": "Mean coordination number",
              "Smax": "Largest cluster fraction", "bond_persist": "Bond persistence",
              "Rg_ratio": "R$_g$ / R$_g$(t=0)", "mobility": "Mobility (MSD / free MSD)"}
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    k, amax = 0, 0.0
    for reg in regimes:
        d = df[(df.regime == reg) & (df["eval"] == "end") & df.dn.isna() & df.F.isna()]
        if "tags" in d:   # tagged runs (_core, _seed2, ...) are not part of the original sweep lines
            d = d[d.tags.fillna("") == ""]
        for T in sorted(d.T_name.unique()):
            s = d[d.T_name == T].sort_values("A")
            if len(s) < 2 or k >= len(pal):
                continue
            ax.plot(s.A, s[metric], "-o", c=pal[k], lw=2, ms=6, mec="white",
                    label=f"{reg}, after {T:g} τ$_B$")
            amax = max(amax, s.A.max())
            k += 1
    if threshold is not None:
        ax.axhline(threshold, ls="--", c="#888", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ticks = [v for v in (0, 1, 2, 3, 5, 10, 20, 50, 100) if v <= max(amax, 1)]
    ax.set_xlim(-0.15, max(amax, 1) * 1.15)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_xlabel("attraction A (k$_B$T)")
    ax.set_title(labels.get(metric, metric), loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    return fig

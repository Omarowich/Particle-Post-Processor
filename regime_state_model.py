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
import pickle
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

TAUB_S = 13.514          # Brownian time in seconds (same constant as the rest of the app)
REGIMES = ("NAF", "ARF", "AAF", "ASF")   # Regime I-IV
# Folder names stay NAF / ARF / AAF / ASF (they are the data keys); everything shown to the user
# uses the paper's terminology: Regime I-IV.
REGIME_NAMES = {
    "NAF": "Brownian aggregation, no acoustic force",
    "ARF": "acoustic radiation force",
    "AAF": "post-acoustic relaxation",
    "ASF": "acoustic streaming flow",
}
REGIME_NUMBER = {"NAF": "I", "ARF": "II", "AAF": "III", "ASF": "IV"}


def regime_label(regime: str, long: bool = False) -> str:
    """'Regime II' (long: 'Regime II (acoustic radiation force)') for a folder name like ARF."""
    r = str(regime).upper()
    if r not in REGIME_NUMBER:
        return str(regime)
    return f"Regime {REGIME_NUMBER[r]}" + (f" ({REGIME_NAMES[r]})" if long else "")
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
        "dt": _tag_value(m.group(5), "dt", int, 0),
        "N": _tag_value(m.group(5), "N", int, 0) or np.nan,       # particle number tag, e.g. _N256
        # anything else in the name (e.g. a tag like N256 or test2), for the run filters
        "extra": "_".join(t for t in (m.group(5) or "").lstrip("_").split("_")
                          if t and not re.fullmatch(r"(?i)(CS|BD|AO|core|q[\d.]+|phi[\d.]+|dt\d+|seed\d+)", t)),
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


def _run_stamp(rd) -> tuple:
    """Changes when a run's data files change (new or re-run), so cached numbers are redone."""
    out = []
    for f in sorted(os.listdir(rd)):
        if f.endswith(".csv"):
            st_ = os.stat(os.path.join(rd, f))
            out.append((f, int(st_.st_mtime), st_.st_size))
    return tuple(out)


def _run_rows(p, rd, rc, box, common_time_tauB, lag_tauB, map_times_tauB) -> list:
    x, y, t = read_run(rd)
    T = t[-1]
    evals = [("end", T)]
    if T > 1.1 * common_time_tauB * TAUB_S:     # every regime, so maps work at the common time too
        evals.append(("common", common_time_tauB * TAUB_S))
    for tm in sorted(set(float(v) for v in map_times_tauB)):
        te = tm * TAUB_S
        if te < 0.97 * T and not any(abs(te - e) < 0.02 * TAUB_S for _, e in evals):
            evals.append((f"t{tm:g}", te))
    rows = []
    for tag, te in evals:
        r = dict(p)
        r["eval"] = tag
        r["T_actual_tauB"] = float(T / TAUB_S)
        r.update(_evaluate(x, y, t, te, rc, box, lag_tauB * TAUB_S))
        rows.append(r)
    return rows


def compute_observables(root: str, regimes=REGIMES, *, rc: float = 2.9, box: float = 50.0,
                        common_time_tauB: float = 1.0, lag_tauB: float = 0.5,
                        progress=None, map_times_tauB=()) -> pd.DataFrame:
    """One row per run at its last frame ('end'); runs longer than common_time_tauB also get a row at
    that time ('common'), and one row ('t<time>') for every time in map_times_tauB the run reaches,
    so maps can be drawn at any of those times (and played over time)."""
    runs = discover_runs(root, regimes)
    cache_file = os.path.join(root, "_observables_cache.pkl")
    try:
        with open(cache_file, "rb") as fh:
            cache = pickle.load(fh)
    except Exception:
        cache = {}
    settings = (round(rc, 4), round(box, 3), round(common_time_tauB, 4), round(lag_tauB, 4),
                tuple(sorted(set(round(float(v), 4) for v in map_times_tauB))))
    rows, fresh, changed = [], {}, False
    for i, (p, rd) in enumerate(runs):
        if progress:
            progress(i / max(len(runs), 1), p["regime"] + "/" + p["name"])
        key = (p["regime"], p["name"], _run_stamp(rd), settings)
        got = cache.get(key)
        if got is None:
            got = _run_rows(p, rd, rc, box, common_time_tauB, lag_tauB, map_times_tauB)
            changed = True
        fresh[key] = got
        rows.extend(dict(r) for r in got)
    if changed or len(fresh) != len(cache):
        try:      # only the runs that still exist, for the current settings
            keep = {k: v for k, v in cache.items() if k[3] != settings}
            keep.update(fresh)
            with open(cache_file + ".tmp", "wb") as fh:
                pickle.dump(keep, fh)
            os.replace(cache_file + ".tmp", cache_file)
        except OSError:
            pass
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


TIMESTEP_TEST_DT = (2000, 8000, 16000, 64000)   # dt tags of the time-step convergence runs (not the standard steps)
CHECK_TAGS = ("phs", "wca")                      # runs with another core (tag in the name): checks, not map runs


def standard_ao_steps(q):
    return 32000 if q >= 0.2 else (128000 if q >= 0.1 else 256000)


def is_check_run(d: pd.DataFrame) -> pd.Series:
    """Time-step and core checks: kept in the table, left out of maps and boundary fits."""
    tags = d.tags.fillna("") if "tags" in d else pd.Series("", index=d.index)
    bad = tags.str.contains("|".join(f"dt{v}(?:_|$)" for v in TIMESTEP_TEST_DT), regex=True)
    bad |= tags.str.contains("|".join(f"_{t}(?:_|$)" for t in CHECK_TAGS), regex=True)
    if {"potential", "q", "dt"} <= set(d.columns):
        ao = (d.potential == "AO") & d.q.notna() & d.dt.notna()
        std = d.q.where(ao, 0.2).map(standard_ao_steps)
        bad |= ao & (d.dt != std)
    return bad


def boundary_data(df: pd.DataFrame, regime="ARF", param="dn", common_time_tauB=1.0, potential="LJ"):
    d = df[(df.regime == regime) & np.isfinite(df[param])].copy()
    d = d[~is_check_run(d)]   # time-step / core check runs: keep them out of the fit
    if "potential" in d and potential:   # one boundary per attraction (LJ or AO)
        d = d[d.potential.fillna("LJ") == potential]
    d = d[np.isclose(d.t_eval_tauB, common_time_tauB, atol=max(0.05, 0.03 * common_time_tauB))]
    # one row per run: a row evaluated at that time beats the run's end row (same frame anyway)
    d = d.assign(_end=(d["eval"] == "end").astype(int)).sort_values("_end")
    return d.drop_duplicates(["regime", "name"]).drop(columns="_end")


AXIS_LABELS = {"dn": "inward displacement d$_n$ (nm)", "F": "streaming force F$_0$ (N/m³)", "t_map": "time (τ$_B$)",
               "T_name": "run length (τ$_B$)", "phi_p": "φ$_p$", "q": "q", "N": "number of particles"}
X_CANDIDATES = ("dn", "F", "t_map", "phi_p", "q", "N")


def fmt_val(v):
    """2,730 for field values, 0.75 for times."""
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}"


def axis_label(p):
    return AXIS_LABELS.get(p, p)


def map_axes(df, regime):
    """Parameters that can go on the map's x axis for a regime: the field (d_n / F0), time, and any other
    run parameter with at least two values (e.g. a particle-number tag)."""
    d = df[df.regime == regime]
    out = []
    for c in X_CANDIDATES:
        if c == "t_map":
            if d.t_eval_tauB.round(2).nunique() >= 2:
                out.append(c)
        elif c in d and pd.to_numeric(d[c], errors="coerce").dropna().nunique() >= 2:
            out.append(c)
    return out


def map_data(df, regime, x, t_tauB=1.0, potential="LJ"):
    """Runs for a map. x = a run parameter: one row per run at time t_tauB. x = 't_map': every run at every
    time it was evaluated (A against time)."""
    if x != "t_map":
        return boundary_data(df, regime, x, t_tauB, potential)
    d = df[df.regime == regime].copy()
    d = d[~is_check_run(d)]
    if "potential" in d and potential:
        d = d[d.potential.fillna("LJ") == potential]
    nt = nominal_time(d)
    d["t_map"] = nt.round(2).map(snap_times(nt.values))
    d = d[d.t_map > 0]
    d = d.assign(_end=(d["eval"] == "end").astype(int)).sort_values("_end")
    return d.drop_duplicates(["name", "t_map"]).drop(columns="_end")


def snap_times(values, rel=0.03):
    """Merge times within 3 % of each other (a run evaluated at 1 τB and a run that ends at 1.01 τB are the
    same time on a map); each group is named by its simplest value (1 rather than 1.01)."""
    v = sorted(set(np.round(np.asarray(values, float), 2)))
    groups, out = [], {}
    for t in v:
        if groups and t <= groups[-1][0] * (1 + rel) + 1e-9:
            groups[-1].append(t)
        else:
            groups.append([t])
    for g in groups:
        rep = min(g, key=lambda x: (len(f"{x:g}"), x))
        for t in g:
            out[t] = rep
    return out


def nominal_time(d):
    """The time a row was meant for: the requested map time ('t0.25' -> 0.25) when it has one, else the
    frame time (runs are evaluated at the nearest saved frame, e.g. 0.26 for 0.25)."""
    tag = d["eval"].astype(str) if "eval" in d else pd.Series("", index=d.index)
    nom = pd.to_numeric(tag.str.extract(r"^t([0-9.]+)$")[0], errors="coerce")
    return nom.fillna(d.t_eval_tauB).astype(float)


def map_times(df, regime=None):
    """Times (τB) at which runs were evaluated (for the map's time slider), near-identical times merged."""
    d = df if regime is None else df[df.regime == regime]
    return sorted(set(snap_times(nominal_time(d).values).values()))


OUTCOMES = {
    "fluid": dict(label="not fluid vs fluid (z)", lo="FLUID", hi="BAND / NETWORK"),
    "network": dict(label="network vs not (largest cluster)", lo="NO NETWORK", hi="NETWORK"),
    "ordered": dict(label="ordered vs not (ψ6)", lo="DISORDERED / FLUID", hi="ORDERED"),
    "custom": dict(label="a measure above a value", lo="BELOW", hi="ABOVE"),
}


def outcome_y(d, outcome="fluid", metric=None, threshold=None):
    """0/1 per run for the boundary model."""
    if outcome == "network":
        return d.state.str.endswith("network").astype(int).values
    if outcome == "ordered":
        return d.state.str.startswith("ordered").astype(int).values
    if outcome == "custom":
        return (d[metric].astype(float) >= float(threshold)).astype(int).values
    return (d.state != "fluid").astype(int).values


def fit_boundary(df, regime="ARF", param="dn", common_time_tauB=1.0, potential="LJ", loo=True,
                 data=None, outcome="fluid", metric=None, threshold=None) -> BoundaryModel:
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    d = map_data(df, regime, param, common_time_tauB, potential) if data is None else data
    y = outcome_y(d, outcome, metric, threshold)
    if len(d) < 6 or y.min() == y.max():
        allr = df[df.regime == regime].drop_duplicates("name")
        lens = sorted(allr.T_actual_tauB.round(2).unique()) if len(allr) else []
        hint = (f" The {len(allr)} {regime_label(regime)} runs in use last {', '.join(f'{v:g}' for v in lens)} τB, so none "
                f"reaches t = {common_time_tauB} τB: lower the map time or include longer (e.g. Python) runs in the filter."
                if len(allr) and max(lens) < 0.95 * common_time_tauB else
                " The fit needs at least 6 runs at that time, some fluid and some not.")
        raise ValueError(f"Need {potential} runs of both outcomes in {regime_label(regime)} at t = {common_time_tauB} τB "
                         f"(found {len(d)} runs, {int(y.sum())} non-fluid)." + hint)
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
        for i in (range(len(y)) if loo else []):
            tr = np.arange(len(y)) != i
            if y[tr].min() == y[tr].max():
                continue
            hits += int(_gpc(0).fit(Xs[tr], y[tr]).predict(Xs[[i]])[0] == y[i])
    m = BoundaryModel(regime, param, d, gpc, mu, sd, hits, kernel=str(gpc.kernel_))
    m.potential = potential
    m.outcome = OUTCOMES.get(outcome, OUTCOMES["fluid"]) if outcome != "custom" else \
        dict(label=f"{metric} ≥ {threshold:g}", lo=f"{metric} < {threshold:g}", hi=f"{metric} ≥ {threshold:g}")
    m.y = y
    grid = np.logspace(np.log10(d[param].min()), np.log10(d[param].max()), 400)
    for a in sorted(d.A.unique()):
        p = m.proba(grid, np.full_like(grid, a))
        # boundary = where the model last crosses from below to above 50 % (an uncertain patch far from the
        # crossing, e.g. between widely spaced low values, must not count as the boundary)
        below = np.where(p < 0.5)[0]
        if len(below) == len(p):
            b = grid[-1]
        elif not len(below):
            b = grid[0]
        else:
            b = grid[min(below[-1] + 1, len(grid) - 1)]
        m.boundary.append((float(a), float(b)))
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
    yy = getattr(model, "y", (d.state != "fluid").astype(int).values)
    for _, r in d[np.asarray(yy) == 1].nsmallest(n_repeat, "dist").iterrows():
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
            first.setdefault((p["regime"], len(x[0])), []).append((key, x[1] - x[0], p.get("source", "")))
    for (reg, _), lst in first.items():
        groups = {}
        for key, dx, src in lst:      # runs whose first step agrees to 1e-6 µm share their random kicks
            groups.setdefault(hashlib.md5(np.round(dx, 6).tobytes()).hexdigest(), []).append((key, src))
        for g in groups.values():
            if len(g) < 2:
                continue
            srcs = sorted({s_ or "?" for _, s_ in g})
            base = [re.sub(r"_seed\d+", "", k.split("/")[-1]) for k, _ in g]
            repeats = len(base) != len(set(base))        # same settings, only the seed differs -> real problem
            ex = ", ".join(k.split("/")[-1] for k, _ in g[:3])
            if repeats:
                issue = ("PROBLEM - repeats of the same settings have identical random kicks, so they are not "
                         "independent repeats: " + ex)
            else:
                issue = ("note - different settings share one set of random kicks (same seed). Fine for comparing "
                         "settings, but these runs are not independent of each other: " + ex)
            rows.append(dict(run=f"{regime_label(reg)}: {len(g)} of {len(lst)} runs ({', '.join(srcs)})", issue=issue))
    return pd.DataFrame(rows, columns=["run", "issue"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "xtick.color": "#555",
                         "ytick.color": "#555", "axes.labelcolor": "#222"})


MAP_STYLE = dict(show_uncertainty=True, show_plan=True, shade=True, boundary_line=True, counts=True,
                 region_labels=True, marker_size=80, color_by="state", cmap="viridis",
                 state_colors=None, lo_color="#2a78d6", hi_color="#eb6834", line_color="#555555")


def fig_state_map(model: BoundaryModel, plan: pd.DataFrame | None = None, figsize=(12, 5), per_run=False, style=None):
    """Left: runs and the fitted boundary; right (optional): model uncertainty and suggested runs.
    per_run=True draws every run as its own dot (seeds side by side), otherwise one dot per parameter point
    (most common state; 'k/n' = how many of the n runs share it). style: see MAP_STYLE (colour the dots by
    state or by any measured column, switch panels, shading, line, labels and suggestions on/off, colours)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    _style(plt)
    st_ = {**MAP_STYLE, **(style or {})}
    scol = {**STATE_COLORS, **(st_["state_colors"] or {})}
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
    unit = {"dn": "nm", "F": "N/m³", "t_map": "τB"}.get(prm, "")
    xlabel = axis_label(prm) + "; tested values evenly spaced"
    oc = getattr(model, "outcome", OUTCOMES["fluid"])
    two = st_["show_uncertainty"]
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=figsize if two else (figsize[0] * 0.55, figsize[1]),
                             constrained_layout=True, squeeze=False)
    ax0 = axes[0, 0]
    if st_["shade"]:
        ax0.contourf(GU, GA, pn, levels=[0, 0.5, 1], colors=[st_["lo_color"], st_["hi_color"]], alpha=0.08)
    if st_["boundary_line"]:
        ax0.contour(GU, GA, pn, levels=[0.5], colors=st_["line_color"], linewidths=1.2, linestyles="--")
        ax0.text(pos(bstar) + 0.08, A_lo - 0.75 * pad, f"fitted boundary ≈ {fmt_val(bstar)} {unit}",
                 ha="left", va="center", fontsize=8, color=st_["line_color"])
    if st_["region_labels"]:
        ax0.text(pos(bstar) / 2 - 0.2, A_lo - 0.75 * pad, oc["lo"], ha="center", va="center", fontsize=9,
                 color=st_["lo_color"], weight="bold")
        ax0.text((pos(bstar) + len(TK) - 1) / 2, A_hi + 0.75 * pad, oc["hi"], ha="center", va="center",
                 fontsize=9, color=st_["hi_color"], weight="bold")
    ms = float(st_["marker_size"])
    by = st_["color_by"]
    if per_run:
        dd = d.sort_values(["A", prm, "name"]).copy()
        dd["k"] = dd.groupby(["A", prm]).cumcount()
        dd["n"] = dd.groupby(["A", prm]).name.transform("count")
        dd["xp"] = pos(dd[prm]) + (dd.k - (dd.n - 1) / 2) * min(0.13, 0.8 / dd.n.max())
    else:
        g = d.groupby(["A", prm])
        dd = g.state.agg(lambda v: v.value_counts().index[0]).reset_index()
        dd = dd.merge(g.size().reset_index(name="n"), on=["A", prm])
        dd = dd.merge(g.state.agg(lambda v: (v == v.value_counts().index[0]).mean()).reset_index(name="f"), on=["A", prm])
        if by != "state":
            dd = dd.merge(g[by].mean().reset_index(), on=["A", prm])
        dd["xp"] = pos(dd[prm])
        ms_ = ms
    if by == "state":
        hollow = (dd["source"] == "COMSOL") if (per_run and "source" in dd) else pd.Series(False, index=dd.index)
        for c in sorted(dd.state.unique()):
            s_ = dd[dd.state == c]
            h_ = hollow[s_.index]
            col = scol.get(c, "#888")
            if (~h_).any():
                ax0.scatter(s_.xp[~h_], s_.A[~h_], c=col, marker=STATE_MARKERS.get(c, "o"),
                            s=ms * (0.6 if per_run else 1), edgecolors="white", linewidths=1, label=c, zorder=3)
            if h_.any():
                ax0.scatter(s_.xp[h_], s_.A[h_], facecolors="white", edgecolors=col, marker=STATE_MARKERS.get(c, "o"),
                            s=ms * 0.6, linewidths=1.6, zorder=3, label=None if (~h_).any() else c)
        if per_run and hollow.any() and (~hollow).any():
            ax0.scatter([], [], facecolors="white", edgecolors="#555", s=ms * 0.6, linewidths=1.6, label="hollow = COMSOL")
    else:
        sc = ax0.scatter(dd.xp, dd.A, c=dd[by].astype(float), cmap=st_["cmap"], s=ms * (0.6 if per_run else 1),
                         edgecolors="white", linewidths=1, zorder=3)
        fig.colorbar(sc, ax=ax0, label=METRIC_LABELS.get(by, by), shrink=0.8)
    if st_["counts"] and not per_run:
        for _, r in dd[dd.n > 1].iterrows():
            ax0.annotate(f"{int(round(r.f * r.n))}/{r.n}", (r.xp, r.A), xytext=(5, 5),
                         textcoords="offset points", fontsize=6.5, color="#555")
    ax0.set_title(("State over time" if prm == "t_map" else f"State after {d.t_eval_tauB.round(1).iloc[0]:g} τ$_B$")
                  + f" ({regime_label(model.regime)}, {getattr(model, 'potential', 'LJ')}; boundary: {oc['label']})",
                  loc="left", fontsize=10, pad=30)
    if two:
        ax1 = axes[0, 1]
        cs = ax1.contourf(GU, GA, H, levels=np.linspace(0, 1, 11), cmap=ListedColormap(_BLUE_RAMP))
        ax1.scatter(pos(d[prm]), d.A, c="#555", s=16, zorder=3, label="existing run")
        if st_["show_plan"] and plan is not None and len(plan):
            col = [c for c in plan.columns if c.startswith(prm)][0]
            new, rep = plan[plan.kind == "new"], plan[plan.kind == "repeat"]
            ax1.scatter(pos(new[col]), new.A_kT, marker="*", s=220, c="#eb6834", edgecolors="white",
                        linewidths=1, zorder=4, label="suggested new run")
            ax1.scatter(pos(rep[col]), rep.A_kT, marker="o", s=170, facecolors="none",
                        edgecolors="#eb6834", linewidths=2, zorder=4, label="repeat with new seed")
        fig.colorbar(cs, ax=ax1, label="model uncertainty (0 = sure, 1 = coin flip)")
        ax1.set_title("Where the model is unsure" + (", and what to run next" if st_["show_plan"] else ""),
                      loc="left", fontsize=11, pad=30)
    elif st_["show_plan"] and plan is not None and len(plan):
        col = [c for c in plan.columns if c.startswith(prm)][0]
        new = plan[plan.kind == "new"]
        ax0.scatter(pos(new[col]), new.A_kT, marker="*", s=200, c="#eb6834", edgecolors="white", zorder=4,
                    label="suggested new run")
    for a in axes.ravel():
        a.set_xticks(np.arange(len(TK)))
        a.set_xticklabels([f"{v:g}" for v in TK], rotation=45 if len(TK) > 12 else 0)
        a.set_xlim(-0.4, len(TK) - 0.6)
        a.set_ylim(A_lo - pad, A_hi + pad)
        a.set_yticks(np.unique(np.round(d.A.values, 3)))
        a.set_xlabel(xlabel)
        a.set_ylabel("attraction A (k$_B$T)")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if a.get_legend_handles_labels()[0]:
            a.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
                     borderaxespad=0.2, handletextpad=0.3, columnspacing=1.0)
    return fig


METRIC_LABELS = {"nonfluid": "fraction of runs not fluid", "z_mean": "mean coordination number",
                 "psi6": "local order ψ6", "Smax": "largest cluster fraction", "bond_persist": "bond persistence",
                 "Rg_ratio": "Rg / Rg(t=0)", "mobility": "mobility (MSD / free MSD)", "proba": "model P(not fluid)"}


def cell_table(d: pd.DataFrame, param: str, value: str = "nonfluid") -> pd.DataFrame:
    """Mean of a measure per (A, parameter) point; value 'nonfluid' = share of runs that are not fluid."""
    v = (d.state != "fluid").astype(float) if value == "nonfluid" else d[value]
    return d.assign(_v=v).groupby(["A", param]).agg(v=("_v", "mean"), n=("_v", "size")).reset_index()


def fig_cell_heatmap(d: pd.DataFrame, param: str, value: str = "nonfluid", title="", cmap="viridis",
                     boundaries=None, figsize=(7.5, 5)):
    """Heat map on the tested values (evenly spaced), one cell per (A, parameter) point, with the
    number of runs in each cell; boundaries = {label: [(A, value*), ...]} drawn on top."""
    import matplotlib.pyplot as plt
    _style(plt)
    t = cell_table(d, param, value)
    TK, AS = np.sort(t[param].unique()), np.sort(t.A.unique())
    M = np.full((len(AS), len(TK)), np.nan)
    N = np.zeros_like(M)
    for _, r in t.iterrows():
        i, j = np.searchsorted(AS, r.A), np.searchsorted(TK, r[param])
        M[i, j], N[i, j] = r.v, r.n
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    vmin, vmax = (0, 1) if value in ("nonfluid", "Smax", "psi6", "bond_persist") else (np.nanmin(M), np.nanmax(M))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=(-0.5, len(TK) - 0.5, -0.5, len(AS) - 0.5))
    for i in range(len(AS)):
        for j in range(len(TK)):
            if N[i, j]:
                ax.text(j, i, f"{M[i, j]:.2f}\n(n={int(N[i, j])})" if N[i, j] > 1 else f"{M[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if (M[i, j] - vmin) / max(vmax - vmin, 1e-12) < 0.55 else "black")
    if boundaries:
        LTK = np.log10(TK)
        for (lab, pts), col in zip(boundaries.items(), ("#e34948", "#111111", "#1baf7a", "#eda100")):
            pts = [(a, b) for a, b in pts if a in AS]
            if pts:
                ax.plot([np.interp(np.log10(b), LTK, np.arange(len(TK))) for a, b in pts],
                        [np.searchsorted(AS, a) for a, b in pts], "--", c=col, lw=2, label=f"boundary {lab}")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3)
    ax.set_xticks(range(len(TK))); ax.set_xticklabels([f"{v:g}" for v in TK])
    ax.set_yticks(range(len(AS))); ax.set_yticklabels([f"{v:g}" for v in AS])
    ax.set_xlabel(axis_label(param) + "; tested values")
    ax.set_ylabel("attraction A (k$_B$T)")
    fig.colorbar(im, ax=ax, label=METRIC_LABELS.get(value, value))
    if title:
        ax.set_title(title, loc="left", fontsize=11, pad=24 if boundaries else 6)
    return fig


def surface_figure(d: pd.DataFrame, param: str, value: str = "proba", model: BoundaryModel | None = None,
                   title=""):
    """Plotly 3D elevation over (log parameter, A): the model probability of 'not fluid' or a measured
    value (interpolated between the runs), with every run as a point on top."""
    import plotly.graph_objects as go
    from scipy.interpolate import griddata
    lx = np.log10(d[param].values.astype(float))
    gx = np.linspace(lx.min(), lx.max(), 60)
    ga = np.linspace(d.A.min(), d.A.max(), 40)
    GX, GA = np.meshgrid(gx, ga)
    if value == "proba" and model is not None:
        Z = model.gpc.predict_proba((np.c_[GX.ravel(), GA.ravel()] - model.mu) / model.sd)[:, 1].reshape(GX.shape)
        pts_z = (d.state != "fluid").astype(float).values
        zlab = "P(not fluid)"
    else:
        t = cell_table(d, param, value if value != "proba" else "nonfluid")
        Z = griddata(np.c_[np.log10(t[param].values.astype(float)), t.A.values], t.v.values, (GX, GA), method="linear")
        col = (d.state != "fluid").astype(float) if value in ("proba", "nonfluid") else d[value]
        pts_z = np.asarray(col, float)
        zlab = METRIC_LABELS.get(value, value)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=10 ** GX, y=GA, z=Z, colorscale="Viridis", opacity=0.85, showscale=True,
                             colorbar=dict(title=zlab), name="surface"))
    hover = [f"{n}<br>{s}" for n, s in zip(d.name, d.state)]
    fig.add_trace(go.Scatter3d(x=d[param], y=d.A, z=pts_z, mode="markers", name="runs", hovertext=hover,
                               hoverinfo="text",
                               marker=dict(size=3.5, color=[STATE_COLORS.get(s, "#888") for s in d.state],
                                           line=dict(color="white", width=0.5))))
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0), title=title,
                      scene=dict(xaxis=dict(title=axis_label(param).replace("$", "").replace("_", ""), type="log"),
                                 yaxis=dict(title="A (kBT)"), zaxis=dict(title=zlab)))
    return fig


def fig_boundary_compare(models: dict, figsize=(8, 5)):
    """The fitted 50 % boundaries of several models (e.g. LJ and AO) on one plot, with their runs."""
    import matplotlib.pyplot as plt
    _style(plt)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cols = {"LJ": "#2a78d6", "AO": "#eb6834"}
    for lab, m in models.items():
        c = cols.get(lab, "#555")
        b = np.array(m.boundary)
        ax.plot(b[:, 1], b[:, 0], "-", c=c, lw=2.2, label=f"{lab}: 50 % boundary")
        d = m.data
        nf = pd.Series(np.asarray(getattr(m, "y", (d.state != "fluid").astype(int).values)) == 1, index=d.index)
        jit = 1 + (0.02 if lab == "AO" else -0.02)
        ax.scatter(d[param_of(m)][~nf] * jit, d.A[~nf], facecolors="white", edgecolors=c, s=26, lw=1.2, zorder=3)
        ax.scatter(d[param_of(m)][nf] * jit, d.A[nf], c=c, s=26, zorder=3)
    ax.scatter([], [], facecolors="white", edgecolors="#555", s=26, label="fluid run")
    ax.scatter([], [], c="#555", s=26, label="not fluid")
    ax.set_xscale("log")
    p = param_of(next(iter(models.values())))
    ax.set_xlabel(axis_label(p))
    ax.set_ylabel("attraction A (k$_B$T; AO matched by B$_2$*)")
    m0 = next(iter(models.values()))
    ax.set_title("Fluid / band boundary" + ("" if m0.param == "t_map" else
                 f" at t = {m0.data.t_eval_tauB.round(1).iloc[0]:g} τ$_B$"), loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return fig


def param_of(m):
    return m.param


STATE_ORDER = ["fluid", "disordered clusters", "ordered clusters", "disordered network", "ordered network"]


@dataclass
class PhaseMap:
    regime: str
    param: str
    data: pd.DataFrame
    gpc: object
    classes: list
    mu: np.ndarray
    sd: np.ndarray
    potential: str = "LJ"

    def proba(self, xvals, A_vals):
        X = np.c_[np.log10(xvals), A_vals]
        return self.gpc.predict_proba((X - self.mu) / self.sd)


class _OneState:
    def predict_proba(self, X):
        return np.ones((len(X), 1))


def fit_phase_map(d: pd.DataFrame, regime: str, param: str, potential="LJ") -> PhaseMap:
    """One model for all states at once (Gaussian-process classifier, one-vs-rest): for any point it gives the
    probability of each state, so the map can show every region and the borders between them."""
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    classes = [c for c in STATE_ORDER if c in set(d.state)] + sorted(set(d.state) - set(STATE_ORDER))
    if not classes:
        raise ValueError("no runs for this selection")
    X = np.c_[np.log10(d[param].values.astype(float)), d.A.values]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    if len(classes) == 1:     # every run in the same state: the whole map is that state
        return PhaseMap(regime, param, d, _OneState(), classes, mu, sd, potential)
    y = np.array([classes.index(s_) for s_ in d.state])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpc = GaussianProcessClassifier(C(1.0, (0.1, 10)) * RBF([1.0, 1.0], (0.3, 100)), n_restarts_optimizer=2,
                                        random_state=0, multi_class="one_vs_rest").fit((X - mu) / sd, y)
    return PhaseMap(regime, param, d, gpc, classes, mu, sd, potential)


def fig_phase_map(pm: PhaseMap, show_runs=True, per_run=True, region_labels=True, confidence=False,
                  borders=True, colors=None, title=None, figsize=(8.5, 5.6), even_spacing=True,
                  x_values=None, A_values=None):
    """Smooth phase diagram: every state as a coloured region (most likely state), smooth borders where two
    states are equally likely, optional run dots (COMSOL hollow) and fading where the model is unsure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgb
    _style(plt)
    col = {**STATE_COLORS, **(colors or {})}
    d, prm = pm.data, pm.param
    TK = np.sort(np.unique(np.r_[d[prm].unique(), list(x_values or [])]).astype(float))
    LTK = np.log10(TK)
    if even_spacing:
        pos = lambda v: np.interp(np.log10(v), LTK, np.arange(len(TK)))
        U = np.linspace(-0.45, len(TK) - 0.55, 500)
        GLx = np.interp(U, np.arange(len(TK)), LTK)
    else:
        pos = lambda v: np.log10(v)
        U = np.linspace(LTK[0] - 0.05, LTK[-1] + 0.05, 500)
        GLx = U
    A_all = np.r_[d.A.values, list(A_values or [])].astype(float)
    A_lo, A_hi = A_all.min(), A_all.max()
    pad = 0.12 * (A_hi - A_lo or 1)
    UA = np.linspace(A_lo - pad, A_hi + pad, 260)
    GU, GA = np.meshgrid(U, UA)
    GL, _ = np.meshgrid(GLx, UA)
    P = pm.gpc.predict_proba((np.c_[GL.ravel(), GA.ravel()] - pm.mu) / pm.sd).reshape(len(UA), len(U), -1)
    lab = P.argmax(-1)
    conf = P.max(-1)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    rgb = np.array([to_rgb(col.get(c, "#999")) for c in pm.classes])
    img = rgb[lab]
    alpha = (0.25 + 0.55 * np.clip((conf - 1 / len(pm.classes)) / (1 - 1 / len(pm.classes)), 0, 1)) if confidence \
        else np.full(conf.shape, 0.55)
    img = np.concatenate([img, alpha[..., None]], -1)
    ax.imshow(img, origin="lower", aspect="auto", extent=(U[0], U[-1], UA[0], UA[-1]), interpolation="bilinear", zorder=0)
    if borders and len(pm.classes) > 1:
        for i in range(len(pm.classes)):
            other = np.delete(P, i, axis=-1).max(-1)
            ax.contour(GU, GA, P[..., i] - other, levels=[0], colors="#333", linewidths=1.3, zorder=2)
    if region_labels:
        for i, c in enumerate(pm.classes):
            m = lab == i
            if m.sum() < 0.01 * m.size:
                continue
            # label at the point deepest inside the region
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]     # the plot edge counts as a border
            yi, xi = np.unravel_index(dist.argmax(), dist.shape)
            ax.text(U[xi], UA[yi], c.upper().replace(" ", "\n"), ha="center", va="center", fontsize=9, weight="bold",
                    color="#222", alpha=0.75, zorder=4)
    if show_runs:
        dd = d.sort_values(["A", prm, "name"]).copy()
        if per_run:
            dd["k"] = dd.groupby(["A", prm]).cumcount()
            dd["n"] = dd.groupby(["A", prm]).name.transform("count")
            step = 0.12 if even_spacing else 0.012
            dd["xp"] = pos(dd[prm]) + (dd.k - (dd.n - 1) / 2) * min(step, (0.8 if even_spacing else 0.08) / dd.n.max())
        else:
            dd = dd.drop_duplicates(["A", prm]).assign(xp=lambda q: pos(q[prm]))
        hollow = (dd["source"] == "COMSOL") if "source" in dd else pd.Series(False, index=dd.index)
        for c in pm.classes:
            s_ = dd[dd.state == c]
            h_ = hollow[s_.index]
            ax.scatter(s_.xp[~h_], s_.A[~h_], c=col.get(c, "#888"), marker=STATE_MARKERS.get(c, "o"), s=30,
                       edgecolors="white", linewidths=0.7, zorder=5)
            ax.scatter(s_.xp[h_], s_.A[h_], facecolors="white", edgecolors=col.get(c, "#888"),
                       marker=STATE_MARKERS.get(c, "o"), s=30, linewidths=1.3, zorder=5)
    from matplotlib.patches import Patch
    hand = [Patch(facecolor=col.get(c, "#999"), alpha=0.6, label=c) for c in pm.classes]
    if show_runs and "source" in d and (d.source == "COMSOL").any() and (d.source != "COMSOL").any():
        hand.append(plt.Line2D([], [], ls="", marker="o", mfc="white", mec="#555", label="hollow = COMSOL run"))
    ax.legend(handles=hand, frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    if even_spacing:
        ax.set_xticks(np.arange(len(TK)))
        ax.set_xticklabels([f"{v:g}" for v in TK], rotation=45 if len(TK) > 12 else 0)
        ax.set_xlabel(axis_label(prm) + "; tested values evenly spaced")
    else:
        ax.set_xticks(LTK[:: max(1, len(LTK) // 10)])
        ax.set_xticklabels([f"{v:g}" for v in TK[:: max(1, len(TK) // 10)]])
        ax.set_xlabel(axis_label(prm) + " (log scale)")
    ax.set_yticks(np.unique(np.round(A_all, 3)))
    ax.set_ylabel("attraction A (k$_B$T" + ("; AO matched by B$_2$*)" if pm.potential == "AO" else ")"))
    ax.set_xlim(U[0], U[-1]); ax.set_ylim(UA[0], UA[-1])
    ax.set_title(title or (("States over time" if prm == "t_map" else
                            f"States after {d.t_eval_tauB.round(1).iloc[0]:g} τ$_B$") +
                           f" ({regime_label(pm.regime)}, {pm.potential})"), loc="left", fontsize=11, pad=36)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return fig


def phase_map_frames(df, regime, param, potential="LJ", times=None, progress=None, **figkw):
    """The phase diagram at every evaluation time, on fixed axes: [(time, PNG bytes, n runs)]. Times with no runs
    are skipped."""
    import io
    import matplotlib.pyplot as plt
    times = times or map_times(df, regime)
    per_t = {tm: boundary_data(df, regime, param, tm, potential) for tm in times}
    per_t = {tm: d for tm, d in per_t.items() if len(d)}
    if not per_t:
        raise ValueError("no runs at any evaluation time for this selection")
    xs = sorted(set().union(*[set(d[param]) for d in per_t.values()]))
    As = sorted(set().union(*[set(d.A) for d in per_t.values()]))
    out = []
    for i, (tm, d) in enumerate(per_t.items()):
        if progress:
            progress(i / len(per_t), tm)
        pm = fit_phase_map(d, regime, param, potential)
        fig = fig_phase_map(pm, x_values=xs, A_values=As,
                            title=f"{regime_label(regime)}, {potential}: states after {tm:g} τ$_B$ ({d.name.nunique()} runs)",
                            **figkw)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        out.append((tm, buf.getvalue(), d.name.nunique()))
    return out


def frames_to_gif(frames, seconds_per_frame=1.0, loop=True):
    """GIF bytes from phase_map_frames output (the last frame is held twice as long)."""
    import io
    from PIL import Image
    ims = [Image.open(io.BytesIO(p)).convert("RGB") for _, p, _ in frames]
    W, H = max(i.width for i in ims), max(i.height for i in ims)
    ims = [i if i.size == (W, H) else i.resize((W, H)) for i in ims]
    dur = [int(1000 * seconds_per_frame)] * len(ims)
    dur[-1] *= 2
    buf = io.BytesIO()
    ims[0].save(buf, format="GIF", save_all=True, append_images=ims[1:], duration=dur, loop=0 if loop else 1)
    return buf.getvalue()


def map_animation(df, regime, param, potential, value="nonfluid", times=None, title=""):
    """Plotly animation of the map over the evaluation times: a heat map of a measure per (A, parameter)
    point, or (value='states') every run as a dot coloured by its state. Slider + play button."""
    import plotly.graph_objects as go
    times = times or map_times(df, regime)
    frames, TKs, ASs = [], set(), set()
    per_t = {}
    for tm in times:
        d = boundary_data(df, regime, param, tm, potential)
        if len(d):
            per_t[tm] = d
            TKs |= set(d[param]); ASs |= set(d.A)
    if not per_t:
        raise ValueError("no runs at any evaluation time for this selection")
    TK, AS = sorted(TKs), sorted(ASs)
    xs = [f"{v:g}" for v in TK]

    def traces(d):
        if value == "states":
            dd = d.sort_values(["A", param, "name"]).copy()
            dd["k"] = dd.groupby(["A", param]).cumcount()
            dd["n"] = dd.groupby(["A", param]).name.transform("count")
            xi = np.searchsorted(TK, dd[param].values) + (dd.k - (dd.n - 1) / 2).values * 0.12
            return [go.Scatter(x=xi, y=dd.A, mode="markers", hovertext=dd.name + "<br>" + dd.state, hoverinfo="text",
                               marker=dict(size=10, color=[STATE_COLORS.get(s_, "#888") for s_ in dd.state],
                                           line=dict(color="white", width=1)))]
        t = cell_table(d, param, value)
        M = np.full((len(AS), len(TK)), np.nan)
        for _, r in t.iterrows():
            M[AS.index(r.A), TK.index(r[param])] = r.v
        zr = (0, 1) if value in ("nonfluid", "Smax", "psi6", "bond_persist") else (None, None)
        return [go.Heatmap(z=M, x=list(range(len(TK))), y=list(range(len(AS))), colorscale="Viridis", zmin=zr[0],
                           zmax=zr[1], colorbar=dict(title=METRIC_LABELS.get(value, value)),
                           texttemplate="%{z:.2f}", hoverongaps=False)]
    keys = sorted(per_t)
    fig = go.Figure(data=traces(per_t[keys[0]]),
                    frames=[go.Frame(data=traces(per_t[tm]), name=f"{tm:g}") for tm in keys])
    steps = [dict(method="animate", label=f"{tm:g}", args=[[f"{tm:g}"], dict(mode="immediate",
             frame=dict(duration=600, redraw=True), transition=dict(duration=0))]) for tm in keys]
    fig.update_layout(
        height=560, title=title or f"{regime_label(regime)}, {potential}: map over time",
        xaxis=dict(title=axis_label(param).replace("$", "").replace("_", "") + ", tested values", tickmode="array",
                   tickvals=list(range(len(TK))), ticktext=xs, range=[-0.6, len(TK) - 0.4]),
        yaxis=dict(title="A (kBT)", tickmode="array", tickvals=(list(range(len(AS))) if value != "states" else AS),
                   ticktext=[f"{a:g}" for a in AS]),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.15, xanchor="left",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(active=0, steps=steps, x=0.15, len=0.85, y=-0.12, currentvalue=dict(prefix="t = ", suffix=" τB"))],
        plot_bgcolor="white")
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
                    label=f"{regime_label(reg)}, after {T:g} τ$_B$")
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

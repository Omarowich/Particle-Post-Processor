"""run_filter.py - one run filter for every page of the analyser.

Reads the parameters from the run folder names (A = TkB, run length = TauB, d_n, F0, Python/COMSOL,
LJ/AO, q, phi_p, time step, seed, extra tags such as N256) and shows a multiselect for every
parameter that has more than one value among the runs, all selected by default.

    import run_filter as rf
    metas = rf.metas_from_paths(paths)              # or rf.meta(name, regime)
    keep = rf.widgets(metas, key="multi", container=st.sidebar)
    paths = [p for p, k in zip(paths, keep) if k]
"""
from __future__ import annotations

import math
import zlib
import os

import regime_state_model as rsm

# (column, label, number format) in the order they are shown
PARAMS = [
    ("regime", "Regime", None),
    ("source", "Source", None),
    ("potential", "Attraction", None),
    ("A", "A / TkB (k_BT)", "{:g}"),
    ("T_name", "Run length / TauB (τ_B)", "{:g}"),
    ("dn", "d_n (nm)", "{:g}"),
    ("F", "F₀ (N/m³)", "{:g}"),
    ("q", "q (AO)", "{:g}"),
    ("phi_p", "φ_p (AO)", "{:.3f}"),
    ("dt", "Time step (steps per τ_B)", "{:d}"),
    ("seed", "Seed", "{:d}"),
    ("extra", "Other tag", None),
]
NONE = "–"
LABELS = {"BD": "Python", "COMSOL": "COMSOL", "LJ": "Lennard-Jones", "AO": "AO depletion",
          "NAF": "Regime I (NAF)", "ARF": "Regime II (ARF)", "AAF": "Regime III (AAF)", "ASF": "Regime IV (ASF)"}


def meta(name, regime=""):
    """Parameters of one run folder name (None if the name is not a run name)."""
    p = rsm.parse_run_name(regime.upper() if regime else "", name)
    if p is None:
        return None
    if p["source"] == "BD" and not p.get("dt"):
        p["dt"] = 4000              # Python LJ runs without a dt tag use the default step
    return p


def metas_from_paths(paths):
    """One parameter dict per run folder path (regime = the parent folder name)."""
    out = []
    for path in paths:
        path = str(path).rstrip("\\/")
        out.append(meta(os.path.basename(path), os.path.basename(os.path.dirname(path))))
    return out


def _val(m, col):
    v = None if m is None else m.get(col)
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)) or (col in ("dt", "seed") and v == 0):
        return NONE
    return v


def _fmt(col, fmt):
    def f(v):
        if v == NONE:
            return "none / not set"
        if fmt and not isinstance(v, str):
            try:
                return fmt.format(int(v) if "d}" in fmt else v)
            except (ValueError, TypeError):
                return str(v)
        return LABELS.get(v, str(v))
    return f


def _sort_key(v):
    return (1, 0.0, "") if v == NONE else ((0, float(v), "") if not isinstance(v, str) else (0, 0.0, v))


PARAMS_3D = [
    ("H", "Chamber height H (µm)", "{:g}"),
    ("drho", "Δρ (kg/m³; 0 = density-matched)", "{:g}"),
    ("field", "Acoustic field / lid", None),
    ("N", "Particles N", "{:d}"),
    ("box", "Box side (µm)", "{:g}"),
]


def meta3d(name, regime="", cfg=None):
    """meta() plus the 3D settings: chamber height, Δρ, field / lid, N, box side (from the name, the run's
    sim_config.json when given)."""
    import re
    m = meta(name, regime.split("_")[0] if regime else "")
    if m is None:
        return None
    ex = m.get("extra") or ""
    cfg = cfg or {}
    g = re.search(r"_H([0-9.]+)|^H([0-9.]+)|3D_H([0-9.]+)", "_" + ex)
    m["H"] = float(cfg["H"]) if cfg.get("H") not in (None, "", 0) else (float(next(x for x in g.groups() if x)) if g else NONE)
    g = re.search(r"drho([0-9.]+)", ex)
    m["drho"] = float(cfg.get("drho", g.group(1) if g else 0) or 0)
    f = str(cfg.get("arf3d_file") or "")
    if "SSAW" in f.upper() or "ssaw" in ex.lower():
        fr = re.search(r"SSAW(\d+)", f.upper() + ex.upper())
        lid = "hard" if "HARD" in (f + ex).upper() else "PDMS" if "PDMS" in (f + ex).upper() else "absorbing" if "ABSORB" in (f + ex).upper() else ""
        m["field"] = f"IDT (SSAW{' ' + fr.group(1) + ' MHz' if fr else ''}){', ' + lid + ' lid' if lid else ''}"
    elif f or "lid" in ex.lower():
        m["field"] = ("side-wall drive, PDMS lid (computed 3D field)" if ("impedance" in f or "PDMS" in ex)
                      else "side-wall drive, glass lid (computed 3D field)")
    elif m.get("dn") and m["dn"] == m["dn"] and m["dn"] > 0:
        m["field"] = "fitted 1D standing wave (same at every height)"
    else:
        m["field"] = "no field"
    n_ = re.search(r"(?:^|_)N(\d+)", ex)
    st_ = cfg.get("start") or {}
    n_cfg = (int(st_.get("n_side", 0)) ** 2 * int(st_.get("layers", 1))) if st_.get("kind") in ("grid", "grid3d") else None
    m["N"] = int(n_.group(1)) if n_ else (n_cfg or 144)
    b_ = re.search(r"box(\d+)", ex)
    m["box"] = 2 * float(cfg["box_half"]) if cfg.get("box_half") else (float(b_.group(1)) if b_ else 50.0)
    return m


def widgets(metas, key, container, title="Filter runs", show_name_box=True, expanded=True, skip=(), defaults=None,
            extra_params=()):
    """Draw the filter and return a list of booleans (one per run: passes the filter).
    defaults: {column: [values selected at first]}, e.g. {"source": ["COMSOL"]}; otherwise all are selected."""
    n = len(metas)
    keep = [m is not None for m in metas]
    if not n:
        return keep
    box = container.expander(f"{title} ({n} runs)", expanded=expanded)
    chosen = {}
    for col, label, fmt in list(PARAMS) + list(extra_params):
        if col in skip:
            continue
        vals = sorted({_val(m, col) for m in metas if m is not None}, key=_sort_key)
        if len(vals) < 2:
            continue
        # key includes the value set, so a new folder (other values) starts with everything selected
        k = f"rf_{key}_{col}_{zlib.crc32('|'.join(map(str, vals)).encode()):x}"
        dflt = vals
        if defaults and col in defaults:
            dflt = [v for v in vals if str(v) in set(map(str, defaults[col]))] or vals
        sel = box.multiselect(label, vals, default=dflt, format_func=_fmt(col, fmt), key=k)
        chosen[col] = set(map(str, sel))
    words = []
    if show_name_box:
        words = box.text_input("Name contains (all words, e.g. AO seed1)", "", key=f"rf_{key}_name").lower().split()
    for i, m in enumerate(metas):
        if m is None:
            continue
        if any(str(_val(m, col)) not in s for col, s in chosen.items()):
            keep[i] = False
        elif words and not all(w in m["name"].lower() for w in words):
            keep[i] = False
    box.caption(f"{sum(keep)} of {n} runs pass the filter")
    return keep


def filter_frame(df, key, container, **kw):
    """Same filter for a table with one row (or several rows) per run: needs a 'name' column and the
    parameter columns of rsm.parse_run_name. Returns the filtered table."""
    if df is None or df.empty:
        return df
    runs = df.drop_duplicates(["regime", "name"]) if "regime" in df else df.drop_duplicates("name")
    metas = [meta(r["name"], r.get("regime", "")) for _, r in runs.iterrows()]
    keep = widgets(metas, key, container, **kw)
    ok = {(r.get("regime", ""), r["name"]) for (_, r), k in zip(runs.iterrows(), keep) if k}
    return df[[(r.get("regime", ""), r["name"]) in ok for _, r in df.iterrows()]]


def topic3d(m):
    """Group of a 3D run for the run picker: field on/off, field type, system size."""
    if m is None:
        return "other runs"
    size = f"N = {m['N']}, {m['box']:g} µm box" if m.get("N") not in (None, NONE) else f"{m['box']:g} µm box"
    if m["field"] == "no field":
        return f"No field (gel / crystal) · {size}"
    return f"Band in the field · {m['field']} · {size}"


SHORT_KEYS = [("potential", lambda v: str(v)), ("A", lambda v: f"A {v:g}"), ("q", lambda v: f"q {v:g}"),
              ("phi_p", lambda v: f"φp {v:g}"), ("dn", lambda v: f"{v:g} nm"), ("T_name", lambda v: f"{v:g} τB"),
              ("H", lambda v: f"H {v:g}"), ("drho", lambda v: f"Δρ {v:g}"), ("field", lambda v: str(v)),
              ("N", lambda v: f"N {v}"), ("box", lambda v: f"box {v:g}")]


def short_labels(metas, names):
    """Short run labels: only the parameters that differ between the given runs, plus the seed."""
    def val(m, k):
        return _val(m, k) if m is not None else NONE
    vary = [(k, f) for k, f in SHORT_KEYS if len({str(val(m, k)) for m in metas}) > 1]
    out = []
    for m, nm in zip(metas, names):
        if m is None:
            out.append(nm); continue
        parts = []
        for k, f in vary:
            v = val(m, k)
            if v != NONE:
                try:
                    parts.append(f(v))
                except (ValueError, TypeError):
                    parts.append(str(v))
        seed = m.get("seed")
        parts.append(f"seed {seed}" if seed else "")
        out.append(" · ".join(p_ for p_ in parts if p_) or nm)
    return out

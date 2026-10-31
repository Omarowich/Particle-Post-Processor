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


def widgets(metas, key, container, title="Filter runs", show_name_box=True, expanded=True, skip=(), defaults=None):
    """Draw the filter and return a list of booleans (one per run: passes the filter).
    defaults: {column: [values selected at first]}, e.g. {"source": ["COMSOL"]}; otherwise all are selected."""
    n = len(metas)
    keep = [m is not None for m in metas]
    if not n:
        return keep
    box = container.expander(f"{title} ({n} runs)", expanded=expanded)
    chosen = {}
    for col, label, fmt in PARAMS:
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

"""figure_builder.py - figures from editable recipes, re-made in one go when new runs arrive.

A recipe (JSON, one file per figure, kept in <Data>/_figures/) says which panels a figure has,
which runs each line is drawn from, and how it looks. Two kinds of figure:

  "time"      every panel shows one measure over time; each line = mean over the seeds of one
              parameter set, with the min-max over seeds as a band
  "vs_param"  every panel shows the end value of one measure (or the break-up time) against a
              run parameter (A, d_n, F0, phi_p); each line = one family of runs, a point per value

Recipe layout (all keys optional except panels and lines):
{
  "name": "AO vs LJ", "kind": "time", "title": "...",
  "columns": 4, "panel_size": [4.2, 3.2], "legend_panel": 0, "dpi": 170,
  "x": {"param": "A", "log": false, "lim": [0, 5], "label": "..."},     # vs_param only
  "t_from": 18.0,                                                      # vs_param: average over t > t_from
  "breakup_threshold": 0.5,                                            # metric "breakup"
  "panels": [{"metric": "z", "title": "...", "filter": {"dn": 4000}, "ylim": [0, 6], "log": false,
              "hlines": [2.5], "ylabel": "..."}],
  "lines":  [{"label": "LJ", "color": "#2a78d6", "ls": "-", "lw": 2, "marker": "", "band": true,
              "filter": {"regime": "ARF", "potential": "LJ", "A": 3, "dn": 4000, "T": 10, "q": null,
                         "source": "BD", "tag": "dt4000"}}]
}
A panel filter overrides the line filters for that panel (e.g. the same lines at three d_n).

Command line (re-makes every recipe in the folder):
    python figure_builder.py --data "D:/Work/AIRPODS Paper 2627/Data" --out figures
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re

import numpy as np
import pandas as pd

import paper_figures as pf

RECIPE_DIR = "_figures"
METRICS = dict(pf.LAB)
METRICS["breakup"] = "break-up time (τ$_B$): largest cluster below the threshold"
FILTER_KEYS = ("regime", "source", "potential", "A", "dn", "F", "T", "q", "phi_p", "tag")
PARAMS = {"A": "attraction A (k$_B$T; AO matched by B$_2$*)", "dn": "inward displacement d$_n$ (nm)",
          "F": "streaming body force F$_0$ (N/m³)", "phi_p": "φ$_p$", "T": "run length (τ$_B$)"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8a8980", "#d6338a", "#111111"]
STYLES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}
COLUMN = {"T": "T_name"}


# --------------------------------------------------------------------------- recipes on disk
def recipe_dir(data_root):
    return os.path.join(data_root, RECIPE_DIR)


def slug(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "figure"


def list_recipes(data_root):
    d = recipe_dir(data_root)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def load_recipe(data_root, key):
    with open(os.path.join(recipe_dir(data_root), key + ".json"), encoding="utf-8") as f:
        return json.load(f)


def save_recipe(data_root, recipe):
    os.makedirs(recipe_dir(data_root), exist_ok=True)
    key = slug(recipe.get("name", "figure"))
    with open(os.path.join(recipe_dir(data_root), key + ".json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2, ensure_ascii=False)
    return key


def delete_recipe(data_root, key):
    os.remove(os.path.join(recipe_dir(data_root), key + ".json"))


def new_recipe(kind="time"):
    r = {"name": "new figure", "kind": kind, "title": "", "columns": 3, "panel_size": [4.2, 3.2],
         "legend_panel": 0, "dpi": 170, "panels": [{"metric": "z"}],
         "lines": [{"label": "line 1", "color": PALETTE[0], "ls": "-", "lw": 2, "marker": "", "band": True,
                    "filter": {"regime": "ARF", "source": "BD"}}]}
    if kind == "vs_param":
        r.update({"x": {"param": "A", "log": False}, "t_from": None, "breakup_threshold": 0.5})
        r["lines"][0]["marker"] = "o"
    return r


# --------------------------------------------------------------------------- data
def load_data(data_root, include_comsol=True, progress=None):
    """Observables of every run (Python and, if wanted, COMSOL), cached like the Paper figures page."""
    df, _ = pf.load_series(data_root, progress=progress, sources=("BD", "CS") if include_comsol else ("BD",))
    return df


def match(df, flt):
    """Rows of df whose run parameters match a filter (missing / None / '' = any)."""
    m = pd.Series(True, index=df.index)
    for k, v in (flt or {}).items():
        if v is None or v == "" or v == "any" or k not in FILTER_KEYS:
            continue
        if k == "tag":
            m &= df.tags.str.contains(str(v), regex=False, na=False)
        elif k in ("regime", "source", "potential"):
            m &= df[k] == v
        else:
            col = COLUMN.get(k, k)
            m &= np.isclose(df[col].astype(float), float(v), equal_nan=False)
    return df[m]


def options(df, key, flt=None):
    """Values a filter key can take among the runs that match the other filters (for drop-downs)."""
    d = match(df, {k: v for k, v in (flt or {}).items() if k != key}) if flt else df
    if key == "tag":
        return []
    col = COLUMN.get(key, key)
    if col not in d:
        return []
    vals = d[col].dropna().unique()
    try:
        return sorted(float(v) for v in vals) if key not in ("regime", "source", "potential") else sorted(vals)
    except (TypeError, ValueError):
        return sorted(map(str, vals))


def n_sets(d):
    """Number of distinct parameter sets (runs that differ only in the seed count as one)."""
    if d.empty:
        return 0
    u = d.drop_duplicates("name")
    cols = [c for c in ("regime", "source", "base_tags", "A", "dn", "F", "T_name") if c in u]
    return int(u[cols].apply(lambda r: "|".join("nan" if (isinstance(v, float) and np.isnan(v)) else str(v)
                                               for v in r), axis=1).nunique())


# --------------------------------------------------------------------------- drawing
def _merged(line, panel):
    return {**line.get("filter", {}), **{k: v for k, v in (panel.get("filter") or {}).items() if v not in (None, "")}}


def _style_kw(line):
    return dict(c=line.get("color", PALETTE[0]), ls=line.get("ls", "-"), lw=float(line.get("lw", 2)),
                marker=line.get("marker") or None, ms=float(line.get("ms", 5)))


def _end_values(d, metric, t_from, thr):
    """One value per run: mean of metric over t > t_from (default: last 10 % of the run), or the break-up time."""
    out = []
    for name, e in d.groupby("name"):
        e = e.sort_values("t")
        if metric == "breakup":
            below = e[e.Smax < thr]
            v = below.t.iloc[0] if len(below) else np.nan
            out.append((name, e.seed.iloc[0], v, e.t.max()))
        else:
            tf = t_from if t_from is not None else e.t.max() * 0.9
            out.append((name, e.seed.iloc[0], e[e.t >= tf - 1e-9][metric].mean(), e.t.max()))
    return pd.DataFrame(out, columns=["name", "seed", "v", "T"])


def _plain(axis):
    """Log axis with plain tick labels (0.2, 0.5, 1, 2 ... instead of 2 x 10^-1)."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def render(recipe, df, return_issues=False):
    """Matplotlib figure for a recipe. With return_issues, also a list of warnings (lines without data...)."""
    plt = pf._style()
    r = recipe
    panels = r.get("panels") or [{"metric": "z"}]
    ncol = max(1, min(int(r.get("columns", 3)), len(panels)))
    nrow = int(np.ceil(len(panels) / ncol))
    w, h = r.get("panel_size", [4.2, 3.2])
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), constrained_layout=True, squeeze=False)
    issues = []
    kind = r.get("kind", "time")
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.set_visible(False)
            continue
        p = panels[i]
        metric = p.get("metric", "z")
        for li, line in enumerate(r.get("lines", [])):
            if line.get("hidden"):
                continue
            d = match(df, _merged(line, p))
            lab = line.get("label", f"line {li + 1}")
            if d.empty:
                issues.append(f"panel {i + 1}, '{lab}': no runs match")
                continue
            kw = _style_kw(line)
            if kind == "time":
                if n_sets(d) > 1:
                    issues.append(f"panel {i + 1}, '{lab}': mixes {n_sets(d)} parameter sets - add a filter")
                if metric == "breakup":
                    issues.append("break-up time is for 'end value vs parameter' figures")
                    continue
                piv = d.pivot_table(index="t", columns="name", values=metric)
                t = piv.index.values
                mean = piv.mean(axis=1).values
                ax.plot(t, mean, label=f"{lab} (n={piv.shape[1]})" if line.get("show_n", True) else lab,
                        markevery=max(1, len(t) // 12), **kw)
                if line.get("band", True) and piv.shape[1] > 1:
                    ax.fill_between(t, piv.min(axis=1).values, piv.max(axis=1).values, color=kw["c"], alpha=0.18, lw=0)
            else:
                xp = (r.get("x") or {}).get("param", "A")
                col = COLUMN.get(xp, xp)
                thr = float(r.get("breakup_threshold", 0.5))
                rows = []
                for xv, g in d.groupby(col):
                    ev = _end_values(g, metric, r.get("t_from"), thr)
                    vals = ev.v.values
                    if metric == "breakup":
                        fin = vals[np.isfinite(vals)]
                        if line.get("points", True):
                            for v, T in zip(vals, ev["T"].values):
                                if np.isfinite(v):
                                    ax.scatter(xv, v, s=16, color=kw["c"], alpha=0.4, lw=0)
                                else:
                                    ax.scatter(xv, T, s=30, facecolors="white", edgecolors=kw["c"], lw=1.2, zorder=3)
                        med = np.median(np.where(np.isfinite(vals), vals, np.inf))
                        rows.append((xv, med if np.isfinite(med) else np.nan, np.nan, np.nan))
                    else:
                        rows.append((xv, np.nanmean(vals), np.nanmin(vals), np.nanmax(vals)))
                        if line.get("points", False):
                            ax.scatter([xv] * len(vals), vals, s=14, color=kw["c"], alpha=0.4, lw=0)
                rows = np.array(sorted(rows), float)
                if not len(rows):
                    continue
                if kw["marker"] is None:
                    kw["marker"] = "o"
                ax.plot(rows[:, 0], rows[:, 1], label=lab, **kw)
                if line.get("band", True) and np.isfinite(rows[:, 2]).any():
                    ax.fill_between(rows[:, 0], rows[:, 2], rows[:, 3], color=kw["c"], alpha=0.18, lw=0)
        for y in p.get("hlines") or []:
            ax.axhline(float(y), ls="--", c="#999", lw=0.8)
        ax.set_title(p.get("title") or (METRICS.get(metric, metric) if kind == "time" or metric != "breakup"
                                        else "break-up time"), loc="left")
        if p.get("ylabel"):
            ax.set_ylabel(p["ylabel"])
        if p.get("log"):
            ax.set_yscale("log")
            _plain(ax.yaxis)
        if p.get("ylim"):
            ax.set_ylim(*p["ylim"])
        if kind == "time":
            ax.set_xlabel(p.get("xlabel") or "t (τ$_B$)")
            if r.get("xlim"):
                ax.set_xlim(*r["xlim"])
        else:
            x = r.get("x") or {}
            ax.set_xlabel(x.get("label") or PARAMS.get(x.get("param", "A"), x.get("param", "A")))
            if x.get("log"):
                ax.set_xscale("log")
                _plain(ax.xaxis)
            if x.get("lim"):
                ax.set_xlim(*x["lim"])
        if i == int(r.get("legend_panel", 0)) and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc=r.get("legend_loc", "best"))
    if r.get("title"):
        fig.suptitle(r["title"], x=0.01, ha="left", fontsize=11)
    return (fig, issues) if return_issues else fig


def render_all(data_root, out_dir=None, df=None, keys=None):
    """Re-make every recipe (or the given keys) from the current data. Returns [(key, path or error)]."""
    import matplotlib.pyplot as plt
    out_dir = out_dir or os.path.join(recipe_dir(data_root), "png")
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_root) if df is None else df
    res = []
    for key in keys or list_recipes(data_root):
        try:
            r = load_recipe(data_root, key)
            fig = render(r, df)
            p = os.path.join(out_dir, key + ".png")
            fig.savefig(p, dpi=int(r.get("dpi", 170)), bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            res.append((key, p))
        except Exception as ex:      # one broken recipe must not stop the others
            res.append((key, f"error: {ex}"))
    return res


# --------------------------------------------------------------------------- starter recipes (the deck figures)
B, O, G, Y, V, K = PALETTE[:6]


def starter_recipes():
    """Editable versions of the time / end-value figures of the proposal deck."""
    L = []
    LJ = {"regime": "ARF", "source": "BD", "potential": "LJ", "A": 3, "T": 10}
    AO = {"regime": "ARF", "source": "BD", "potential": "AO", "A": 3, "T": 10, "q": 0.2}
    mets = ("z", "psi6", "Smax", "nn1")
    L.append({"name": "AO vs LJ over time", "kind": "time", "columns": 3, "panel_size": [4.0, 2.8],
              "title": "AO vs LJ, matched by B$_2$* (band = min–max over seeds)",
              "panels": [{"metric": m, "filter": {"dn": dn}, "title": f"{pf.LAB[m]}, d$_n$ = {dn} nm"}
                         for m in mets for dn in (2000, 2750, 4000)],
              "lines": [{"label": "LJ, A = 3 k$_B$T", "color": B, "filter": LJ},
                        {"label": "AO, q = 0.2", "color": O, "filter": AO}]})
    L.append({"name": "Hexagonal fraction", "kind": "time", "columns": 3, "panel_size": [4.0, 3.0],
              "title": "Hexagonal fraction over time",
              "panels": [{"metric": m, "filter": {"dn": dn}, "title": f"{pf.LAB[m]}, d$_n$ = {dn} nm"}
                         for m in ("hex", "hex_strict") for dn in (2000, 2750, 4000)],
              "lines": [{"label": "LJ, A = 3 k$_B$T", "color": B, "filter": LJ},
                        {"label": "AO, q = 0.2", "color": O, "filter": AO}]})
    L.append({"name": "Confinement vs attraction", "kind": "time", "columns": 3,
              "title": "Order from the field, not from attraction (d$_n$ = 4000 nm)",
              "panels": [{"metric": m} for m in ("psi6", "z", "nn1")],
              "lines": [{"label": "Regime I: no field, no attraction", "color": K,
                         "filter": {"regime": "NAF", "source": "BD", "potential": "LJ", "A": 0, "T": 10}},
                        {"label": "field, no attraction", "color": V,
                         "filter": {"regime": "ARF", "source": "BD", "potential": "LJ", "A": 0, "dn": 4000, "T": 10}},
                        {"label": "field + AO", "color": O, "filter": {**AO, "dn": 4000}}]})
    L.append({"name": "Regime I LJ vs AO", "kind": "time", "columns": 3, "panel_size": [5.0, 3.6],
              "title": "Regime I  No field: cluster formation, LJ vs AO at the same B$_2$*",
              "panels": [{"metric": "z", "hlines": [2.5]}, {"metric": "Smax"}, {"metric": "psi6", "hlines": [0.8]}],
              "lines": [{"label": "LJ, A = 3 k$_B$T", "color": B,
                         "filter": {"regime": "NAF", "source": "BD", "potential": "LJ", "A": 3, "T": 10}},
                        {"label": "AO, same B$_2$*", "color": O,
                         "filter": {"regime": "NAF", "source": "BD", "potential": "AO", "A": 3, "T": 10}}]})
    seq = ["#3b0f70", "#5c2a8a", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b"]
    L.append({"name": "Regime III AO over time", "kind": "time", "columns": 4, "panel_size": [4.0, 3.2],
              "title": "Regime III  After the field is switched off: AO matched by B$_2$*",
              "panels": [{"metric": m} for m in ("y_spread", "z", "Smax", "hex")],
              "lines": [{"label": f"matched to A = {A:g}", "color": seq[i % len(seq)], "band": True,
                         "filter": {"regime": "AAF", "source": "BD", "potential": "AO", "A": A}}
                        for i, A in enumerate((1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4))]})
    L.append({"name": "Regime III end state vs A", "kind": "vs_param", "columns": 4, "panel_size": [4.0, 3.2],
              "title": "Regime III  End state after the field is switched off", "t_from": 18.0,
              "x": {"param": "A", "log": False},
              "panels": [{"metric": m} for m in ("y_spread", "z", "Smax", "hex")],
              "lines": [{"label": "Python LJ", "color": B, "marker": "o",
                         "filter": {"regime": "AAF", "source": "BD", "potential": "LJ"}},
                        {"label": "Python AO (matched B$_2$*)", "color": O, "marker": "s",
                         "filter": {"regime": "AAF", "source": "BD", "potential": "AO"}},
                        {"label": "COMSOL LJ", "color": K, "marker": "x", "ls": "", "band": False,
                         "filter": {"regime": "AAF", "source": "CS", "potential": "LJ"}}]})
    L.append({"name": "Regime IV break-up over time", "kind": "time", "columns": 4, "panel_size": [4.2, 3.2],
              "title": "Regime IV  Break-up under streaming, LJ vs AO at the same B$_2$*",
              "panels": [{"metric": m, "filter": {"A": A, "F": F}, "title": f"{pf.LAB[m]}, A = {A}, F$_0$ = {F}",
                          "hlines": [0.5] if m == "Smax" else []}
                         for m in ("Smax", "hex") for (A, F) in ((3, 3000), (3, 10000), (5, 3000), (5, 10000))],
              "lines": [{"label": "LJ", "color": B, "filter": {"regime": "ASF", "source": "BD", "potential": "LJ", "T": 3}},
                        {"label": "AO, same B$_2$*", "color": O,
                         "filter": {"regime": "ASF", "source": "BD", "potential": "AO", "T": 3}}]})
    L.append({"name": "Regime IV break-up vs F0", "kind": "vs_param", "columns": 2, "panel_size": [6.0, 4.4],
              "title": "Regime IV  Break-up against flow strength", "x": {"param": "F", "log": True},
              "breakup_threshold": 0.5,
              "panels": [{"metric": "breakup", "log": True}, {"metric": "hex", "ylim": [0, 1]}],
              "lines": [{"label": f"{p}, A = {A}", "color": c, "ls": ls, "marker": mk, "band": False,
                         "filter": {"regime": "ASF", "source": "BD", "potential": p, "A": A}}
                        for p, c in (("LJ", B), ("AO", O)) for A, ls, mk in ((3, "--", "o"), (5, "-", "s"))]})
    return L


def write_starters(data_root, overwrite=False):
    done = []
    for r in starter_recipes():
        key = slug(r["name"])
        if overwrite or not os.path.isfile(os.path.join(recipe_dir(data_root), key + ".json")):
            save_recipe(data_root, r)
            done.append(key)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Data folder with NAF / ARF / AAF / ASF inside")
    ap.add_argument("--out", default=None, help="PNG folder (default <data>/_figures/png)")
    ap.add_argument("--only", nargs="*", default=None, help="recipe names (file names without .json)")
    ap.add_argument("--starters", action="store_true", help="first write the starter recipes (keeps edited ones)")
    a = ap.parse_args()
    if a.starters:
        print("starter recipes written:", ", ".join(write_starters(a.data)) or "none (all exist)")
    for key, res in render_all(a.data, a.out, keys=a.only):
        print(("  ✓ " if not res.startswith("error") else "  ✗ ") + key + "  " + res)


if __name__ == "__main__":
    main()

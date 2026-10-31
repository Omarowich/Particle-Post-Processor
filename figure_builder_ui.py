"""Streamlit page for figure_builder.py (Output type -> Paper figures -> Figure builder)."""
from __future__ import annotations

import io
import os

import numpy as np

import figure_builder as fb

KINDS = {"time": "over time", "vs_param": "end value against a parameter"}
LS = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot", "": "no line (points only)"}
MARKERS = {"": "none", "o": "circle", "s": "square", "D": "diamond", "^": "triangle", "v": "triangle down",
           "x": "cross", "+": "plus"}
FILTER_LABELS = {"regime": "Regime", "source": "Source", "potential": "Attraction", "A": "A (k_BT)",
                 "dn": "d_n (nm)", "F": "F₀ (N/m³)", "T": "length (τ_B)", "q": "q", "phi_p": "φ_p"}
REGIME_NAMES = {"NAF": "I (NAF)", "ARF": "II (ARF)", "AAF": "III (AAF)", "ASF": "IV (ASF)"}


def _fmt(v):
    if v == "any":
        return "any"
    if isinstance(v, str):
        return {"BD": "Python", "CS": "COMSOL", **REGIME_NAMES}.get(v, v)
    return f"{v:g}"


def _pick(st, label, key, opts, current, help=None):
    """selectbox with 'any' + the values present in the data (+ the saved value, if no run has it any more)."""
    opts = ["any"] + list(opts)
    if current not in (None, "", "any") and not any(_same(current, o) for o in opts[1:]):
        opts.append(current)
    idx = next((i for i, o in enumerate(opts) if _same(current, o)), 0)
    v = st.selectbox(label, opts, index=idx, format_func=_fmt, key=key, help=help)
    return None if v == "any" else v


def _same(a, b):
    if a in (None, "", "any"):
        return b == "any"
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return bool(np.isclose(float(a), float(b)))


def _floats(txt):
    out = []
    for t in str(txt).replace(";", ",").split(","):
        t = t.strip()
        if t:
            try:
                out.append(float(t))
            except ValueError:
                pass
    return out


def page(st, root, df):
    SS = st.session_state
    recipes = fb.list_recipes(root)
    if "fb_ver" not in SS:
        SS.fb_ver, SS.fb_recipe, SS.fb_key = 0, fb.new_recipe(), None

    def load(key, recipe):
        SS.fb_ver += 1
        SS.fb_key, SS.fb_recipe = key, recipe

    st.caption(f"Recipes are kept in `{fb.recipe_dir(root)}`, one JSON file per figure. Edit a figure here, save it, "
               "and **Re-make all figures** redraws every saved recipe from the current runs (also from the command "
               "line: `python figure_builder.py --data <Data folder>`).")
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.4])
    sel = c1.selectbox("Recipe", ["(new figure)"] + recipes,
                       index=(recipes.index(SS.fb_key) + 1) if SS.fb_key in recipes else 0, key=f"fb_sel{SS.fb_ver}")
    if sel != "(new figure)" and sel != SS.fb_key:
        load(sel, fb.load_recipe(root, sel)); st.rerun()
    if sel == "(new figure)" and SS.fb_key is not None:
        load(None, fb.new_recipe()); st.rerun()
    if c2.button("New: over time", key="fb_new_t"):
        load(None, fb.new_recipe("time")); st.rerun()
    if c3.button("New: vs parameter", key="fb_new_p"):
        load(None, fb.new_recipe("vs_param")); st.rerun()
    if c4.button("Add starter recipes (deck figures)", key="fb_starters",
                 help="Writes editable versions of the deck figures; recipes you already have are kept."):
        done = fb.write_starters(root)
        st.success("Added: " + (", ".join(done) if done else "nothing new (all exist)")); st.rerun()

    with st.expander("Re-make all figures", expanded=False):
        out = st.text_input("Save PNGs to", value=os.path.join(fb.recipe_dir(root), "png"), key="fb_out").strip().strip('"')
        if st.button("Re-make all saved recipes now", key="fb_all", type="primary"):
            with st.spinner("drawing…"):
                res = fb.render_all(root, out, df=df)
            for k, p in res:
                (st.error if p.startswith("error") else st.write)(f"{'✗' if p.startswith('error') else '✓'} {k}: {p}")

    r = SS.fb_recipe
    v = SS.fb_ver
    K = lambda s: f"fb{v}_{s}"   # widget keys change when another recipe is loaded

    # ---------------- general
    with st.expander("Figure", expanded=True):
        g1, g2, g3 = st.columns([2, 3, 1.4])
        name = g1.text_input("Name (file name)", r.get("name", "figure"), key=K("name"))
        title = g2.text_input("Title on the figure (optional)", r.get("title", ""), key=K("title"))
        kind = g3.radio("Kind", list(KINDS), index=list(KINDS).index(r.get("kind", "time")), format_func=KINDS.get,
                        key=K("kind"))
        g4, g5, g6, g7, g8 = st.columns(5)
        cols = g4.number_input("Panels per row", 1, 8, int(r.get("columns", 3)), key=K("cols"))
        pw = g5.number_input("Panel width (in)", 2.0, 12.0, float(r.get("panel_size", [4.2, 3.2])[0]), 0.2, key=K("pw"))
        ph = g6.number_input("Panel height (in)", 1.5, 10.0, float(r.get("panel_size", [4.2, 3.2])[1]), 0.2, key=K("ph"))
        leg = g7.number_input("Legend in panel #", 1, 64, int(r.get("legend_panel", 0)) + 1, key=K("leg"))
        dpi = g8.number_input("PNG dpi", 72, 600, int(r.get("dpi", 170)), 10, key=K("dpi"))
        x = dict(r.get("x") or {"param": "A"})
        xlim = None
        if kind == "vs_param":
            h1, h2, h3, h4 = st.columns(4)
            x["param"] = h1.selectbox("x axis: run parameter", list(fb.PARAMS), format_func=lambda p: fb.PARAMS[p],
                                      index=list(fb.PARAMS).index(x.get("param", "A")), key=K("xp"))
            x["log"] = h2.checkbox("log x", bool(x.get("log")), key=K("xlog"))
            tf = h3.text_input("Average over t > (τ_B; empty = last 10 % of each run)",
                               "" if r.get("t_from") is None else f"{r['t_from']:g}", key=K("tf"))
            thr = h4.number_input("Break-up: largest cluster below", 0.05, 0.95, float(r.get("breakup_threshold", 0.5)),
                                  0.05, key=K("thr"))
        else:
            xl = st.text_input("Time range (τ_B), e.g. 0, 10 (empty = all)",
                               ", ".join(f"{a:g}" for a in r.get("xlim") or []), key=K("xlim"))
            xlim = _floats(xl)[:2] or None

    # ---------------- panels
    with st.expander("Panels (one measure each)", expanded=True):
        n_p = st.number_input("Number of panels", 1, 32, len(r.get("panels", [])) or 1, key=K("np"))
        metrics = list(fb.METRICS) if kind == "vs_param" else [m for m in fb.METRICS if m != "breakup"]
        panels = []
        for i in range(int(n_p)):
            p = (r.get("panels") or [{}])[i] if i < len(r.get("panels") or []) else {"metric": "z"}
            pf_ = p.get("filter") or {}
            c = st.columns([2.2, 2.4, 1.2, 1.2, 1.2, 1.2, 0.8, 1.2])
            m = c[0].selectbox(f"Panel {i + 1}", metrics, index=metrics.index(p.get("metric", "z"))
                               if p.get("metric", "z") in metrics else 0, format_func=lambda k: fb.METRICS[k].split(" (")[0]
                               if k != "breakup" else "break-up time", key=K(f"pm{i}"))
            ttl = c[1].text_input("title (empty = measure)", p.get("title", ""), key=K(f"pt{i}"))
            fdn = _pick(c[2], "only d_n", K(f"pdn{i}"), fb.options(df, "dn"), pf_.get("dn"), "same lines, this d_n")
            fA = _pick(c[3], "only A", K(f"pA{i}"), fb.options(df, "A"), pf_.get("A"))
            fF = _pick(c[4], "only F₀", K(f"pF{i}"), fb.options(df, "F"), pf_.get("F"))
            yl = c[5].text_input("y range", ", ".join(f"{a:g}" for a in p.get("ylim") or []), key=K(f"py{i}"))
            lg = c[6].checkbox("log y", bool(p.get("log")), key=K(f"pl{i}"))
            hl = c[7].text_input("dashed lines at", ", ".join(f"{a:g}" for a in p.get("hlines") or []), key=K(f"ph{i}"))
            pp = {"metric": m, "title": ttl, "filter": {k: val for k, val in (("dn", fdn), ("A", fA), ("F", fF)) if val is not None},
                  "ylim": _floats(yl)[:2] or None, "log": lg, "hlines": _floats(hl)}
            panels.append({k: val for k, val in pp.items() if val not in (None, "", [], {}, False)})

    # ---------------- lines
    lines = []
    with st.expander("Lines (which runs, and how they look)", expanded=True):
        n_l = st.number_input("Number of lines", 1, 16, len(r.get("lines", [])) or 1, key=K("nl"))
        for i in range(int(n_l)):
            L = (r.get("lines") or [{}])[i] if i < len(r.get("lines") or []) else \
                {"label": f"line {i + 1}", "color": fb.PALETTE[i % len(fb.PALETTE)], "filter": {}}
            F = dict(L.get("filter") or {})
            st.markdown(f"**Line {i + 1}**")
            a = st.columns([2.4, 0.8, 1.3, 0.9, 1.1, 0.8, 0.8])
            lab = a[0].text_input("label", L.get("label", f"line {i + 1}"), key=K(f"ll{i}"))
            col = a[1].color_picker("colour", L.get("color", fb.PALETTE[i % len(fb.PALETTE)]), key=K(f"lc{i}"))
            ls = a[2].selectbox("line", list(LS), index=list(LS).index(L.get("ls", "-")) if L.get("ls", "-") in LS else 0,
                                format_func=LS.get, key=K(f"ls{i}"))
            lw = a[3].number_input("width", 0.0, 6.0, float(L.get("lw", 2)), 0.25, key=K(f"lw{i}"))
            mk = a[4].selectbox("marker", list(MARKERS), index=list(MARKERS).index(L.get("marker", "")) if L.get("marker", "") in MARKERS else 0,
                                format_func=MARKERS.get, key=K(f"lm{i}"))
            band = a[5].checkbox("band", bool(L.get("band", True)), key=K(f"lb{i}"), help="min–max over seeds")
            hide = a[6].checkbox("hide", bool(L.get("hidden", False)), key=K(f"lh{i}"))
            b = st.columns(10)
            newF = {}
            for j, key in enumerate(("regime", "source", "potential", "A", "dn", "F", "T", "q", "phi_p")):
                newF[key] = _pick(b[j], FILTER_LABELS[key], K(f"lf{i}{key}"), fb.options(df, key, {**F, **newF}), F.get(key))
            newF["tag"] = b[9].text_input("name contains", F.get("tag", "") or "", key=K(f"lt{i}"),
                                          help="e.g. dt2000, N256, seed1").strip() or None
            newF = {k: val for k, val in newF.items() if val is not None}
            d = fb.match(df, newF)
            st.caption(f"→ {d.name.nunique()} runs" + (f", {fb.n_sets(d)} parameter sets" if len(d) else "") +
                       (" — over-time lines need exactly one parameter set (add a filter or a panel filter)"
                        if kind == "time" and len(d) and fb.n_sets(d) > 1 and not any(pp.get("filter") for pp in panels) else ""))
            line = {"label": lab, "color": col, "ls": ls, "lw": lw, "marker": mk, "band": band, "filter": newF}
            if hide:
                line["hidden"] = True
            lines.append(line)

    recipe = {"name": name, "kind": kind, "title": title, "columns": int(cols), "panel_size": [pw, ph],
              "legend_panel": int(leg) - 1, "dpi": int(dpi), "panels": panels, "lines": lines}
    if kind == "vs_param":
        recipe.update({"x": x, "t_from": (_floats(tf) or [None])[0], "breakup_threshold": float(thr)})
    elif xlim:
        recipe["xlim"] = xlim
    SS.fb_recipe = recipe

    # ---------------- preview and saving
    st.subheader("Preview")
    import matplotlib.pyplot as plt
    try:
        fig, issues = fb.render(recipe, df, return_issues=True)
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        for msg in sorted(set(issues))[:12]:
            st.warning(msg)
    except Exception as ex:
        buf = None
        st.error(f"could not draw: {ex}")
    s1, s2, s3, s4 = st.columns(4)
    if s1.button("Save recipe", type="primary", key=K("save")):
        key = fb.save_recipe(root, recipe)
        if SS.fb_key and SS.fb_key != key and SS.fb_key in recipes:
            st.info(f"Saved as a new recipe `{key}` (renamed); the old `{SS.fb_key}` is kept.")
        SS.fb_key = key
        st.success(f"saved {key}.json"); st.rerun()
    if buf is not None:
        s2.download_button("Download PNG", buf.getvalue(), file_name=fb.slug(name) + ".png", mime="image/png", key=K("dl"))
    import json
    s3.download_button("Download recipe (JSON)", json.dumps(recipe, indent=2, ensure_ascii=False).encode(),
                       file_name=fb.slug(name) + ".json", mime="application/json", key=K("dlj"))
    if SS.fb_key in recipes and s4.button(f"Delete recipe {SS.fb_key}", key=K("del")):
        fb.delete_recipe(root, SS.fb_key)
        load(None, fb.new_recipe()); st.rerun()

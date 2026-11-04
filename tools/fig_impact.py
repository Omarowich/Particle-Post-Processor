"""Fig. S (paper 2): impact of the secondary radiation force and Rayleigh streaming. AO q = 0.05, A = 3, d_n = 4000 nm,
3 seeds; 2D at 2 tauB, 3D (PS, H = 4 and 20 um) at 5 tauB. Usage: fig_impact.py <results.csv> <out>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
d = d[d.t == d.groupby("case").t.transform("max")]
LAB = {"base": "neither", "sec_nom": "secondary, nominal", "sec_worst": "secondary, upper bound",
       "st-_nom": "streaming, nominal", "st-_worst": "streaming, upper bound", "st+_worst": "streaming reversed, upper bound"}
COL = {"base": "#555", "sec_nom": "#9ecae1", "sec_worst": "#2a78d6", "st-_nom": "#fdae6b", "st-_worst": "#eb6834", "st+_worst": "#bbb"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6), constrained_layout=True)
panels = [("2D", "psi6", "ψ$_6$", "a  2D (monolayer)"), ("H4", "Smax", "largest cluster S$_{max}$", "b  3D, H = 4 µm"),
          ("H20", "Smax", "largest cluster S$_{max}$", "c  3D, H = 20 µm"), ("H20", "f_on_top", "fraction resting on another", "d  3D, H = 20 µm")]
for a, (sys_, col, ylab, title) in zip(ax, panels):
    keys = [k for k in LAB if f"{sys_}_{k}" in set(d.case)]
    for i, k in enumerate(keys):
        g = d[d.case == f"{sys_}_{k}"][col]
        a.bar(i, g.mean(), yerr=g.std(), color=COL[k], width=0.75, capsize=2, error_kw=dict(lw=0.7))
        a.scatter(np.full(len(g), i), g, s=6, color="k", zorder=3)
    a.set_xticks(range(len(keys))); a.set_xticklabels([LAB[k] for k in keys], rotation=55, ha="right", fontsize=6.3)
    a.set_ylabel(ylab); a.set_title(title, loc="left", fontsize=8, fontweight="bold")
    a.set_ylim(0, 1.08)
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"FigS_impact_secondary-force_streaming.{ext}"), dpi=300)

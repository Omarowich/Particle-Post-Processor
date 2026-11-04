import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("/home/claude/sim/b2c/collapse.csv"); out = sys.argv[1]
MOD = {"LJ 12-6": ("#2a78d6", "o"), "AO q=0.2": ("#008300", "v"), "AO q=0.075": ("#7fbf7f", "<"), "AO q=0.05": ("#eb6834", "s"),
       "2n-n LJ": ("#1baf7a", "^"), "square well": ("#eda100", "D"), "Yukawa": ("#e87ba4", "P")}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True, sharey=True)
for i, reg in enumerate(("I", "III")):
    for j, (col, lab) in enumerate((("B2", "−B$_2^*$ (full potential)"), ("B2c_1.10", "−B$_{2,c}^*$ (attraction within 1.1σ only)"))):
        a = ax[i, j]
        for m, g in R[R.regime == reg].groupby("model"):
            g = g.sort_values("A"); c, mk = MOD.get(m, ("#999", "o"))
            a.errorbar(-g[col], g.psi6, yerr=g.psd.fillna(0), color=c, marker=mk, ms=3.5, lw=1, capsize=1.5, label=m)
        a.set_xscale("log"); a.axhline(0.8, color="#999", ls="--", lw=0.6); a.set_xlabel(lab)
        a.set_title(f"{'abcd'[2 * i + j]}  Regime {reg}, 10 τ$_B$", loc="left", fontsize=8, fontweight="bold")
    ax[i, 0].set_ylabel("ψ$_6$")
h, l = [], []
for a in ax.ravel():
    for hh, ll in zip(*a.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, frameon=False, fontsize=6.8, ncol=7, loc="outside lower center")
for e in ("png", "pdf"): fig.savefig(f"{out}/FigS_contact-stickiness_collapse.{e}", dpi=300)

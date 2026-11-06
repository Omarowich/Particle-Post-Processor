"""Fig. 9 (paper 2): order onset against the full B2* and against the well depth. Usage: python fig_order_onset.py <energy_shell.csv> <out dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv(sys.argv[1]); out = sys.argv[2]
MOD = {"LJ 12-6": ("#2a78d6", "o"), "AO q=0.2": ("#008300", "v"), "AO q=0.075": ("#7fbf7f", "<"), "AO q=0.05": ("#eb6834", "s"),
       "2n-n LJ": ("#1baf7a", "^"), "square well": ("#eda100", "D"), "Yukawa": ("#e87ba4", "P"),
       "2n-n LJ n=8": ("#4b2991", "X"), "2n-n LJ n=12": ("#8f5bd6", "X"), "Yukawa k=2": ("#7a4b1e", "*"), "Yukawa k=4": ("#c08a4f", "*")}
LAB = {"AO q=0.05": "AO q = 0.05", "AO q=0.2": "AO q = 0.2", "AO q=0.075": "AO q = 0.075", "2n-n LJ": "2n–n LJ (n = 57.6)",
       "2n-n LJ n=8": "2n–n LJ n = 8", "2n-n LJ n=12": "2n–n LJ n = 12", "Yukawa": "Yukawa (κ = 17.3)", "Yukawa k=2": "Yukawa κ = 2", "Yukawa k=4": "Yukawa κ = 4"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(6.0, 4.2), constrained_layout=True)
for r, reg in enumerate(("I", "III")):
    G = R[R.regime == reg]
    for c, (col, xl, lg) in enumerate((("B2", "−B$_2^*$", True), ("depth", "well depth −U$_{min}$ (k$_B$T)", False))):
        a = ax[r, c]
        for m, g in G.groupby("model"):
            g = g.sort_values("A"); cl, mk = MOD[m]; x = g[col] if col == "depth" else -g[col]
            a.plot(x, g.psi6, color=cl, marker=mk, ms=3.5, lw=1, label=LAB.get(m, m))
        if lg: a.set_xscale("log")
        a.axhline(0.8, color="#999", ls="--", lw=0.6); a.set_xlabel(xl)
        if c == 0: a.set_ylabel(f"Regime {reg}: ψ$_6$ after 10 τ$_B$")
ax[0, 0].set_title("a  full B$_2^*$: spread 9.6 / 17.7", loc="left", fontsize=8, fontweight="bold")
ax[0, 1].set_title("b  well depth: 1.5 / 1.3", loc="left", fontsize=8, fontweight="bold")
h, l = [], []
for a in ax.flat:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, frameon=False, ncol=4, fontsize=6.2, loc="outside lower center")
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig9_order-onset_B2-vs-well-depth.{e}", dpi=300)
print("saved")

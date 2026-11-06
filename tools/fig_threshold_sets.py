"""Fig. (paper 2, section 3.3): order onset of every model under all 27 threshold sets (fluid threshold z_c,
network threshold S_c, order threshold psi_c), Regimes I and III after 10 tauB.
Usage: python fig_threshold_sets.py <threshold_sets.csv> <out dir>"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv(sys.argv[1]); out = sys.argv[2]
MOD = {"LJ 12-6": ("#2a78d6", "o"), "AO q=0.05": ("#eb6834", "s"), "AO q=0.2": ("#008300", "v"), "2n-n LJ": ("#1baf7a", "^"),
       "square well": ("#eda100", "D"), "Yukawa": ("#e87ba4", "P")}
LAB = {"AO q=0.05": "AO q = 0.05", "AO q=0.2": "AO q = 0.2"}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
for a, reg, top in zip(ax, ("I", "III"), (5.6, 4.15)):
    G = R[R.regime == reg].sort_values(["z_c", "S_c", "psi_c"])
    for m, (c, mk) in MOD.items():
        g = G[G.model == m]
        if g.empty or g.onset_A.isna().all() and reg == "III": continue
        x = np.arange(1, len(g) + 1); y = g.onset_A.values; off = (list(MOD).index(m) - 2.5) * 0.04
        a.plot(x, y + off, color=c, lw=1.0, marker=mk, ms=3, label=LAB.get(m, m))
        nan = np.isnan(y)
        if nan.any(): a.scatter(x[nan], np.full(nan.sum(), top) + off, marker="^", s=12, color=c, clip_on=False)
    for k in (9.5, 18.5): a.axvline(k, color="#bbb", lw=0.6)
    for k, zc in enumerate((2.0, 2.5, 3.0)): a.text(5 + 9 * k, top + 0.2, f"z$_c$ = {zc}", ha="center", fontsize=6.5, color="#555")
    a.set_xlim(0.5, 27.5); a.set_xticks([]); a.set_xlabel("27 threshold sets (S$_c$ and ψ$_c$ vary within each z$_c$ block)")
    a.set_ylabel("onset of order, A (k$_B$T)")
    a.set_title(f"{'a' if reg == 'I' else 'b'}  Regime {reg}", loc="left", fontsize=8, fontweight="bold", pad=14)
ax[0].text(23, 5.25, "short-range models: not ordered up to A = 10", fontsize=5.8, ha="center", va="top", color="#555")
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=6, fontsize=6.5, loc="outside lower center")
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_threshold-sets_order-onset_RegimesI-III.{e}", dpi=300)
print("saved")

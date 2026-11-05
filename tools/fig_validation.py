"""Fig. 5: Python BD vs COMSOL particle tracing, matched cases (LJ 12-6, 2D, N = 144): coordination z, local order
psi6 and largest-cluster fraction S_max of each COMSOL run against the mean +- SD of the Python seeds, at the end and
at half of the common run time. Usage: python fig_validation.py <validation_comsol_vs_bd.csv> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
COL = {"NAF": ("#888888", "Regime I"), "ARF": ("#eb6834", "Regime II"), "AAF": ("#2a78d6", "Regime III")}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
for a, (k, lab, lim) in zip(ax, (("z_mean", "coordination z", (0, 6)), ("psi6", "local order ψ$_6$", (0.4, 1.0)), ("Smax", "largest cluster S$_{max}$", (0, 1.02)))):
    a.plot(lim, lim, color="#999", lw=0.6, zorder=0)
    for reg, (c, nm) in COL.items():
        g = d[d.regime == reg]
        a.errorbar(g[k + "_C"], g[k + "_B"], yerr=g[k + "_Bsd"], fmt="o", ms=2.8, color=c, ecolor=c, elinewidth=0.5, capsize=0, alpha=0.85, label=nm)
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    a.set_xlabel(f"COMSOL {lab}"); a.set_ylabel(f"Python {lab}")
    med = (d[k + "_C"] - d[k + "_B"]).abs().median()
    a.text(0.04, 0.96, f"median |Δ| = {med:.2f}", transform=a.transAxes, va="top", fontsize=6.5)
ax[0].legend(frameon=False, fontsize=6.2, loc="lower right")
for i, a in enumerate(ax): a.set_title("abc"[i], loc="left", fontweight="bold", fontsize=8.5)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig5_validation_Python-vs-COMSOL_matched-cases_LJ.{e}"), dpi=300)
print("saved")

"""Fig. 11b (draft): 3D order against chamber height. Usage: python fig_order3d.py <order3d.csv> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
R = pd.read_csv(sys.argv[1]); out = sys.argv[2]
R = R[~R.name.str.contains("_2000nm_|_4000nm_|drho0")]
R = R[~((R.regime == "II") & (R.dim == 2) & (R.dn == 3000))] if False else R
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.0), constrained_layout=True)
RC = {"I": ("#2a78d6", 10), "II": ("#eb6834", 5), "III": ("#1baf7a", 10)}
for k, (col, lab) in enumerate((("f_xtal", "crystal-like fraction (3D q$_6$ bonds)"), ("psi6_bottom", "ψ$_6$ of the floor layer"))):
    a = ax[k]
    for reg, (c, T) in RC.items():
        s = R[(R.regime == reg) & (R.dim == 3) & (R.t == T) & ((R.regime != "II") | R.dn.isin([2500, 3500]))]
        if reg != "II": s = s[s.A == 3]
        g = s.groupby("H")[col].agg(["mean", "std"])
        a.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=c, marker="o", ms=3.5, lw=1.2, capsize=2,
                   label=f"Regime {reg} ({T} τ$_B$)")
        r2 = R[(R.regime == reg) & (R.dim == 2) & (R.t == T)]
        r2 = r2[r2.A == 3] if reg != "II" else r2[r2.dn.isin([2500, 3000])]
        a.axhline(r2[col].mean(), color=c, ls="--", lw=0.8)
    a.set_xscale("log"); a.set_xticks([2.2, 3, 4, 6, 10, 20]); a.set_xticklabels(["2.2", "3", "4", "6", "10", "20"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.axvspan(2.0, 3.5, color="#e8f3e8", zorder=0, lw=0)
    a.set_xlabel("chamber height H (µm)"); a.set_ylabel(lab); a.set_ylim(0, 1.02)
    a.set_title(f"{'ab'[k]}  {'3D crystal order' if k == 0 else 'order in the floor layer'}", loc="left", fontsize=8.5, fontweight="bold")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper right"); ax[0].text(0.97, 0.55, "dashed: 2D, A = 3", transform=ax[0].transAxes, ha="right", fontsize=6.5, color="#555")
a = ax[2]
for H, ls in ((2.2, ":"), (4.0, "--"), (20.0, "-")):
    s = R[(R.regime == "III") & (R.dim == 3) & (R.H == H) & (R.A == 3)].groupby("t")[["f_xtal", "f_bottom"]].mean()
    a.plot(s.index, s.f_xtal, color="#1baf7a", ls=ls, marker="o", ms=3, label=f"crystal-like, H = {H:g}")
    a.plot(s.index, s.f_bottom, color="#7a2a9a", ls=ls, marker="s", ms=3, label=f"on the floor, H = {H:g}")
a.set_xlabel("time after the field is off (τ$_B$)"); a.set_ylabel("fraction of particles"); a.set_ylim(0, 1.02)
a.set_title("c  Regime III, A = 3", loc="left", fontsize=8.5, fontweight="bold")
a.legend(frameon=False, fontsize=5.6, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
os.makedirs(out, exist_ok=True)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig11b_3D-order_q6_floor-layer.{e}"), dpi=300)
print("saved")

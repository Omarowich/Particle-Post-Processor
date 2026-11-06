"""Fig. 14: when does the 2D picture hold? 3D runs (AO q = 0.05, A = 3, N = 144) against chamber height H.
(a) fraction of particles resting on others; (b) in-plane order psi6 (projected) with the 2D value (band);
(c) Regime II: fraction of particles within 5 um of the node. Mean +- SD over seeds; stars: heavy particles
(drho = 1000 kg/m3, gravitational length 0.1 um) at H = 20 um. Usage: python fig_when2d.py <hscan csv> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]; d["drho"] = d.drho.fillna(52)
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
C = {"I": ("#888888", "Regime I (10 τ$_B$)", 10, 0), "II": ("#eb6834", "Regime II, 3500 nm (5 τ$_B$)", 5, 3500), "III": ("#2a78d6", "Regime III (10 τ$_B$)", 10, 0)}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
for reg, (c, lab, T, dn) in C.items():
    g = d[(d.regime == reg) & (d.t == T) & (d.A == 3) & (d.dn == dn)]
    for k, (col, a) in enumerate((("f_on_top", ax[0]), ("psi6", ax[1]))):
        s = g[(g.dim == 3) & (g.drho == 52)].groupby("H")[col].agg(["mean", "std"])
        a.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=c, marker="o", ms=3.5, lw=1.1, capsize=0, label=lab)
        hv = g[(g.dim == 3) & (g.drho == 1000)][col]
        if len(hv): a.errorbar([20], [hv.mean()], yerr=[hv.std()], color=c, marker="*", ms=9, mec="k", mew=0.4, lw=0, elinewidth=0.8, capsize=0)
    if reg == "II":
        r2 = d[(d.regime == "II") & (d.dim == 2) & (d.t == T) & d.dn.isin([3000, 4000])].psi6
        ax[1].axhspan(r2.mean() - r2.std(), r2.mean() + r2.std(), color=c, alpha=0.15, lw=0); ax[1].text(23, r2.mean() + r2.std() + 0.01, "2D, Regime II", fontsize=5.8, ha="right", color=c)
        s = g[(g.dim == 3) & (g.drho == 52)].groupby("H").band5.agg(["mean", "std"])
        hb = g[(g.dim == 3) & (g.drho == 1000)].band5
        if len(hb): ax[2].errorbar([20], [hb.mean()], yerr=[hb.std()], color=c, marker="*", ms=9, mec="k", mew=0.4, lw=0, elinewidth=0.8)
        ax[2].errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), color=c, marker="o", ms=3.5, lw=1.1, capsize=0, label="3D, 3500 nm")
        b2 = d[(d.regime == "II") & (d.dim == 2) & (d.t == T) & d.dn.isin([3000, 4000])].band5
        ax[2].axhspan(b2.mean() - b2.std(), b2.mean() + b2.std(), color=c, alpha=0.15, lw=0, label="2D, 3000–4000 nm")
for a in ax:
    a.set_xscale("log"); a.set_xticks([2.2, 3, 4, 6, 10, 20]); a.set_xticklabels(["2.2", "3", "4", "6", "10", "20"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.axvspan(2.0, 3.5, color="#e5f4e4", lw=0, zorder=0); a.set_xlim(2.0, 24); a.set_xlabel("chamber height H (µm)")
ax[0].set_ylabel("fraction on top of others"); ax[0].set_ylim(-0.03, 0.7)
ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
ax[0].text(20, 0.12, "heavy\nparticles", fontsize=5.8, ha="center")
ax[1].set_ylabel("in-plane order ψ$_6$"); ax[1].set_ylim(0.2, 1.02); ax[1].text(2.1, 0.25, "green: 2D-like", fontsize=5.8, color="#2e7d32")
ax[2].set_ylabel("fraction within 5 µm of the node"); ax[2].set_ylim(0.8, 1.02); ax[2].legend(frameon=False, fontsize=5.8, loc="lower right")
for i, a in enumerate(ax): a.set_title("abc"[i] + ["   stacking", "   order", "   band formation"][i], loc="left", fontweight="bold", fontsize=7.5)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig14_when-2D-holds_chamber-height_stacking-order-band.{e}"), dpi=300)
print("saved")

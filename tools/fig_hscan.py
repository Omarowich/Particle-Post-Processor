"""Fig. 11 (draft): 2D vs full 3D against chamber height (batches 17a-i). AO q = 0.05, PS (drho = 52), N = 144, 50 um box.
Usage: python fig_hscan.py <hscan csv> <states_RegimeI.csv> <states_RegimeIII.csv> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
hs, sI, sIII, out = sys.argv[1:5]
d = pd.read_csv(hs); d = d[d.name.str.contains("_10TauB_|_5TauB_2500|_5TauB_3500")]
d3 = d[d.dim == 3]; d2 = d[d.dim == 2]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
CA = {1.0: "#9bb7e0", 2.0: "#2a78d6", 3.0: "#eb6834", 5.0: "#7a2a9a"}
CD = {2500.0: "#2a78d6", 3500.0: "#eb6834"}
def ref(states, A, T, col="psi6"):
    s = pd.read_csv(states); s = s[s.model.str.startswith("AO q=0.05") & (s.A == A) & (s.t_tauB == T)]
    return s[col].mean(), s[col].std()
def series(ax_, sub, key, val, col, lab, marker="o"):
    g = sub[sub[key] == val].groupby("H")[col].agg(["mean", "std", "count"]).reset_index()
    if len(g): ax_.errorbar(g.H, g["mean"], yerr=g["std"].fillna(0), color=lab[1], marker=marker, ms=4, lw=1.2, capsize=2, label=lab[0])
for j, (reg, states, T, As) in enumerate((("I", sI, 10, (2.0, 3.0, 5.0)), ("III", sIII, 10, (1.0, 2.0, 3.0)))):
    c = 0 if reg == "I" else 2
    sub = d3[(d3.regime == reg) & (d3.t == T)]
    for A in As:
        series(ax[0, c], sub, "A", A, "psi6", (f"A = {A:g}", CA[A]))
        m, s = ref(states, A, T); ax[0, c].axhspan(m - s, m + s, color=CA[A], alpha=0.18, lw=0)
        series(ax[1, c], sub, "A", A, "f_on_top", (f"A = {A:g}", CA[A]))
    ax[0, c].set_title(f"{'a' if reg == 'I' else 'c'}  Regime {reg} (no field), {T} τ$_B$", loc="left", fontsize=8.5, fontweight="bold")
    ax[0, c].set_ylabel("ψ$_6$ (projected on the floor)"); ax[0, c].set_ylim(0.3, 1.0)
    ax[0, c].text(0.98, 0.97, "bands: 2D (mean ± SD)", transform=ax[0, c].transAxes, ha="right", va="top", fontsize=6.5, color="#555")
# Regime II: band formation at 1 tauB (kinetics) and order at 5 tauB
sub1 = d3[(d3.regime == "II") & (d3.t == 1)]; sub5 = d3[(d3.regime == "II") & (d3.t == 5)]
r1 = d2[(d2.regime == "II") & (d2.t == 1)].groupby("dn")["band5"].mean()
for dn in (2500.0, 3500.0):
    series(ax[0, 1], sub1, "dn", dn, "band5", (f"3D, {dn:.0f} nm", CD[dn]))
    series(ax[1, 1], sub5, "dn", dn, "f_on_top", (f"{dn:.0f} nm", CD[dn]))
ax[0, 1].axhline(r1[2500.0], color=CD[2500.0], ls="--", lw=0.8); ax[0, 1].axhline(np.interp(3500, r1.index, r1.values), color=CD[3500.0], ls="--", lw=0.8)
ax[0, 1].text(0.98, 0.03, "dashed: 2D (3500 nm interpolated)", transform=ax[0, 1].transAxes, ha="right", fontsize=6.5, color="#555")
ax[0, 1].set_ylabel("fraction within 5 µm of the node, 1 τ$_B$"); ax[0, 1].set_ylim(0, 1)
ax[0, 1].set_title("b  Regime II (field on): band growth", loc="left", fontsize=8.5, fontweight="bold")
p2 = d2[(d2.regime == "II") & (d2.t == 5)]["psi6"].mean()
ax[1, 1].text(0.98, 0.03, f"ψ$_6$ at 5 τ$_B$: 2D {p2:.2f}, 3D {sub5.psi6.mean():.2f}", transform=ax[1, 1].transAxes, ha="right", fontsize=6.5, color="#555")
for c in range(3):
    ax[1, c].set_ylabel("fraction resting on another"); ax[1, c].set_ylim(0, 0.7)
    for r in range(2):
        ax[r, c].set_xscale("log"); ax[r, c].set_xticks([2.2, 3, 4, 6, 10, 20]); ax[r, c].set_xticklabels(["2.2", "3", "4", "6", "10", "20"])
        ax[r, c].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[r, c].axvspan(2.0, 3.5, color="#e8f3e8", zorder=0, lw=0)
    ax[1, c].set_xlabel("chamber height H (µm)")
    ax[1, c].axhline(0, color="#999", lw=0.5)
ax[1, 0].text(2.6, 0.62, "room for\none layer", fontsize=6.5, color="#3a7d3a", ha="center")
ax[0, 0].legend(frameon=False, fontsize=6.5, loc="lower left"); ax[0, 1].legend(frameon=False, fontsize=6.5, loc="upper left")
ax[0, 2].legend(frameon=False, fontsize=6.5, loc="lower left")
for c, lab in enumerate("def"):
    ax[1, c].set_title(f"{lab}  stacking ({'5' if c == 1 else '10'} τ$_B$)", loc="left", fontsize=8.5, fontweight="bold")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig11_2D-vs-3D_chamber-height.{ext}"), dpi=300)
print("saved")

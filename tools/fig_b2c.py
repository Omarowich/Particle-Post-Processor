"""Fig. B2c (paper 2): a contact-shell second virial coefficient as a structure parameter.
(a) cumulative B2*(r): how much of each model's stickiness lies within r; all at the same total B2* (LJ A = 3).
(b) onset of order (psi6 = 0.8) per model on the full B2* axis and on the contact-shell B2c* axis.
(c) how well the onsets collapse against the cutoff r_c (max/min of the onset across models).
(d) order against B2c*(1.1 sigma), Regime I."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pd.read_csv("scan.csv"); cum = json.load(open("cumulative.json")); out = sys.argv[1]
MOD = {"LJ 12-6": ("#2a78d6", "o"), "AO q=0.2": ("#008300", "v"), "AO q=0.075": ("#7fbf7f", "<"), "AO q=0.05": ("#eb6834", "s"),
       "2n-n LJ": ("#1baf7a", "^"), "square well": ("#eda100", "D"), "Yukawa": ("#e87ba4", "P")}
LAB = {"AO q=0.05": "AO q = 0.05", "AO q=0.2": "AO q = 0.2", "AO q=0.075": "AO q = 0.075", "2n-n LJ": "2n–n LJ"}
cuts = sorted(float(c[1:]) for c in R.columns if c.startswith("c"))

def onset(g, col):
    g = g.sort_values("A"); x = -g[col].values; y = g.psi6.values
    i = np.where((y[:-1] < 0.8) & (y[1:] >= 0.8))[0]
    if not len(i) or min(x[i[0]], x[i[0] + 1]) <= 0:
        return np.nan
    k = i[0]; return float(np.exp(np.interp(0.8, [y[k], y[k + 1]], [np.log(x[k]), np.log(x[k + 1])])))

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
# a
a = ax[0, 0]
for m, d in cum.items():
    c, mk = MOD[m]; a.plot(d["r"], d["b"], color=c, lw=1.4, label=LAB.get(m, m))
a.axvline(1.1, color="#555", ls="--", lw=0.8); a.text(1.11, 0.6, "r$_c$ = 1.1σ", fontsize=6.5, color="#555")
a.axvspan(1.0, 1.1, color="#eee", zorder=0)
a.set_xlim(0.95, 3.1); a.set_xlabel("r / σ (upper limit of the B$_2$ integral)"); a.set_ylabel("B$_2^*$ counted up to r")
a.set_title("a  where the stickiness sits (all at B$_2^*$ = −10)", loc="left", fontsize=8, fontweight="bold")
a.legend(frameon=False, fontsize=6.3, loc="lower right")
a.annotate("LJ 12-6: only −1.9 of −10\ninside 1.1σ; rest in the tail", xy=(1.1, -1.9), xytext=(1.6, -4.5), fontsize=6.3,
           arrowprops=dict(arrowstyle="->", lw=0.6, color="#2a78d6"), color="#2a78d6")
# b
a = ax[0, 1]
yl, yt = [], []
for reg_i, reg in enumerate(("I", "III")):
    G = R[R.regime == reg]
    for k, (m, g) in enumerate(sorted(G.groupby("model"), key=lambda t: t[0])):
        yy = reg_i * 8 + k
        o1, o2 = onset(g, "B2"), onset(g, "c1.10")
        c, mk = MOD[m]
        if np.isfinite(o1) and np.isfinite(o2):
            a.plot([o1, o2], [yy, yy], color="#ccc", lw=0.8, zorder=0)
        if np.isfinite(o1): a.scatter(o1, yy, facecolors="white", edgecolors=c, marker=mk, s=28, lw=1.1)
        if np.isfinite(o2): a.scatter(o2, yy, color=c, marker=mk, s=28)
        yl.append(yy); yt.append(f"{reg}: {LAB.get(m, m)}")
a.set_yticks(yl); a.set_yticklabels(yt, fontsize=6.3); a.set_xscale("log"); a.invert_yaxis()
a.set_xlabel("attraction at the onset of order (ψ$_6$ = 0.8)")
a.scatter([], [], facecolors="white", edgecolors="#555", s=24, label="−B$_2^*$ (full)"); a.scatter([], [], color="#555", s=24, label="−B$_{2,c}^*$ (r ≤ 1.1σ)")
a.legend(frameon=False, fontsize=6.5, loc="lower right")
a.set_title("b  onset of order per model", loc="left", fontsize=8, fontweight="bold")
# c
a = ax[1, 0]
for reg, ls in (("I", "-"), ("III", "--")):
    G = R[R.regime == reg]; sp = []
    for c_ in cuts:
        o = np.array([onset(g, f"c{c_:.2f}") for _, g in G.groupby("model")])
        o = o[np.isfinite(o)]
        sp.append(o.max() / o.min() if len(o) >= 3 else np.nan)
    full = np.array([onset(g, "B2") for _, g in G.groupby("model")]); full = full[np.isfinite(full)]
    a.plot(cuts, sp, color="#333", ls=ls, marker="o", ms=2.5, lw=1.1, label=f"Regime {reg}")
    a.axhline(full.max() / full.min(), color="#999", ls=ls, lw=0.8)
    a.text(1.505, full.max() / full.min(), f"full B$_2^*$ ({reg})", fontsize=6, color="#777", va="center", ha="right")
a.axvspan(1.08, 1.12, color="#fde0c5", zorder=0); a.text(1.1, 1.15, "Lindemann\n~10 %", fontsize=6, ha="center", color="#b35806")
a.set_yscale("log"); a.set_xlabel("cutoff r$_c$ / σ"); a.set_ylabel("spread of the onset across models (max/min)")
a.set_title("c  how well the models collapse vs the cutoff", loc="left", fontsize=8, fontweight="bold"); a.legend(frameon=False, fontsize=6.5, loc="upper left")
# d
a = ax[1, 1]
for m, g in R[R.regime == "I"].groupby("model"):
    g = g.sort_values("A"); c, mk = MOD[m]
    a.errorbar(-g["c1.10"], g.psi6, yerr=g.psd.fillna(0), color=c, marker=mk, ms=3.5, lw=1, capsize=1.5, label=LAB.get(m, m))
    a.plot(-g["B2"], g.psi6, color=c, lw=0.7, ls=":", alpha=0.7)
a.set_xscale("log"); a.axhline(0.8, color="#999", ls="--", lw=0.6)
a.set_xlabel("−B$_{2,c}^*$ (solid) · −B$_2^*$ (dotted, LJ 12-6 shifted right)"); a.set_ylabel("ψ$_6$ after 10 τ$_B$")
a.set_title("d  Regime I: order collapses on B$_{2,c}^*$", loc="left", fontsize=8, fontweight="bold"); a.legend(frameon=False, fontsize=6.3, loc="lower right")
for e in ("png", "pdf"): fig.savefig(f"{out}/FigB2c_contact-shell-virial_structure.{e}", dpi=300)

"""Fig. 7: range, not form, at matched B2* in all four regimes (2D, N = 144, 3 seeds, mean +- SD).
(a) Regime I and (b) Regime III: psi6 after 10 tauB against the state point A (all models at the B2* of LJ 12-6 at A).
(c) Regime II: psi6 against time at d_n = 2000 nm, A = 3; (d) Regime II: psi6 at 10 tauB against d_n.
(e) Regime IV: time at which the largest cluster falls below 50 %, (f) psi6 of the fragments at 3 tauB; A = 3.
Usage: python fig_range_form_all.py <_tools dir with the four csv files> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
T, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
def norm(m): return m.replace("AO q = ", "AO q=").replace("-", "–") if "LJ" in m and "2n" in m else m.replace("AO q = ", "AO q=")
STY = {"LJ 12-6": ("#2a78d6", "o", "LJ 12-6 (λ* 1.26–1.38)"), "AO q=0.05": ("#eb6834", "s", "AO q = 0.05"),
       "2n–n LJ": ("#1baf7a", "^", "2n–n LJ"), "square well": ("#d39200", "D", "square well"), "Yukawa": ("#e87ba4", "v", "Yukawa"),
       "AO q=0.075": ("#a33b20", "s", "AO q = 0.075"), "AO q=0.2": ("#7a2a9a", "P", "AO q = 0.2 (λ* 1.03)")}
ORDER = ["LJ 12-6", "AO q=0.05", "2n–n LJ", "square well", "Yukawa", "AO q=0.075", "AO q=0.2"]
B2 = {1: -0.56, 2: -3.6, 3: -10.0, 5: -59}
fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
def endplot(a, f, t_end, title, amax=5.2):
    d = pd.read_csv(os.path.join(T, f)); d["model"] = d.model.map(norm)
    d = d[np.isclose(d.t_tauB, t_end) & (d.A <= amax)]
    for i, m in enumerate(ORDER):
        g = d[d.model == m].groupby("A").psi6.agg(["mean", "std"])
        if g.empty: continue
        c, mk, lab = STY[m]
        a.errorbar(g.index + (i - 3) * 0.03, g["mean"], yerr=g["std"], color=c, marker=mk, ms=3.2, lw=1.0, capsize=0, elinewidth=0.6, label=lab)
    a.axhline(0.8, color="#999", ls="--", lw=0.6); a.set_ylim(0.45, 1.0); a.set_xlim(-0.2, amax)
    a.set_xlabel("state point A (all models at the B$_2^*$ of LJ at A)"); a.set_ylabel("ψ$_6$ after 10 τ$_B$")
    sec = a.secondary_xaxis("top"); sec.set_xticks(list(B2)); sec.set_xticklabels([f"{v:g}" for v in B2.values()], fontsize=6)
    sec.set_xlabel("B$_2^*$", fontsize=6.5, labelpad=1)
    a.set_title(title, loc="left", fontweight="bold", fontsize=7.5, pad=16)
endplot(ax[0, 0], "states_RegimeI.csv", 10.0, "a   Regime I (no field)")
endplot(ax[0, 1], "states_RegimeIII.csv", 10.0, "b   Regime III (field off)")
ax[0, 1].legend(frameon=False, fontsize=5.6, loc="lower right")
# Regime II
d = pd.read_csv(os.path.join(T, "regimeII_rangeform_series.csv")); d["model"] = d.model.map(norm)
a = ax[0, 2]
for m in ORDER:
    g = d[(d.model == m) & (d.dn == 2000)].groupby("t_tauB").psi6.agg(["mean", "min", "max"])
    if g.empty: continue
    c = STY[m][0]; a.fill_between(g.index, g["min"], g["max"], color=c, alpha=0.15, lw=0); a.plot(g.index, g["mean"], color=c, lw=1.1)
a.axhline(0.8, color="#999", ls="--", lw=0.6); a.set_ylim(0.45, 1.0); a.set_xlabel("t in the field (τ$_B$)"); a.set_ylabel("ψ$_6$")
a.set_title("c   Regime II, d$_n$ = 2000 nm, A = 3", loc="left", fontweight="bold", fontsize=7.5, pad=16)
a = ax[1, 0]; e = d[np.isclose(d.t_tauB, 10.0)]
for i, m in enumerate([x for x in ORDER if x in e.model.unique()]):
    g = e[e.model == m].groupby("dn").psi6.agg(["mean", "std"]); c, mk, lab = STY[m]
    a.errorbar(g.index + (i - 1.5) * 40, g["mean"], yerr=g["std"], color=c, marker=mk, ms=3.2, lw=1.0, capsize=0, elinewidth=0.6)
a.axhline(0.8, color="#999", ls="--", lw=0.6); a.set_ylim(0.75, 1.0); a.set_xticks([2000, 3000, 4000])
a.set_xlabel("field amplitude d$_n$ (nm)"); a.set_ylabel("ψ$_6$ after 10 τ$_B$")
a.set_title("d   Regime II, A = 3", loc="left", fontweight="bold", fontsize=7.5)
# Regime IV
b = pd.read_csv(os.path.join(T, "breakup_RegimeIV.csv")); b["model"] = b.model.map(norm)
ms = [x for x in ORDER if x in b.model.unique()]
for a, col, lab, ttl in ((ax[1, 1], "t_break_tauB", "break-up time (τ$_B$)", "e   Regime IV: break-up, A = 3"),
                         (ax[1, 2], "psi6_end", "ψ$_6$ of the fragments at 3 τ$_B$", "f   Regime IV: fragment order")):
    for i, m in enumerate(ms):
        c, mk, _ = STY[m]
        for F, off, alpha in ((1000, -0.17, 0.45), (3000, 0.17, 1.0)):
            v = b[(b.model == m) & (b.F == F)][col].dropna()
            a.scatter(np.full(len(v), i + off), v, s=10, color=c, alpha=alpha, marker=mk, lw=0)
            if len(v): a.errorbar(i + off, v.mean(), yerr=v.std(), color="k", marker="_", ms=7, lw=0.7, capsize=0)
    a.set_xticks(range(len(ms))); a.set_xticklabels([STY[m][2].split(" (")[0] for m in ms], rotation=30, ha="right", fontsize=6)
    a.set_ylabel(lab); a.set_title(ttl, loc="left", fontweight="bold", fontsize=7.5)
ax[1, 2].axhline(0.8, color="#999", ls="--", lw=0.6)
ax[1, 1].text(0.02, 0.97, "pale: F = 1000, full: F = 3000", transform=ax[1, 1].transAxes, fontsize=5.8, va="top")
for e_ in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig7_range-vs-form_RegimesI-IV.{e_}"), dpi=300)
print("saved")

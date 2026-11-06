"""Fig. (paper 2, section 4.x): range vs form in 3D. Regime III (10 tauB after the field is off, from the COMSOL band) for AO
q = 0.05, square well and LJ 12-6 at matched B2*, in a 2D-like chamber (H = 3 um) and a tall one (H = 20 um), PS.
(a) crystal-like fraction, (b) in-plane order of the floor layer, (c, d) 3D views at A = 3, H = 3 um.
Bond cutoff 1.25 r_min (2.5 um for AO and SW, 2.8 um for LJ, whose minimum sits at 2.24 um).
Usage: python fig_range_form_3d.py <night_29.npz> <out dir>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp
S, out = np.load(sys.argv[1]), sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
M = {"AO": ("#eb6834", "s", (2, 2.5, 3), "AO q = 0.05"), "SW": ("#eda100", "D", (2, 2.5, 3), "square well"), "LJ": ("#2a78d6", "o", (2, 3, 4, 5), "LJ 12-6")}
fig = plt.figure(figsize=(7.2, 4.2))
for j, (col, yl, t) in enumerate(((3, "crystal-like fraction", "a  crystal-like, Regime III"), (6, "floor-layer ψ$_6$", "b  in-plane order of the floor layer"))):
    ax = fig.add_axes([0.08 + 0.48 * j, 0.58, 0.38, 0.34])
    for m, (c, mk, As, l) in M.items():
        for H, ls, mf in ((3, "-", c), (20, "--", "white")):
            y = [S[f"AAF|{m}|{A:g}|{H}|ser"][-1, col] for A in As]
            ax.plot(As, y, color=c, ls=ls, marker=mk, ms=3.5, mfc=mf, lw=1.2, label=f"{l}, H = {H} µm")
    ax.set_xlabel("attraction A (k$_B$T), matched B$_2^*$"); ax.set_ylabel(yl); ax.set_title(t, **lab); ax.set_xticks([2, 2.5, 3, 4, 5])
    if j == 1: ax.legend(fontsize=5.3, frameon=False, ncol=2, loc="lower right")
    ax.set_ylim(0, 0.85 if j == 0 else 1.02)
for j, (m, t) in enumerate((("AO", "c  AO q = 0.05, A = 3, H = 3 µm"), ("LJ", "d  LJ 12-6, A = 3, H = 3 µm"))):
    a = fig.add_axes([0.0 + 0.5 * j, 0.0, 0.5, 0.46], projection="3d"); k = f"AAF|{m}|3|3"; P = S[k + "|snap"]; fl = S[k + "|xf"]
    persp(a, P, fl, ((-25, 25), (-12, 12), (0, 3)), t, s=9, zoom=1.45, elev=62, azim=-90, note=f"crystal-like {fl.mean():.0%}, 10 τ$_B$ after switch-off")
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_range-vs-form_3D_RegimeIII_H3-H20.{e}", dpi=300)
print("saved")

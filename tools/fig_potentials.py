"""Fig. 1b (paper 2): pair potentials of all attraction models at equal B2* (A = 2 and 3), Coulomb core included."""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1]); import airpods_bd as bd
from verify_ao import potential_from_force, b2_reduced
M = json.load(open(os.path.join(sys.argv[1], "model_match_q0.05.json"))); out = sys.argv[2]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
r = np.linspace(1.5, 10, 200001)
core = bd.radial_force(r, 0.0, None)
def U(fr): return potential_from_force(r, fr)
MOD = [("LJ 12-6", "#2a78d6", "-", lambda A: bd.radial_force(r, A, None)),
       ("AO q = 0.05", "#eb6834", "-", lambda A: bd.radial_force(r, 0.0, (0.05, M["points"][str(A)]["phi_p"]))),
       ("2n–n LJ (n = 57.6)", "#1baf7a", "--", lambda A: core + bd.att_force(r, {"model": "LJ2n", "eps": M["LJ2n"]["eps"][str(A)], "n": M["LJ2n"]["n"]})),
       ("square well (w = 33 nm)", "#eda100", "-.", lambda A: core + bd.att_force(r, {"model": "SW", "eps": M["SW"]["eps"][str(A)], "width": M["SW"]["width"]})),
       ("Yukawa (1/κ = 58 nm)", "#e87ba4", ":", lambda A: core + bd.att_force(r, {"model": "YUK", "eps": M["YUK"]["eps"][str(A)], "kappa": M["YUK"]["kappa"]}))]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
A = 2
RC = 2.2  # contact shell r_c = 1.1 sigma (sigma = 2a = 2 um)
for lab, col, ls, f in MOD:
    u = U(f(A)); b2 = b2_reduced(r, u); e = -u[r > 1.9].min(); lam = np.sqrt(1 + (1 - b2) / (np.exp(e) - 1))
    sh = r <= RC; b2c = b2_reduced(r[sh], u[sh])
    ax[0].plot(r - 2, u, color=col, ls=ls, lw=1.6, label=f"{lab}: B$_2^*$ = {b2:.2f}, B$_{{2,c}}^*$ = {b2c:.2f}, λ* = {lam:.3f}")
    ax[1].plot(r - 2, u, color=col, ls=ls, lw=1.6)
ax[0].axvspan(-0.02, RC - 2, color="#ddd", alpha=0.6, lw=0, zorder=0)
ax[0].axvline(RC - 2, color="#555", ls="--", lw=0.7)
ax[0].text(RC - 2 + 0.03, 1.05, "r$_c$ = 1.1σ: B$_{2,c}^*$ counts only\nthe attraction left of this line", fontsize=6.3, color="#444", va="top")
ax[0].set_xlim(-0.02, 2.5); ax[0].set_ylim(-5.5, 1.5); ax[0].axhline(0, color="#999", lw=0.5)
ax[0].set_xlabel("surface separation r − 2a (µm)"); ax[0].set_ylabel("U(r) / k$_B$T")
ax[0].legend(frameon=False, fontsize=6.3, loc="lower right")
ax[0].set_title(f"a  All models at the B$_2^*$ of LJ A = {A}", loc="left", fontsize=8.5, fontweight="bold")
ax[1].set_xlim(-0.005, 0.15); ax[1].set_ylim(-5.5, 1.5); ax[1].axhline(0, color="#999", lw=0.5)
ax[1].set_xlabel("r − 2a (µm)"); ax[1].set_title("b  Close-up: the short-range models (all inside r$_c$)", loc="left", fontsize=8.5, fontweight="bold")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig1b_pair-potentials_equal-B2.{ext}"), dpi=300)
print("saved")

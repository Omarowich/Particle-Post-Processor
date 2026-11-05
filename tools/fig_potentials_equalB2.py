"""Fig. 2: pair potentials of the five attraction models at the same B2* (that of LJ 12-6 at A = 2), Coulomb core
included. (a) full range; (b) close-up of the short-range models. Usage: python fig_potentials_equalB2.py <simulation dir> <out dir>"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SIM, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, SIM); import airpods_bd as bd
from verify_ao import potential_from_force, b2_reduced
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
M = json.load(open(os.path.join(SIM, "model_match_q0.05.json")))
r = np.linspace(1.5, 10, 200001); core = bd.radial_force(r, 0.0, None)
phi = lambda A: M["AO_phi"][str(A)] if not isinstance(M["AO_phi"][str(A)], dict) else M["AO_phi"][str(A)]["phi_p"]
MOD = [("LJ 12-6", "#2a78d6", "-", lambda A: bd.radial_force(r, A, None)),
       ("AO, q = 0.05", "#eb6834", "-", lambda A: bd.radial_force(r, 0.0, (0.05, phi(A)))),
       ("2n–n LJ, n = 57.6", "#1baf7a", "--", lambda A: core + bd.att_force(r, {"model": "LJ2n", "eps": M["LJ2n"]["eps"][str(A)], "n": M["LJ2n"]["n"]})),
       ("square well, w = 33 nm", "#d39200", "-.", lambda A: core + bd.att_force(r, {"model": "SW", "eps": M["SW"]["eps"][str(A)], "width": M["SW"]["width"]})),
       ("Yukawa, 1/κ = 58 nm", "#e87ba4", ":", lambda A: core + bd.att_force(r, {"model": "YUK", "eps": M["YUK"]["eps"][str(A)], "kappa": M["YUK"]["kappa"]}))]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
(d, e), (g, h) = axs
A = 2
for lab, col, ls, f in MOD:
    u = potential_from_force(r, f(A)); b2 = b2_reduced(r, u); eps = -u[r > 1.9].min(); lam = np.sqrt(1 + (1 - b2) / (np.exp(eps) - 1))
    d.plot(r - 2, u, color=col, ls=ls, lw=1.4, label=f"{lab}  (λ* = {lam:.3f})"); e.plot(r - 2, u, color=col, ls=ls, lw=1.4)
for ax in (d, e): ax.axhline(0, color="#999", lw=0.5); ax.set_ylim(-5.5, 1.5)
d.set_xlim(-0.02, 2.5); d.set_xlabel("surface separation r − 2a (µm)"); d.set_ylabel("U(r) / k$_B$T")
d.legend(frameon=False, fontsize=6.2, loc="lower right", title=f"all at B$_2^*$ = {b2:.2f}", title_fontsize=6.4)
e.set_xlim(-0.005, 0.15); e.set_xlabel("r − 2a (µm)")
d.set_title("a   all models at equal B$_2^*$", loc="left", fontsize=7.5, fontweight="bold")
e.set_title("b   close-up of the short-range models", loc="left", fontsize=7.5, fontweight="bold")
# (c, d) how core and attraction add inside contact: AO at A = 3 (phi_p = 0.182), Coulomb vs PHS core
ao = (0.05, 0.182)
for core, col, nm in (("coulomb", "#2a78d6", "Coulomb core"), ("phs", "#7a2a9a", "PHS core")):
    Fc = bd.radial_force(r, 0.0, None, core); Ft = bd.radial_force(r, 0.0, ao, core); Fa = Ft - Fc
    u = potential_from_force(r, Ft); i = u.argmin()
    g.plot(r - 2, Fc, color=col, ls="--", lw=1.1, label=f"{nm}: repulsion")
    g.plot(r - 2, Ft, color=col, lw=1.5, label=f"{nm}: total")
    h.plot(r - 2, u, color=col, lw=1.5, label=f"{nm}: depth {-u[i]:.1f} k$_B$T")
    h.plot(r[i] - 2, u[i], "o", color=col, ms=4)
g.plot(r - 2, Fa, color="#eb6834", lw=1.5, label="AO attraction (held at contact value for r < 2a)")
for ax in (g, h):
    ax.axvline(0, color="#999", ls=":", lw=0.7); ax.axhline(0, color="#999", lw=0.5); ax.set_xlim(-0.06, 0.12)
    ax.set_xlabel("r − 2a (µm)")
g.set_ylim(-150, 200); g.set_ylabel("radial force (k$_B$T/µm, + = repulsive)")
g.legend(frameon=False, fontsize=5.8, loc="upper right")
h.set_ylim(-10, 4); h.set_ylabel("U(r) / k$_B$T"); h.legend(frameon=False, fontsize=6.2, loc="upper right")
g.set_title("c   forces add: AO (A = 3) with each core", loc="left", fontsize=7.5, fontweight="bold")
h.set_title("d   resulting well", loc="left", fontsize=7.5, fontweight="bold")
h.annotate("pair slides 42 nm inside contact", xy=(-0.042, -8.66), xytext=(-0.05, -3.5), fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig2_pair-potentials_equal-B2_and_core.{ext}"), dpi=300)
print("saved")

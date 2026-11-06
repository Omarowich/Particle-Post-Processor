"""Fig. (paper 2, section 4.2): a lossy lid flattens the trap. (a) F_y along the floor (z = 1 um) for the side-wall drive with
a PDMS lid at H = 4-20 um (water) and the old H = 6 um PDMS field; glass for reference. (b) where the particles end up
(y histogram at the end, Regime II, A = 3, d_n = 4000 nm). (c-e) 3D views.
Usage: python fig_flat_trap.py <_fields dir> <night_32.npz> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp, zcolor
FD, S, out = sys.argv[1], np.load(sys.argv[2]), sys.argv[3]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
fig = plt.figure(figsize=(7.2, 4.4))
ax = fig.add_axes([0.07, 0.60, 0.40, 0.33])
cols = {4: "#eb6834", 6: "#e34948", 10: "#7a2a9a", 20: "#33415c"}
for H, c in cols.items():
    d = np.load(os.path.join(FD, f"auto_side20_H{H}_box50_pdms_water.npz")); ys, zs = d["ys"] * 1e6, d["zs"] * 1e6
    ax.plot(ys, d["Fy"][:, np.argmin(abs(zs - 1))], color=c, lw=1.2, label=f"PDMS, H = {H} µm")
d = np.load(os.path.join(FD, "auto_side20_H4_box50_glass_water.npz")); ys, zs = d["ys"] * 1e6, d["zs"] * 1e6
ax.plot(ys, d["Fy"][:, np.argmin(abs(zs - 1))], color="#999", lw=1.0, ls=":", label="glass, H = 4 µm")
f = np.load(os.path.join(FD, "field_H6_impedance_PS.npz")); ys, zs, F = f["ys"] * 1e6, f["zs"] * 1e6, f["F"]
s = 18.4 / np.abs(F[:, :, np.argmin(abs(zs - zs.max() / 2)), 1]).max(); ix = len(f["xs"]) // 2
ax.plot(ys, F[ix, :, np.argmin(abs(zs - 1)), 1] * s, color="#e34948", lw=1.2, ls="--", label="PDMS, H = 6 µm, old field")
ax.axhspan(-1, 1, color="#2a78d6", alpha=0.10, lw=0); ax.text(-24, -3.5, "|F$_y$| < 1 k$_B$T/µm", fontsize=5.6, color="#2a78d6")
ax.set_xlim(-25, 25); ax.set_ylim(-20, 20); ax.set_xlabel("y (µm)"); ax.set_ylabel("F$_y$ at z = 1 µm (k$_B$T/µm)")
ax.legend(fontsize=5.0, frameon=False, loc="upper right", ncol=1); ax.set_title("a  force along the floor", **lab)
ax = fig.add_axes([0.57, 0.60, 0.41, 0.33]); yb = np.arange(-25, 26, 2.5); yc = (yb[:-1] + yb[1:]) / 2
for k, c, ls, l in (("H6_drho52_PDMSlid", "#e34948", "--", "old field, 5 τ$_B$"), ("H6_drho52_PDMSlid_long", "#7f1d1d", "--", "old field, 20 τ$_B$"),
                     ("H6_drho52_PDMSlid_noattr", "#f4a3a3", ":", "old field, no attraction"), ("H4_drho52_sidepdmsradonlyW", "#eb6834", "-", "water, H = 4 µm"),
                     ("H6_drho52_sidepdmsradonlyW", "#e34948", "-", "water, H = 6 µm"), ("H20_drho52_sidepdmsradonlyW", "#33415c", "-", "water, H = 20 µm")):
    ax.step(yc, S[k + "|yhist"], where="mid", color=c, ls=ls, lw=1.1, label=l)
ax.set_xlim(-25, 25); ax.set_xlabel("y (µm)"); ax.set_ylabel("particles per 2.5 µm"); ax.legend(fontsize=5.2, frameon=False, loc="upper left")
ax.set_title("b  where the particles end up (PDMS lid)", **lab)
for j, (k, t) in enumerate((("H6_drho52_PDMSlid", "c  old field, H = 6 µm, 5 τ$_B$"), ("H6_drho52_PDMSlid_long", "d  same, 20 τ$_B$"),
                            ("H4_drho52_sidepdmsradonlyW", "e  water, H = 4 µm, 5 τ$_B$"))):
    a = fig.add_axes([0.0 + 0.333 * j, 0.0, 0.333, 0.50], projection="3d"); P = S[k + "|snap"]; H = 4 if "H4" in k else 6
    persp(a, P, np.zeros(len(P), bool), ((-25, 25), (-25, 25), (0, H)), t, s=7, color=zcolor(P, 3.5), zoom=1.3, elev=58, azim=-90)
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_flat-trap_PDMS-lid_two-side-bands.{e}", dpi=300)
print("saved")

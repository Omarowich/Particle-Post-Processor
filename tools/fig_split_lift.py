"""Fig. (paper 2, section 4.2): how the side-wall streaming splits the band and then lifts it (legacy fluid set, H = 20 um,
streaming only). (a-d) 3D views at 0, 0.06, 0.3 and 1 tauB, colour = height; (e) streaming along the floor and at z = 3 um
across the band; (f) fraction at the node and fraction above z = 3 um against time, with thermal noise
(3 seeds) and without (noise-free control).
Usage: python fig_split_lift.py <night_split_frames.npz> <streaming18_series.npz> <split_nonoise_series.npz> <auto_side20_H20_box50_glass.npz> <out>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp, zcolor
FR, S18, SN, FL, out = (np.load(a) if a.endswith(".npz") else a for a in sys.argv[1:6])
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
fig = plt.figure(figsize=(7.2, 4.3))
for j, T in enumerate(("0", "0.06", "0.3", "1")):
    a = fig.add_axes([0.25 * j - 0.01, 0.50, 0.27, 0.50], projection="3d"); P = FR[f"stream_noise|{T}"]
    persp(a, P, np.zeros(len(P), bool), ((-25, 25), (-25, 25), (0, 10)), f"{'abcd'[j]}  {T} τ$_B$", s=6, color=zcolor(P, 8.0), zoom=1.25, elev=18, azim=-20,
          note=f"at node {np.mean(abs(P[:, 1]) < 3):.0%}, above 3 µm {np.mean(P[:, 2] > 3):.0%}")
ax = fig.add_axes([0.07, 0.09, 0.36, 0.33])
ys, zs, U, W = FL["ys"] * 1e6, FL["zs"] * 1e6, FL["U"], FL["W"]
for z, ls in ((1.0, "-"), (3.0, "--")):
    iz = np.argmin(abs(zs - z)); ax.plot(ys, U[:, iz], color="#2a78d6", ls=ls, lw=1.2, label=f"u$_y$, z = {z:g} µm"); ax.plot(ys, W[:, iz], color="#e34948", ls=ls, lw=1.2, label=f"u$_z$, z = {z:g} µm")
ax.axhline(0, color="#999", lw=0.5); ax.axvline(0, color="#999", lw=0.5, ls=":"); ax.set_xlim(-25, 25)
ax.set_xlabel("y (µm), node at 0"); ax.set_ylabel("streaming (µm/s)"); ax.legend(fontsize=5.4, frameon=False, ncol=2, loc="lower right")
ax.set_title("e  flow at the node: down, then outward", **lab)
ax = fig.add_axes([0.55, 0.09, 0.43, 0.33])
for q, c, l in (("node", "#2a78d6", "at node (|y| < 3 µm)"), ("up", "#e34948", "above z = 3 µm")):
    for s in "123":
        k = f"sideglass|20|streamonly|legacy|{s}"; ax.plot(S18[k + "|t"], S18[k + "|" + q], color=c, lw=0.9, alpha=0.8, label=l if s == "1" else None)
    ax.plot(SN["streamonly|t"], SN["streamonly|" + q], color=c, lw=1.3, ls="--")
ax.plot([], [], color="#555", lw=0.9, label="thermal noise (3 seeds)"); ax.plot([], [], color="#555", lw=1.3, ls="--", label="no noise")
ax.set_xscale("symlog", linthresh=0.1); ax.set_xlim(0, 3); ax.set_xticks([0, 0.05, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0", "0.05", "0.1", "0.3", "1", "3"])
ax.set_ylim(0, 1.02); ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("fraction"); ax.legend(fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.set_title("f  split first, lift later; noise plays no part", **lab)
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_streaming-split-then-lift_side-wall-H20.{e}", dpi=300)
print("saved")

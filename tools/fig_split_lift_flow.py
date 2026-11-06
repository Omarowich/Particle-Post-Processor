"""Fig. (paper 2, section 4.2): the side-wall streaming in a thin and a tall chamber, and what it does to the band
(legacy fluid set, glass lid, streaming only, AO q = 0.05, A = 3). Rows: H = 4 and 20 um. (a, d) streaming in the y-z
cross-section (streamlines, colour = speed) with the paths of the centres of the two band halves (y < 0, y > 0) from
0 to 3 tauB; (b, e) band width std(y) and mean height against time; (c, f) 3D view at 3 tauB, colour = height.
Usage: python fig_split_lift_flow.py <split_lift_H4_H20_frames.npz> <_fields dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp, zcolor
S, FD, out = np.load(sys.argv[1]), sys.argv[2], sys.argv[3]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
fig = plt.figure(figsize=(7.2, 5.4))
for r, (H, top) in enumerate(((4, 0.57), (20, 0.095))):
    d = np.load(os.path.join(FD, f"auto_side20_H{H}_box50_glass.npz")); ys, zs, U, W = d["ys"] * 1e6, d["zs"] * 1e6, d["U"], d["W"]
    sp = np.hypot(U, W); h = 0.40 if H == 20 else 0.20
    ax = fig.add_axes([0.07, top + (0.0 if H == 20 else 0.12), 0.40, h])
    ax.pcolormesh(ys, zs, sp.T, cmap="Blues", vmin=0, vmax=4.2, shading="auto", rasterized=True)
    ax.streamplot(ys, zs, U.T, W.T, color="#33415c", density=(1.4, 0.7 if H == 20 else 0.45), linewidth=0.45, arrowsize=0.5, broken_streamlines=False)
    t = S[f"H{H}|t"]
    for side, c in (("L", "#e34948"), ("R", "#eb6834")):
        cm = S[f"H{H}|com{side}"]; ax.plot(cm[:, 0], cm[:, 1], color=c, lw=1.4); ax.scatter(cm[0, 0], cm[0, 1], color=c, s=12, marker="o", zorder=4)
        ax.scatter(cm[-1, 0], cm[-1, 1], color=c, s=18, marker=">", zorder=4)
    P = S[f"H{H}|3"]; ax.scatter(P[:, 1], P[:, 2], s=2, color="#e34948", alpha=0.35, zorder=3)
    ax.axvline(0, color="#999", ls=":", lw=0.6); ax.set_xlim(-25, 25); ax.set_ylim(0, H); ax.set_xlabel("y (µm), node at 0"); ax.set_ylabel("z (µm)")
    ax.set_title(f"{'ad'[r]}  H = {H} µm: streaming, paths of the two halves (0 → 3 τ$_B$)", **lab)
    ax = fig.add_axes([0.56, top + 0.05, 0.17, 0.30])
    ax.plot(t, S[f"H{H}|width"], color="#2a78d6", lw=1.2, label="band width std(y)"); ax.plot(t, S[f"H{H}|zmean"], color="#e34948", lw=1.2, label="mean height")
    ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("µm"); ax.set_xlim(0, 3); ax.legend(fontsize=5.3, frameon=False, loc="upper left")
    ax.set_title(f"{'be'[r]}  width and height", **lab)
    a = fig.add_axes([0.74, top - 0.02, 0.26, 0.46], projection="3d"); P = S[f"H{H}|3"]
    persp(a, P, np.zeros(len(P), bool), ((-25, 25), (-25, 25), (0, H)), f"{'cf'[r]}  3 τ$_B$", color=zcolor(P, max(3.0, H * 0.4)), zoom=1.2, elev=20, azim=-25)
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_streaming-thin-vs-tall_converge-vs-split.{e}", dpi=300)
print("saved")

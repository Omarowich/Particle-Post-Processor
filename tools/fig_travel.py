import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/particle-post-processor/simulation"); import streaming2d as s2
out = sys.argv[1]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(2, 3, figsize=(7.4, 3.8), constrained_layout=True)
for r_, (drive, lab, bh, fn) in enumerate((("idt", "two IDTs (standing SAW), 100 µm", 50.0, "stream_idt20_H20_box100_absorb_E.npz"),
                                            ("idt_travel", "one IDT (travelling SAW), 400 µm", 200.0, "stream_idt_travel20_H20_box400_absorb.npz"))):
    ys, zs, p, om, rho0, c0 = s2.first_order(drive, 20.0, bh, 20.0, "absorb")
    d = np.load("/home/claude/sim/stream/" + fn)
    Y, Z = ys * 1e6, zs * 1e6; P = np.abs(p).T
    a = ax[r_, 0]; a.imshow(P / P.max(), extent=(Y[0], Y[-1], 0, Z[-1]), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    a.set_title(f"{'ad'[r_]}  |p|, {lab}", loc="left", fontsize=7, fontweight="bold"); a.set_ylabel("z (µm)")
    a = ax[r_, 1]; Fy, Fz = d["Fy"], d["Fz"]; sy, sz = 6, 6
    a.imshow(np.hypot(Fy, Fz).T, extent=(Y[0], Y[-1], 0, Z[-1]), origin="lower", aspect="auto", cmap="Blues")
    sy = max(1, len(Y) // 16); sz = 6
    a.quiver(Y[::sy][:, None].repeat(len(Z[::sz]), 1).T, Z[::sz][:, None].repeat(len(Y[::sy]), 1), Fy[::sy, ::sz].T, Fz[::sy, ::sz].T, color="#2a78d6", width=0.005)
    a.set_title(f"{'be'[r_]}  radiation force (max {np.hypot(Fy, Fz).max():.0f} k$_B$T/µm)", loc="left", fontsize=7.5, fontweight="bold")
    a = ax[r_, 2]; U, W = d["U"], d["W"]
    a.streamplot(Y, Z, U.T, W.T, color=np.hypot(U, W).T, cmap="Oranges", density=1.0, linewidth=0.8, arrowsize=0.6)
    a.set_xlim(Y[0], Y[-1]); a.set_ylim(0, Z[-1])
    a.set_title(f"{'cf'[r_]}  streaming (max {np.hypot(U, W).max():.1f} µm/s)", loc="left", fontsize=7.5, fontweight="bold")
for a in ax[1]: a.set_xlabel("y (µm)")
for e in ("png", "pdf"): fig.savefig(f"{out}/FigS_standing-vs-travelling-SAW_streaming.{e}", dpi=300)

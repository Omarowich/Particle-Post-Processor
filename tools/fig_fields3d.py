"""Fig. 4: the 3D chamber and its computed radiation-force fields (water, H = 20 um).
(a, d) cross-section schematics of the two drives; (b, c, e, f) Gor'kov force on the y-z cross-section for side-wall drive with glass and PDMS lids and
IDT drive with absorbing and glass lids (arrows: (F_y, F_z); colour: |F|), all scaled to the 2D fit (F0 = 18.4 kBT/um).
Usage: python fig_fields3d.py <simulation dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
SIM, out = sys.argv[1], sys.argv[2]; sys.path.insert(0, SIM)
import streaming2d as s2
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
OR, RED, BLU = "#eb6834", "#d62f2f", "#2a78d6"
CASES = [("side", "glass", 25, "side wall, glass lid"), ("side", "pdms", 25, "side wall, PDMS lid"),
         ("idt", "absorb", 50, "IDT, absorbing lid"), ("idt", "glass", 50, "IDT, glass lid")]
H = 20.0
R = [s2.compute(d, H_um=H, box_half_um=bh, freq_MHz=20.0, lid=l, fluid="water") for d, l, bh, _ in CASES]
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], hspace=0.5, wspace=0.28)
from matplotlib.patches import Rectangle
def schematic(ax, idt, letter):
    W = 50 if idt else 25
    ax.add_patch(Rectangle((-W, 0), 2*W, H, fc="#d4e9f7", ec="none"))
    ax.plot([-W, W], [H, H], color="#777", lw=2.5, solid_capstyle="butt")
    ax.text(0, H + 2.5, "lid: glass / PDMS / absorbing", ha="center", fontsize=6, color="#555")
    rng = np.random.default_rng(3); xs = rng.uniform(-0.25*W, 0.25*W, 9)
    ax.scatter(np.sort(xs)*0 + np.linspace(-0.2*W, 0.2*W, 9), np.full(9, 1.0), s=7, color="#1f3a5f", zorder=3)
    ax.plot([0, 0], [0, H], color="#666", ls=":", lw=0.6)
    ax.annotate("", xy=(0.82*W, H*0.55), xytext=(0.82*W, H*0.85), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="k"))
    ax.text(0.86*W, H*0.7, "g", fontsize=7)
    if not idt:
        ax.plot([-W, W], [0, 0], color="#333", lw=2.0, solid_capstyle="butt")
        for s_ in (-1, 1):
            ax.plot([s_*W, s_*W], [0, H], color=OR, lw=4, solid_capstyle="butt")
            ax.annotate("", xy=(s_*(W + 2), H/2), xytext=(s_*(W + 11), H/2), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2, mutation_scale=9, shrinkA=0, shrinkB=0))
        ax.text(0, -6, "side walls vibrate along y (d$_n$)", ha="center", fontsize=6, color=OR)
        ax.set_xlim(-W - 12, W + 12)
    else:
        ax.add_patch(Rectangle((-W - 20, -5), 2*W + 40, 5, fc="#d9cfe6", ec="none"))
        ax.plot([-W, -W], [0, H], color="#333", lw=1.2); ax.plot([W, W], [0, H], color="#333", lw=1.2)
        yy = np.linspace(-W, W, 300); ax.plot(yy, 0.8*np.sin(2*np.pi*yy/(2*W)), color=RED, lw=0.8)
        for s_ in (-1, 1):
            for k in range(4): ax.plot([s_*(W + 6 + 3*k)]*2, [-0.5, 1.5], color="#b8860b", lw=1.0)
            ax.text(s_*(W + 10), 3.5, "IDT", ha="center", fontsize=6, color="#b8860b")
        th = np.radians(21.8)
        for y0, sgn in ((-30, 1), (30, -1)):
            ax.annotate("", xy=(y0 + sgn*12*np.tan(th), 12), xytext=(y0, 0.5), arrowprops=dict(arrowstyle="-|>", color=BLU, lw=0.8))
        ax.text(-12, 7, "θ$_R$", color=BLU, fontsize=6.5)
        ax.text(0, -10, "LiNbO$_3$ floor carries a standing SAW", ha="center", fontsize=6, color="#6a4c93")
        ax.set_xlim(-W - 20, W + 20)
    ax.set_ylim(-13, H + 5); ax.set_aspect("auto"); ax.set_axis_off()
    ax.set_title(f"{letter}   {'IDT drive' if idt else 'side-wall drive'}", loc="left", fontsize=7.2, fontweight="bold")
schematic(fig.add_subplot(gs[0, 0]), False, "a"); schematic(fig.add_subplot(gs[1, 0]), True, "d")
# (b-e) force fields
vmax = 40
for i, ((d, l, bh, title), r) in enumerate(zip(CASES, R)):
    ax = fig.add_subplot(gs[i // 2, 1 + i % 2]); L_ = "bcef"[i]
    y, z = r["ys"] * 1e6, r["zs"] * 1e6; Fy, Fz = r["Fy"], r["Fz"]; M = np.hypot(Fy, Fz)
    im = ax.pcolormesh(y, z, M.T, cmap="Reds", vmin=0, vmax=vmax, shading="auto", rasterized=True)
    sy, sz = max(1, len(y) // 14), max(1, len(z) // 6)
    Y, Z = np.meshgrid(y[::sy], z[::sz], indexing="ij")
    ax.quiver(Y, Z, Fy[::sy, ::sz], Fz[::sy, ::sz], color="k", scale=300, width=0.006)
    ax.axvline(0, color="#555", ls=":", lw=0.6)
    ax.set_xlim(y[0], y[-1]); ax.set_ylim(0, H); ax.set_aspect("auto")
    ax.set_title(f"{L_}   {title}", loc="left", fontsize=7.2, fontweight="bold")
    if i % 2 == 0: ax.set_ylabel("z (µm)")
    if i // 2 == 1: ax.set_xlabel("y (µm)")
    fz = Fz[np.argmin(np.abs(y)), :]; ax.text(0.98, 0.04, f"F$_z$ at node: {fz.min():+.1f} … {fz.max():+.1f}", transform=ax.transAxes, ha="right", fontsize=5.6,
                                             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
cb = fig.colorbar(im, ax=[x for x in fig.axes if x.get_label() != "<colorbar>"][2:], shrink=0.7, pad=0.02, extend="max"); cb.set_label("|F$_{rad}$| (k$_B$T/µm), d$_n$ = 4000 nm")
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig4_3D-chamber_radiation-force-fields_water_H20.{ext}"), dpi=300, bbox_inches="tight")
print("saved")

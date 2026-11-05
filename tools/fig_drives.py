"""Fig. 2 (paper 2): how the acoustic field is made. Side-wall drive (bulk acoustic wave, the COMSOL set-up) vs IDT
drive (standing surface acoustic wave leaking into the liquid at the Rayleigh angle). Usage: fig_drives.py <demo_fields.npy> <out>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
D = np.load(sys.argv[1], allow_pickle=True).item(); out = sys.argv[2]
c0, cs = 1850.0, 3990.0; th = np.arcsin(c0 / cs)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.4, 4.0), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1], height_ratios=[0.45, 1])
F6 = np.load(sys.argv[3], allow_pickle=True).item()["pdms_H6"] if len(sys.argv) > 3 else None

def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=1.1, **kw))

# ---- a: side-wall schematic
ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.55, 1.25); ax.set_aspect("equal")
ax.add_patch(Rectangle((-1, 0), 2, 0.8, fc="#eaf2fb", ec="none"))
ax.add_patch(Rectangle((-1.12, -0.1), 0.12, 1.0, fc="#999", ec="none")); ax.add_patch(Rectangle((1.0, -0.1), 0.12, 1.0, fc="#999", ec="none"))
ax.add_patch(Rectangle((-1.12, 0.8), 2.24, 0.1, fc="#cfd8dc", ec="none")); ax.add_patch(Rectangle((-1.12, -0.1), 2.24, 0.1, fc="#cfd8dc", ec="none"))
for s in (-1, 1):
    arrow(ax, (s * 1.36, 0.4), (s * 1.16, 0.4), color="#e34948"); arrow(ax, (s * 1.16, 0.25), (s * 1.36, 0.25), color="#e34948")
yy = np.linspace(-1, 1, 200)
for z in (0.15, 0.4, 0.65):
    ax.plot(yy, z + 0.08 * np.cos(np.pi * yy / 2 + np.pi / 2) * 0, color="#2a78d6", lw=0)
ax.plot(yy, 0.4 + 0.3 * np.abs(np.sin(np.pi * yy / 2)), color="#2a78d6", lw=1.2)
ax.plot([0, 0], [0, 0.8], color="#2a78d6", ls="--", lw=0.9); ax.text(0, 0.86, "node", ha="center", fontsize=6.5, color="#2a78d6")
ax.text(0, -0.25, "walls vibrate along y; standing wave\nacross y, nodes λ$_{liquid}$/2 ≈ 50 µm apart",
        ha="center", va="top", fontsize=6.5)
ax.text(0, 1.18, "glass lid", ha="center", fontsize=6.5, color="#555")
ax.set_title("a  side-wall drive (bulk wave)", loc="left", fontsize=8, fontweight="bold")
# ---- b, c: BAW fields
d = D["baw"]; P = d["p"].T
ax = fig.add_subplot(gs[0, 1])
ax.imshow(np.real(P) / np.abs(P).max(), extent=(d["ys"][0], d["ys"][-1], 0, d["zs"][-1]), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("y (µm)"); ax.set_ylabel("z (µm)"); ax.set_title("b  Re p, glass lid", loc="left", fontsize=8, fontweight="bold")
ax = fig.add_subplot(gs[0, 2])
ax.imshow(np.abs(P) / np.abs(P).max(), extent=(d["ys"][0], d["ys"][-1], 0, d["zs"][-1]), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
ax.set_xlabel("y (µm)"); ax.set_title("c  |p|, glass lid", loc="left", fontsize=8, fontweight="bold")
if F6 is not None:
    ax = fig.add_subplot(gs[0, 3]); P6 = np.abs(F6["p"]).T
    ax.imshow(P6 / P6.max(), extent=(F6["ys"][0], F6["ys"][-1], 0, F6["zs"][-1]), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("y (µm)"); ax.set_title("d  |p|, PDMS lid, H = 6", loc="left", fontsize=8, fontweight="bold")
# ---- d: IDT schematic
ax = fig.add_subplot(gs[1, 0]); ax.set_axis_off(); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-0.6, 1.25); ax.set_aspect("equal")
ax.add_patch(Rectangle((-1.4, -0.22), 2.8, 0.22, fc="#d8cfe8", ec="none")); ax.text(0, -0.13, "LiNbO$_3$", ha="center", va="center", fontsize=6.5)
for s in (-1, 1):
    for i in range(4):
        ax.add_patch(Rectangle((s * 1.18 - 0.02 + 0.05 * i * s, 0), 0.025, 0.05, fc="#b8860b", ec="none"))
    ax.text(s * 1.25, 0.1, "IDT", ha="center", fontsize=6.5)
    arrow(ax, (s * 1.05, 0.03), (s * 0.75, 0.03), color="#e34948")
ax.add_patch(Rectangle((-0.7, 0), 1.4, 0.9, fc="#eaf2fb", ec="none"))
ax.add_patch(Rectangle((-0.78, 0.9), 1.56, 0.12, fc="#f6d8c4", ec="none")); ax.text(0, 1.1, "lid (glass or PDMS)", ha="center", fontsize=6.5, color="#555")
yy = np.linspace(-0.7, 0.7, 200); ax.plot(yy, 0.02 * np.sin(2 * np.pi * yy / 0.7), color="#e34948", lw=0.8)
for s, y0 in ((1, -0.65), (-1, 0.65)):
    L = 0.85 / np.cos(th)
    ax.plot([y0, y0 + s * L * np.sin(th)], [0.02, 0.02 + L * np.cos(th)], color="#2a78d6", lw=1.0)
    arrow(ax, (y0 + s * 0.3 * np.sin(th), 0.3 * np.cos(th)), (y0 + s * 0.6 * np.sin(th), 0.6 * np.cos(th)), color="#2a78d6")
ax.text(-0.52, 0.32, f"θ$_R$ = {np.degrees(th):.1f}°", fontsize=6.5, color="#2a78d6")
ax.text(0, -0.3, "SAW leaks into the liquid at sin θ$_R$ = c$_0$/c$_{SAW}$;\nnodes λ$_{SAW}$/2 = 100 µm apart at 20 MHz",
        ha="center", va="top", fontsize=6.5)
ax.set_title("e  IDT drive (surface acoustic wave)", loc="left", fontsize=8, fontweight="bold")
# ---- e, f: SAW fields (tall chamber, absorbing lid vs glass lid)
panels = (("saw_travel", "f  Re p, one IDT", "re"), ("saw_absorb", "g  |p|, absorbing lid", "abs"),
          ("saw_hard", "h  |p|, glass lid", "abs"))
for k, (key, lab, kind) in enumerate(panels):
    d = D[key]; P = d["p"].T
    ax = fig.add_subplot(gs[1, 1 + k])
    img = np.real(P) / np.abs(P).max() if kind == "re" else np.abs(P) / np.abs(P).max()
    ax.imshow(img, extent=(d["ys"][0], d["ys"][-1], 0, d["zs"][-1]), origin="lower", aspect="equal",
              cmap="RdBu_r" if kind == "re" else "magma", vmin=-1 if kind == "re" else 0, vmax=1)
    ax.set_xlim(-100, 100)
    if kind == "abs":
        for yn in (-99.75, 0, 99.75):
            ax.axvline(yn, color="w", ls="--", lw=0.6)
    else:
        for yc in (-70, 10):          # propagation direction: up and towards +y at theta_R from vertical
            ax.annotate("", xy=(yc + 70 * np.tan(th), 110), xytext=(yc, 40), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))
        ax.text(-95, 140, f"θ$_R$ = {np.degrees(th):.1f}°", fontsize=6.5, va="top")
    ax.set_xlabel("y (µm)"); ax.set_title(lab, loc="left", fontsize=8, fontweight="bold")
    if k == 0:
        ax.set_ylabel("z (µm)")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig3_acoustic-drives_side-wall-vs-IDT.{ext}"), dpi=300)

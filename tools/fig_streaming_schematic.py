"""Fig. 2c (paper 2): how the acoustic field drives streaming. (a) side-wall drive: boundary-driven Rayleigh rolls
between floor and lid; (b) IDT drive: boundary streaming at the vibrating floor plus Eckart jets along the leaky beams;
(c) the streaming model used in the 3D runs (bd_engine.rayleigh_streaming), H = 20 um, with the pressure node."""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
sys.path.insert(0, "/home/claude/particle-post-processor/simulation"); import bd_engine as be
out = sys.argv[1]; th = np.arcsin(1850 / 3990)
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
def arr(ax, a, b, c="#2a78d6", lw=1.0, ls="-", ms=8): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, ls=ls))
def roll(ax, cx, cy, rx, ry, cw, c="#eb6834"):
    t = np.linspace(0.15, 2 * np.pi - 0.15, 60) * (1 if cw else -1)
    x, y = cx + rx * np.cos(t), cy + ry * np.sin(t); ax.plot(x, y, color=c, lw=1.0)
    arr(ax, (x[-2], y[-2]), (x[-1], y[-1]), c=c, ms=7)
# a
ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.75, 0.9)
ax.add_patch(Rectangle((-1, 0), 2, 0.75, fc="#eaf2fb", ec="none"))
for y0 in (-0.07, 0.75): ax.add_patch(Rectangle((-1.05, y0), 2.1, 0.07, fc="#cfd8dc", ec="none"))
for s in (-1, 1): ax.add_patch(Rectangle((1.0 if s > 0 else -1.07, -0.07), 0.07, 0.89, fc="#999", ec="none"))
ax.plot([0, 0], [0, 0.75], color="#111", ls=":", lw=0.8)
for cx, cwb in ((-0.75, True), (-0.25, False), (0.25, True), (0.75, False)):
    roll(ax, cx, 0.19, 0.2, 0.14, cwb); roll(ax, cx, 0.56, 0.2, 0.14, not cwb)
ax.text(0, -0.15, "the oscillating flow along floor and lid has a\nthin viscous boundary layer (~0.2 µm); its\ntime-averaged stress drives rolls (Rayleigh)", ha="center", va="top", fontsize=6.3)
ax.set_title("a  side-wall drive", loc="left", fontsize=8, fontweight="bold")
# b
ax = fig.add_subplot(gs[1]); ax.set_axis_off(); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.75, 0.9)
ax.add_patch(Rectangle((-1.3, -0.1), 2.6, 0.1, fc="#d8cfe8", ec="none"))
ax.add_patch(Rectangle((-0.8, 0), 1.6, 0.75, fc="#eaf2fb", ec="none")); ax.add_patch(Rectangle((-0.85, 0.75), 1.7, 0.07, fc="#cfd8dc", ec="none"))
L = 0.75 / np.cos(th)
for s, y0 in ((1, -0.7), (-1, 0.7)):
    ax.plot([y0, y0 + s * L * np.sin(th)], [0, 0.75], color="#2a78d6", lw=4, alpha=0.18)
    arr(ax, (y0 + s * 0.15 * np.sin(th), 0.15), (y0 + s * 0.55 * np.sin(th), 0.55), c="#e34948", lw=1.2)
for cx, cw in ((-0.45, False), (0.45, True)):
    roll(ax, cx, 0.12, 0.18, 0.08, cw, c="#eb6834")
ax.text(0.0, 0.42, "Eckart jets", ha="center", fontsize=6, color="#e34948")
ax.text(0, -0.15, "the vibrating floor drives boundary rolls; sound\nabsorbed along the tilted beams pushes the liquid\nwith them (Eckart); walls and lid close the loop", ha="center", va="top", fontsize=6.3)
ax.set_title("b  IDT drive", loc="left", fontsize=8, fontweight="bold")
# c: model
ax = fig.add_subplot(gs[2])
y = np.linspace(-25, 25, 120); z = np.linspace(0, 20, 60); Y, Z = np.meshgrid(y, z)
P = np.c_[np.zeros(Y.size), Y.ravel(), Z.ravel()]
u = be.rayleigh_streaming(P, {"A": 0.54, "k": 0.062, "H": 20.0, "sign": -1.0})
U, W = u[:, 1].reshape(Y.shape), u[:, 2].reshape(Y.shape)
sp = np.hypot(U, W)
ax.streamplot(y, z, U, W, color=sp, cmap="Oranges", density=1.1, linewidth=0.8, arrowsize=0.7)
ax.axvline(0, color="#111", ls=":", lw=0.8); ax.set_xlim(-25, 25); ax.set_ylim(0, 20)
ax.set_xlabel("y (µm)"); ax.set_ylabel("z (µm)")
ax.set_title("c  model used in the 3D runs, H = 20 µm", loc="left", fontsize=8, fontweight="bold")
ax.text(0.02, 0.02, f"max {sp.max():.2f} µm/s (nominal)", transform=ax.transAxes, fontsize=6)
os.makedirs(out, exist_ok=True)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig2c_streaming_side-wall-vs-IDT.{e}"), dpi=300)

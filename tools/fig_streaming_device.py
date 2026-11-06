"""Fig. 15 (paper 2, section 4.2): does the device's own streaming break the band?
(a, b) how the side-wall and IDT drives make streaming; (c) largest cluster vs time for every batch-18 run with
radiation force + computed streaming (solid: water fluid set, dashed: legacy set, 3 seeds); (d-f) side views at
3 tauB with the computed streaming (streamlines, colour = speed on one scale).
Usage: python fig_streaming_device.py <streaming18_series.npz> <_fields dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
S = np.load(sys.argv[1]); FD = sys.argv[2]; out = sys.argv[3]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
fig = plt.figure(figsize=(7.2, 4.3), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], width_ratios=[1, 1, 1, 0.05])
def arr(ax, a, b, c="#2a78d6", lw=1.0, ls="-", ms=8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=c, ls=ls))
def roll(ax, cx, cy, rx, ry, cw, c="#eb6834"):
    t = np.linspace(0.15, 2 * np.pi - 0.15, 60) * (1 if cw else -1)
    x, y = cx + rx * np.cos(t), cy + ry * np.sin(t); ax.plot(x, y, color=c, lw=1.0)
    arr(ax, (x[-2], y[-2]), (x[-1], y[-1]), c=c, ms=7)
# a  side-wall
ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.62, 0.86)
ax.add_patch(Rectangle((-1, 0), 2, 0.75, fc="#eaf2fb", ec="none"))
for y0 in (-0.07, 0.75): ax.add_patch(Rectangle((-1.05, y0), 2.1, 0.07, fc="#cfd8dc", ec="none"))
for s in (-1, 1): ax.add_patch(Rectangle((1.0 if s > 0 else -1.07, -0.07), 0.07, 0.89, fc="#eb6834", ec="none"))
ax.plot([0, 0], [0, 0.75], color="#111", ls=":", lw=0.8)
for cx, cwb in ((-0.75, True), (-0.25, False), (0.25, True), (0.75, False)):
    roll(ax, cx, 0.19, 0.2, 0.14, cwb); roll(ax, cx, 0.56, 0.2, 0.14, not cwb)
ax.text(0, -0.13, "sound runs along floor and lid; the viscous\nboundary layers drive rolls between them\n(Rayleigh), two per half-wavelength",
        ha="center", va="top", fontsize=6)
ax.set_title("a  side-wall drive, glass lid", loc="left", fontsize=7.5, fontweight="bold")
# b  IDT
th = np.arcsin(1481 / 3990)
ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.62, 0.86)
ax.add_patch(Rectangle((-1.15, -0.1), 2.3, 0.1, fc="#eb6834", ec="none", alpha=0.85))
ax.add_patch(Rectangle((-0.85, 0), 1.7, 0.75, fc="#eaf2fb", ec="none"))
ax.plot([-0.85, 0.85], [0.78, 0.78], color="#7a2a9a", lw=1.4, ls=(0, (3, 2)))
ax.plot([0, 0], [0, 0.75], color="#111", ls=":", lw=0.8)
L = 0.75 / np.cos(th)
for s, y0 in ((1, -0.75), (-1, 0.75)):
    ax.plot([y0, y0 + s * L * np.sin(th)], [0, 0.75], color="#2a78d6", lw=4, alpha=0.18)
for cx, cw in ((-0.62, False), (0.62, True)):
    roll(ax, cx, 0.12, 0.16, 0.08, cw)
ax.text(0, -0.13, "the vibrating floor drives boundary rolls, strongest\nwhere the leaky beams leave it; the absorbing\nlid sends nothing back",
        ha="center", va="top", fontsize=6)
ax.set_title("b  IDT drive, absorbing lid", loc="left", fontsize=7.5, fontweight="bold")
# c  largest cluster vs time
ax = fig.add_subplot(gs[0, 2])
CAS = [("sideglass", "20", "#e34948", "side wall, H = 20"), ("sideglass", "4", "#eb6834", "side wall, H = 4"),
       ("idtabsorb", "20", "#2a78d6", "IDT, H = 20"), ("idtabsorb", "4", "#7a2a9a", "IDT, H = 4")]
for dr, H, c, lab in CAS:
    for s in "123":
        k = f"{dr}|{H}|both|legacy|{s}"
        if k + "|t" in S: ax.plot(S[k + "|t"], S[k + "|S"], color=c, lw=0.8, ls="--", alpha=0.7)
    k = f"{dr}|{H}|both|water|1"
    ax.plot(S[k + "|t"], S[k + "|S"], color=c, lw=1.4, label=lab)
ax.plot([], [], color="#555", lw=1.4, label="water"); ax.plot([], [], color="#555", lw=0.8, ls="--", label="legacy set")
ax.set_xlim(0, 3); ax.set_ylim(0, 1.05); ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("S$_{max}$")
ax.legend(fontsize=5.6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.94), ncol=2, handlelength=1.6, columnspacing=0.8)
ax.set_title("c  largest cluster, radiation + streaming", loc="left", fontsize=7.5, fontweight="bold")
# d-f side views
vmax = 3.0
PAN = [("d  side wall, H = 20 µm, water", "auto_side20_H20_box50_glass_water.npz", "sideglass|20|both|water|1", "side"),
       ("e  same, legacy fluid set", "auto_side20_H20_box50_glass.npz", "sideglass|20|both|legacy|1", "side"),
       ("f  IDT, H = 20 µm, water", "auto_idt20_H20_box100_absorb_water.npz", "idtabsorb|20|both|water|1", "idt")]
for j, (title, fn, key, kind) in enumerate(PAN):
    a = fig.add_subplot(gs[1, j]); d = np.load(os.path.join(FD, fn), allow_pickle=True)
    ys, zs, U, W = d["ys"] * 1e6, d["zs"] * 1e6, d["U"], d["W"]; sp = np.hypot(U, W)
    H, half = zs.max(), ys.max()
    a.pcolormesh(ys, zs, sp.T, cmap="Blues", vmin=0, vmax=vmax, shading="auto", rasterized=True)
    a.streamplot(ys, zs, U.T, W.T, color="#33415c", density=(1.2, 0.8), linewidth=0.45, arrowsize=0.5, broken_streamlines=False)
    P = S[key + "|snap"]
    a.scatter(P[:, 1], P[:, 2], s=5 if half < 30 else 3, color="#e34948", edgecolor="white", linewidth=0.2, zorder=3)
    a.axvline(0, color="#999", ls=":", lw=0.6)
    src = dict(color="#eb6834", lw=3.0, solid_capstyle="butt", zorder=4, clip_on=False)
    if kind == "side":
        a.plot([-half, -half], [0, H], **src); a.plot([half, half], [0, H], **src)
        a.plot([-half, half], [H, H], color="#555", lw=2.0, zorder=4, clip_on=False)
    else:
        a.plot([-half, half], [0, 0], **src)
        a.plot([-half, half], [H, H], color="#7a2a9a", lw=1.4, ls=(0, (3, 2)), zorder=4, clip_on=False)
    a.set_xlim(-half, half); a.set_ylim(0, H); a.set_xlabel("y (µm)")
    if j == 0: a.set_ylabel("z (µm)")
    a.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
    a.text(0.98, 0.97, f"max {sp.max():.1f} µm/s · 3 τ$_B$", transform=a.transAxes, ha="right", va="top", fontsize=6,
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2), zorder=5)
cax = fig.add_subplot(gs[1, 3])
sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, vmax)); sm.set_array([])
fig.colorbar(sm, cax=cax, label="streaming speed (µm/s)")
os.makedirs(out, exist_ok=True)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig15_device-streaming_band-survival_water-vs-legacy.{e}"), dpi=300)
print("saved")

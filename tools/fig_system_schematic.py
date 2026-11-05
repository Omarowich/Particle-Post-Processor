"""Fig. 1: the system and the models.
(a) Regimes I-IV (top view schematics). (b) Side-wall drive: standing wave between two vibrating walls, node in the middle.
(c) Depletion: overlapping depletion layers and the resulting short-range potential.
Usage: python fig_system_schematic.py <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
out = sys.argv[1]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
WAT, PAR, SRC, RED, GRY = "#d4e9f7", "#1f3a5f", "#eb6834", "#d62f2f", "#666666"
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(7.2, 2.5))
top = fig.add_gridspec(1, 3, width_ratios=[2.35, 0.95, 1.0], wspace=0.12)
def letter(ax, s, x=0.0, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

# ---------- (a) regimes ----------
a = fig.add_subplot(top[0]); a.set_xlim(0, 4); a.set_ylim(-0.12, 1.27); a.set_axis_off(); letter(a, "a", -0.02, 0.93)
W, H = 0.66, 0.70
def chamber(cx):
    a.add_patch(FancyBboxPatch((cx - W/2, 0.5 - H/2), W, H, boxstyle="round,pad=0,rounding_size=0.06", fc=WAT, ec="#7fb2d6", lw=0.6))
def dots(cx, x, y, s=7):
    a.scatter(cx + x, 0.5 + y, s=s, color=PAR, lw=0, zorder=3)
def clusters(n_cl, per, spread, xr, yr):
    P = []
    for _ in range(n_cl):
        c = np.array([rng.uniform(-xr, xr), rng.uniform(-yr, yr)])
        P.append(c + rng.normal(0, spread, (per, 2)))
    return np.vstack(P)
def transducer(cx, y, up):
    a.add_patch(Rectangle((cx - 0.16, y - 0.025), 0.32, 0.05, fc=SRC, ec="none"))
names = [("Regime I", "no field"), ("Regime II", "radiation force on"), ("Regime III", "field off"), ("Regime IV", "streaming")]
for i, (nm, sub) in enumerate(names):
    cx = 0.5 + i
    a.text(cx, 1.17, nm, ha="center", fontsize=7.5, fontweight="bold"); a.text(cx, 1.07, sub, ha="center", fontsize=6.3, color=GRY)
    chamber(cx)
    if i == 0:
        P = clusters(7, 7, 0.022, 0.22, 0.22)
    elif i == 1:
        P = np.c_[rng.uniform(-0.26, 0.26, 55), rng.normal(0, 0.022, 55)]
        a.plot([cx - W/2, cx + W/2], [0.5, 0.5], color=GRY, ls=":", lw=0.6, zorder=2)
        transducer(cx, 0.5 + H/2 + 0.05, up=True); transducer(cx, 0.5 - H/2 - 0.05, up=False)
    elif i == 2:
        P = np.c_[rng.uniform(-0.26, 0.26, 55), rng.normal(0, 0.045, 55)]
    else:
        x = rng.uniform(-0.26, 0.26, 55); P = np.c_[x, 0.16*np.cos(np.pi*x/0.52)**2 - 0.08 + rng.normal(0, 0.02, 55)]
        transducer(cx, 0.5 - H/2 - 0.05, up=False)
        for s in (-1, 1):   # two streaming rolls
            t = np.linspace(0, 2*np.pi, 80)
            a.plot(cx + s*0.14 + 0.1*np.cos(t), 0.5 + 0.17*np.sin(t), color=RED, lw=0.6, ls=(0, (2, 1.5)), zorder=2)
            a.annotate("", xy=(cx + s*0.14 - s*0.1*0.0, 0.5 + 0.17), xytext=(cx + s*0.14 + s*0.03, 0.5 + 0.165),
                       arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.6, mutation_scale=6))
    P = P[(np.abs(P[:, 0]) < W/2 - 0.03) & (np.abs(P[:, 1]) < H/2 - 0.03)]
    dots(cx, P[:, 0], P[:, 1])
a.text(2.0, -0.1, "top view (x–y); y points across the band", ha="center", fontsize=6.3, color=GRY)

# ---------- (b) side-wall drive ----------
b = fig.add_subplot(top[1]); b.set_xlim(-0.75, 0.75); b.set_ylim(-0.9, 0.95); b.set_axis_off(); b.set_aspect("equal"); letter(b, "b", -0.05, 0.93)
yy = np.linspace(-0.5, 0.5, 200); amp = np.abs(np.sin(np.pi*yy))
b.imshow(np.tile(amp[:, None], (1, 10)), extent=(-0.5, 0.5, -0.5, 0.5), origin="lower", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("p", ["#f3f9fd", "#6fb3e0"]), aspect="auto", zorder=0)
for s in (-1, 1):
    b.add_patch(Rectangle((-0.5, s*0.5 - (0.04 if s > 0 else 0.0)), 1.0, 0.04, fc="#222", ec="none"))
    b.annotate("", xy=(0.0, s*0.78), xytext=(0.0, s*0.52), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3, mutation_scale=8))
b.text(0.08, 0.66, "$d_n$", color=RED, va="center", fontsize=7); b.text(0.08, -0.66, "$d_n$", color=RED, va="center", fontsize=7)
b.plot([-0.5, 0.5], [0, 0], color=GRY, ls=":", lw=0.7)
b.text(-0.48, 0.03, "node", color=GRY, fontsize=6.3, va="bottom")
P = np.c_[rng.uniform(-0.42, 0.42, 11), rng.uniform(-0.40, 0.40, 11)]
b.scatter(P[:, 0], P[:, 1], s=26, color=PAR, lw=0, zorder=3)
b.text(0, 0.88, "side-wall drive", ha="center", fontsize=6.8, fontweight="bold")
b.text(0, -0.88, "black: vibrating walls; shade: |p|", ha="center", fontsize=5.8, color=GRY)

# ---------- (c) depletion ----------
c = fig.add_subplot(top[2]); c.set_xlim(-1.35, 1.35); c.set_ylim(-1.65, 1.42); c.set_aspect("equal"); c.set_axis_off(); letter(c, "c", -0.05, 0.93)
R, d, L = 0.42, 0.47, 0.17   # colloid radius, half distance, depletion-layer thickness (schematic, not to scale)
for s in (-1, 1):
    c.add_patch(Circle((s*d, 0.35), R + L, fc="#eaf4fb", ec="#5aa9d6", lw=0.6, ls="--", zorder=1))
th = np.linspace(-np.pi, np.pi, 400)
# lens where the two depletion layers overlap
x0 = 0.0; h = np.sqrt((R + L)**2 - d**2)
t = np.linspace(-np.arcsin(h/(R+L)), np.arcsin(h/(R+L)), 60)
lens = np.r_[np.c_[-d + (R+L)*np.cos(t), 0.35 + (R+L)*np.sin(t)], np.c_[d - (R+L)*np.cos(t[::-1]), 0.35 + (R+L)*np.sin(t[::-1])]]
c.add_patch(Polygon(lens, fc=RED, ec="none", alpha=0.85, zorder=2))
for s in (-1, 1):
    g = Circle((s*d, 0.35), R, fc="#b9b9b9", ec="#555", lw=0.6, zorder=3); c.add_patch(g)
    c.add_patch(Circle((s*d - 0.15, 0.5), 0.14, fc="white", ec="none", alpha=0.45, zorder=3))
    c.annotate("", xy=(s*0.12, 0.35), xytext=(s*0.40, 0.35), arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6, mutation_scale=9), zorder=4)
n = 0
while n < 26:
    p = rng.uniform([-1.25, -0.45], [1.25, 1.15])
    if min(np.hypot(p[0] - d, p[1] - 0.35), np.hypot(p[0] + d, p[1] - 0.35)) < R + L + 0.05: continue
    s = np.linspace(0, 1, 30); ang = rng.uniform(0, np.pi)
    xs, ys = 0.16*s - 0.08, 0.025*np.sin(3*np.pi*s)
    c.plot(p[0] + xs*np.cos(ang) - ys*np.sin(ang), p[1] + xs*np.sin(ang) + ys*np.cos(ang), color="#2a9fd6", lw=0.7, zorder=1); n += 1
c.text(0, 1.30, "depletion attraction", ha="center", fontsize=6.8, fontweight="bold")
# energy sketch
ia = c.inset_axes([0.14, 0.02, 0.8, 0.3])
r = np.linspace(0.95, 1.8, 300); U = np.where(r < 1, 4, np.where(r < 1.3, -1.0*(1 - (r - 1)/0.3)**2, 0)); U[r < 1] = np.nan
ia.plot(r, U, color=RED, lw=1.2); ia.plot([1, 1], [-1.0, 1.0], color=RED, lw=1.2)
ia.axhline(0, color="k", lw=0.5); ia.set_xlim(0.85, 1.8); ia.set_ylim(-1.3, 1.1)
ia.set_xticks([1, 1.3]); ia.set_xticklabels(["2a", "2a(1+q)"], fontsize=5.6); ia.set_yticks([])
ia.set_xlabel("distance r (range not to scale)", fontsize=5.6, labelpad=1); ia.set_ylabel("U", fontsize=6, labelpad=1)
for s in ("top", "right"): ia.spines[s].set_visible(False)

for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig1_system-regimes-drive-depletion.{ext}"), dpi=300, bbox_inches="tight")
print("saved")

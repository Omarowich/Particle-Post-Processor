"""Fig. 1: the system. (a) chamber cross-section (y-z): vibrating side walls, standing pressure wave, radiation force
towards the node, band on the floor, gravity, lid. (b) top view of the 2D model: start grid -> band.
(c) depletion schematic (Omar's drawing). (d) AO pair potential: range 2a(1+q), depth set by phi_p.
(e) protocol: Regimes I-IV.  Usage: python fig_system_schematic.py <depletion.png> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
img = plt.imread(sys.argv[1]); out = sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
OR, BL, GR, RD = "#eb6834", "#3b6fb6", "#888888", "#d62f2f"
fig = plt.figure(figsize=(7.2, 4.9))
gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1], hspace=0.45, wspace=1.4)
def lab(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

# (a) cross-section
a = fig.add_subplot(gs[0, :4]); lab(a, "a")
L, H = 50.0, 20.0
a.add_patch(Rectangle((-L/2, 0), L, H, fc="#eef5fb", ec="none"))
a.plot([-L/2, L/2], [0, 0], color="k", lw=1.6); a.plot([-L/2, L/2], [H, H], color="#555", lw=1.6)
for s in (-1, 1):
    a.plot([s*L/2]*2, [0, H], color=OR, lw=3.5, solid_capstyle="butt")
    a.annotate("", xy=(s*(L/2+0.6), H/2), xytext=(s*(L/2+4.5), H/2), arrowprops=dict(arrowstyle="<->", color=OR, lw=1.0))
a.text(L/2+2.5, H/2+1.0, "vibrating wall,\ndisplacement $d_n$", color=OR, ha="left", va="bottom", fontsize=6.5)
y = np.linspace(-L/2, L/2, 200)
a.plot(y, H*0.62 + 4.2*np.sin(np.pi*y/L), color=BL, lw=1.0)
a.plot(y, H*0.62 - 4.2*np.sin(np.pi*y/L), color=BL, lw=1.0, ls="--")
a.text(L/2-1.5, H*0.62+5.0, "pressure $p(y)$", color=BL, ha="right", fontsize=6.5)
a.axvline(0, color=GR, ls=":", lw=0.7); a.text(0.6, H-1.2, "node", color=GR, fontsize=6.5, va="top")
for yy in (-19, -13, -7, 7, 13, 19):
    F = -np.sin(2*np.pi*yy/50.7)
    a.add_patch(FancyArrowPatch((yy, 5.5), (yy + 4.5*F, 5.5), arrowstyle="-|>", mutation_scale=7, color=RD, lw=1.0))
a.text(-13, 7.2, "radiation force", color=RD, ha="center", fontsize=6.5)
for yy in np.arange(-3.0, 3.01, 2.0):
    a.add_patch(Circle((yy, 1.0), 1.0, fc="#9a9a9a", ec="k", lw=0.4))
for yy in (-17.5, 15.5):
    a.add_patch(Circle((yy, 1.0), 1.0, fc="#9a9a9a", ec="k", lw=0.4))
a.add_patch(FancyArrowPatch((21.5, 12), (21.5, 8), arrowstyle="-|>", mutation_scale=7, color="k", lw=0.8))
a.text(22.3, 10, "$g$", fontsize=7, va="center")
a.text(0, H+1.2, "lid (glass / PDMS / absorbing; 3D only)", color="#555", ha="center", va="bottom", fontsize=6.5)
a.annotate("", xy=(-L/2+1.2, 0), xytext=(-L/2+1.2, H), arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
a.text(-L/2+2.0, H/2, "$H$", fontsize=7, va="center")
a.set_xlim(-L/2-6, L/2+14); a.set_ylim(-2.5, H+6); a.set_aspect("equal")
a.set_xlabel("y (µm)"); a.set_yticks([0, H]); a.set_ylabel("z (µm)"); a.set_xticks([-25, 0, 25])
for s in ("top", "right"): a.spines[s].set_visible(False)

# (c) depletion drawing
c = fig.add_subplot(gs[0, 4:]); lab(c, "c", x=0.02); c.imshow(img); c.set_axis_off()

# (b) top view of the 2D model
b = fig.add_subplot(gs[1, :2]); lab(b, "b")
g = np.arange(-22.0, 22.01, 4.0); X, Y = np.meshgrid(g, g)
b.scatter(X, Y, s=4, color="#bbbbbb", lw=0)
rng = np.random.default_rng(3); bx = rng.uniform(-24, 24, 144); by = rng.normal(0, 2.2, 144)
b.scatter(bx, by, s=4, color="#333333", lw=0)
b.axhline(0, color=GR, ls=":", lw=0.6)
b.set_aspect("equal"); b.set_xlim(-26, 26); b.set_ylim(-26, 26); b.set_xticks([-25, 0, 25]); b.set_yticks([-25, 0, 25])
b.set_xlabel("x (µm)"); b.set_ylabel("y (µm)")
b.set_title("2D model: N = 144, grid start (grey)\n→ band at the node (black)", fontsize=6.5)

# (d) AO potential
d = fig.add_subplot(gs[1, 2:4]); lab(d, "d")
q, sig = 0.05, 2.0
r = np.linspace(sig, sig*(1+q)*1.04, 300)
for phi, A in ((0.129, 1), (0.182, 3), (0.244, 7)):
    x = r/(sig*(1+q)); U = -phi*((1+q)/q)**3*(1 - 1.5*x + 0.5*x**3); U[x > 1] = 0
    d.plot(r, U, lw=1.1, label=f"A = {A} ($\\phi_p$ = {phi})")
d.plot([sig, sig], [-12, 2], color="k", lw=1.4)
d.axhline(0, color="k", lw=0.5)
d.annotate("", xy=(sig, 1.0), xytext=(sig*(1+q), 1.0), arrowprops=dict(arrowstyle="<->", lw=0.6))
d.text(sig*(1+q/2), 1.5, "2$a q$", ha="center", fontsize=6.5)
d.set_xlim(1.97, 2.13); d.set_ylim(-12, 3)
d.set_xlabel("centre distance r (µm)"); d.set_ylabel("U / $k_BT$")
d.set_title("AO depletion, q = $R_g/a$ = 0.05", fontsize=6.5); d.legend(fontsize=5.5, frameon=False, loc="lower right")
for s in ("top", "right"): d.spines[s].set_visible(False)

# (e) protocol
e = fig.add_subplot(gs[1, 4:]); lab(e, "e", x=0.02)
R = [("I", "no field", "#dddddd"), ("II", "radiation force on", "#f6c9b5"),
     ("III", "field off, from the band", "#cfe0f1"), ("IV", "streaming flow", "#d6ecd2")]
for i, (n, t, col) in enumerate(R):
    yv = 3 - i
    e.add_patch(Rectangle((0, yv - 0.38), 1, 0.76, fc=col, ec="none"))
    e.text(0.05, yv, f"Regime {n}", fontweight="bold", va="center", fontsize=6.5)
    e.text(0.05, yv - 0.26, t, va="center", fontsize=5.8)
e.set_xlim(0, 1); e.set_ylim(-0.6, 3.6); e.set_axis_off()
e.set_title("simulation protocols", fontsize=6.5)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig1_system-schematic_chamber-depletion-AO-regimes.{ext}"), dpi=300, bbox_inches="tight")
print("saved")

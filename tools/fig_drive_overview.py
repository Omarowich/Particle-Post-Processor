"""Fig. 2b (paper 2): the drive x lid combinations used in the 3D runs, H = 20 um. Top: schematics of the two drives with
the wave paths; below: pressure amplitude |p| and radiation force (arrows) for glass, PDMS and absorbing lids.
Usage: fig_drive_overview.py <drive_overview.npy> <out>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
D = np.load(sys.argv[1], allow_pickle=True).item(); out = sys.argv[2]
th = np.arcsin(1850 / 3990)
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.2, 7.6), constrained_layout=True)
gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1, 1, 1])

def arr(ax, a, b, c="#2a78d6", ls="-", lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=lw, color=c, ls=ls))

# --- schematics
for col, drive in enumerate(("side", "idt")):
    ax = fig.add_subplot(gs[0, col]); ax.set_axis_off(); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.45, 1.2); ax.set_aspect("equal")
    ax.add_patch(Rectangle((-1, 0), 2, 0.75, fc="#eaf2fb", ec="none"))
    ax.add_patch(Rectangle((-1.1, 0.75), 2.2, 0.1, fc="#cfd8dc", ec="none")); ax.text(0, 0.93, "lid: glass / PDMS / absorbing", ha="center", fontsize=6.5, color="#555")
    if drive == "side":
        ax.add_patch(Rectangle((-1.1, -0.08), 2.2, 0.08, fc="#cfd8dc", ec="none"))
        for s in (-1, 1):
            ax.add_patch(Rectangle((s * 1.0 - (0.1 if s > 0 else 0) + (0 if s > 0 else -0.1) + (0.1 if s > 0 else 0), -0.08), 0.1 * 1, 0.93, fc="#999", ec="none")) if False else None
            ax.add_patch(Rectangle((1.0 if s > 0 else -1.1, -0.08), 0.1, 0.93, fc="#999", ec="none"))
            arr(ax, (s * 1.42, 0.45), (s * 1.17, 0.45), c="#e34948"); arr(ax, (s * 1.17, 0.3), (s * 1.42, 0.3), c="#e34948")
        for z in (0.2, 0.4, 0.6):
            arr(ax, (-0.9, z), (0.9, z)); arr(ax, (0.9, z - 0.05), (-0.9, z - 0.05), ls="--", lw=0.8)
        ax.plot([0, 0], [0, 0.75], color="#111", ls=":", lw=0.9)
        ax.text(0, -0.2, "walls move along y; sound runs horizontally\nand bounces between the walls → standing wave\nacross y, nodes λ$_{liquid}$/2 = 50 µm apart (COMSOL set-up)",
                ha="center", va="top", fontsize=6.5)
        ax.set_title("a  side-wall drive (bulk acoustic wave)", loc="left", fontsize=8, fontweight="bold")
    else:
        ax.add_patch(Rectangle((-1.5, -0.12), 3.0, 0.12, fc="#d8cfe8", ec="none")); ax.text(1.3, -0.07, "LiNbO$_3$", ha="center", va="center", fontsize=6)
        for s in (-1, 1):
            for i in range(4):
                ax.add_patch(Rectangle((s * 1.25 + 0.05 * i - 0.08, 0.0), 0.025, 0.05, fc="#b8860b", ec="none"))
            arr(ax, (s * 1.15, 0.03), (s * 0.95, 0.03), c="#e34948")
        L = 0.75 / np.cos(th)
        for s, y0 in ((1, -0.75), (-1, 0.75)):
            y1 = y0 + s * L * np.sin(th)
            arr(ax, (y0, 0.0), (y1, 0.75))
            arr(ax, (y1, 0.75), (y1 + s * L * np.sin(th) * 0.9, 0.75 - 0.75 * 0.9), ls="--", lw=0.8)
        ax.text(-0.72, 0.18, f"θ$_R$ = {np.degrees(th):.1f}°", fontsize=6.5, color="#2a78d6")
        ax.text(0, -0.2, "floor carries a standing surface wave; sound leaks\nup at θ$_R$ (solid), a reflecting lid sends it back\n(dashed); nodes λ$_{SAW}$/2 = 100 µm apart at 20 MHz",
                ha="center", va="top", fontsize=6.5)
        ax.set_title("b  IDT drive (standing surface acoustic wave)", loc="left", fontsize=8, fontweight="bold")

# --- fields
LID = {"glass": "glass lid (reflects)", "pdms": "PDMS lid (Z = 1 MRayl)", "absorb": "absorbing lid (Z = ρ$_0$c$_0$)"}
NOTE = {("side", "glass"): "uniform in height, no vertical force\n= the 2D model, one band",
        ("side", "pdms"): "sound leaks out: decays from the walls,\npushes up at the band (flat well if H ≲ 15 µm)",
        ("side", "absorb"): "leaks even more: same picture,\nweaker trap in the middle",
        ("idt", "glass"): "beams reflect: standing wave in height,\npushes particles down",
        ("idt", "pdms"): "most sound leaves; partial reflection\npushes particles up",
        ("idt", "absorb"): "beams leave: almost uniform in height\n= cleanest field"}
lab = iter("cdefgh")
for r, lid in enumerate(("glass", "pdms", "absorb")):
    for c, drive in enumerate(("side", "idt")):
        d = D[f"{drive}_{lid}"]; ys, zs = d["ys"], d["zs"]; P = np.abs(d["p"]).T; F = d["F"]
        ax = fig.add_subplot(gs[1 + r, c])
        ax.imshow(P / P.max(), extent=(ys[0], ys[-1], zs[0], zs[-1]), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        sy, sz = max(1, len(ys) // 16), max(1, len(zs) // 5)
        Y, Z = np.meshgrid(ys[sy // 2::sy], zs[sz // 2::sz])
        Fy, Fz = F[sy // 2::sy, sz // 2::sz, 1].T, F[sy // 2::sy, sz // 2::sz, 2].T
        ax.quiver(Y, Z, Fy, Fz, color="#7fd3ff", pivot="mid", width=0.006, scale=None)
        nodes = (0.0,) if drive == "side" else (0.0,)
        for yn in nodes:
            ax.axvline(yn, color="w", ls=":", lw=0.8)
        iz = len(zs) // 2; band = np.abs(ys) < 3
        fz = F[band, iz, 2].mean() / np.abs(F[:, iz, 1]).max()
        ax.text(0.98, 0.04, f"F$_z$/F$_y$ at band {fz:+.2f}", transform=ax.transAxes, color="#111", fontsize=6, ha="right",
                bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
        ax.set_title(f"{next(lab)}  {'side-wall' if drive == 'side' else 'IDT'}, {LID[lid]}", loc="left", fontsize=7.5, fontweight="bold")
        ax.text(0.0, 1.0, NOTE[(drive, lid)].replace("\n", " "), transform=ax.transAxes, fontsize=6, color="#555", va="bottom",
                ha="left") if False else None
        ax.set_xlabel(NOTE[(drive, lid)].replace("\n", " ") if r < 2 else "y (µm) · " + NOTE[(drive, lid)].replace("\n", " "), fontsize=6.3, color="#444")
        ax.set_ylabel("z (µm)" if c == 0 else "")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig2b_drive-and-lid-overview_H20.{ext}"), dpi=300)
for k, d in D.items():
    iz = len(d["zs"]) // 2; band = np.abs(d["ys"]) < 3
    print(k, round(d["F"][band, iz, 2].mean() / np.abs(d["F"][:, iz, 1]).max(), 3))

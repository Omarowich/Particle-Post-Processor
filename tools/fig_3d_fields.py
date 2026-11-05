"""Fig. 4: 3D chamber fields (water), y-z cross-sections at H = 20 um, one acoustic amplitude (in-plane force at
mid-height = F0 at d_n = 4000 nm). Columns: side-wall drive + glass lid; IDT + absorbing, glass and PDMS lids.
Top: radiation force (colour = |F|, arrows = direction). Bottom: computed streaming (colour = speed, streamlines).
Usage: python fig_3d_fields.py <simulation dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[1]); import streaming2d as s2
out = sys.argv[2]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})
CASES = [("side", "glass", 25.0, "side-wall drive, glass lid"), ("idt", "absorb", 50.0, "IDT, absorbing lid"),
         ("idt", "glass", 50.0, "IDT, glass lid"), ("idt", "pdms", 50.0, "IDT, PDMS lid")]
H = 20.0
R = [s2.compute(d, H_um=H, box_half_um=bh, freq_MHz=20.0, lid=l, fluid="water") for d, l, bh, _ in CASES]
fmax = max(np.hypot(r["Fy"], r["Fz"]).max() for r in R); umax = max(np.hypot(r["U"], r["W"]).max() for r in R)
fig, axT = plt.subplots(4, 2, figsize=(7.2, 6.4), constrained_layout=True)
ax = axT.T
for c, ((d, l, bh, title), r) in enumerate(zip(CASES, R)):
    ys, zs = r["ys"] * 1e6, r["zs"] * 1e6
    F = np.hypot(r["Fy"], r["Fz"]); U = np.hypot(r["U"], r["W"])
    a = ax[0, c]
    im1 = a.pcolormesh(ys, zs, F.T, cmap="Reds", vmin=0, vmax=fmax, shading="auto", rasterized=True)
    sy, sz = max(1, len(ys) // (8 if bh < 30 else 14)), max(1, len(zs) // 6)
    Y, Z = np.meshgrid(ys[::sy], zs[::sz], indexing="ij")
    Fy, Fz = r["Fy"][::sy, ::sz], r["Fz"][::sy, ::sz]; n = np.hypot(Fy, Fz) + 1e-12
    a.quiver(Y, Z, Fy / n, Fz / n, color="#333", scale=22, width=0.006, headwidth=4)
    a.set_ylabel(title.replace(", ", ",\n") + "\n\nz (µm)", fontsize=6.6)
    b = ax[1, c]
    im2 = b.pcolormesh(ys, zs, U.T, cmap="Blues", vmin=0, vmax=umax, shading="auto", rasterized=True)
    b.streamplot(ys, zs, r["U"].T, r["W"].T, color="#2c4a6e", density=(0.8 if bh < 30 else 1.3, 0.6), linewidth=0.45,
                 arrowsize=0.6, broken_streamlines=False)
    for x in (a, b):
        x.set_xlim(-50, 50); x.set_ylim(0, H); x.set_aspect(1.8)
        for y0 in ((0,) if d == "side" else (-50, 0, 50)): x.axvline(y0, color="#888", ls=":", lw=0.6)
    b.set_ylabel("z (µm)")
    if c == 3: a.set_xlabel("y (µm)"); b.set_xlabel("y (µm)")
    print(title, "f = %.1f MHz" % (r["omega"] / 2 / np.pi / 1e6), "|F|max %.1f" % F.max(), "|u|max %.2f" % U.max())
fig.colorbar(im1, ax=ax[0, :], shrink=0.6, location="top", label="radiation force |F$_{rad}$| (k$_B$T/µm); arrows: direction")
fig.colorbar(im2, ax=ax[1, :], shrink=0.6, location="top", label="streaming speed (µm/s); lines: streamlines")
for i, x in enumerate(axT.ravel()): x.text(0.01, 0.97, "abcdefgh"[i], transform=x.transAxes, fontsize=8, fontweight="bold", va="top", color="k")
fig.text(0.5, -0.01, "z stretched 1.8×; dotted: pressure nodes. Water, H = 20 µm, one acoustic amplitude (in-plane force at mid-height = F$_0$ at d$_n$ = 4000 nm).", ha="center", fontsize=6.3)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig4_3D-chamber-fields_force-and-streaming_water_H20.{e}"), dpi=300, bbox_inches="tight")
print("saved")

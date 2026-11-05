"""Fig. 12b (draft): where the motion of the streaming-only runs comes from. Side views (y-z, all x) of the computed
streaming (streamlines, colour = speed) with the particles of the run on top, at 0, 0.3, 1 and 3 tauB.
Usage: python fig_streaming_overlay.py <asf_streamonly.npz> <_fields dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = np.load(sys.argv[1]); FD = sys.argv[2]; out = sys.argv[3]
CASES = [("sideglass20", "auto_side20_H20_box50_glass.npz", "side wall, glass lid, H = 20 µm"),
         ("sideglass4", "auto_side20_H4_box50_glass.npz", "side wall, glass lid, H = 4 µm"),
         ("idtabsorb20", "auto_idt20_H20_box100_absorb.npz", "IDT, absorbing lid, H = 20 µm"),
         ("idtabsorb4", "auto_idt20_H4_box100_absorb.npz", "IDT, absorbing lid, H = 4 µm")]
TS = ["0", "0.3", "1", "3"]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
fig, ax = plt.subplots(4, 4, figsize=(7.2, 6.6), constrained_layout=True,
                       gridspec_kw={"height_ratios": [1.6, 0.8, 1.6, 0.8]})
vmax = 3.0
for r, (key, fn, title) in enumerate(CASES):
    d = np.load(os.path.join(FD, fn))
    ys, zs, U, W = d["ys"] * 1e6, d["zs"] * 1e6, d["U"], d["W"]      # um, um/s at d_n = 4000 nm
    sp = np.hypot(U, W)
    H = zs.max(); half = ys.max()
    for c, T in enumerate(TS):
        a = ax[r, c]
        a.pcolormesh(ys, zs, sp.T, cmap="Blues", vmin=0, vmax=vmax, shading="auto", rasterized=True)
        a.streamplot(ys, zs, U.T, W.T, color="#33415c", density=(1.6, 0.9 if H > 10 else 0.5), linewidth=0.5,
                     arrowsize=0.6, broken_streamlines=False)
        P = S[f"{key}|{T}"]
        a.scatter(P[:, 1], P[:, 2], s=6 if half < 30 else 3, color="#e34948", edgecolor="white", linewidth=0.2, zorder=3)
        a.axvline(0, color="#999", ls=":", lw=0.6)
        # where the sound comes from (orange) and what the lid does (grey solid = reflecting glass, purple dashed = absorbing)
        src = dict(color="#eb6834", lw=3.2, solid_capstyle="butt", zorder=4, clip_on=False)
        if key.startswith("side"):
            a.plot([-half, -half], [0, H], **src); a.plot([half, half], [0, H], **src)
            a.plot([-half, half], [H, H], color="#555", lw=2.2, zorder=4, clip_on=False)
        else:
            a.plot([-half, half], [0, 0], **src)
            a.plot([-half, half], [H, H], color="#7a2a9a", lw=1.6, ls=(0, (3, 2)), zorder=4, clip_on=False)
        a.set_xlim(-half, half); a.set_ylim(0, H)
        if r == 0: a.set_title(f"{T} τ$_B$", fontsize=8, fontweight="bold")
        if c == 0: a.set_ylabel("z (µm)")

    ax[r, 0].text(0.02, 0.97, title, transform=ax[r, 0].transAxes, fontsize=6.5, fontweight="bold", va="top", zorder=5,
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5)).set_in_layout(False)
fig.supxlabel("y (µm), across the band (dotted: pressure node).  Orange: vibrating surface that makes the sound "
              "(side wall: both side walls move in y; IDT: the floor carries the standing SAW from IDTs outside the box).  "
              "Lid: grey = glass (reflects), purple dashed = absorbing.", fontsize=6.3, wrap=True)
sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, vmax)); sm.set_array([])
fig.colorbar(sm, ax=ax[:, -1], shrink=0.5, label="streaming speed (µm/s), d$_n$ = 4000 nm")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig12b_streaming-field-with-particles_side-views.{e}"), dpi=300)
print("saved")

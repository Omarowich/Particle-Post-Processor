"""Fig. 11c (draft): how the 3D band stacks. (a) schematic, (b-d) average trajectories of three groups of particles,
(e) side views. Data: per-particle trajectories of the 3D Regime II runs (AO q = 0.05, A = 3, H = 20 um, 3500 nm).
Usage: python fig_stacking_mechanism.py <traj_regII.npz> <out dir> [run key, default H20_3500]"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
S = np.load(sys.argv[1]); out = sys.argv[2]; key = sys.argv[3] if len(sys.argv) > 3 else "H20_3500"
P = S[key]; t = S["t_" + key]; a = 1.0

def ontop(Q):
    d = Q[:, None] - Q[None]; lat = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(lat, np.inf)
    return ((d[..., 2] > a) & (lat < 2.2 * a)).any(1)

k1, kend = np.searchsorted(t, 1.0 - 1e-6), len(t) - 1
top_end = ontop(P[kend]); h = P[..., 2] - a; ay = np.abs(P[..., 1])
floor_band_1 = (ay[k1] < 5) & (h[k1] < 0.5)
G = {"squeezed up from the floor": (top_end & floor_band_1, "#eb6834"),
     "arrived late, rode up": (top_end & ~floor_band_1, "#7a2a9a"),
     "stayed on the floor": (~top_end, "#2a78d6")}
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 0.75])
# (a) schematic
ax = fig.add_subplot(gs[0, :2]); ax.set_aspect("equal"); ax.axis("off")
ax.plot([-7.5, 7.5], [0, 0], color="#555", lw=1.2); ax.text(7.4, -0.55, "floor", ha="right", fontsize=6.5, color="#555")
ax.axvline(0, color="#999", ls=":", lw=0.8); ax.text(0.2, -0.75, "node", ha="left", fontsize=6.5, color="#777")
xs = np.arange(-6, 6.01, 2.0)
for x in xs:
    ax.add_patch(Circle((x, 1), 0.97, fc="#9bb7e0", ec="#2a78d6", lw=0.6))
ax.add_patch(Circle((0, 1 + np.sqrt(3)), 0.97, fc="#f6b796", ec="#eb6834", lw=0.8))
ax.add_patch(Circle((0, 1), 0.97, fc="white", ec="#eb6834", lw=0.8, ls="--"))
ax.add_artist(FancyArrowPatch((0, 1.3), (0, 1 + np.sqrt(3) - 0.3), arrowstyle="-|>", mutation_scale=8, color="#eb6834", lw=1))
for s in (-1, 1):
    ax.add_artist(FancyArrowPatch((s * 7.6, 1), (s * 6.1, 1), arrowstyle="-|>", mutation_scale=9, color="#333", lw=1.2))
ax.text(-7.5, 1.7, "radiation force\nsqueezes inward", fontsize=6.3, color="#333")
ax.add_patch(Circle((4.3, 3.3), 0.97, fc="#d9c2ea", ec="#7a2a9a", lw=0.8))
ax.add_artist(FancyArrowPatch((6.3, 2.9), (5.1, 3.2), arrowstyle="-|>", mutation_scale=8, color="#7a2a9a", lw=1))
ax.text(-7.6, 4.0, "squeezed out of the floor layer, or arriving late and riding up:\nlands in a hollow, 3 bonds ≈ 15 k$_B$T, locked where it is", fontsize=6, color="#b24a1c", ha="left", va="bottom")
ax.set_xlim(-8, 8); ax.set_ylim(-0.9, 5.4)
ax.set_title("a  how the band stacks (side view across the node)", loc="left", fontsize=8, fontweight="bold")
# (b) mean path in (|y|, height)
ax = fig.add_subplot(gs[0, 2:])
for lab, (m, c) in G.items():
    if m.sum() == 0: continue
    yy, hh = ay[:, m].mean(1), h[:, m].mean(1)
    ax.plot(yy, hh, color=c, lw=1.3); ax.scatter(yy[::10], hh[::10], color=c, s=10, zorder=3)
    ax.annotate("", xy=(yy[-1], hh[-1]), xytext=(yy[-4], hh[-4]), arrowprops=dict(arrowstyle="-|>", color=c, lw=1))
ax.set_xlabel("mean distance from the node |y| (µm)"); ax.set_ylabel("mean height above contact (µm)")
ax.set_title("b  average path (dots every 0.5 τ$_B$)", loc="left", fontsize=8, fontweight="bold")
# (c, d) mean |y| and height vs time
for j, (arr, lab, tt) in enumerate(((ay, "distance from the node |y| (µm)", "c  approach to the node"),
                                    (h, "height above contact (µm)", "d  height"))):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    for lab2, (m, c) in G.items():
        if m.sum() == 0: continue
        mu, se = arr[:, m].mean(1), arr[:, m].std(1) / np.sqrt(m.sum())
        ax.plot(t, mu, color=c, lw=1.3, label=f"{lab2} (n = {m.sum()})"); ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.2, lw=0)
    ax.set_xlabel("time in the field (τ$_B$)"); ax.set_ylabel(lab); ax.set_title(tt, loc="left", fontsize=8, fontweight="bold")
    if j == 0: ax.legend(frameon=False, fontsize=5.8, loc="upper right")
# (e) side views
for j, T in enumerate((0.0, 1.0, 2.0, t[-1])):
    ax = fig.add_subplot(gs[2, j]); k = np.searchsorted(t, T - 1e-6)
    for lab, (m, c) in G.items():
        ax.scatter(P[k, m, 1], P[k, m, 2], s=4, color=c, alpha=0.8, lw=0)
    ax.set_xlim(-25, 25); ax.set_ylim(0, 7); ax.set_aspect(2.0)
    ax.set_xlabel("y (µm)"); ax.set_title(f"{'e' if j == 0 else ''}  side view, {t[k]:.1f} τ$_B$", loc="left", fontsize=7.5, fontweight="bold")
    if j == 0: ax.set_ylabel("z (µm)")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig11c_3D-band_stacking-mechanism_{key}.{e}"), dpi=300)
print({k: int(v[0].sum()) for k, v in G.items()})

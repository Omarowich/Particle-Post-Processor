"""Fig. 8d (paper 2): a PDMS lid flattens the side-wall standing wave in thin chambers (lossy waveguide).
Usage: fig_pdms_flat.py <pdms_flat.npy> <ydist json> <out dir>"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = np.load(sys.argv[1], allow_pickle=True).item(); Y = json.load(open(sys.argv[2])); out = sys.argv[3]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
# a, b: |p| in the y-z plane, H = 6 um
for k, (lid, lab) in enumerate((("hard", "glass lid"), ("pdms", "PDMS lid"))):
    ax = fig.add_subplot(gs[0, 0]) if k == 0 else None
    if k == 0:
        sub = gs[0, 0].subgridspec(2, 1, hspace=0.05); axs = [fig.add_subplot(sub[i]) for i in range(2)]; ax.remove()
    d = D[f"{lid}_H6"]; P = np.abs(d["p"]).T; P = P / P.max()
    a = axs[k]
    im = a.imshow(P, extent=(d["ys"][0], d["ys"][-1], d["zs"][0], d["zs"][-1]), origin="lower", aspect="auto",
                  cmap="magma", vmin=0, vmax=1)
    a.set_ylabel("z (µm)"); a.set_yticks([0, 3, 6])
    a.text(0.02, 0.85, lab, transform=a.transAxes, color="w", fontsize=7, fontweight="bold")
    if k == 0:
        a.set_title("a  |p| / max, H = 6 µm (side-wall drive)", loc="left", fontsize=8, fontweight="bold"); a.set_xticks([])
    else:
        a.set_xlabel("y (µm)")
# c: force profile, mid-height, normalised
ax = fig.add_subplot(gs[0, 1])
cols = {"hard_H6": ("#555", "glass lid, H = 6"), "pdms_H40": ("#9ecae1", "PDMS, H = 40"), "pdms_H20": ("#4292c6", "PDMS, H = 20"),
        "pdms_H6": ("#eb6834", "PDMS, H = 6"), "pdms_H3": ("#a50f15", "PDMS, H = 3")}
for key, (c, lab) in cols.items():
    d = D[key]; iz = len(d["zs"]) // 2; F = d["F"][:, iz, 1]
    ax.plot(d["ys"], F / np.abs(F).max(), color=c, lw=1.3, ls="--" if "hard" in key else "-", label=lab)
ax.axhline(0, color="#bbb", lw=0.5); ax.set_xlabel("y (µm)"); ax.set_ylabel("F$_y$ / max|F$_y$| (mid-height)")
ax.set_title("b  force across the chamber", loc="left", fontsize=8, fontweight="bold"); ax.legend(frameon=False, fontsize=6.5)
# d: potential
ax = fig.add_subplot(gs[0, 2])
for key, (c, lab) in cols.items():
    d = D[key]; iz = len(d["zs"]) // 2; F = d["F"][:, iz, 1] / np.abs(d["F"][:, iz, 1]).max()
    U = -np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(d["ys"]))]); U -= U.min()
    ax.plot(d["ys"], U / U.max(), color=c, lw=1.3, ls="--" if "hard" in key else "-")
ax.axvspan(-10, 10, color="#eb6834", alpha=0.08, lw=0)
ax.set_xlabel("y (µm)"); ax.set_ylabel("potential U / max (Gor'kov)")
ax.set_title("c  the well gets a flat bottom", loc="left", fontsize=8, fontweight="bold")
# e: decay length vs H
ax = fig.add_subplot(gs[1, 0])
Hs = np.array([3, 4, 6, 10, 20, 40]); L = []
for H in Hs:
    r = []
    for lid in ("hard", "pdms"):
        d = D[f"{lid}_H{H}"]; iz = len(d["zs"]) // 2; P = np.abs(d["p"][:, iz])
        r.append(P[0] / P[np.argmin(np.abs(d["ys"] + 12.5))])
    L.append(12.5 / np.log(r[1] / r[0]))
ax.plot(Hs, L, "o-", color="#eb6834", label="computed (PDMS lid)")
hh = np.linspace(2, 42, 50); ax.plot(hh, 2 * hh * 1.0e6 / (1180 * 2030), "--", color="#555", label="lossy waveguide, 2HZ/ρ$_0$c$_0$")
ax.axhline(25, color="#bbb", lw=0.6, ls=":"); ax.text(41, 26, "half width", ha="right", fontsize=6.5, color="#777")
ax.set_xlabel("chamber height H (µm)"); ax.set_ylabel("decay length of |p| from the wall (µm)")
ax.set_title("d  decay length ∝ H", loc="left", fontsize=8, fontweight="bold"); ax.legend(frameon=False, fontsize=6.5)
# f: particle positions across y after 5 tauB
ax = fig.add_subplot(gs[1, 1:])
bins = np.arange(-25, 25.1, 1.0)
sty = {"1D field, H = 6 µm": ("#999", "--"), "hard lid (computed), H = 20 µm": ("#555", "-"),
       "PDMS lid, H = 20 µm": ("#4292c6", "-"), "PDMS lid, H = 6 µm": ("#eb6834", "-")}
for lab, (c, ls) in sty.items():
    v = np.array(Y[lab]["y"]); h, _ = np.histogram(v, bins, density=True)
    ax.step(bins[:-1] + 0.5, h, where="mid", color=c, ls=ls, lw=1.3, label=f"{lab} ({Y[lab]['n']} runs)")
ax.set_xlabel("y (µm)"); ax.set_ylabel("particle density (after 5 τ$_B$)")
ax.set_title("e  where the particles are: one band vs two stalled fronts", loc="left", fontsize=8, fontweight="bold")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig8d_PDMS-lid_flat-well.{ext}"), dpi=300)
print([round(x, 1) for x in L])

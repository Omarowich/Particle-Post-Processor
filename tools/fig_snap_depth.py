"""Fig. 9: Regime I snapshots after 10 tauB (seed 1), LJ 12-6 (top) and AO q = 0.05 (bottom), ordered by well depth.
Colour = local lattice orientation (arg psi6 / 6), grey = fewer than 2 neighbours. Usage: python fig_snap_depth.py <npz> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial import cKDTree
S = np.load(sys.argv[1]); out = sys.argv[2]
plt.rcParams.update({"font.size": 7})
ROWS = [("LJ 12-6", "LJ", [(2, 2.0, -3.6), (3, 3.0, -10.0), (4, 4.0, None), (5, 5.0, -59)]),
        ("AO q = 0.05", "AO", [(1, 4.0, -0.56), (2, 5.0, -3.6), (3, 5.7, -10.0), (5, 6.8, -59)])]
cm = plt.get_cmap("twilight")
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
for r, (lab, key, cols) in enumerate(ROWS):
    for c, (A, depth, b2) in enumerate(cols):
        a = ax[r, c]; P = S[f"{key}|{A}"]
        b = cKDTree(P).query_pairs(2.9, output_type="ndarray"); n = len(P); z = np.bincount(b.ravel(), minlength=n)
        d = P[b[:, 1]] - P[b[:, 0]]; e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0])); psi = np.zeros(n, complex)
        np.add.at(psi, b[:, 0], e); np.add.at(psi, b[:, 1], e)
        ang = (np.angle(psi) % (2 * np.pi)) / (2 * np.pi)
        col = [cm(v) if zz >= 2 else (0.75, 0.75, 0.75, 1) for v, zz in zip(ang, z)]
        ok = z >= 2; p6 = np.abs(psi[ok] / z[ok]).mean()
        a.add_collection(EllipseCollection(2, 2, 0, units="xy", offsets=P, transOffset=a.transData, facecolors=col, edgecolors="none"))
        a.set_xlim(-25.5, 25.5); a.set_ylim(-25.5, 25.5); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_linewidth(0.5); s.set_color("#999")
        a.set_title(f"A = {A}: depth {depth:g} k$_B$T" + (f", B$_2^*$ = {b2:g}" if b2 is not None else "") + f"\nψ$_6$ = {p6:.2f}", fontsize=6.6)
        if c == 0: a.set_ylabel(lab, fontsize=8, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 60)); sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=0.6, label="lattice orientation (°); grey: < 2 neighbours")
for e_ in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig9_RegimeI_snapshots_LJ-vs-AO_by-well-depth_10tauB.{e_}"), dpi=300)
print("saved")

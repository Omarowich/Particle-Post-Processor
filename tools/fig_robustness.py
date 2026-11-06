"""Fig. 13: robustness tests of section 3.3, one per panel (Regime I, seed 1, 10 tauB; colour = lattice orientation).
(a, b) density: AO q = 0.05, A = 2 at area fraction 0.28 and 0.50; (c) system size: N = 576 in a 100 um box (AO q = 0.2,
A = 3); (d) 10 % polydispersity, AO q = 0.05, A = 7 (PHS core). Usage: python fig_robustness.py <Data dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from scipy.spatial import cKDTree
D, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 7})
P_ = [("a   density 0.28 (N = 225)", "NAF_density/2Tkb_20TauB_core_BD_AO_q0.05_phi0.16_dt256000_N225_seed1", 25, None, "AO q = 0.05, A = 2"),
      ("b   density 0.50 (N = 400)", "NAF_density/2Tkb_20TauB_core_BD_AO_q0.05_phi0.16_dt256000_N400_seed1", 25, None, "AO q = 0.05, A = 2"),
      ("c   system size: N = 576", "NAF_N576/3Tkb_10TauB_core_BD_AO_q0.2_phi0.633_dt32000_N576_seed1", 50, None, "AO q = 0.2, A = 3"),
      ("d   10 % polydisperse", "NAF/7Tkb_10TauB_core_BD_AO_q0.05_phi0.244_pd0.1_dt256000_seed1", 25, 0.1, "AO q = 0.05, A = 7 (12.3 k$_B$T)")]
cm = plt.get_cmap("twilight")
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.35), constrained_layout=True)
for a, (ttl, run, half, pd, lab) in zip(ax, P_):
    rd = os.path.join(D, run)
    X = np.loadtxt(rd + "/datax.csv", delimiter=",", skiprows=1); Y = np.loadtxt(rd + "/datay.csv", delimiter=",", skiprows=1)
    t = X[1::2, 0] / 13.514; k = int(np.argmin(np.abs(t - 10))); P = np.c_[X[0::2][k], Y[0::2][k]]; n = len(P)
    R = np.ones(n)
    if pd:
        rng = np.random.default_rng(1 + 5000); R = 1 + pd * np.clip(rng.standard_normal(4096), -2, 2)[:n]
    b = cKDTree(P).query_pairs(2.9, output_type="ndarray"); z = np.bincount(b.ravel(), minlength=n)
    d = P[b[:, 1]] - P[b[:, 0]]; e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0])); psi = np.zeros(n, complex)
    np.add.at(psi, b[:, 0], e); np.add.at(psi, b[:, 1], e)
    col = [cm((np.angle(p) % (2 * np.pi)) / (2 * np.pi)) if zz >= 2 else (0.75, 0.75, 0.75, 1) for p, zz in zip(psi, z)]
    ok = z >= 2; p6 = np.abs(psi[ok] / z[ok]).mean()
    a.add_collection(EllipseCollection(2 * R, 2 * R, 0, units="xy", offsets=P, transOffset=a.transData, facecolors=col, edgecolors="none"))
    a.set_xlim(-half - 0.5, half + 0.5); a.set_ylim(-half - 0.5, half + 0.5); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_color("#999"); s.set_linewidth(0.5)
    a.set_title(ttl, loc="left", fontsize=6.8, fontweight="bold")
    a.set_xlabel(f"{lab}\nz = {z.mean():.2f}, ψ$_6$ = {p6:.2f}", fontsize=6.3)
for e_ in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig13_robustness_density-size-polydispersity_snapshots.{e_}"), dpi=300)
print("ok")

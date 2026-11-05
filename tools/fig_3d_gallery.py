"""Fig. S (raw results): snapshots of the 3D runs of the chamber-height scan (batches 17a-i), seed 1.
Rows: Regime I (10 tauB), II (3500 nm, 5 tauB), III (10 tauB); each as a top view (colour = height) and a side view (y-z).
Columns: H = 2.2, 3, 4, 6, 10, 20 um. Usage: python fig_3d_gallery.py <hscan.npz> <out dir>"""
import sys, os, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
S = np.load(sys.argv[1]); out = sys.argv[2]
HS = ["2.2", "3", "4", "6", "10", "20"]
ROWS = [("I", "NAF_3D", r"^3Tkb_10TauB_(core_)?BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "10"),
        ("II", "ARF_3D", r"^3Tkb_5TauB_3500nm_BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "5"),
        ("III", "AAF_3D", r"^3Tkb_10TauB_(core_)?BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "10")]
def find(sub, pat, T):
    for k in S.files:
        s, n, t = k.split("|")
        if s == sub and t == T and re.match(pat, n): return S[k]
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5})
fig, ax = plt.subplots(6, 6, figsize=(7.2, 7.6), gridspec_kw={"height_ratios": [1, 0.32] * 3}, constrained_layout=True)
cm = plt.get_cmap("viridis")
for r, (reg, sub, pat, T) in enumerate(ROWS):
    for c, H in enumerate(HS):
        P = find(sub, pat.replace("{H}", re.escape(H)), T)
        at, asd = ax[2 * r, c], ax[2 * r + 1, c]
        for a_ in (at, asd):
            a_.set_xticks([]); a_.set_yticks([])
        if P is None:
            at.text(0.5, 0.5, "–", ha="center", transform=at.transAxes); continue
        h = P[:, 2] - 1.0; o = np.argsort(P[:, 2])
        col = cm(np.clip(h[o] / 4.0, 0, 1))
        at.add_collection(EllipseCollection(2, 2, 0, units="xy", offsets=P[o][:, :2], transOffset=at.transData, facecolors=col, edgecolors="none"))
        at.set_xlim(-25, 25); at.set_ylim(-25, 25); at.set_aspect("equal")
        o2 = np.argsort(P[:, 0])
        asd.add_collection(EllipseCollection(2, 2, 0, units="xy", offsets=P[o2][:, 1:3], transOffset=asd.transData, facecolors=cm(np.clip(h[o2] / 4.0, 0, 1)), edgecolors="none", alpha=0.8))
        asd.set_xlim(-25, 25); asd.set_ylim(0, min(float(H), 8.0) + 0.1); asd.set_aspect("equal")
        asd.axhline(float(H), color="#999", lw=0.6) if float(H) <= 8 else None
        if r == 0: at.set_title(f"H = {H} µm", fontsize=7.5, fontweight="bold")
        if c == 0:
            at.set_ylabel(f"Regime {reg}\n{'3500 nm, ' if reg == 'II' else ''}{T} τ$_B$\ntop view", fontsize=7)
            asd.set_ylabel("side", fontsize=6.5)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 4)); sm.set_array([])
fig.colorbar(sm, ax=ax[:, -1], shrink=0.4, label="height above floor contact (µm)")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"FigS_3D-runs_snapshots_height-scan.{e}"), dpi=300)
print("saved")

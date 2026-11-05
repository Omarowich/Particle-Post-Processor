"""Fig. 13 (draft): Regime IV in 3D (batches 22, 22b) vs 2D. Top views (x-y, colour = height) at 0, 1 and 3 tauB.
Rows: 2D F = 3000 (reference); 3D COMSOL flow at the 2D speed (F = 1000, 3000; H = 4, 20); 3D body force with wall
friction (F = 3000, 10000, 30000; H = 4, 20). Usage: python fig_regIV_3d.py <regIV3d.npz> <regIV2d.npz> <out dir>"""
import sys, os, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
S3 = np.load(sys.argv[1]); S2 = np.load(sys.argv[2]); out = sys.argv[3]
TS = ["0", "1", "3"]
def get3(F, H, kind, T):
    for k in S3.files:
        n, t = k.split("|")
        if t == T and f"_{F}F_" in n and f"_H{H}_" in n and kind in n: return S3[k]
ROWS = [("2D\nF 3000", lambda T: S2[f"2D_3000|{T}"])]
for kind, lab in (("comsolflow", "3D 2D-speed"), ("bodyforce", "3D body force")):
    for F in ((1000, 3000) if kind == "comsolflow" else (3000, 10000, 30000)):
        for H in (4, 20):
            ROWS.append((f"{lab}\nF {F:g}, H {H}", lambda T, F=F, H=H, kind=kind: get3(F, H, kind, T)))
cm = plt.get_cmap("viridis")
plt.rcParams.update({"font.size": 6.5})
fig, ax = plt.subplots(len(TS), len(ROWS), figsize=(7.2, 2.4), constrained_layout=True)
for c, (lab, fn) in enumerate(ROWS):
    for r, T in enumerate(TS):
        a = ax[r, c]; a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal"); a.set_xlim(-25, 25); a.set_ylim(-25, 25)
        P = fn(T)
        if P is None: continue
        col = cm(np.clip((P[:, 2] - 1) / 4, 0, 1))
        a.add_collection(EllipseCollection(2, 2, 0, units="xy", offsets=P[:, :2], transOffset=a.transData, facecolors=col, edgecolors="none"))
        if r == 0: a.set_title(lab, fontsize=5.6)
        if c == 0: a.set_ylabel(f"{T} τ$_B$", fontsize=7)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 4)); sm.set_array([])
fig.colorbar(sm, ax=ax[:, -1], shrink=0.8, label="height (µm)")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig13_RegimeIV_3D-vs-2D_top-views.{e}"), dpi=300)
print("saved")

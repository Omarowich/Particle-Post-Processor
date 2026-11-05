"""Fig. 13 (3D view, draft): Regime IV in 3D vs 2D in perspective. Rows: runs (as Fig. 13 columns); columns: 0, 1, 3 tauB.
Box drawn to the chamber (50 x 50 x H um; the 2D run on a 1-um plane); colour = height. Uses the app's 3D renderer.
Usage: python fig_regIV_3d_view.py <regIV3d.npz> <regIV2d.npz> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import viewer3d as v3
S3 = np.load(sys.argv[1]); S2 = np.load(sys.argv[2]); out = sys.argv[3]
TS = ["0", "1", "3"]
def get3(F, H, kind, T):
    for k in S3.files:
        n, t = k.split("|")
        if t == T and f"_{F}F_" in n and f"_H{H}_" in n and kind in n: return S3[k]
ROWS = [("2D, F = 3000", 2.2, lambda T: S2[f"2D_3000|{T}"])]
for kind, lab in (("comsolflow", "3D, 2D speed"), ("bodyforce", "3D, body force")):
    for F in ((1000, 3000) if kind == "comsolflow" else (3000, 10000, 30000)):
        for H in (4, 20):
            ROWS.append((f"{lab}, F = {F}, H = {H} µm", float(H), lambda T, F=F, H=H, kind=kind: get3(F, H, kind, T)))
cm = plt.get_cmap("viridis")
plt.rcParams.update({"font.size": 6.5})
fig = plt.figure(figsize=(7.2, 1.45 * len(ROWS)), constrained_layout=True)
gs = fig.add_gridspec(len(ROWS), len(TS))
for r, (lab, H, fn) in enumerate(ROWS):
    for c, T in enumerate(TS):
        ax = fig.add_subplot(gs[r, c], projection="3d")
        P = fn(T)
        if P is None: ax.set_axis_off(); continue
        cols = [cm(min(max((z - 1.0) / 4.0, 0), 1)) for z in P[:, 2]]
        v3._draw_3d(ax, P, cols, 25.0, H, 28, -60, 1.0, True, 1.4, 1.0, False, "#ffffff", 2.0, None, "none", 0.45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        if r == 0: ax.set_title(f"{T} τ$_B$", fontsize=8, fontweight="bold")
        if c == 0: ax.text2D(-0.05, 0.5, lab.replace(", H", "\nH"), transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=6.3)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 4)); sm.set_array([])
fig.colorbar(sm, ax=fig.axes, shrink=0.2, label="height above contact (µm)")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig13_RegimeIV_3D-vs-2D_3Dview.{e}"), dpi=250)
print("saved")

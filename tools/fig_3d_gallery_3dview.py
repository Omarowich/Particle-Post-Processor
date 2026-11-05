"""Fig. S (raw results, 3D view): perspective views of the 3D runs of the chamber-height scan, seed 1.
Rows: Regime I (10 tauB), II (3500 nm, 5 tauB), III (10 tauB); columns: H = 2.2 ... 20 um. Colour = height; heights stretched 2x so stacking is visible.
Uses viewer3d's renderer (same look as the app). Usage: python fig_3d_gallery_3dview.py <hscan.npz> <out dir>"""
import sys, os, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import viewer3d as v3
S = np.load(sys.argv[1]); out = sys.argv[2]
HS = ["2.2", "3", "4", "6", "10", "20"]
ROWS = [("I", "NAF_3D", r"^3Tkb_10TauB_(core_)?BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "10"),
        ("II", "ARF_3D", r"^3Tkb_5TauB_3500nm_BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "5"),
        ("III", "AAF_3D", r"^3Tkb_10TauB_(core_)?BD_AO_q0\.05_phi0\.182_dt256000_3D_H{H}_drho52_seed1$", "10")]
def find(sub, pat, T):
    for k in S.files:
        s, n, t = k.split("|")
        if s == sub and t == T and re.match(pat, n): return S[k]
plt.rcParams.update({"font.size": 7})
fig = plt.figure(figsize=(7.2, 4.4), constrained_layout=True)
gs = fig.add_gridspec(3, 6)
cm = plt.get_cmap("viridis")
for r, (reg, sub, pat, T) in enumerate(ROWS):
    for c, H in enumerate(HS):
        ax = fig.add_subplot(gs[r, c], projection="3d")
        P = find(sub, pat.replace("{H}", re.escape(H)), T)
        if P is None:
            ax.set_axis_off(); continue
        cols = [cm(min(max((z - 1.0) / 4.0, 0), 1)) for z in P[:, 2]]
        v3._draw_3d(ax, P, cols, 25.0, float(H), 28, -60, 1.0, True, 1.2, 1.0, False, "#ffffff", 2.0, None, "none", 0.45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
        if r == 0: ax.set_title(f"H = {H} µm", fontsize=7.5, fontweight="bold")
        if c == 0: ax.text2D(-0.08, 0.5, f"Regime {reg}\n{'3500 nm, ' if reg == 'II' else ''}{T} τ$_B$", transform=ax.transAxes,
                             rotation=90, va="center", ha="right", fontsize=7)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 4)); sm.set_array([])
fig.colorbar(sm, ax=fig.axes, shrink=0.5, label="height above floor contact (µm)")
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"FigS_3D-runs_snapshots_height-scan_3Dview.{e}"), dpi=300)
print("saved")

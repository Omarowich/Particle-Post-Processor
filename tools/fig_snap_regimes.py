import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, os.environ["HOME"] + "/mnt/AIRPODS Analyser/particle-pp-test")
import regime_state_model as m
D = os.environ["HOME"] + "/mnt/AIRPODS Paper 2627/Data/"
rows = [("Regime I (no field)\nA = 3, 30 τ$_B$", "NAF/3Tkb_30TauB_core_BD_seed1", "NAF/3Tkb_30TauB_core_BD_AO_q0.05_phi0.182_dt256000_seed1", "AO q = 0.05"),
        ("Regime II (field on)\nA = 3, d$_n$ = 4000 nm, 10 τ$_B$", "ARF/3Tkb_10TauB_4000nm_BD_seed1", "ARF/3Tkb_10TauB_4000nm_BD_AO_q0.2_phi0.633_dt32000_seed1", "AO q = 0.2"),
        ("Regime III (field off)\nA = 2, 20 τ$_B$", "AAF/2Tkb_20TauB_BD_seed1", "AAF/2Tkb_20TauB_BD_AO_q0.2_phi0.51_dt32000_seed1", "AO q = 0.2"),
        ("Regime IV (streaming)\nA = 3, F = 3000, 3 τ$_B$", "ASF/3Tkb_3TauB_3000F_BD_dt32000_seed1", "ASF/3Tkb_3TauB_3000F_BD_AO_q0.2_phi0.633_dt32000_seed1", "AO q = 0.2")]
def psi6(P, rc=2.9):
    d = P[:, None] - P[None]; r = np.hypot(d[..., 0], d[..., 1]); np.fill_diagonal(r, 99)
    ang = np.arctan2(d[..., 1], d[..., 0]); nb = r < rc
    z = np.where(nb, np.exp(6j * ang), 0).sum(1); n = nb.sum(1)
    psi = np.where(n > 0, z / np.maximum(n, 1), 0)
    return np.abs(psi), n, (np.degrees(np.angle(psi)) / 6.0) % 60.0
plt.rcParams.update({"font.size": 7})
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.3), constrained_layout=True)
for i, (title, lj, ao, aolab) in enumerate(rows):
    for j, (run, lab) in enumerate(((lj, "LJ"), (ao, aolab))):
        x, y, t = m.read_run(D + run); P = np.c_[x[-1], y[-1]]
        p6, n, th = psi6(P); ax = axs[j, i]
        for (xx, yy), a, nn in zip(P, th, n):
            ax.add_patch(Circle((xx, yy), 1.0, fc=plt.cm.twilight(a / 60.0) if nn > 0 else "#bbbbbb", ec="none"))
        ax.set_xlim(-26, 26); ax.set_ylim(-26, 26); ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((-25, -25), 50, 50, fill=False, ec="#c3c2b7", lw=1))
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_ylabel("LJ 12-6" if j == 0 else "AO", fontsize=8, fontweight="bold")
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title((title + "\n" if j == 0 else "") + f"{lab}: z = {n.mean():.1f}, ψ$_6$ = {p6[n > 0].mean() if (n > 0).any() else 0:.2f}", fontsize=6.8, fontweight="normal")
sm = plt.cm.ScalarMappable(cmap="twilight", norm=plt.Normalize(0, 60))
fig.colorbar(sm, ax=axs, orientation="vertical", shrink=0.6, label="lattice orientation (°); grey: no neighbour")
out = os.environ["HOME"] + "/mnt/AIRPODS Paper 2627/proposal and paper/5 figures/6 range vs form (2D)/Fig8_snapshots_RegimeI-IV_LJ-vs-AO_matched-B2.png"
fig.savefig(out, dpi=300); fig.savefig(out[:-4] + ".pdf"); print("ok")

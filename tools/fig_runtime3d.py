"""Fig. 3b (draft): 3D run time, Python BD vs COMSOL particle tracing. Values: medians over the finished 3D jobs
(Data/_sim_jobs status files, laptop, 4 runs at once) and Anam's COMSOL 3D run (Data/_runtime/comsol_runtimes_anam.csv).
Usage: python fig_runtime3d.py <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]
ROWS = [("COMSOL 3D (Anam)", 10, 780.0, 20300, "#7a2a9a"),
        ("Python 3D, no field", 144, 0.17, 44, "#2a78d6"),
        ("Python 3D, 1D field", 144, 0.19, 47, "#2a78d6"),
        ("Python 3D, computed\nfield + streaming", 144, 0.38, 98, "#2a78d6"),
        ("Python 3D, COMSOL\nin-plane flow", 144, 0.49, 124, "#2a78d6"),
        ("Python 3D, no field", 4000, 1.78, 455, "#1baf7a")]
SCAL = [(144, 44), (1000, 142), (1960, 359), (2890, 421), (4000, 455)]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]; y = np.arange(len(ROWS))[::-1]
for yy, (lab, n, ms, s, c) in zip(y, ROWS):
    a.barh(yy, s, color=c, height=0.62)
    a.text(s * 1.15, yy, f"{s:,.0f} s  ({ms:g} ms/step, N = {n:,})".replace(",", " "), va="center", fontsize=6.5)
a.set_yticks(y); a.set_yticklabels([r[0] for r in ROWS], fontsize=6.8); a.set_xscale("log"); a.set_xlim(10, 2e6)
a.set_xlabel("wall time per simulated τ$_B$ (s)")
a.set_title("a  3D models, time per τ$_B$", loc="left", fontsize=8.5, fontweight="bold")
a.set_xlabel("wall time per simulated τ$_B$ (s)   (Python ≈ 460× faster with 14× more particles)")
a = ax[1]
n, s = np.array(SCAL).T
a.loglog(n, s, "o-", color="#1baf7a", ms=4, label="Python 3D, no field (laptop, 1 core per run)")
a.loglog([10], [20300], "*", color="#7a2a9a", ms=11, label="COMSOL 3D, 10 particles (40-core workstation)")
nn = np.array([100, 5000]); a.loglog(nn, 44 * nn / 144, ":", color="#999", lw=0.8, label="∝ N")
a.set_xlabel("number of particles N"); a.set_ylabel("wall time per τ$_B$ (s)"); a.set_ylim(20, 1e5)
a.set_title("b  scaling with N", loc="left", fontsize=8.5, fontweight="bold"); a.legend(frameon=False, fontsize=6.2, loc="upper right")
os.makedirs(out, exist_ok=True)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig3b_runtime_3D_Python-vs-COMSOL.{e}"), dpi=300)
print("saved")

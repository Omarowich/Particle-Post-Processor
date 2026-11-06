"""Fig. 3 (paper 2): computational cost, COMSOL particle tracing vs Python Brownian dynamics.
Inputs (Data/_runtime): comsol_runtimes_nas.csv (MATLAB comsol_runtimes.m), python_runtimes.csv, benchmark_runtimes.csv.
Usage: python fig_runtime.py <runtime folder> <output folder>"""
import sys, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
C_COMSOL, C_PY = "#eb6834", "#2a78d6"
POT = {"LJ": ("LJ, 4k steps/τ$_B$", "#2a78d6", "o"),
       "AOq0.2": ("AO q 0.2, 32k", "#1baf7a", "s"),
       "AOq0.05": ("AO q 0.05, 256k", "#eda100", "D")}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": "#555", "xtick.color": "#333", "ytick.color": "#333",
                     "font.family": "DejaVu Sans"})

c = pd.read_csv(os.path.join(src, "comsol_runtimes_nas.csv"))
c = c[np.isfinite(c.s_per_tauB)].copy()
c["regime"] = c.regime.replace({"running simulations": "ARF"})
c = c.drop_duplicates(subset=["file", "solution_time_s"])
p = pd.read_csv(os.path.join(src, "python_runtimes.csv"))
b = pd.read_csv(os.path.join(src, "benchmark_runtimes.csv"))

HOST = {"DESKTOP-0T21IEI": ("o", "workstation, 2× Xeon, 40 cores"), "Omarowich-lap": ("^", "laptop, Intel Core 10th gen, 8 cores"),
        "m013.tum.vpn.lrz.de": ("s", "MacBook, Apple M2")}
def hosts(df):
    if "host" not in df: return [("o", df)]
    return [(HOST.get(h, ("o", h))[0], g) for h, g in df.groupby("host")]
REG = [("NAF", "I\nno field"), ("ARF", "II\nfield on"), ("AAF", "III\nfield off"), ("ASF", "IV\nstreaming")]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

# (a) cost per tau_B by regime
a = ax[0]; rng = np.random.default_rng(1)
for i, (r, lab) in enumerate(REG):
    cr = c[c.regime == r]; v = cr.s_per_tauB.values
    for mk, g in hosts(cr):
        a.scatter(i - 0.15 + rng.uniform(-0.09, 0.09, len(g)), g.s_per_tauB, s=11 if mk == "o" else 16, marker=mk,
                  color=C_COMSOL, alpha=0.75, lw=0, zorder=2)
    a.plot([i - 0.28, i - 0.02], [np.median(v)] * 2, color=C_COMSOL, lw=2, solid_capstyle="round", zorder=3)
    pv = p.loc[(p.regime == r) & (p.potential == "LJ"), "s_per_tauB"].values
    a.scatter(i + 0.15 + rng.uniform(-0.06, 0.06, len(pv)), pv, s=6, color=C_PY, alpha=0.5, lw=0, zorder=2)
    a.plot([i + 0.02, i + 0.28], [np.median(pv)] * 2, color=C_PY, lw=2, solid_capstyle="round", zorder=3)
    a.text(i, 3e5, f"×{np.median(v) / np.median(pv):.0f}", ha="center", va="bottom", fontsize=7.5, color="#333")
a.set_yscale("log"); a.set_ylim(1, 1e6); a.set_xlim(-0.55, 3.55)
a.set_xticks(range(4)); a.set_xticklabels([l for _, l in REG], fontsize=6.5)
a.set_ylabel("wall time per τ$_B$ (s)")
for y, t in ((60, "1 min"), (3600, "1 h"), (86400, "1 day")):
    a.axhline(y, color="#ccc", lw=0.5, zorder=0); a.text(3.6, y, t, fontsize=6.5, color="#888", ha="left", va="center", clip_on=False)
a.set_title("a  Cost per τ$_B$, LJ, by regime", loc="left", fontsize=8.5, fontweight="bold")

# (b) Regime II: cost vs field amplitude
a = ax[1]
r2 = c[c.regime == "ARF"].copy()
r2["dn"] = r2.run.str.extract(r"([\d.]+)nm")[0].astype(float)
r2 = r2[np.isfinite(r2.dn)]
for mk, g in hosts(r2):
    a.scatter(g.dn, g.s_per_tauB, s=10 if mk == "o" else 16, marker=mk, color=C_COMSOL, alpha=0.75, lw=0, zorder=2)
med = r2.groupby("dn").s_per_tauB.median()
a.plot(med.index, med.values, color=C_COMSOL, lw=1.6, zorder=3, label=f"COMSOL ({len(c)} runs in a)")
py2 = p[(p.regime == "ARF") & (p.potential == "LJ")]
a.axhline(py2.s_per_tauB.median(), color=C_PY, lw=1.6, label=f"Python BD ({(p.potential == 'LJ').sum()} runs in a)")
a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1, 1e6)
a.set_xlabel("field amplitude d$_n$ (nm)"); a.set_ylabel("wall time per τ$_B$ (s)")
if "host" in c:
    for h, (mk, lab) in HOST.items():
        if (c.host == h).any():
            a.scatter([], [], marker=mk, s=14, color="#888", lw=0, label=f"COMSOL on {lab} ({(c.host == h).sum()})")
fig.legend(*a.get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=3, fontsize=6.5,
           handlelength=1.4, columnspacing=1.2)
a.set_title("b  Regime II vs field amplitude", loc="left", fontsize=8.5, fontweight="bold")

# (c) Python benchmark: step cost vs runs at once
a = ax[2]
for k, (lab, col, mk) in POT.items():
    g = b[b.potential == k].groupby("runs_at_once").ms_per_step.agg(["mean", "min", "max"])
    a.errorbar(g.index, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], color=col, marker=mk,
               ms=4, lw=1.4, capsize=2, label=lab)
a.set_xticks([1, 2, 4]); a.set_ylim(0, 1.6); a.set_xlim(0.6, 4.4)
a.set_xlabel("runs at the same time"); a.set_ylabel("ms per step (N = 144)")
a.legend(frameon=False, loc="lower right", fontsize=6.5, borderaxespad=0.1)
a.set_title("c  Python (all pairs), per step", loc="left", fontsize=8.5, fontweight="bold")

# (d) cost per step vs particle number
a = ax[3]
bs_f = os.path.join(src, "bench_scaling.csv")
if os.path.isfile(bs_f):
    bs = pd.read_csv(bs_f)
    ST = {"dense 2D": ("#888888", "o", "Python, all pairs, 2D"), "neighbour list 2D": (C_PY, "o", "Python, neighbour list, 2D"),
          "neighbour list 3D": ("#184f95", "^", "Python, neighbour list, 3D")}
    for (pot, meth), g in bs.groupby(["potential", "method"]):
        col, mk, lab = ST[meth]
        a.plot(g.N, g.ms_per_step, color=col, marker=mk, ms=3.5, lw=1.3, ls="-" if pot.startswith("AO") else "--")
        if pot.startswith("AO") or meth == "dense 2D":
            a.text(g.N.iloc[-1] * 1.25, g.ms_per_step.iloc[-1], lab.replace("Python, ", ""), fontsize=6.3, color=col, va="center")
    a.plot([], [], color="#555", ls="-", lw=1.3, label="Python, AO q = 0.05"); a.plot([], [], color="#555", ls="--", lw=1.3, label="Python, LJ 12-6")
an_f = os.path.join(src, "comsol_runtimes_anam.csv")
cc = [c[np.isfinite(c.ms_per_step)]]
if os.path.isfile(an_f):
    an = pd.read_csv(an_f); an = an[np.isfinite(an.ms_per_step)].drop_duplicates(subset=["file", "solution_time_s"])
    an2 = an[(an.box_um == 50) & np.isfinite(an.n_particles)]
    a.scatter(an2.n_particles * np.exp(np.random.default_rng(2).uniform(-0.06, 0.06, len(an2))), an2.ms_per_step,
              s=9, color=C_COMSOL, alpha=0.6, lw=0, label="COMSOL, 2D, 50 µm box (Anam, ours)")
    d3 = an[an.run.str.contains("_3D", na=False)]
    if len(d3):
        a.scatter([10] * len(d3), d3.ms_per_step, s=60, marker="*", color=C_COMSOL, lw=0, label="COMSOL, 3D (N = 10)")
cn = c[np.isfinite(c.ms_per_step)]
a.scatter(144 * np.exp(np.random.default_rng(3).uniform(-0.06, 0.06, len(cn))), cn.ms_per_step, s=9, color=C_COMSOL, alpha=0.6, lw=0)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(7, 9e4); a.set_ylim(0.02, 3000)
a.set_xlabel("particle number N"); a.set_ylabel("ms per step")
a.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
a.set_title("d  Cost per step vs N", loc="left", fontsize=8.5, fontweight="bold")

os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"Fig6_runtime_COMSOL-vs-Python.{ext}"), dpi=300)
print("saved")

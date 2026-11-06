"""Fig. 5: validation of the Python model. (a-d) coordination z and (e-h) band width std(y) against time, COMSOL
(one run) vs Python (mean and min-max of 3 seeds), LJ 12-6, same start, Regime II. (i, j) time-step convergence
(LJ A = 3, 4000 nm). (k) two-particle pair distribution vs the Boltzmann factor of the coded potential (AO q = 0.05,
A = 3). (l) first-step displacements: COMSOL repeats the same noise, the Python seeds do not.
Usage: python fig_validation_overlap.py <validation_series.npz> <validation_g2p.npz> <simulation dir> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = np.load(sys.argv[1]); G = np.load(sys.argv[2]); SIM, out = sys.argv[3], sys.argv[4]
sys.path.insert(0, SIM); import airpods_bd as bd; from verify_ao import potential_from_force
plt.rcParams.update({"font.size": 6.8, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
OR, BL, GR = "#eb6834", "#2a78d6", "#888888"
fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
gs = fig.add_gridspec(3, 4)
CASES = ((0, 2000), (0, 4000), (5, 2000), (5, 4000))
L = iter("abcdefghijkl")
for j, (A, dn) in enumerate(CASES):
    C = S[f"C|{A}|{dn}"]; B = [S[k] for k in S.files if k.startswith(f"B|{A}|{dn}|")]
    t = B[0][:, 0]; n = min(len(b) for b in B)
    for i, (col, lab) in enumerate(((1, "coordination z"), (3, "band width std(y) (µm)"))):
        ax = fig.add_subplot(gs[i, j]); M = np.array([b[:n, col] for b in B])
        ax.fill_between(t[:n], M.min(0), M.max(0), color=BL, alpha=0.2, lw=0)
        ax.plot(t[:n], M.mean(0), color=BL, lw=1.6, label="Python, 3 seeds")
        ax.plot(C[:, 0], C[:, col], color=OR, lw=1.1, label="COMSOL")
        ax.set_title(f"{next(L)}   A = {A}, d$_n$ = {dn} nm", loc="left", fontsize=7, fontweight="bold")
        if j == 0: ax.set_ylabel(lab)
        if i == 1: ax.set_xlabel("t (τ$_B$)")
        if i == 0 and j == 0: ax.legend(frameon=False, fontsize=6)
# time step
R = [S[k] for k in S.files if k.startswith("TSref|")]; n = min(len(r) for r in R); t = R[0][:n, 0]
for j, (col, lab) in enumerate(((1, "coordination z"), (2, "local order ψ$_6$"))):
    ax = fig.add_subplot(gs[2, j]); M = np.array([r[:n, col] for r in R])
    ax.fill_between(t, M.min(0), M.max(0), color=GR, alpha=0.25, lw=0, label="τ$_B$/4000, 3 seeds (range)")
    for k, c, ls, lab2 in (("TS|dt2000", "#d39200", "-", "τ$_B$/2000"), ("TS|dt8000", OR, ":", "τ$_B$/8000")):
        if k in S.files: ax.plot(S[k][:, 0], S[k][:, col], color=c, ls=ls, lw=1.2, label=lab2)
    ax.set_xlim(0, 2); ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel(lab)
    ax.set_title(f"{next(L)}   time step, LJ A = 3, 4000 nm", loc="left", fontsize=7, fontweight="bold")
    if j == 0: ax.legend(frameon=False, fontsize=5.8, loc="lower right")
# two-particle g(r)
ax = fig.add_subplot(gs[2, 2])
r = np.linspace(1.8, 2.6, 4001); U = potential_from_force(np.linspace(1.5, 10, 200001), bd.radial_force(np.linspace(1.5, 10, 200001), 0.0, (0.05, 0.182)))
Ur = np.interp(r, np.linspace(1.5, 10, 200001), U)
ax.semilogy(r - 2, np.exp(-Ur), color=GR, lw=3, alpha=0.6, label="exp(−U/k$_B$T)")
for st, c, ls in ((32000, "#d39200", "-"), (256000, BL, "--")):
    h = G[f"AO_q0.05_A3|{st}"]; rr, cnt = h[:, 0], h[:, 1]; g = cnt / rr; g /= g[(rr > 2.2) & (rr < 2.5)].mean()
    ax.semilogy(rr - 2, g, color=c, ls=ls, lw=1.0, label=f"two particles, τ$_B$/{st:,}".replace(",", " "))
ax.set_xlim(-0.06, 0.2); ax.set_ylim(0.05, 2e3); ax.set_xlabel("r − 2a (µm)"); ax.set_ylabel("pair distribution g(r)")
ax.set_title(f"{next(L)}   two particles, AO q = 0.05, A = 3", loc="left", fontsize=7, fontweight="bold"); ax.legend(frameon=False, fontsize=5.8)
# noise
ax = fig.add_subplot(gs[2, 3]); kc, kb = S["kick_C"], S["kick_B"]
lim = 1.1 * max(np.abs(kc).max(), np.abs(kb).max())
ax.plot([-lim, lim], [-lim, lim], color="#ccc", lw=0.8)
ax.scatter(kb[:, 0], kb[:, 1], s=4, color=BL, label=f"Python, two seeds (r = {np.corrcoef(kb.T)[0, 1]:.2f})")
ax.scatter(kc[:, 0], kc[:, 1], s=4, color=OR, label=f"COMSOL, two runs (r = {np.corrcoef(kc.T)[0, 1]:.3f})")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("first-step Δx, run 1 (µm)"); ax.set_ylabel("same particle, run 2 (µm)")
ax.set_title(f"{next(L)}   Brownian noise", loc="left", fontsize=7, fontweight="bold"); ax.legend(frameon=False, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig5_validation_Python-vs-COMSOL_overlap_timestep_noise.{e}"), dpi=300)
print("saved")

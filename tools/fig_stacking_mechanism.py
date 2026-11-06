"""Fig. (paper 2, section 4.1): why a 3D band stacks. (a) share of particles on top of others against buoyant density for
four field amplitudes (Regime II, H = 20 um, 3 tauB); (b) the same against the energy ratio squeeze/lift;
(c) on top against time with and without thermal noise (same jittered start); (d, e) 3D views at 5 tauB.
Usage: python fig_stacking_mechanism.py <night_27.npz> <night_31.npz> <out dir>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp, zcolor
S27, S31, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
fig = plt.figure(figsize=(7.2, 4.3))
DN = {1000: "#c6dbef", 2000: "#6baed6", 3500: "#2171b5", 5000: "#08306b"}; DR = (52, 150, 300, 600, 1000)
def top(k): return S27[k + "|ser"][-1, 4]
k_ = 0.124
def ratio(dn, dr):   # radiation energy gained moving from y = 3 to 1 um, over the cost of lifting one layer (1.63 um)
    F0 = 18.4 * (dn / 4000) ** 2; return (F0 / k_) * (np.cos(k_ * 1) - np.cos(k_ * 3)) / (1.63 * 0.53 * dr / 52)
ax = fig.add_axes([0.07, 0.60, 0.25, 0.33])
for dn, c in DN.items():
    ax.plot(DR, [top(f"{dn}|{dr}|1") for dr in DR], color=c, marker="o", ms=3, lw=1.2, label=f"{dn} nm")
ax.set_xscale("log"); ax.set_xticks(DR); ax.set_xticklabels([str(d) for d in DR]); ax.set_xlabel("Δρ (kg/m$^3$)"); ax.set_ylabel("on top of others")
ax.legend(fontsize=5.6, frameon=False, title="d$_n$", title_fontsize=5.6); ax.set_ylim(0, 0.6); ax.set_title("a  stacking vs weight and field", **lab)
ax = fig.add_axes([0.40, 0.60, 0.25, 0.33])
for dn, c in DN.items():
    ax.scatter([ratio(dn, dr) for dr in DR], [top(f"{dn}|{dr}|1") for dr in DR], color=c, s=12, zorder=3)
ax.axvline(1, color="#999", ls="--", lw=0.6); ax.set_xscale("log"); ax.set_ylim(0, 0.6)
ax.set_xlabel("squeeze energy / lift cost"); ax.set_ylabel("on top of others"); ax.set_title("b  energy balance", **lab)
ax = fig.add_axes([0.73, 0.60, 0.25, 0.33])
for k, c, l in (("noise", "#e34948", "with thermal noise"), ("nonoise", "#33415c", "no noise")):
    s = S31[k + "|ser"]; ax.plot(s[:, 0], s[:, 4], color=c, lw=1.3); ax.plot(s[:, 0], s[:, 5], color=c, lw=1.0, ls="--")
ax.text(4.9, 0.84, "at node, noise", color="#e34948", fontsize=5.6, ha="right"); ax.text(4.9, 0.55, "at node, no noise", color="#33415c", fontsize=5.6, ha="right")
ax.text(4.9, 0.40, "on top, noise", color="#e34948", fontsize=5.6, ha="right"); ax.text(4.9, 0.03, "on top, no noise", color="#33415c", fontsize=5.6, ha="right")
ax.set_xlabel("t (τ$_B$)"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 5); ax.set_ylabel("fraction")
ax.set_title("c  same start, noise on/off", **lab)
for j, (k, t) in enumerate((("noise", "d  with thermal noise, 5 τ$_B$"), ("nonoise", "e  without thermal noise, 5 τ$_B$"))):
    a = fig.add_axes([0.0 + 0.5 * j, 0.02, 0.5, 0.48], projection="3d"); P = S31[k + "|snap"]
    persp(a, P, np.zeros(len(P), bool), ((-25, 25), (-12, 12), (0, 6)), t, s=10, color=zcolor(P, 4.0), zoom=1.55,
          note=f"on top {np.mean(P[:, 2] > 2.5):.0%}, within 3 µm of node {np.mean(abs(P[:, 1]) < 3):.0%}")
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_stacking-mechanism_squeeze-vs-weight_noise-control.{e}", dpi=300)
print("saved")

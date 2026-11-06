"""Fig. (paper 2, section 4.3): gel or crystal in a 3D chamber, by true well depth, with the shallow-well runs continued to
100 tauB. (a-d) 3D views of the last frame (dark red = crystal-like); (e) crystal-like fraction against time; (f) at the end.
Usage: python fig_gel_3d_long.py <sec43_render_data.npz> <night_26.npz> <out dir>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from persp3d import persp
S, L, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
DEP = {"coul": {"A1": 4.0, "A2": 5.0, "A3": 5.6, "A5": 6.8}, "phs1000": {"A2": 7.4, "A3": 8.7, "A7": 12.3, "A10": 14.0},
       "phs1000long": {"A7": 12.3, "A10": 14.0}, "phs4000": {"A3": 8.7, "A7": 12.3}}
fig = plt.figure(figsize=(7.2, 3.9))
pan = [(L, "A2", "a  5.0 k$_B$T, 100 τ$_B$"), (L, "A3", "b  5.6 k$_B$T, 100 τ$_B$"), (L, "A5", "c  6.8 k$_B$T, 100 τ$_B$"), (S, "gel|phs1000long|A7", "d  12.3 k$_B$T (PHS core), 30 τ$_B$")]
for j, (D, k, t) in enumerate(pan):
    a = fig.add_axes([0.25 * j, 0.47, 0.25, 0.52], projection="3d"); P = D[k + "|snap"]; fl = D[k + ("|xf" if D is L else "|xflag")]
    sm = D[k + "|ser"][-1, 1] if D is L else D[k + "|S"][-1]
    persp(a, P, fl, ((-25, 25), (-25, 25), (0, 20)), t, s=6, elev=22, azim=-58, note=f"crystal-like {fl.mean():.2f}, S$_{{max}}$ {sm:.2f}")
ax = fig.add_axes([0.07, 0.10, 0.40, 0.30]); cm = plt.get_cmap("viridis")
for core, ls in (("coul", "-"), ("phs1000", "--"), ("phs1000long", "--"), ("phs4000", ":")):
    for A, dep in DEP[core].items():
        k = f"gel|{core}|{A}"
        if k + "|t" not in S: continue
        t, fx = S[k + "|t"], S[k + "|fx"]
        if core == "coul" and A in ("A2", "A3", "A5"):
            s = L[A + "|ser"]; t = np.r_[t, 20 + s[1:, 0]]; fx = np.r_[fx, s[1:, 3]]
        ax.plot(t, fx, color=cm((dep - 4) / 10.5), lw=1.2, ls=ls)
sm_ = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(4, 14.5)); sm_.set_array([])
fig.colorbar(sm_, ax=ax, pad=0.01, aspect=12, label="true well depth (k$_B$T)")
for ls, l in (("-", "Coulomb, N = 1000"), ("--", "PHS, N = 1000"), (":", "PHS, N = 4000")): ax.plot([], [], color="#555", ls=ls, label=l)
ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("crystal-like fraction"); ax.set_xlim(0, 100); ax.set_ylim(0, 0.45)
ax.legend(fontsize=5.6, frameon=False, loc="upper left"); ax.set_title("e  crystal-like fraction vs time", **lab)
ax = fig.add_axes([0.62, 0.10, 0.36, 0.30])
d = sorted((DEP["coul"][A], (L[A + "|ser"][-1, 3] if A in ("A2", "A3", "A5") else S[f"gel|coul|{A}|fx"][-1])) for A in DEP["coul"])
ax.plot(*zip(*d), marker="o", color="#2a78d6", lw=1.0, ms=3.5, label="Coulomb, 100 τ$_B$ (4.0: 20 τ$_B$)")
for core, mk, c, l in (("phs1000", "s", "#e34948", "PHS, 10 τ$_B$"), ("phs1000long", "D", "#a3222b", "PHS, 30 τ$_B$"), ("phs4000", "^", "#eb6834", "PHS, N = 4000, 10 τ$_B$")):
    d = sorted((dep, S[f"gel|{core}|{A}|fx"][-1]) for A, dep in DEP[core].items() if f"gel|{core}|{A}|fx" in S)
    ax.plot(*zip(*d), marker=mk, color=c, lw=1.0, ms=3.5, label=l)
ax.axvspan(4, 5, color="#2a78d6", alpha=0.08, lw=0); ax.text(4.5, 0.30, "2D order\nonset", ha="center", va="top", fontsize=5.5, color="#2a78d6")
ax.set_xlabel("true well depth (k$_B$T)"); ax.set_ylabel("crystal-like fraction, end"); ax.set_xlim(3.5, 14.5); ax.set_ylim(0, 0.45)
ax.legend(fontsize=5.3, frameon=False, loc="upper right"); ax.set_title("f  crystal-like fraction vs well depth", **lab)
for e in ("png", "pdf"): fig.savefig(f"{out}/Fig_3D-gel-vs-crystal_by-well-depth_100tauB.{e}", dpi=300)
print("saved")

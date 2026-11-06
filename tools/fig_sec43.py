"""Figs. 16 and 17 (paper 2, section 4.3). 16: Regime III and IV in 3D - perspective views of the last frame
(dark red = crystal-like, >= 4 coherent q6 bonds, 2.2 um cutoff) and time curves. 17: 3D gel vs crystal by true
well depth - perspective views and crystal-like fraction vs time / depth.
Usage: python fig_sec43.py <sec43_render_data.npz> <out dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = np.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
XT, OT = "#a3222b", "#9fb3c8"
from persp3d import persp
lab = dict(fontsize=7.2, fontweight="bold", loc="left", pad=24)
# ---------------- Fig 16 ----------------
fig = plt.figure(figsize=(7.2, 4.0))
top = [(0.0, "III|H3", "a  III, H = 3 µm, 10 τ$_B$", 6), (0.25, "III|H20", "b  III, H = 20 µm, 10 τ$_B$", 6),
       (0.5, "IV|comsol|F1000|H4", "c  IV, F = 1000, H = 4 µm, 3 τ$_B$", 6),
       (0.75, "IV|comsol|F1000|H20", "d  IV, F = 1000, H = 20 µm, 3 τ$_B$", 6)]
for x, k, title, zt in top:
    ax = fig.add_axes([x, 0.52, 0.25, 0.48], projection="3d")
    P = S[k + "|snap"]; fl = S[k + "|xflag"]
    persp(ax, P, fl, ((-25, 25), (-16, 16), (0, zt)), title,
          note=f"crystal-like {fl.mean():.2f}, S$_{{max}}$ {S[k + '|S'][-1]:.2f}")
# e: III crystal-like fraction vs t, by height
ax = fig.add_axes([0.06, 0.11, 0.26, 0.26])
cols = {"2.2": "#33415c", "3": "#2a78d6", "4": "#53a7ee", "6": "#eb6834", "20": "#e34948"}
for H, c in cols.items():
    k = f"III|H{H}"; ax.plot(S[k + "|t"], S[k + "|fx"], color=c, lw=1.2, label=f"H = {H}" + (" (= 10)" if H == "20" else ""))
ax.set_xlabel("t (τ$_B$), field off"); ax.set_ylabel("crystal-like fraction"); ax.set_ylim(0, 0.9); ax.set_xlim(0, 10)
ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4); ax.set_title("e  Regime III from a 3D band", **lab)
# f: IV S_max vs t, 2D vs 3D (COMSOL flow, no-slip profile)
ax = fig.add_axes([0.40, 0.11, 0.26, 0.26])
for F, c in (("1000", "#2a78d6"), ("3000", "#e34948")):
    for s in "123":
        k = f"IV2D|F{F}|s{s}"; m = S[k + "|t"] <= 3.0
        ax.plot(S[k + "|t"][m], S[k + "|S"][m], color=c, lw=0.7, ls="--", alpha=0.6)
    for H, lw, ls in (("4", 1.4, "-"), ("20", 1.4, ":")):
        k = f"IV|comsol|F{F}|H{H}"; ax.plot(S[k + "|t"], S[k + "|S"], color=c, lw=lw, ls=ls)
for c, l in (("#2a78d6", "F = 1000"), ("#e34948", "F = 3000")): ax.plot([], [], color=c, lw=1.4, label=l)
for ls, lw, l in (("--", 0.7, "2D, 3 seeds"), ("-", 1.4, "3D, H = 4"), (":", 1.4, "3D, H = 20")): ax.plot([], [], color="#555", ls=ls, lw=lw, label=l)
ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("S$_{max}$"); ax.set_xlim(0, 3); ax.set_ylim(0, 1.05)
ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4); ax.set_title("f  Regime IV, COMSOL flow, 2D vs 3D", **lab)
# g: IV body force with wall friction
ax = fig.add_axes([0.73, 0.11, 0.26, 0.26])
for F, c in (("3000", "#53a7ee"), ("10000", "#2a78d6"), ("30000", "#33415c")):
    for H, ls in (("4", "-"), ("20", ":")):
        k = f"IV|body|F{F}|H{H}"; ax.plot(S[k + "|t"], S[k + "|S"], color=c, lw=1.3, ls=ls, label=f"F = {F}" if H == "4" else None)
ax.plot([], [], color="#555", ls="-", label="H = 4"); ax.plot([], [], color="#555", ls=":", label="H = 20")
ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("S$_{max}$"); ax.set_xlim(0, 3); ax.set_ylim(0, 1.05)
ax.legend(fontsize=5.4, frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.8, handlelength=1.4); ax.set_title("g  Regime IV, body force + wall friction", **lab)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig16_RegimeIII-IV-in-3D_perspective-and-curves.{e}"), dpi=300)
# ---------------- Fig 17 ----------------
lab = dict(fontsize=7.2, fontweight="bold", loc="left")
DEP = {"coul": {"A1": 4.0, "A2": 5.0, "A3": 5.6, "A5": 6.8}, "phs1000": {"A2": 7.4, "A3": 8.7, "A7": 12.3, "A10": 14.0},
       "phs1000long": {"A7": 12.3, "A10": 14.0}, "phs4000": {"A3": 8.7, "A7": 12.3}}
fig = plt.figure(figsize=(7.2, 3.9))
pan = [(0.0, "gel|coul|A2", "a  5.0 k$_B$T (Coulomb core), 20 τ$_B$"), (0.25, "gel|coul|A3", "b  5.6 k$_B$T (Coulomb core), 20 τ$_B$"),
       (0.5, "gel|phs1000long|A7", "c  12.3 k$_B$T (PHS core), 30 τ$_B$"), (0.75, "gel|phs4000|A3", "d  8.7 k$_B$T, N = 4000, 10 τ$_B$")]
for x, k, title in pan:
    ax = fig.add_axes([x, 0.47, 0.25, 0.52], projection="3d"); P = S[k + "|snap"]; fl = S[k + "|xflag"]
    L = 50 if "4000" in k else 25
    persp(ax, P, fl, ((-L, L), (-L, L), (0, 20)), title, elev=22, azim=-58,
          note=f"crystal-like {fl.mean():.2f}, S$_{{max}}$ {S[k + '|S'][-1]:.2f}")
ax = fig.add_axes([0.07, 0.12, 0.40, 0.29])
cm = plt.get_cmap("viridis")
for core, ls in (("coul", "-"), ("phs1000", "--"), ("phs1000long", "--"), ("phs4000", ":")):
    for A, dep in DEP[core].items():
        k = f"gel|{core}|{A}"
        if k + "|t" not in S: continue
        ax.plot(S[k + "|t"], S[k + "|fx"], color=cm((dep - 4) / 10.5), lw=1.2, ls=ls)
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(4, 14.5)); sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.01, aspect=12, label="true well depth (k$_B$T)")
for ls, l in (("-", "Coulomb, N = 1000"), ("--", "PHS, N = 1000"), (":", "PHS, N = 4000")): ax.plot([], [], color="#555", ls=ls, label=l)
ax.set_xlabel("t (τ$_B$)"); ax.set_ylabel("crystal-like fraction"); ax.set_xlim(0, 30); ax.set_ylim(0, 0.12)
ax.legend(fontsize=5.6, frameon=False, loc="upper right"); ax.set_title("e  crystal-like fraction vs time", **lab)
ax = fig.add_axes([0.62, 0.12, 0.36, 0.29])
for core, mk, c, l in (("coul", "o", "#2a78d6", "Coulomb, 20 τ$_B$"), ("phs1000", "s", "#e34948", "PHS, 10 τ$_B$"),
                       ("phs1000long", "D", "#a3222b", "PHS, 30 τ$_B$"), ("phs4000", "^", "#eb6834", "PHS, N = 4000, 10 τ$_B$")):
    d = sorted((dep, S[f"gel|{core}|{A}|fx"][-1]) for A, dep in DEP[core].items() if f"gel|{core}|{A}|fx" in S)
    ax.plot(*zip(*d), marker=mk, color=c, lw=1.0, ms=3.5, label=l)
ax.axvspan(4, 5, color="#2a78d6", alpha=0.08, lw=0); ax.text(4.5, 0.112, "2D order\nonset", ha="center", va="top", fontsize=5.5, color="#2a78d6")
ax.set_xlabel("true well depth (k$_B$T)"); ax.set_ylabel("crystal-like fraction, end"); ax.set_xlim(3.5, 14.5); ax.set_ylim(0, 0.12)
ax.legend(fontsize=5.4, frameon=False, loc="upper right"); ax.set_title("f  crystal-like fraction vs well depth", **lab)
for e in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig17_3D-gel-vs-crystal_by-well-depth.{e}"), dpi=300)
print("saved")

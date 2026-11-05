"""Fig. 8e (batch 16, IDT lids) and Fig. 12 (batch 18, computed streaming), drafts.
Usage: python fig_lids_streaming.py <obs_16_18.csv> <out dir>"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
d["tag"] = d.name.str.extract(r"drho52_?(.*)_seed")[0]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})

# ---- Fig 8e: IDT standing SAW, 20 MHz, N = 576 in 100 um box, AO q = 0.05, A = 3, 4000 nm
L = d[d.tag.str.startswith("N576_box100_SSAW20", na=False)].copy(); L["lid"] = L.tag.str.split("_").str[-1]
LC = {"hard": "#2a78d6", "pdms": "#eb6834", "absorb": "#1baf7a"}; LN = {"hard": "glass (hard)", "pdms": "PDMS", "absorb": "absorbing"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
for lid in ("hard", "pdms", "absorb"):
    for H, ls in ((6.0, "--"), (20.0, "-")):
        g = L[(L.lid == lid) & (L.H == H)].groupby("t")[["band5", "f_on_top", "h_mean"]].agg(["mean", "std"])
        for k, v in enumerate(("band5", "f_on_top", "h_mean")):
            ax[k].errorbar(g.index, g[(v, "mean")], yerr=g[(v, "std")].fillna(0), color=LC[lid], ls=ls, marker="o", ms=3, capsize=2, lw=1.2,
                           label=f"{LN[lid]}, H = {H:g} µm")
for k, (lab, tt) in enumerate((("fraction within 5 µm of a node", "a  band growth"), ("fraction resting on another", "b  stacking"),
                               ("mean particle height (µm)", "c  height (floor contact = 1)"))):
    ax[k].set_ylabel(lab); ax[k].set_xlabel("time in the field (τ$_B$)"); ax[k].set_title(tt, loc="left", fontsize=8.5, fontweight="bold")
ax[2].set_yscale("log"); ax[2].set_yticks([1, 2, 5, 10, 20]); ax[2].set_yticklabels(["1", "2", "5", "10", "20"])
ax[0].legend(frameon=False, fontsize=6, loc="lower right")
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig8e_IDT-lids_band-stacking-height.{ext}"), dpi=300)

# ---- Fig 12: computed streaming (batch 18), band start, A = 3, 3 tauB
S = d[d.tag.str.contains("stream|both", na=False) & (d.t == 3)].copy()
S["drive"] = np.where(S.tag.str.startswith("side"), "side wall, glass lid", "IDT 20 MHz, absorbing lid")
S["mode"] = np.where(S.tag.str.contains("streamonly"), "streaming only", "streaming + radiation")
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
grp = [(dr, H) for dr in ("side wall, glass lid", "IDT 20 MHz, absorbing lid") for H in (4.0, 20.0)]
x = np.arange(len(grp)); w = 0.38
for k, (v, lab, tt) in enumerate((("band5", "fraction within 5 µm of the node", "a  band kept?"), ("Smax", "largest cluster fraction", "b  aggregate intact?"),
                                  ("f_on_top", "fraction resting on another", "c  stacking"))):
    for i, (mode, col) in enumerate((("streaming + radiation", "#eb6834"), ("streaming only", "#2a78d6"))):
        m = [S[(S.drive == dr) & (S.H == H) & (S["mode"] == mode)][v].mean() for dr, H in grp]
        s = [S[(S.drive == dr) & (S.H == H) & (S["mode"] == mode)][v].std() for dr, H in grp]
        ax[k].bar(x + (i - 0.5) * w, m, w, yerr=s, color=col, capsize=2, label=mode)
    ax[k].set_xticks(x); ax[k].set_xticklabels([f"{'side' if dr.startswith('side') else 'IDT'}\nH = {H:g}" for dr, H in grp])
    ax[k].set_ylabel(lab); ax[k].set_ylim(0, 1.05); ax[k].set_title(tt, loc="left", fontsize=8.5, fontweight="bold")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="outside lower center")
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig12_computed-streaming_band-breakup.{ext}"), dpi=300)
print("saved")

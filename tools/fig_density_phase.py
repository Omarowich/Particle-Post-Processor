"""Fig. 9 (paper 2): Regime I phase diagrams against colloid area fraction, AO q = 0.05 vs LJ 12-6, 10 tauB.
Uses the app's phase-diagram code (regime_state_model.fit_phase_map / fig_phase_map)."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, sys.argv[4] if len(sys.argv) > 4 else "/home/claude/particle-post-processor")
import regime_state_model as rsm
d = pd.read_csv(sys.argv[1]); r = pd.read_csv(sys.argv[2]); out = sys.argv[3]; T = 10.0
d["N"] = d.folder.str.extract(r"_N(\d+)_")[0].astype(int)
r = r[r.model.isin(["AO q=0.05", "LJ 12-6"])].copy(); r["N"] = 144
a = pd.concat([d, r]); a = a[np.isclose(a.t_tauB, T) & (a.A <= 7)].copy()
a["phi_area"] = (a.N * np.pi / 2500).round(2); a["name"] = a.folder; a["t_eval_tauB"] = T
phi_net = np.pi / 2.9 ** 2
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True, sharey=True)
for k, (m, lab) in enumerate((("AO q=0.05", "AO q = 0.05"), ("LJ 12-6", "LJ 12-6"))):
    g = a[a.model == m].reset_index(drop=True)
    pm = rsm.fit_phase_map(g, "NAF", "phi_area", "AO" if m.startswith("AO") else "LJ")
    rsm.fig_phase_map(pm, ax=ax[k], legend=(k == 0), A_values=sorted(a.A.unique()), x_note=(phi_net, "network label\nnot meaningful →"),
                      title=f"{'ab'[k]}  {lab}, Regime I, {T:g} τ$_B$")
    ax[k].set_xlabel("colloid area fraction")
ax[1].set_ylabel("")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"): fig.savefig(os.path.join(out, f"Fig9_RegimeI_density-phase-diagram.{ext}"), dpi=300)
print(a.groupby(["model", "phi_area"]).A.unique())

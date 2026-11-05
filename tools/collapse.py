"""Does a contact-shell second virial coefficient B2c* (attraction counted only within r_c of contact) collapse the
order of all attraction models onto one curve, where the full B2* does not?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/particle-post-processor"); sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import sim_gui as sg, airpods_bd as bd
T = "/mnt/user-data/uploads/AIRPODS Paper 2627/Data/_tools/"
base = {"core": "coulomb", "core_Z": 100, "q": 0.05, "phi_p": 0.1, "A": 0.0, "lj_cutoff_sigma": 3.0}
EXTRA = {"2n-n LJ": ("LJ2n", "n", 57.55), "Yukawa": ("YUK", "kappa", 17.28), "square well": ("SW", "width", 0.03319)}
r = np.linspace(1.5, 12, 120001)
cache = {}
NEW = {"2n-n LJ n=8": ("LJ2n", "n", 8.0, {1: 1.32, 2: 2.421, 3: 3.436, 5: 5.4}),
       "2n-n LJ n=12": ("LJ2n", "n", 12.0, {1: 1.82, 2: 3.005, 3: 4.022, 5: 5.949}),
       "Yukawa k=2": ("YUK", "kappa", 2.0, {1: 1.837, 2: 3.206, 3: 4.379, 5: 6.522}),
       "Yukawa k=4": ("YUK", "kappa", 4.0, {1: 2.605, 2: 4.002, 3: 5.117, 5: 7.137})}
def states(f):
    """State table, with the batch-19 intermediate-range models appended to Regime I."""
    d = pd.read_csv(T + f)
    if "RegimeI." in f:
        try:
            n = pd.read_csv(T + "obs_19.csv")
            lab = {"LJ2n_n8": "2n-n LJ n=8", "LJ2n_n12": "2n-n LJ n=12", "YUK_k2": "Yukawa k=2", "YUK_k4": "Yukawa k=4"}
            n["model"] = n.name.str.extract(r"BD_(LJ2n_n\d+|YUK_k\d+)")[0].map(lab)
            n["A"] = n.name.str.extract(r"^(\d+)Tkb")[0].astype(float); n["t_tauB"] = n.t
            d = pd.concat([d, n[["model", "A", "t_tauB", "z", "Smax", "psi6"]]], ignore_index=True)
        except FileNotFoundError:
            pass
    return d
def cfg(model, A):
    if model in NEW:
        pot, pk, pv, E = NEW[model]; return {**base, "potential": pot, pk: pv, "eps": E[int(A)]}
    if model == "LJ 12-6":
        return {**base, "potential": "LJ", "A": A}
    if model.startswith("AO"):
        q = float(model.split("=")[1]); phi, _ = sg.match_phi({**base, "q": q}, A); return {**base, "potential": "AO", "q": q, "phi_p": phi}
    pot, pk, pv = EXTRA[model]; c = {**base, "potential": pot, pk: pv, "eps": 1.0}
    e, _ = sg.match_eps(c, A); return {**c, "eps": e}
def b2s(model, A, cuts):
    key = (model, A)
    if key not in cache:
        _, U, _ = sg.potential_curve(cfg(model, A), r); cache[key] = U
    U = cache[key]; out = []
    for rc in cuts:
        Uc = np.where(r <= rc, U, 0.0) if rc else U
        f = 1 - np.exp(-np.clip(Uc, -50, 700))
        out.append((np.trapezoid(f * r, r) + r[0] ** 2 / 2) / (bd.SIGMA ** 2 / 2))
    return out
if __name__ == '__main__':
    CUTS = [None, 2.1, 2.2, 2.4, 2.6]          # full, 1.05, 1.10, 1.20, 1.30 sigma
    rows = []
    if __name__ != "__main__":
        pass
    for f, reg, T_ in (("states_RegimeI.csv", "I", 10.0), ("states_RegimeIII.csv", "III", 10.0)):
        d = states(f); d = d[np.isclose(d.t_tauB, T_) & (d.A > 0)]
        for (m, A), g in d.groupby(["model", "A"]):
            try:
                bs = b2s(m, float(A), CUTS)
            except Exception as ex:
                continue
            rows.append(dict(regime=reg, model=m, A=A, psi6=g.psi6.mean(), psd=g.psi6.std(), z=g.z.mean(),
                             **{("B2" if c is None else f"B2c_{c/2:.2f}"): b for c, b in zip(CUTS, bs)}))
    R = pd.DataFrame(rows); R.to_csv("/home/claude/sim/b2c/collapse.csv", index=False)
    # onset of order (psi6 = 0.8 crossing) per model, in each variable; spread across models = how well it collapses
    cols = [c for c in R.columns if c.startswith("B2")]
    for reg in ("I", "III"):
        print("Regime", reg)
        on = {}
        for m, g in R[R.regime == reg].groupby("model"):
            g = g.sort_values("A")
            for c in cols:
                x = -g[c].values; y = g.psi6.values
                i = np.where((y[:-1] < 0.8) & (y[1:] >= 0.8))[0]
                if len(i):
                    k = i[0]; xx = np.exp(np.interp(0.8, [y[k], y[k + 1]], [np.log(max(x[k], 1e-3)), np.log(max(x[k + 1], 1e-3))]))
                    on.setdefault(c, {})[m] = -xx
        for c, dct in on.items():
            v = np.array(list(dct.values()))
            print(f"  {c:10s} onset per model: " + ", ".join(f"{k}: {x:.2f}" for k, x in dct.items()) +
                  f"  -> spread (max/min of |onset|) = {np.abs(v).max() / np.abs(v).min():.2f}")

"""Cutoff scan for the contact-shell virial coefficient and the cumulative B2*(r) curves (data for Fig. B2c)."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/sim/b2c")
import collapse as C                       # reuses cfg(), potential cache, state tables
cuts = np.round(np.arange(1.02, 1.52, 0.02), 2)
rows = []
for f, reg in (("states_RegimeI.csv", "I"), ("states_RegimeIII.csv", "III")):
    d = C.states(f); d = d[np.isclose(d.t_tauB, 10.0) & (d.A > 0)]
    for (m, A), g in d.groupby(["model", "A"]):
        try:
            bs = C.b2s(m, float(A), [None] + [2.0 * c for c in cuts])
        except Exception:
            continue
        rows.append(dict(regime=reg, model=m, A=A, psi6=g.psi6.mean(), psd=g.psi6.std(), B2=bs[0],
                         **{f"c{c:.2f}": b for c, b in zip(cuts, bs[1:])}))
R = pd.DataFrame(rows); R.to_csv("/home/claude/sim/b2c/scan.csv", index=False)
# cumulative B2*(r) at matched A = 3 for every model
cum = {}
for m in ("LJ 12-6", "AO q=0.05", "2n-n LJ", "Yukawa", "square well", "2n-n LJ n=8", "2n-n LJ n=12", "Yukawa k=2", "Yukawa k=4"):
    C.b2s(m, 3.0, [None]); U = C.cache[(m, 3.0)]; r = C.r
    f = 1 - np.exp(-np.clip(U, -50, 700))
    cumv = (np.concatenate([[0], np.cumsum(0.5 * (f[1:] * r[1:] + f[:-1] * r[:-1]) * np.diff(r))]) + r[0] ** 2 / 2) / (C.bd.SIGMA ** 2 / 2)
    sel = slice(None, None, 50)
    cum[m] = dict(r=(r[sel] / C.bd.SIGMA).tolist(), b=cumv[sel].tolist())
json.dump(cum, open("/home/claude/sim/b2c/cumulative.json", "w"))
print(len(R), "rows")

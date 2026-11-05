"""Lindemann window centred on the well minimum: attraction counted only for |r - r_min| <= w*sigma.
B2w* = (2/sigma^2) * integral over the window of (1 - exp(-bU)) r dr (attractive part only, no core term)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/sim/b2c"); import collapse as C
def onset(g, col):
    g = g.sort_values("A"); x = -g[col].values; y = g.psi6.values
    i = np.where((y[:-1] < 0.8) & (y[1:] >= 0.8))[0]
    if not len(i) or min(x[i[0]], x[i[0] + 1]) <= 0: return np.nan
    k = i[0]; return float(np.exp(np.interp(0.8, [y[k], y[k + 1]], [np.log(x[k]), np.log(x[k + 1])])))
S = C.bd.SIGMA; W = (0.05, 0.1, 0.15, 0.2, 0.3)
rows = []
for fn, reg in (("states_RegimeI.csv", "I"), ("states_RegimeIII.csv", "III")):
    d = C.states(fn); d = d[np.isclose(d.t_tauB, 10.0) & (d.A > 0)]
    for (m, A), g in d.groupby(["model", "A"]):
        try: C.b2s(m, float(A), [None])
        except Exception: continue
        U = C.cache[(m, float(A))]; r = C.r; out = r >= S * 0.98
        rmin = r[out][np.argmin(U[out])]
        row = dict(regime=reg, model=m, A=A, psi6=g.psi6.mean(), rmin=rmin / S)
        f = 1 - np.exp(-np.clip(U, -50, 700))
        for w in W:
            msk = (np.abs(r - rmin) <= w * S) & (U < 0)
            row[f"w{w:g}"] = np.trapezoid(np.where(msk, f, 0) * r, r) / (S ** 2 / 2)
        rows.append(row)
R = pd.DataFrame(rows); R.to_csv("window_min.csv", index=False)
print(R[R.regime == "I"].groupby("model").rmin.mean().round(3).to_dict())
for reg in ("I", "III"):
    G = R[R.regime == reg]
    for w in W:
        o = {m: onset(g, f"w{w:g}") for m, g in G.groupby("model")}; v = np.array([x for x in o.values() if np.isfinite(x)])
        print(reg, f"±{w}σ spread {v.max()/v.min():5.2f} n={len(v)}", {k: round(x, 2) for k, x in o.items() if np.isfinite(x)})

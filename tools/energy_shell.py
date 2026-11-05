"""Energy-shell alternatives to B2c*: count the attraction only where the pair energy is near the bottom of the well.
 E1(f): integral of (1 - exp(-bU)) r dr over r where U(r) <= f * U_min   (f = 0.9: within 10 % of the well depth)
 E2(d): same over r where U(r) <= U_min + d k_BT (thermally accessible bottom of the well)
Plus the core (r < sigma, +1) as in B2*. Onset spread across models in Regimes I and III."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/sim/b2c"); import collapse as C
def coef(U, mask):
    f = np.where(mask, 1 - np.exp(-np.clip(U, -50, 700)), 0.0)
    return (np.trapezoid(f * C.r, C.r) + C.r[0] ** 2 / 2) / (C.bd.SIGMA ** 2 / 2)
def onset(g, col):
    g = g.sort_values("A"); x = -g[col].values; y = g.psi6.values
    i = np.where((y[:-1] < 0.8) & (y[1:] >= 0.8))[0]
    if not len(i) or min(x[i[0]], x[i[0] + 1]) <= 0: return np.nan
    k = i[0]; return float(np.exp(np.interp(0.8, [y[k], y[k + 1]], [np.log(x[k]), np.log(x[k + 1])])))
VAR = {f"f{f:g}": ("frac", f) for f in (0.99, 0.95, 0.9, 0.8, 0.5, 0.2)}
VAR.update({f"d{d:g}": ("kT", d) for d in (0.5, 1, 2, 3)})
rows = []
for fn, reg in (("states_RegimeI.csv", "I"), ("states_RegimeIII.csv", "III")):
    d = C.states(fn); d = d[np.isclose(d.t_tauB, 10.0) & (d.A > 0)]
    for (m, A), g in d.groupby(["model", "A"]):
        try: C.b2s(m, float(A), [None])
        except Exception: continue
        U = C.cache[(m, float(A))]; core = C.r < C.bd.SIGMA; Um = U[~core].min()
        row = dict(regime=reg, model=m, A=A, psi6=g.psi6.mean(), B2=coef(U, np.ones_like(U, bool)), depth=-Um)
        for k, (kind, v) in VAR.items():
            att = (U <= v * Um) if kind == "frac" else (U <= Um + v)
            row[k] = coef(U, core | (att & ~core))
        rows.append(row)
R = pd.DataFrame(rows); R.to_csv("energy_shell.csv", index=False)
for reg in ("I", "III"):
    G = R[R.regime == reg]
    for col in ["B2", "depth"] + list(VAR):
        if col == "depth":
            o = {m: (lambda g: (lambda gg: float(np.interp(0.8, gg.psi6.values, gg.depth.values)) if gg.psi6.is_monotonic_increasing else np.nan)(g.sort_values("A")))(g) for m, g in G.groupby("model")}
        else:
            o = {m: onset(g, col) for m, g in G.groupby("model")}
        v = np.array([x for x in o.values() if np.isfinite(x)])
        print(reg, f"{col:6s} spread {v.max() / v.min():5.2f}  n={len(v)}", {k: round(x, 2) for k, x in o.items() if np.isfinite(x)})

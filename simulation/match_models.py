"""Match the extra attraction models to AO (q = 0.075) at equal B2* and equal Noro-Frenkel range lambda*."""
import numpy as np, json, sys
from scipy.optimize import brentq, fsolve
sys.path.insert(0, '.')
import airpods_bd as bd
from verify_ao import potential_from_force, b2_reduced
r = np.linspace(1.5, 10, 200001)
def U_of(fr): return potential_from_force(r, fr)
core = bd.radial_force(r, 0.0, None)
def B_lj(A): return b2_reduced(r, U_of(bd.radial_force(r, A, None)))
def B_ao(q, phi): return b2_reduced(r, U_of(bd.radial_force(r, 0.0, (q, phi))))
def U_att(att): return U_of(core + bd.att_force(r, att))
def B_att(att): return b2_reduced(r, U_att(att))
def depth(U): return -U[r >= 1.9].min()
def lam(B, e): return np.sqrt(1 + (1 - B) / (np.exp(e) - 1))
Q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.075
As = [2, 2.5, 3, 4]
out = {"q_ref": Q, "points": {}}
ref = {}
for A in As:
    t = B_lj(A)
    phi = brentq(lambda p: B_ao(Q, p) - t, 0.01, 3)
    e = depth(U_of(bd.radial_force(r, 0.0, (Q, phi))))
    ref[A] = (t, phi, e, lam(t, e))
    print(f"A={A}: B2*={t:7.2f}  AO q={Q} phi_p={phi:.3f} depth={e:.2f} kBT lambda*={lam(t,e):.4f}")
t3, _, e3, l3 = ref[3]
PAR = {"LJ2n": ("n", 50.0), "SW": ("width", 2 * Q), "YUK": ("kappa", 1 / Q)}
for m, (p, guess) in PAR.items():
    def eqs(x):
        att = {"model": m, "eps": x[0], p: x[1]}
        B = B_att(att); e = depth(U_att(att))
        return [B - t3, lam(B, e) - l3]
    sol = fsolve(eqs, [e3, guess], full_output=True)
    e_m, par = sol[0]
    print(f"{m}: {p} = {par:.4g}  (residual {np.abs(sol[1]['fvec']).max():.1e})")
    out[m] = {p: float(par), "eps": {}}
    for A in As:
        t = ref[A][0]
        eps = brentq(lambda e: B_att({"model": m, "eps": e, p: par}) - t, 0.1, 40)
        U = U_att({"model": m, "eps": eps, p: par}); d = depth(U)
        out[m]["eps"][str(A)] = float(eps)
        print(f"    A={A}: eps={eps:.3f}  depth={d:.2f}  lambda*={lam(t, d):.4f}")
for A in As:
    out["points"][str(A)] = {"B2": float(ref[A][0]), "phi_p": float(ref[A][1]), "depth_AO": float(ref[A][2]), "lambda_AO": float(ref[A][3])}
json.dump(out, open(f"model_match_q{Q:g}.json", "w"), indent=1)

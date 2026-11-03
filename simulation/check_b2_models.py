"""Two-particle check of the extra attraction models: sampled B2* vs the potential, for two time steps."""
import sys, json, time, numpy as np
sys.path.insert(0, '.')
import airpods_bd as bd
from verify_ao import potential_from_force, b2_reduced, g_from_samples
m = json.load(open('model_match_q0.075.json')); A = '3'
which = sys.argv[1]
cases = {"AO": (dict(ao=(0.075, m['points'][A]['phi_p'])), (128000, 256000)),
         "LJ2n": (dict(att={"model": "LJ2n", "eps": m['LJ2n']['eps'][A], "n": m['LJ2n']['n']}), (128000, 256000)),
         "YUK": (dict(att={"model": "YUK", "eps": m['YUK']['eps'][A], "kappa": m['YUK']['kappa']}), (128000, 256000)),
         "SW": (dict(att={"model": "SW", "eps": m['SW']['eps'][A], "width": m['SW']['width']}), (256000, 512000))}
kw, steps_list = cases[which]
def force(r):
    if 'att' in kw: return bd.radial_force(r, 0.0, None) + bd.att_force(r, kw['att'])
    return bd.radial_force(r, 0.0, kw['ao'])
R = 4.0; n_pairs = 2000; T = float(sys.argv[2]) if len(sys.argv) > 2 else 10
rf = np.linspace(1.5, R, 40001); b2_th = b2_reduced(rf, potential_from_force(rf, force(rf)))
for steps in steps_list:
    t0 = time.time(); rng = np.random.default_rng(7); dt = bd.TAUB / steps
    rho = rng.uniform(-R / np.sqrt(2), R / np.sqrt(2), size=(n_pairs, 2)); rr = np.linalg.norm(rho, axis=1)
    rho[rr < 2.2] *= (2.6 / rr[rr < 2.2])[:, None]
    n = int(T * steps); burn = n // 5; S = []
    for s in range(n):
        r = np.linalg.norm(rho, axis=1)
        move = 2 * bd.D * dt * (force(r) / r)[:, None] * rho
        mag = np.linalg.norm(move, axis=1); big = mag > bd.MAX_STEP_UM
        move[big] *= (bd.MAX_STEP_UM / mag[big])[:, None]
        rho = rho + move + np.sqrt(4 * bd.D * dt) * rng.standard_normal(rho.shape)
        r = np.linalg.norm(rho, axis=1); out = r > R
        rho[out] *= ((2 * R - r[out]) / r[out])[:, None]
        if s >= burn and s % 200 == 0: S.append(np.linalg.norm(rho, axis=1))
    c, g = g_from_samples(np.concatenate(S), R=R, dr=0.005, r_norm=(0.7 * R, 0.95 * R))
    b2 = (np.sum((1 - g) * c * (c[1] - c[0])) + 1.5 ** 2 / 2) / (bd.SIGMA ** 2 / 2)
    print(f"{which:5s} dt=tauB/{steps:<7d} B2* sim {b2:7.2f}  potential {b2_th:7.2f}  ({b2/b2_th:.0%})  [{time.time()-t0:.0f} s]", flush=True)

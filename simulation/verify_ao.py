"""verify_ao.py - shows that the AO interaction in airpods_bd.py is the AO in your COMSOL model.

Three checks, cheapest first. Run all:   python verify_ao.py
                         or one:         python verify_ao.py --check force | b2 | comsol

0. match   phi_p that gives AO the same B2* as LJ with the potentials exactly as simulated.
1. force   The coded AO force equals the analytic AO force and minus the derivative of
           the AO potential, everywhere (should agree to rounding error).
2. b2      Two particles, simulated with the same force function and integrator.
           From where they spend their time we get g(r) and B2*; they must match
           exp(-U/kT) and B2* computed from the potential. Done for AO and for its
           matched LJ partner, so it also checks that the two have the same B2*.
3. comsol  One COMSOL AO run (run_ao_validation.m) vs 3 Python seeds with identical
           settings, compared at several times (like the LJ validation).

Figures go to  ao_verification/ next to this file.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import airpods_bd as bd

AO = bd.AO_Q02             # q, phi_p of the Python AO runs (matched to LJ A = 3 with the real core)
AO_COMSOL = (0.2, 0.664)   # q, phi_p in your COMSOL AO model
A_MATCH = 3.0              # LJ partner
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ao_verification")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ao_potential(r, q, phi, a=bd.A_P):
    """Textbook AO potential (kBT) between two spheres, hard core not included."""
    d = q * a
    R = a + d
    n_p = phi / (4 * np.pi / 3 * d ** 3)
    r = np.asarray(r, float)
    V = (4 * np.pi / 3) * R ** 3 * (1 - 3 * r / (4 * R) + r ** 3 / (16 * R ** 3))
    return np.where(r < 2 * R, -n_p * V, 0.0)


def potential_from_force(r, fr):
    """U(r) = integral_r^inf F dr  (F + = repulsive), on a fine increasing grid."""
    U = np.zeros_like(r)
    U[:-1] = np.cumsum((0.5 * (fr[1:] + fr[:-1]) * np.diff(r))[::-1])[::-1]
    return U


def b2_reduced(r, U):
    """2D reduced second virial coefficient B2/B2_HS (B2_HS = pi sigma^2 / 2) for U on r >= r[0];
    below r[0] the core makes exp(-U) = 0, which is added analytically."""
    trap = getattr(np, "trapezoid", None) or np.trapz
    B2 = np.pi * (trap((1 - np.exp(-U)) * r, r) + r[0] ** 2 / 2)
    return B2 / (np.pi * bd.SIGMA ** 2 / 2)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


# ---------------------------------------------------------------------------
# 1. force
# ---------------------------------------------------------------------------
def check_force():
    print("\n1. AO force: code vs analytic formula vs -dU/dr")
    q, phi = AO
    d = q * bd.A_P
    r = np.linspace(2.0 * bd.A_P, 2 * (bd.A_P + d) + 0.2, 4001)
    f_code = bd.radial_force(r, 0.0, AO, core="none")
    n_p = phi / (4 * np.pi / 3 * d ** 3)
    f_formula = np.where(r < 2 * (bd.A_P + d), -n_p * np.pi * ((bd.A_P + d) ** 2 - r ** 2 / 4), 0.0)
    U = ao_potential(r, q, phi)
    f_deriv = -np.gradient(U, r)
    inside = r < 2 * (bd.A_P + d) - 1e-3
    err_formula = np.max(np.abs(f_code - f_formula))
    err_deriv = np.max(np.abs(f_code[inside] - f_deriv[inside]))
    scale = np.max(np.abs(f_code))
    print(f"   contact force {f_code[0]:.3f} kBT/um, range ends at r = {2 * (bd.A_P + d):.2f} um")
    print(f"   max |code - formula| = {err_formula:.2e} kBT/um  ({err_formula / scale:.1e} of the peak)")
    print(f"   max |code - (-dU/dr)| = {err_deriv:.2e} kBT/um  ({err_deriv / scale:.1e} of the peak, finite-difference error)")
    print(f"   force at the cutoff: {bd.radial_force(np.array([2 * (bd.A_P + d) - 1e-9]), 0, AO, core="none")[0]:.2e} (continuous -> 0)")
    U_code = potential_from_force(r, f_code)
    print(f"   depth at contact from the code: {-U_code[0]:.3f} kBT   (textbook AO: {-U[0]:.3f} kBT)")
    ok = err_formula / scale < 1e-10 and err_deriv / scale < 1e-3
    print("   PASS" if ok else "   FAIL")

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax[0].plot(r, f_formula, c="#2a78d6", lw=3, label="analytic AO force")
    ax[0].plot(r, f_code, "--", c="#eb6834", lw=1.5, label="airpods_bd.py")
    ax[0].plot(r[inside][::200], f_deriv[inside][::200], "o", c="#1baf7a", ms=4, label="−dU/dr (textbook U)")
    ax[0].set_xlabel("centre distance r (µm)"); ax[0].set_ylabel("force (k$_B$T/µm, + = repulsive)")
    ax[0].legend(frameon=False); ax[0].set_title("AO force", loc="left")
    ax[1].plot(r, U, c="#2a78d6", lw=3, label="textbook AO potential")
    ax[1].plot(r, U_code, "--", c="#eb6834", lw=1.5, label="integrated from code force")
    ax[1].set_xlabel("centre distance r (µm)"); ax[1].set_ylabel("U (k$_B$T)")
    ax[1].legend(frameon=False); ax[1].set_title("AO potential", loc="left")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "1_ao_force_check.png"), dpi=200)
    return ok


# ---------------------------------------------------------------------------
# 2. two-particle g(r) and B2
# ---------------------------------------------------------------------------
def two_particle_run(A, ao, *, steps_per_tauB, T_tauB, n_pairs=2000, R=10.0, seed=7):
    """Relative coordinate of two particles (same force, same Euler-Maruyama step as the
    full simulation): d rho = 2 D F dt + sqrt(4 D dt) xi, reflected at |rho| = R.
    n_pairs independent pairs are run side by side. Returns sampled distances."""
    rng = np.random.default_rng(seed)
    dt = bd.TAUB / steps_per_tauB
    rho = rng.uniform(-R / np.sqrt(2), R / np.sqrt(2), size=(n_pairs, 2))
    rr = np.linalg.norm(rho, axis=1)
    bad = rr < 2.2
    rho[bad] *= (3.0 / rr[bad])[:, None]
    n_steps = int(T_tauB * steps_per_tauB)
    burn = n_steps // 5
    samples = []
    for s in range(n_steps):
        r = np.linalg.norm(rho, axis=1)
        f = bd.radial_force(r, A, ao)
        move = 2 * bd.D * dt * (f / r)[:, None] * rho
        mag = np.linalg.norm(move, axis=1)
        big = mag > bd.MAX_STEP_UM
        move[big] *= (bd.MAX_STEP_UM / mag[big])[:, None]
        rho = rho + move + np.sqrt(4 * bd.D * dt) * rng.standard_normal(rho.shape)
        r = np.linalg.norm(rho, axis=1)
        out = r > R
        rho[out] *= ((2 * R - r[out]) / r[out])[:, None]
        if s >= burn and s % 100 == 0:
            samples.append(np.linalg.norm(rho, axis=1))
    return np.concatenate(samples)


def g_from_samples(rs, R=10.0, dr=0.01, r_norm=(7.0, 9.5)):
    edges = np.arange(1.5, R + dr / 2, dr)
    h, _ = np.histogram(rs, edges)
    c = 0.5 * (edges[1:] + edges[:-1])
    g = h / (2 * np.pi * c * dr)                    # density per area
    norm = g[(c > r_norm[0]) & (c < r_norm[1])].mean()  # U = 0 there
    return c, g / norm


def check_b2(T_tauB=30, n_pairs=4000):
    """AO: step-size series in a small disk (fast equilibration), extrapolated to dt -> 0.
    LJ partner: same, as a reference that converges already at the normal step."""
    print("\n2. Two particles: sampled B2* vs the potential, for decreasing time step")
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    out = {}
    cases = [(f"AO q={AO[0]:g}, phi={AO[1]:g}", 0.0, AO, 5.0, (4000, 16000, 32000), "#eb6834"),
             (f"LJ A={A_MATCH:g}", A_MATCH, None, 10.0, (4000, 16000), "#2a78d6")]
    for k, (label, A, ao, R, steps_list, col) in enumerate(cases):
        rfine = np.linspace(1.5, R, 40001)
        U = potential_from_force(rfine, bd.radial_force(rfine, A, ao))
        b2_th = b2_reduced(rfine, U)
        ax[k].plot(rfine, np.exp(-U), c="#888", lw=3, label="exp(−U/k$_B$T), code potential")
        xs, ys = [], []
        for steps in steps_list:
            t0 = time.time()
            T = T_tauB * (1 if steps <= 16000 else 0.5)
            rs = two_particle_run(A, ao, steps_per_tauB=steps, T_tauB=T, n_pairs=n_pairs, R=R)
            c, g = g_from_samples(rs, R=R, dr=0.01, r_norm=(0.65 * R, 0.95 * R))
            b2 = (np.sum((1 - g) * c * (c[1] - c[0])) + 1.5 ** 2 / 2) / (bd.SIGMA ** 2 / 2)
            xs.append(1 / steps); ys.append(b2)
            print(f"   {label:<22} dt = tau_B/{steps:<6} B2* sim = {b2:7.2f}   potential = {b2_th:7.2f}"
                  f"   ({b2 / b2_th:5.0%})   [{time.time() - t0:.0f} s]")
            ax[k].plot(c, g, "-", lw=1, label=f"simulation, τ$_B$/{steps}")
        if len(xs) >= 2:
            p = np.polyfit(xs[-2:], ys[-2:], 1)
            print(f"   {label:<22} extrapolated to dt -> 0: B2* = {p[1]:.2f}  ({p[1] / b2_th:.0%} of the potential)")
            out[label] = (ys, p[1], b2_th)
        ax[k].set_yscale("log"); ax[k].set_xlim(1.8, 3.0 if ao else 6.5)
        win = (rfine > 1.8) & (rfine < (3.0 if ao else 6.5)); ax[k].set_ylim(0.05, 3 * np.exp(-U[win]).max())
        ax[k].set_xlabel("centre distance r (µm)"); ax[k].set_ylabel("g(r)")
        ax[k].set_title(label, loc="left"); ax[k].legend(frameon=False, fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "2_two_particle_g_and_B2.png"), dpi=200)
    return out


# ---------------------------------------------------------------------------
# 3. COMSOL AO run vs Python
# ---------------------------------------------------------------------------
COMSOL_NAME = "3Tkb_1TauB_4000nm_AO_q0.2_phi0.664"
COMSOL_STEPS = 32000       # Python step for this comparison; set the COMSOL run to the same step


def check_match():
    """phi_p that gives AO the same B2* as LJ, with the potentials exactly as simulated (core included)."""
    from scipy.optimize import brentq
    print("\n0. AO <-> LJ matching with the simulated potentials (Coulomb core, Z = 100)")
    r = np.linspace(1.5, 10, 200001)
    B = lambda A, ao: b2_reduced(r, potential_from_force(r, bd.radial_force(r, A, ao)))
    print(f"   {'LJ A':<6}{'B2*':>9}" + "".join(f"{'phi_p (q=' + format(q, 'g') + ')':>16}" for q in (0.1, 0.2, 0.3)))
    for A in (2.5, 3, 3.5, 10):
        t = B(A, None)
        row = f"   {A:<6g}{t:9.2f}"
        for q in (0.1, 0.2, 0.3):
            try:
                row += f"{brentq(lambda p: B(0, (q, p)) - t, 0.05, 5):16.3f}"
            except ValueError:
                row += f"{'-':>16}"
        print(row)
    print(f"   COMSOL model uses phi_p = {AO_COMSOL[1]} at q = 0.2 -> B2* = {B(0, AO_COMSOL):.2f} (LJ A=3: {B(3, None):.2f})")


def observables_over_time(x, y, t, times_tauB):
    out = []
    for tt in times_tauB:
        k = min(int(np.searchsorted(t, tt * bd.TAUB - 1e-9)), len(t) - 1)
        out.append(bd.observables(x[:k + 1], y[:k + 1]))
    return out


def check_comsol(comsol_root, seeds=(1, 2, 3), bd_root=r"D:\Work\AIRPODS Paper 2627\Data"):
    print("\n3. COMSOL AO run vs Python (same AO, same field, same start)")
    folder = os.path.join(comsol_root, "ARF", COMSOL_NAME)
    if not os.path.isfile(os.path.join(folder, "datax.csv")):
        print(f"   no COMSOL run at {folder}\n   run run_ao_validation.m first (one run, 1 tau_B).")
        return None
    xc, yc, tc = bd.read_comsol(folder)
    times = [0.25, 0.5, 0.75, 1.0]
    oc = observables_over_time(xc, yc, tc, times)
    ob = []
    for s in seeds:
        name = bd.run_name(A_MATCH, 1, 4000, AO_COMSOL, s, COMSOL_STEPS)
        f = os.path.join(bd_root, "ARF", name)
        if os.path.isfile(os.path.join(f, "datax.csv")):
            x, y, t = bd.read_comsol(f)
        else:
            x, y, t = bd.simulate(0.0, 4000, 1, ao=AO_COMSOL, steps_per_tauB=COMSOL_STEPS, seed=s)
            bd.write_comsol_csv(f, x, y, t)
        ob.append(observables_over_time(x, y, t, times))
    keys = ["z", "Smax", "psi6", "y_spread"]
    print(f"   {'t (tau_B)':<10}" + "".join(f"{k:>22}" for k in keys))
    worst = 0
    for i, tt in enumerate(times):
        row = f"   {tt:<10}"
        for k in keys:
            v = np.array([o[i][k] for o in ob])
            m, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0
            row += f"   {m:6.2f}±{sd:4.2f} | {oc[i][k]:5.2f}"
            if sd > 0:
                worst = max(worst, abs(oc[i][k] - m) / sd)
        print(row)
    print("   (Python mean ± spread over seeds | COMSOL)")
    print(f"   largest COMSOL distance from the Python mean: {worst:.1f} seed-spreads"
          + ("   PASS (within 3)" if worst <= 3 else "   look closer (core shape? time step?)"))

    plt = _plt()
    fig, ax = plt.subplots(1, len(keys), figsize=(13, 3.2), constrained_layout=True)
    for j, k in enumerate(keys):
        for o in ob:
            ax[j].plot(times, [p[k] for p in o], "-o", c="#86b6ef", ms=4)
        ax[j].plot(times, [p[k] for p in oc], "-s", c="#eb6834", ms=6, lw=2, label="COMSOL")
        ax[j].plot([], [], "-o", c="#86b6ef", label="Python seeds")
        ax[j].set_title(k, loc="left"); ax[j].set_xlabel("t (τ$_B$)")
    ax[0].legend(frameon=False)
    fig.savefig(os.path.join(OUT, "3_comsol_vs_python_AO.png"), dpi=200)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", choices=["match", "force", "b2", "comsol", "all"], default="all")
    ap.add_argument("--comsol", default=r"D:\Work\AIRPODS Paper 2627\Simulations\revision runs",
                    help="folder containing ARF\\" + COMSOL_NAME)
    ap.add_argument("--b2-taub", type=float, default=30, help="length of the two-particle runs (tau_B)")
    a = ap.parse_args()
    if a.check in ("match", "all"):
        check_match()
    if a.check in ("force", "all"):
        check_force()
    if a.check in ("b2", "all"):
        check_b2(T_tauB=a.b2_taub)
    if a.check in ("comsol", "all"):
        check_comsol(a.comsol)
    print(f"\nFigures: {OUT}")


if __name__ == "__main__":
    main()

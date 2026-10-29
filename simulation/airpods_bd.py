"""airpods_bd.py - fast Brownian-dynamics replacement for COMSOL Study 3 (ARF / NAF).

Same system as the COMSOL model: 144 particles (radius 1 um) starting on a
12 x 12 grid with 4 um spacing in a 50 x 50 um box with walls, overdamped in
water-glycerol (eta = 2.9 mPa s, T = 293 K), with
    * Brownian motion (a real, settable random seed per run)
    * pair attraction:  Lennard-Jones (eps = A kBT, sigma = 2a)  or  AO (q, phi_p)
    * the same repulsive core as COMSOL (Coulomb, Z = 100, only below r = 2a)
    * acoustic radiation force of a 1D standing wave along y, node at y = 0:
          F_y = -F0 * sin(k y),   F0 = F0_REF * (d_n / 4000 nm)^2
      F0_REF and k were fitted to the COMSOL A = 0 ARF runs (see ARF_* below);
      force ~ amplitude^2, which the 2000/4000/5000 nm runs confirm.

Output is written exactly like the COMSOL export (datax.csv / datay.csv with
alternating position and time rows), so the Particle Post Processor reads it.

Examples
    python airpods_bd.py --A 3 --dn 4000 --T 1                         # one LJ run
    python airpods_bd.py --A 3 --dn 2750 --T 2 --ao 0.2 0.664          # one AO run
    python airpods_bd.py --plan validation --out "D:\\...\\BD runs"
    python airpods_bd.py --plan validation --out "D:\\...\\BD runs" --compare "D:\\...\\raw csv data"
    python airpods_bd.py --plan revision --out "D:\\...\\BD runs"
    python airpods_bd.py --plan aaf --workers 6                         # regime III, 6 runs at a time

Regime III (AAF, after the acoustic force): the field is off and the run starts from the
band in aaf_start.csv (frame 0 of your COMSOL AAF runs), so Python and COMSOL start identically.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

# ---- physical constants and model parameters (taken from the COMSOL model) ----
KB = 1.380649e-23
T_K = 293.0
ETA = 2.9e-3            # Pa s
A_P = 1.0               # particle radius, um
D = KB * T_K / (6 * np.pi * ETA * A_P * 1e-6) * 1e12   # um^2/s  (= 0.0740)
TAUB = A_P ** 2 / D     # s (= 13.51)
BOX = 25.0              # walls at +-25 um (particle centres, as in COMSOL output)
SIGMA = 2.0 * A_P       # LJ sigma = 2 r_p
GRID = np.arange(-22.0, 22.0 + 1e-9, 4.0)              # 12 x 12 start positions

# ---- acoustic radiation force, fitted to COMSOL ARF runs at A = 0 ----
ARF_F0_REF = 18.4       # kBT/um at d_n = 4000 nm
ARF_K = 0.124           # 1/um  (half wavelength ~ 25 um, node at y = 0)
ARF_DN_REF = 4000.0     # nm

# ---- repulsive core: same as COMSOL "Particle-Particle Interaction 2" ----
# Coulomb force between particles of charge number Z = 100 (Particle Properties 1),
# applied only below the cutoff 2 r_p. In kBT/um: F = CORE_K / r^2 for r < 2a.
CORE = "coulomb"        # "coulomb" (= COMSOL), "phs" (smooth pseudo-hard sphere) or "wca" (soft, old)
CORE_Z = 100
CORE_K = (CORE_Z * 1.602176634e-19) ** 2 / (4 * np.pi * 8.8541878128e-12) / (KB * T_K) * 1e6  # kBT*um
CORE_EPS = 10.0         # kBT, only for CORE = "wca"
CORE_SIG = 2.0 * A_P / 2 ** (1 / 6)

MAX_STEP_UM = 0.1       # safety cap on the deterministic move per step

# ---- settings used when you press Run without command-line arguments (e.g. in PyCharm) ----
DEFAULT_ARGS = ["--plan", "validation",
                "--out", r"D:\Work\AIRPODS Paper 2627\Data",
                "--compare", r"D:\Work\AIRPODS Paper 2627\Data"]
# other options:  ["--plan", "revision"]
#                 ["--plan", "ao"]
#                 ["--A", "3", "--dn", "2750", "--T", "2", "--ao", "0.2", "0.664", "--steps", "4000", "--seed", "2"]


def arf_force(y, dn_nm):
    if dn_nm <= 0:
        return np.zeros_like(y)
    F0 = ARF_F0_REF * (dn_nm / ARF_DN_REF) ** 2
    return -F0 * np.sin(ARF_K * y)


def radial_force(r, A, ao=None, core=None):
    """Pair force magnitude along the line of centres, kBT/um, + = repulsive.
    The single place where the pair interaction is defined (used by the
    simulation and by verify_ao.py)."""
    r = np.asarray(r, float)
    fr = np.zeros_like(r)

    # repulsive core, acts only for r < 2a
    core = CORE if core is None else core
    m = r < 2.0 * A_P
    if m.any() and core == "coulomb":
        fr[m] += CORE_K / r[m] ** 2
    elif core == "phs":
        # pseudo-hard-sphere Mie(50,49) core (Jover et al., J. Chem. Phys. 137, 144505 (2012)),
        # eps = 1.4737 kBT; cut where it reaches zero, placed at r = 2a
        sig = 2.0 * A_P * 49 / 50
        mm = r < 2.0 * A_P
        if mm.any():
            C = 50 * (50 / 49) ** 49 * 1.4737
            x = sig / r[mm]
            fr[mm] += C / r[mm] * (50 * x ** 50 - 49 * x ** 49)
    elif m.any() and core == "wca":
        s6 = (CORE_SIG / r[m]) ** 6
        fr[m] += 24 * CORE_EPS / r[m] * (2 * s6 ** 2 - s6)

    if ao is None:
        if A > 0:
            m = r < 3.0 * SIGMA                  # LJ tail beyond 6 um is < 0.6 % of the well
            s6 = (SIGMA / r[m]) ** 6
            fr[m] += 24 * A / r[m] * (2 * s6 ** 2 - s6)
    else:
        q, phi = ao
        delta = q * A_P
        n_p = phi / (4 * np.pi / 3 * delta ** 3)      # 1/um^3
        m = r < 2 * (A_P + delta)
        rc = np.maximum(r[m], 2 * A_P)
        fr[m] -= n_p * np.pi * ((A_P + delta) ** 2 - rc ** 2 / 4)   # kBT/um, attractive
    return fr


def pair_forces(P, A, ao=None, core=None):
    """Return (N,2) forces in kBT/um from all pair interactions."""
    dx = P[:, None, :] - P[None, :, :]            # i - j
    r2 = (dx ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    fr = radial_force(r, A, ao, core)
    return ((fr / r)[:, :, None] * dx).sum(1)


def simulate(A=0.0, dn_nm=4000.0, T_tauB=1.0, *, ao=None, steps_per_tauB=4000, out_per_tauB=100,
             seed=1, core=None, verbose=True, start=None, progress=None):
    """Return x, y (frames x N, um) and t (s). start: (N, 2) start positions; default the 12 x 12 grid.
    progress(step, n_steps, n_capped) is called at every saved frame (raise inside it to stop)."""
    rng = np.random.default_rng(seed)
    if start is None:
        gx, gy = np.meshgrid(GRID, GRID, indexing="ij")
        P = np.c_[gx.ravel(), gy.ravel()].astype(float)
    else:
        P = np.array(start, float).copy()
    dt = TAUB / steps_per_tauB
    n_steps = int(round(T_tauB * steps_per_tauB))
    every = steps_per_tauB // out_per_tauB
    if verbose and steps_per_tauB * T_tauB > 2e5:
        print(f"   {int(steps_per_tauB * T_tauB):,} steps - this run takes a few minutes")
    if steps_per_tauB % out_per_tauB:
        raise ValueError("steps_per_tauB must be a multiple of out_per_tauB")
    frames = [P.copy()]
    times = [0.0]
    noise = np.sqrt(2 * D * dt)
    clipped = 0
    t0 = time.time()
    for s in range(1, n_steps + 1):
        F = pair_forces(P, A, ao, core)
        F[:, 1] += arf_force(P[:, 1], dn_nm)
        move = D * F * dt
        mag = np.linalg.norm(move, axis=1)
        big = mag > MAX_STEP_UM
        if big.any():
            clipped += int(big.sum())
            move[big] *= (MAX_STEP_UM / mag[big])[:, None]
        P += move + noise * rng.standard_normal(P.shape)
        # reflecting walls
        P = np.where(P > BOX, 2 * BOX - P, P)
        P = np.where(P < -BOX, -2 * BOX - P, P)
        if s % every == 0:
            frames.append(P.copy())
            times.append(s * dt)
            if progress is not None:
                progress(s, n_steps, clipped)
    if verbose:
        frac = clipped / (n_steps * len(P))
        note = ""
        if frac > 1e-4:   # more than 1 in 10,000 particle moves capped: the step is too big
            note = f"  WARNING: force cap hit {clipped} times ({frac:.2%} of moves) - increase --steps"
        print(f"   {n_steps} steps in {time.time() - t0:.1f} s{note}")
    F = np.array(frames)
    return F[:, :, 0], F[:, :, 1], np.array(times)


def write_comsol_csv(folder, x, y, t):
    """Same layout as the COMSOL export: header 1..N, then position row / time row pairs."""
    os.makedirs(folder, exist_ok=True)
    n = x.shape[1]
    header = ",".join(str(i) for i in range(1, n + 1))
    for name, arr in (("datax.csv", x), ("datay.csv", y)):
        rows = np.empty((2 * len(t), n))
        rows[0::2] = arr
        rows[1::2] = t[:, None]
        np.savetxt(os.path.join(folder, name), rows, delimiter=",", header=header, comments="", fmt="%.10g")


def run_name(A, T_tauB, dn_nm, ao=None, seed=1, steps=4000, regime=None):
    base = f"{A:g}Tkb_{T_tauB:g}TauB" + (f"_{dn_nm:g}nm" if dn_nm > 0 else ("" if regime == "AAF" else "_core"))
    base += "_BD"
    if ao:
        base += f"_AO_q{ao[0]:g}_phi{ao[1]:g}"
    if steps != STEPS:
        base += f"_dt{steps}"
    return base + f"_seed{seed}"


# ---------------------------------------------------------------------------
# run plans
# ---------------------------------------------------------------------------
PLAN_TAUB = 10      # run length for the revision / AO plans (BD is cheap, so use the long runs)
PLAN_SEEDS = (1, 2, 3)
STEPS = 4000        # time steps per tau_B; the stiff COMSOL core needs >= 4000 (2000 -> force cap hits)
# AO polymer amount matched to LJ A = 3 by B2*, computed with THIS code's potentials
# (Coulomb core included): verify_ao.py --check match prints the table.
AO_Q02 = (0.2, 0.633)
AO_Q01 = (0.1, 0.368)
# AO wells are narrow (0.4 um at q = 0.2) next to a stiff core: the simple time step
# under-samples contact unless it is small (verify_ao.py --check b2). At q = 0.2,
# tau_B/4000 gives only half the B2*, tau_B/32000 gives 93 %. q = 0.1 is twice as narrow.
AO_STEPS = {0.2: 32000, 0.1: 128000}
# phi_p (q = 0.2) with the same B2* as LJ at each A, core included (verify_ao.py --check match)
AO_MATCH_Q02 = {1: 0.357, 2: 0.510, 2.5: 0.573, 3: 0.633, 3.5: 0.690, 4: 0.747, 5: 0.859}
AAF_A = (0, 1, 2, 2.5, 3, 3.5, 4, 5)   # same attraction values as the COMSOL AAF runs
AAF_TAUB = 20                           # same length as the COMSOL AAF runs
AAF_START = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aaf_start.csv")


def plan(name):
    """(regime, A, d_n nm, T tau_B, ao, seed, steps_per_tauB)"""
    T, S, N = PLAN_TAUB, PLAN_SEEDS, STEPS
    if name == "validation":     # reproduce existing COMSOL runs (those are 1 tau_B long), 3 seeds each
        return [("ARF", A, dn, 1, None, s, N) for A in (0, 5) for dn in (2000, 4000) for s in S]
    if name == "revision":       # the must-do list, with real seeds
        L = []
        L += [("ARF", 3, 4000, 2, None, 1, st) for st in (2000, 4000, 8000)]                 # time-step check
        L += [("ARF", A, dn, T, None, s, N) for A in (0, 2.5, 5) for dn in (2250, 2750) for s in S]  # boundary
        L += [("ARF", A, dn, T, None, s, N) for (A, dn) in ((0, 3000), (3, 3000), (1, 5000)) for s in S]
        L += [("NAF", 0, 0, T, None, s, N) for s in S]                                    # no-field control
        L += [("ARF", 0, 5000, T, None, s, N) for s in S]                                 # confinement only
        L += [("ARF", 0, 4000, T, None, s, N) for s in S]                                 # confinement only, same field as AO
        return L
    if name == "ao":             # AO vs LJ partners (matched to A = 3); AO needs much smaller steps
        L = []
        for dn in (2000, 2750, 4000):
            for s in S:
                L.append(("ARF", 3, dn, T, AO_Q02, s, AO_STEPS[0.2]))
                L.append(("ARF", 3, dn, T, None, s, N))
        L += [("ARF", 3, 2750, 2, AO_Q01, s, AO_STEPS[0.1]) for s in S]   # q = 0.1 is costly: 2 tau_B
        return L
    if name in ("aaf", "aaf_lj", "aaf_ao"):   # regime III: field off, start from the COMSOL AAF band
        L = []
        if name != "aaf_ao":
            L += [("AAF", A, 0, AAF_TAUB, None, s, N) for A in AAF_A for s in S]
        if name != "aaf_lj":
            L += [("AAF", A, 0, AAF_TAUB, (0.2, AO_MATCH_Q02[A]), s, AO_STEPS[0.2]) for A in AAF_A if A in AO_MATCH_Q02 for s in S]
        return L
    if name == "ao_dt":          # full 144-particle AO time-step check (reference: the tau_B/32000 runs of plan "ao")
        return [("ARF", 3, 4000, 2, AO_Q02, s, st) for st in (16000, 64000) for s in S]
    raise ValueError(f"unknown plan {name}")


# ---------------------------------------------------------------------------
# quick comparison with COMSOL runs (same observables as the post-processor)
# ---------------------------------------------------------------------------
def observables(x, y, rc=2.9):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P = np.c_[x[-1], y[-1]]
    n = len(P)
    b = cKDTree(P).query_pairs(rc, output_type="ndarray")
    z = np.bincount(b.ravel(), minlength=n)
    _, lab = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(n, n)), directed=False)
    psi = np.zeros(n, complex)
    if len(b):
        d = P[b[:, 1]] - P[b[:, 0]]
        e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0]))
        np.add.at(psi, b[:, 0], e)
        np.add.at(psi, b[:, 1], e)
    ok = z >= 2
    P0 = np.c_[x[0], y[0]]
    rg = lambda Q: np.sqrt(((Q - Q.mean(0)) ** 2).sum(1).mean())
    return dict(z=z.mean(), Smax=np.bincount(lab).max() / n,
                psi6=np.abs(psi[ok] / z[ok]).mean() if ok.sum() > 5 else np.nan,
                y_spread=y[-1].std(), Rg_ratio=rg(P) / rg(P0))


def read_comsol(folder):
    rx = np.loadtxt(os.path.join(folder, "datax.csv"), delimiter=",", skiprows=1)
    ry = np.loadtxt(os.path.join(folder, "datay.csv"), delimiter=",", skiprows=1)
    return rx[0::2], ry[0::2], rx[1::2, 0]


def _run_job(job):
    """One run of a plan (top level so it can run in a separate process)."""
    i, n, out, (reg, A, dn, T, ao, seed, steps) = job
    name = run_name(A, T, dn, ao, seed, steps, reg)
    folder = os.path.join(out, reg, name)
    print(f"[{i}/{n}] {reg}/{name}", flush=True)
    if os.path.isfile(os.path.join(folder, "datax.csv")):
        print(f"   [{i}] already done, reading it", flush=True)
        x, y, t = read_comsol(folder)
    else:
        start = np.loadtxt(AAF_START, delimiter=",", skiprows=1) if reg == "AAF" else None
        x, y, t = simulate(0.0 if ao else A, dn, T, ao=ao, steps_per_tauB=steps, seed=seed, start=start)
        write_comsol_csv(folder, x, y, t)
    return (reg, A, dn, T, ao), observables(x, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--A", type=float, default=0.0, help="LJ attraction (kBT); for AO runs the matched LJ value (name only)")
    ap.add_argument("--dn", type=float, default=4000.0, help="inward displacement d_n (nm); 0 = no acoustic field")
    ap.add_argument("--T", type=float, default=1.0, help="run length (tau_B)")
    ap.add_argument("--ao", type=float, nargs=2, metavar=("Q", "PHI"), help="use AO with size ratio Q and phi_p PHI")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000,
                    help="time steps per tau_B (multiple of 100); use 4000+ for AO q=0.2, 5000+ for q=0.1")
    ap.add_argument("--plan", choices=["validation", "revision", "ao", "ao_dt", "aaf", "aaf_lj", "aaf_ao"], help="run a whole list")
    ap.add_argument("--out", default=r"D:\Work\AIRPODS Paper 2627\Data",
                    help="output root (the merged Data folder); runs go to <out>/<ARF|NAF|AAF>/<name>/")
    ap.add_argument("--compare", help="folder with the COMSOL runs (raw csv data) to compare against")
    ap.add_argument("--workers", type=int, default=1, help="runs at the same time (use about your number of CPU cores)")
    import sys
    a = ap.parse_args(None if len(sys.argv) > 1 else DEFAULT_ARGS)

    runs = plan(a.plan) if a.plan else [("ARF" if a.dn > 0 else "NAF", a.A, a.dn, a.T,
                                         tuple(a.ao) if a.ao else None, a.seed, a.steps)]
    print(f"D = {D:.4f} um^2/s, tau_B = {TAUB:.2f} s, {len(runs)} run(s) -> {os.path.abspath(a.out)}")
    jobs = [(i, len(runs), a.out, r) for i, r in enumerate(runs, 1)]
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_run_job, jobs))
    else:
        results = [_run_job(j) for j in jobs]

    if a.compare:
        import collections
        print("\nBD (mean of seeds) vs COMSOL, at the end of the run:")
        print(f"{'run':<26}{'':>8}{'z':>7}{'Smax':>7}{'psi6':>7}{'y_spread':>10}{'Rg/Rg0':>8}")
        groups = collections.defaultdict(list)
        for key, o in results:
            groups[key].append(o)
        for (reg, A, dn, T, ao), os_ in groups.items():
            if ao:
                continue
            label = f"{A:g}Tkb_{T:g}TauB" + (f"_{dn:g}nm" if dn > 0 else "")
            m = {k: np.nanmean([o[k] for o in os_]) for k in os_[0]}
            sd = {k: np.nanstd([o[k] for o in os_]) for k in os_[0]}
            print(f"{label:<26}{'BD':>8}" + "".join(f"{m[k]:7.2f}" if k != 'y_spread' else f"{m[k]:10.2f}" for k in m)
                  + f"   (± {sd['z']:.2f} z, ± {sd['psi6']:.2f} psi6 over {len(os_)} seeds)")
            cf = os.path.join(a.compare, reg, label + "_CS")       # merged Data folder
            if not os.path.isdir(cf):
                cf = os.path.join(a.compare, reg, label)           # old raw csv data folder
            if os.path.isdir(cf):
                o = observables(*read_comsol(cf)[:2])
                print(f"{'':<26}{'COMSOL':>8}" + "".join(f"{o[k]:7.2f}" if k != 'y_spread' else f"{o[k]:10.2f}" for k in o))
            else:
                print(f"{'':<26}{'COMSOL':>8}   (no folder {cf})")


if __name__ == "__main__":
    main()

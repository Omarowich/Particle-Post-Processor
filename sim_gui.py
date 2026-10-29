"""Helpers for the analyser's "Run simulation" page: previews, B2* matching, runtime estimate, jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(HERE, "simulation")
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)
import sim_runner as sr  # noqa: E402

bd = sr.bd

POTENTIALS = {"LJ": "Lennard-Jones (ε = A k_BT, σ = 2a)", "AO": "Asakura–Oosawa depletion (q, φ_p)",
              "none": "no attraction (core only)"}
CORES = {"coulomb": "Coulomb, Z = 100, below 2a (= COMSOL)", "phs": "pseudo-hard sphere (Mie 50/49)",
         "wca": "soft WCA (old)", "none": "no core"}
STARTS = {"grid": "square grid (like COMSOL)", "random": "random, minimum distance",
          "aaf": "band of the COMSOL AAF runs (regime III)", "run": "a frame of an existing run",
          "file": "CSV file with x, y columns"}


def _with(c):
    sr.apply_physics({**sr.DEFAULTS, **c})


def potential_curve(c, r=None):
    """r (µm), U(r)/kBT, F(r) for the chosen interaction, from the simulation's own force function."""
    _with(c)
    r = np.linspace(1.6 * bd.A_P, 8.0 * bd.A_P, 6001) if r is None else r
    ao = (float(c["q"]), float(c["phi_p"])) if c["potential"] == "AO" else None
    A = float(c["A"]) if c["potential"] == "LJ" else 0.0
    core = c["core"] if c["core"] != "none" else "off"
    fr = bd.radial_force(r, A, ao, core)
    U = np.zeros_like(r)
    U[:-1] = np.cumsum((0.5 * (fr[1:] + fr[:-1]) * np.diff(r))[::-1])[::-1]
    return r, U, fr


def b2_star(c):
    """Reduced 2D second virial coefficient of the chosen interaction (core included)."""
    _with(c)
    r = np.linspace(1.5 * bd.A_P, 10 * bd.A_P, 100001)
    _, U, _ = potential_curve(c, r)
    trap = getattr(np, "trapezoid", None) or np.trapz
    B2 = np.pi * (trap((1 - np.exp(-np.clip(U, -50, 700))) * r, r) + r[0] ** 2 / 2)
    return B2 / (np.pi * bd.SIGMA ** 2 / 2)


def match_phi(c, A_target):
    """phi_p that gives AO (at c['q']) the same B2* as LJ at A_target."""
    from scipy.optimize import brentq
    target = b2_star({**c, "potential": "LJ", "A": A_target})
    f = lambda p: b2_star({**c, "potential": "AO", "phi_p": p}) - target
    try:
        return brentq(f, 1e-3, 20.0, xtol=1e-4), target
    except ValueError:
        return float("nan"), target


def suggested_steps(c):
    if c["potential"] == "AO":
        q = float(c["q"])
        return 32000 if q >= 0.2 else (128000 if q >= 0.1 else 256000)
    return 4000


def start_preview(c, seed=1):
    _with(c)
    return sr.start_positions(c["start"], seed)


def seconds_per_step(n_particles, potential="LJ"):
    """Measured on this computer (pair forces dominate)."""
    P = np.random.default_rng(0).uniform(-20, 20, (n_particles, 2))
    ao = (0.2, 0.633) if potential == "AO" else None
    t0 = time.perf_counter()
    for _ in range(30):
        bd.pair_forces(P, 3.0, ao)
    return (time.perf_counter() - t0) / 30 * 1.15


def run_names(c):
    return [sr.run_folder_name({**sr.DEFAULTS, **c}, int(s)) for s in c["seeds"]]


# --------------------------------------------------------------------------- jobs
def jobs_dir(out_root):
    return os.path.join(out_root, "_sim_jobs")


def launch(c):
    """Write the job settings and start sim_runner.py in the background. Returns the job folder."""
    c = {**sr.DEFAULTS, **c}
    jd = os.path.join(jobs_dir(c["out_root"]), time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(jd, exist_ok=True)
    with open(os.path.join(jd, "config.json"), "w") as f:
        json.dump(c, f, indent=2)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00000200 | 0x08000000   # new process group, no console window
    else:
        kw["start_new_session"] = True
    with open(os.path.join(jd, "stdout.txt"), "w") as out:
        subprocess.Popen([sys.executable, os.path.join(SIM_DIR, "sim_runner.py"), os.path.join(jd, "config.json")],
                         cwd=SIM_DIR, stdout=out, stderr=subprocess.STDOUT, **kw)
    return jd


def list_jobs(out_root):
    d = jobs_dir(out_root)
    if not os.path.isdir(d):
        return []
    out = []
    for j in sorted(os.listdir(d), reverse=True):
        jd = os.path.join(d, j)
        try:
            with open(os.path.join(jd, "config.json")) as f:
                c = json.load(f)
        except Exception:
            continue
        st = []
        for s in c.get("seeds", []):
            try:
                with open(os.path.join(jd, f"status_seed{s}.json")) as f:
                    st.append({"seed": s, **json.load(f)})
            except Exception:
                st.append({"seed": s, "state": "waiting", "frac": 0.0})
        state = ("failed" if os.path.exists(os.path.join(jd, "failed")) else
                 "stopped" if os.path.exists(os.path.join(jd, "STOP")) and not os.path.exists(os.path.join(jd, "done")) else
                 "done" if os.path.exists(os.path.join(jd, "done")) else "running")
        out.append({"id": j, "dir": jd, "config": c, "status": st, "state": state})
    return out


def stop(job_dir):
    open(os.path.join(job_dir, "STOP"), "w").close()


def read_log(job_dir):
    txt = ""
    for f in ("log.txt", "stdout.txt"):
        p = os.path.join(job_dir, f)
        if os.path.isfile(p):
            with open(p, errors="replace") as fh:
                txt += fh.read()
    return txt


def load_config_from_run(folder):
    p = os.path.join(folder, "sim_config.json")
    with open(p) as f:
        return json.load(f)

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

POTENTIALS = {"LJ": "Lennard-Jones 12-6 (ε = A k_BT, σ = 2a)", "AO": "Asakura–Oosawa depletion (q, φ_p)",
              "LJ2n": "2n–n Lennard-Jones (Noro–Frenkel; n, ε)", "SW": "continuous square well (width, ε)",
              "YUK": "attractive Yukawa (κ, ε)", "none": "no attraction (core only)"}
ATT_PARAM = {"LJ2n": ("n", "exponent n", 57.55), "SW": ("width", "well width (µm)", 0.03319),
             "YUK": ("kappa", "κ (1/µm)", 17.28)}
FIELD_SHAPES = {"1D": "fitted 1D standing wave (same at every height)", "side": "computed: side-wall drive (bulk wave)",
                "idt": "computed: IDT drive (standing SAW from the floor)",
                "idt_travel": "computed: one IDT, travelling SAW (no nodes)", "file": "field file (.npz or COMSOL .csv)"}
LIDS = {"glass": "glass (reflects)", "pdms": "PDMS (Z = 1 MRayl)", "absorb": "absorbing (Z = ρ₀c₀)"}
CORES = {"coulomb": "Coulomb, Z = 100, below 2a (= COMSOL)", "phs": "pseudo-hard sphere (Mie 50/49)",
         "wca": "soft WCA (old)", "none": "no core"}
STARTS = {"grid": "square grid (like COMSOL)", "grid3d": "stacked square layers (3D)", "random": "random, minimum distance",
          "aaf": "band of the COMSOL Regime III runs", "run": "a frame of an existing run",
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
    att = sr.att_of(c) if c["potential"] in ATT_PARAM else None
    if att:
        fr = fr + bd.att_force(r, att)
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


def match_eps(c, A_target):
    """Well depth eps of an extra model (LJ2n / SW / YUK, range fixed) with the same B2* as LJ at A_target."""
    from scipy.optimize import brentq
    target = b2_star({**c, "potential": "LJ", "A": A_target})
    f = lambda e: b2_star({**c, "eps": e}) - target
    try:
        return brentq(f, 0.05, 60.0, xtol=1e-4), target
    except ValueError:
        return float("nan"), target


def suggested_steps(c):
    if c["potential"] == "AO":
        q = float(c["q"])
        return 32000 if q >= 0.2 else (128000 if q >= 0.1 else 256000)
    if c["potential"] in ATT_PARAM:          # short-range models: same step as AO q = 0.05
        return 256000
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


# --------------------------------------------------------------------------- batches
SCENARIOS = {
    "I": "Regime I: no field, grid start",
    "II": "Regime II: acoustic radiation force (d_n), grid start",
    "III": "Regime III: field off, start from the COMSOL band",
    "IV": "Regime IV: streaming flow (F₀), start from the COMSOL band",
}


def expand_batch(base, spec, match_cache=None):
    """List of setting overrides (one per parameter set) for a batch.

    spec: {"scenario": "I".."IV", "A": [...], "dn": [...], "F": [...], "potentials": ["LJ", "AO", "LJ2n", "SW", "YUK"],
           "q": [0.2], "attp": {"LJ2n": [57.55], ...}, "H": [4, 20] (3D only), "keep_start": False,
           "T_tauB": 10, "seeds": [1, 2, 3], "auto_steps": True, "out_per_tauB": 100}
    LJ2n / SW / YUK are matched to each A by B2* (eps from the matching, at each range parameter).
    AO is matched to each A by B2* (phi_p from the matching, at each q)."""
    match_cache = {} if match_cache is None else match_cache
    sc = spec.get("scenario", "II")
    fields = {"I": [{"dn_nm": 0.0, "flow_F": 0.0}],
              "II": [{"dn_nm": float(d), "flow_F": 0.0} for d in spec.get("dn") or [4000]],
              "III": [{"dn_nm": 0.0, "flow_F": 0.0}],
              "IV": [{"dn_nm": 0.0, "flow_F": float(F)} for F in spec.get("F") or [3000]]}[sc]
    start = dict(base.get("start", {"kind": "grid"}))
    if sc in ("III", "IV"):
        # Regime III/IV: the COMSOL band, or the start set in Settings when it is a run frame (e.g. a 3D band)
        start = start if (spec.get("keep_start") and start.get("kind") == "run") else {"kind": "aaf"}
    elif start.get("kind") in ("aaf", "run"):
        start = {"kind": "grid", "n_side": 12, "spacing": 4.0}
    regime = {"I": "NAF", "II": "ARF", "III": "AAF", "IV": "ASF"}[sc]
    if int(base.get("dim", 2)) == 3:
        regime += "_3D"
    heights = [float(h) for h in spec.get("H") or []] if int(base.get("dim", 2)) == 3 else []
    items = []
    for A in spec.get("A") or [3.0]:
        for pot in spec.get("potentials") or ["LJ"]:
            if pot == "AO":
                variants = [("q", float(q)) for q in (spec.get("q") or [0.2])]
            elif pot in ATT_PARAM:
                pk = ATT_PARAM[pot][0]
                variants = [(pk, float(v)) for v in (spec.get("attp", {}).get(pot) or [ATT_PARAM[pot][2]])]
            else:
                variants = [(None, None)]
            for vk, vv in variants:
                q = vv if pot == "AO" else None
                it = {"potential": pot, "A": float(A), "start": start, "regime": regime,
                      "T_tauB": float(spec.get("T_tauB", base.get("T_tauB", 10))),
                      "seeds": [int(s) for s in spec.get("seeds") or [1, 2, 3]],
                      "out_per_tauB": int(spec.get("out_per_tauB", base.get("out_per_tauB", 100)))}
                if pot == "AO":
                    key = (float(q), float(A))
                    if key not in match_cache:
                        phi, _ = match_phi({**base, "potential": "AO", "q": float(q)}, float(A))
                        match_cache[key] = float(phi)
                    it.update({"q": float(q), "phi_p": round(match_cache[key], 3)})
                elif pot in ATT_PARAM:
                    key = (pot, vv, float(A))
                    if key not in match_cache:
                        e, _ = match_eps({**base, "potential": pot, vk: vv, "eps": 1.0, "A": 0.0}, float(A))
                        match_cache[key] = float(e)
                    it.update({vk: vv, "eps": round(match_cache[key], 3)})
                if spec.get("auto_steps", True):
                    it["steps_per_tauB"] = suggested_steps({**base, **it})
                for f in fields:
                    for h in heights or [None]:
                        items.append({**it, **f, **({"H": h} if h is not None else {})})
    return items


def batch_runs(base, items):
    """[(regime, run name, exists?, steps)] for every run of a batch."""
    out = []
    for it in items:
        c = {**sr.DEFAULTS, **base, **it}
        for s in c["seeds"]:
            reg, name = sr.run_folder_name(c, int(s))
            ex = os.path.isfile(os.path.join(c["out_root"], reg, name, "datax.csv"))
            out.append((reg, name, ex, int(c["steps_per_tauB"]) * float(c["T_tauB"]), c["potential"]))
    return out


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
        keys = ([f"{k}-{s}" for k, b in enumerate(c["batch"]) for s in b.get("seeds", c.get("seeds", []))]
                if c.get("batch") else c.get("seeds", []))
        for s in keys:
            try:
                with open(os.path.join(jd, f"status_seed{s}.json")) as f:
                    st.append({"seed": s, **json.load(f)})
            except Exception:
                st.append({"seed": s, "state": "waiting", "frac": 0.0})
        state = ("failed" if os.path.exists(os.path.join(jd, "failed")) else
                 "stopped" if os.path.exists(os.path.join(jd, "STOP")) and not os.path.exists(os.path.join(jd, "done")) else
                 "done" if os.path.exists(os.path.join(jd, "done")) else "running")
        if state == "running":   # no progress written for 15 min: the window was closed / the process killed
            try:
                last = max(os.path.getmtime(os.path.join(jd, f)) for f in os.listdir(jd))
                if time.time() - last > 900:
                    state = "stopped"
            except ValueError:
                pass
        if state != "running":   # job is over: a run still marked running/waiting either finished or never ran
            for r in st:
                if r.get("state") in ("running", "waiting"):
                    fin = r.get("folder") and os.path.isfile(os.path.join(r["folder"], "datax.csv"))
                    r.update({"state": "done", "frac": 1.0} if fin else {"state": "not run", "frac": 0.0})
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

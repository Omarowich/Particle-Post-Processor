"""sim_runner.py - runs airpods_bd.py simulations from a settings file (used by the analyser's
"Run simulation" page, but works on its own too):

    python sim_runner.py job_folder/config.json
    python sim_runner.py "<Data>/_batches/my_batch.json"      (a batch saved from the app's Batch tab)

The job folder gets  log.txt, status_seed<N>.json  (progress)  and  done / failed  markers.
A config with a "batch" list runs many settings in one job: every entry overrides some settings
(A, q, phi_p, dn_nm, flow_F, potential, start, seeds, ...) and its seeds are run; status files are
then status_seed<entry>-<seed>.json.
Put a file called STOP in the job folder to stop all its runs after the current frame.
Every finished run folder gets a copy of its settings (sim_config.json), so it can be repeated.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import airpods_bd as bd  # noqa: E402

# settings that can be changed, with the values of the COMSOL model
DEFAULTS = {
    "potential": "LJ",            # "LJ", "AO" or "none" (core only)
    "A": 3.0,                     # LJ depth (kBT); for AO the matched LJ value, used in the run name
    "q": 0.2, "phi_p": 0.633,     # AO size ratio and polymer volume fraction
    "core": "coulomb",            # "coulomb" (COMSOL), "phs", "wca", "none"
    "core_Z": 100,
    "lj_cutoff_sigma": 3.0,       # LJ cut off at this many sigma
    "dn_nm": 4000.0,              # field amplitude (inward wall displacement); 0 = field off
    "flow_F": 0.0,                # acoustic streaming parameter F (as in COMSOL); 0 = no streaming
    "arf_F0_ref": bd.ARF_F0_REF, "arf_k": bd.ARF_K,
    "start": {"kind": "grid", "n_side": 12, "spacing": 4.0},
    "T_tauB": 1.0, "steps_per_tauB": 4000, "out_per_tauB": 100,
    "seeds": [1],
    "box_half": bd.BOX,
    "T_K": bd.T_K, "eta": bd.ETA, "radius_um": bd.A_P,
    "max_step_um": bd.MAX_STEP_UM,
    "out_root": r"D:\Work\AIRPODS Paper 2627\Data", "regime": "auto", "tag": "", "workers": 1,
}


def apply_physics(c):
    """Set the module constants of airpods_bd from the settings (each process does this itself)."""
    bd.T_K, bd.ETA, bd.A_P = float(c["T_K"]), float(c["eta"]), float(c["radius_um"])
    bd.D = bd.KB * bd.T_K / (6 * np.pi * bd.ETA * bd.A_P * 1e-6) * 1e12
    bd.TAUB = bd.A_P ** 2 / bd.D
    bd.SIGMA = 2.0 * bd.A_P
    bd.BOX = float(c["box_half"])
    bd.ARF_F0_REF, bd.ARF_K = float(c["arf_F0_ref"]), float(c["arf_k"])
    bd.CORE_Z = c["core_Z"]
    bd.CORE_K = (bd.CORE_Z * 1.602176634e-19) ** 2 / (4 * np.pi * 8.8541878128e-12) / (bd.KB * bd.T_K) * 1e6
    bd.CORE = c["core"]
    bd.CORE_SIG = 2.0 * bd.A_P / 2 ** (1 / 6)
    bd.MAX_STEP_UM = float(c["max_step_um"])
    if float(c.get("lj_cutoff_sigma", 3.0)) != 3.0:
        _patch_lj_cutoff(float(c["lj_cutoff_sigma"]))


def _patch_lj_cutoff(n_sigma):
    orig = bd.radial_force

    def radial_force(r, A, ao=None, core=None):
        fr = orig(r, 0.0, ao, core)          # core (+ AO)
        if ao is None and A > 0:
            r = np.asarray(r, float)
            m = r < n_sigma * bd.SIGMA
            s6 = (bd.SIGMA / r[m]) ** 6
            fr[m] += 24 * A / r[m] * (2 * s6 ** 2 - s6)
        return fr
    bd.radial_force = radial_force


def start_positions(st, run_seed):
    kind = st.get("kind", "grid")
    if kind == "grid":
        n, a = int(st.get("n_side", 12)), float(st.get("spacing", 4.0))
        g = (np.arange(n) - (n - 1) / 2) * a
        gx, gy = np.meshgrid(g, g, indexing="ij")
        return np.c_[gx.ravel(), gy.ravel()]
    if kind == "random":
        rng = np.random.default_rng(int(st.get("seed", run_seed)) + 1000)
        n, dmin, L = int(st.get("n", 144)), float(st.get("min_dist", 2.2)), bd.BOX - bd.A_P
        P = []
        tries = 0
        while len(P) < n:
            p = rng.uniform(-L, L, 2)
            if not P or np.min(np.hypot(*(np.array(P) - p).T)) >= dmin:
                P.append(p)
            tries += 1
            if tries > 200000:
                raise RuntimeError(f"could only place {len(P)} of {n} particles at min. distance {dmin} µm")
        return np.array(P)
    if kind == "aaf":
        return np.loadtxt(bd.AAF_START, delimiter=",", skiprows=1)
    if kind == "file":
        return np.loadtxt(st["path"], delimiter=",", skiprows=1)[:, :2]
    if kind == "run":   # a frame of an existing run
        x, y, t = bd.read_comsol(st["folder"])
        k = int(np.clip(np.searchsorted(t, float(st.get("t_tauB", 1e9)) * bd.TAUB - 1e-9), 0, len(t) - 1))
        return np.c_[x[k], y[k]]
    raise ValueError(f"unknown start kind {kind}")


def regime_of(c):
    if c["regime"] != "auto":
        return c["regime"]
    if float(c.get("flow_F", 0)) > 0:
        return "ASF"
    if float(c["dn_nm"]) > 0:
        return "ARF"
    return "AAF" if c["start"].get("kind") in ("aaf", "run") else "NAF"


def run_folder_name(c, seed):
    ao = (float(c["q"]), float(c["phi_p"])) if c["potential"] == "AO" else None
    reg = regime_of(c)
    name = bd.run_name(float(c["A"]) if c["potential"] != "none" else 0.0, float(c["T_tauB"]), float(c["dn_nm"]),
                       ao, seed, int(c["steps_per_tauB"]), reg, float(c.get("flow_F", 0)))
    tag = c.get("tag", "").strip().strip("_")
    if tag:
        name = name.replace(f"_seed{seed}", f"_{tag}_seed{seed}")
    return reg, name


def _status(job_dir, key, **kw):
    """Write the progress file. Never lets a locked file (Windows, while the page reads it) stop a run."""
    p = os.path.join(job_dir, f"status_seed{key}.json")
    tmp = p + f".{os.getpid()}.tmp"
    for _ in range(5):
        try:
            with open(tmp, "w") as f:
                json.dump(kw, f)
            os.replace(tmp, p)
            return
        except OSError:
            time.sleep(0.05)


def run_one(args):
    c, seed, job_dir = args[:3]
    key = args[3] if len(args) > 3 else seed
    apply_physics(c)
    reg, name = run_folder_name(c, seed)
    folder = os.path.join(c["out_root"], reg, name)
    if os.path.isfile(os.path.join(folder, "datax.csv")):
        _status(job_dir, key, state="exists", name=name, folder=folder, frac=1.0)
        return name, "already existed"
    P0 = start_positions(c["start"], seed)
    ao = (float(c["q"]), float(c["phi_p"])) if c["potential"] == "AO" else None
    A = float(c["A"]) if c["potential"] == "LJ" else 0.0
    t0 = time.time()
    stop = os.path.join(job_dir, "STOP")

    def progress(s, n, capped):
        if os.path.exists(stop):
            raise KeyboardInterrupt("stopped")
        el = time.time() - t0
        _status(job_dir, key, state="running", name=name, folder=folder, frac=s / n, step=s, n_steps=n,
                elapsed=el, eta=el / max(s, 1) * (n - s), capped=capped)

    _status(job_dir, key, state="running", name=name, folder=folder, frac=0.0)
    try:
        x, y, t = bd.simulate(A, float(c["dn_nm"]), float(c["T_tauB"]), ao=ao,
                              steps_per_tauB=int(c["steps_per_tauB"]), out_per_tauB=int(c["out_per_tauB"]),
                              seed=seed, core=c["core"] if c["core"] != "none" else "off", verbose=False,
                              start=P0, progress=progress, flow_F=float(c.get("flow_F", 0)))
    except KeyboardInterrupt:
        _status(job_dir, key, state="stopped", name=name, folder=folder, frac=0.0)
        return name, "stopped"
    bd.write_comsol_csv(folder, x, y, t)
    with open(os.path.join(folder, "sim_config.json"), "w") as f:
        json.dump({**c, "seeds": [seed], "run_seed": seed, "D_um2_s": bd.D, "tauB_s": bd.TAUB,
                   "runtime_s": time.time() - t0, "airpods_bd": os.path.join(HERE, "airpods_bd.py")}, f, indent=2)
    _status(job_dir, key, state="done", name=name, folder=folder, frac=1.0, elapsed=time.time() - t0)
    return name, "done"


def expand_jobs(c, job_dir):
    """(settings, seed, job folder, status key) for every run of a job, batch or single."""
    if c.get("batch"):
        base = {k: v for k, v in c.items() if k != "batch"}
        jobs = []
        for k, b in enumerate(c["batch"]):
            ck = {**base, **b}
            jobs += [(ck, int(s), job_dir, f"{k}-{int(s)}") for s in ck["seeds"]]
        return jobs
    return [(c, int(s), job_dir, int(s)) for s in c["seeds"]]


def main(cfg_path):
    job_dir = os.path.dirname(os.path.abspath(cfg_path))
    with open(cfg_path) as f:
        c = {**DEFAULTS, **json.load(f)}
    if os.path.basename(os.path.dirname(job_dir)) != "_sim_jobs":
        # started by hand on a saved settings / batch file: make a job folder like the app does
        job_dir = os.path.join(c["out_root"], "_sim_jobs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "config.json"), "w") as f:
            json.dump(c, f, indent=2)
        print(f"job folder: {job_dir}  (progress also shows in the app's Jobs tab)")
    log = open(os.path.join(job_dir, "log.txt"), "a", buffering=1)
    log.write(f"started {time.ctime()}  pid {os.getpid()}\n")
    jobs = expand_jobs(c, job_dir)
    log.write(f"{len(jobs)} run(s), {int(c.get('workers', 1))} at the same time\n")
    try:
        if int(c.get("workers", 1)) > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(int(c["workers"]), len(jobs))) as ex:
                for name, res in ex.map(run_one, jobs):
                    log.write(f"{name}: {res}\n")
        else:
            for j in jobs:
                name, res = run_one(j)
                log.write(f"{name}: {res}\n")
        open(os.path.join(job_dir, "done"), "w").close()
        log.write(f"finished {time.ctime()}\n")
    except Exception as ex:
        log.write(traceback.format_exc())
        if isinstance(ex, MemoryError) or "BrokenProcessPool" in type(ex).__name__ or "Memory" in type(ex).__name__:
            log.write("\nOUT OF MEMORY (or a run process was killed): the computer ran out of RAM. Start it again with "
                      "fewer 'runs at the same time', and close other simulations first. Finished runs are kept and "
                      "skipped on the next start.\n")
        open(os.path.join(job_dir, "failed"), "w").close()


if __name__ == "__main__":
    main(sys.argv[1])

"""sim_runner.py - runs airpods_bd.py simulations from a settings file (used by the analyser's
"Run simulation" page, but works on its own too):

    python sim_runner.py job_folder/config.json
    python sim_runner.py "<Data>/_batches/my_batch.json"      (a batch saved from the app's Batch tab)
    python sim_runner.py "<Data>/_batches/runs.csv" [--workers 6] [--priority 1,2] [--out "<Data>"]
    python sim_runner.py a.json b.csv c.csv                     (several files, one after the other)
    python sim_runner.py --queue "<Data>/_batches/queue" [--watch]

Queue: every .json / .csv in the queue folder is run in name order (prefix 1_, 2_ ... to set the order) and
then moved to queue/done (or queue/failed). Files added while it runs are picked up too. With --watch it keeps
waiting for new files instead of stopping when the queue is empty (stop with Ctrl+C).

A CSV has one row per parameter set; its columns are setting names (regime, potential, A, q, phi_p,
dn_nm, flow_F, T_tauB, steps_per_tauB, seeds, core, tag, ...). Empty cells keep the default, seeds are
separated by spaces ("1 2 3"), unknown columns (priority, runs, est_core_hours, purpose) are ignored
except "priority", which --priority filters on. Comma or semicolon separated (German Excel) both work.
Results go to --out, else to the folder above "_batches" if the CSV sits in one.

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
    "potential": "LJ",            # "LJ", "AO", "none" (core only), or "LJ2n" / "SW" / "YUK" (need eps + n / width / kappa)
    "A": 3.0,                     # LJ depth (kBT); for AO the matched LJ value, used in the run name
    "q": 0.2, "phi_p": 0.633,     # AO size ratio and polymer volume fraction
    "core": "coulomb",            # "coulomb" (COMSOL), "phs", "wca", "none"
    "core_Z": 100,
    "lj_cutoff_sigma": 3.0,       # LJ cut off at this many sigma
    "dn_nm": 4000.0,              # field amplitude (inward wall displacement); 0 = field off
    "flow_F": 0.0,                # acoustic streaming parameter F (as in COMSOL); 0 = no streaming
    "flow_profile": "uniform",    # 3D: in-plane streaming "uniform" in height or "parabolic" (no slip at floor and lid)
    "flow_hele_shaw": False,      # 3D: treat F as a body force between floor and lid: in-plane speed reduced by the wall
                                  # friction, factor K^2 / (K^2 + 12/H^2), K = 2 pi / flow_vortex_um (Brinkman / Hele-Shaw)
    "flow_vortex_um": 50.0,       # size of one streaming vortex pair (um) for that factor
    "rayleigh_A": 0.0,            # 3D: Rayleigh streaming rolls between floor and lid, um/s (0 = off)
    "secondary_E0": 0.0,          # secondary radiation force, acoustic energy density in kBT/um^3 (0 = off)
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
    if kind == "grid3d":   # stacked square layers on the floor (3D runs)
        import bd_engine as be
        return be.grid_start(int(st.get("n_side", 10)), float(st.get("spacing", 5.0)), dim=3,
                             layers=int(st.get("layers", 1)), dz=float(st.get("dz", 2.0)))
    if kind == "aaf":
        return np.loadtxt(bd.AAF_START, delimiter=",", skiprows=1)
    if kind == "file":
        return np.loadtxt(st["path"], delimiter=",", skiprows=1)[:, :2]
    if kind == "run":   # a frame of an existing run
        x, y, t = bd.read_comsol(st["folder"])
        k = int(np.clip(np.searchsorted(t, float(st.get("t_tauB", 1e9)) * bd.TAUB - 1e-9), 0, len(t) - 1))
        zf = os.path.join(st["folder"], "dataz.csv")
        if os.path.isfile(zf):          # a 3D run: keep the heights (e.g. a 3D band)
            z = np.loadtxt(zf, delimiter=",", skiprows=1)[0::2]
            return np.c_[x[k], y[k], z[k]]
        return np.c_[x[k], y[k]]
    raise ValueError(f"unknown start kind {kind}")


def regime_of(c):
    if c["regime"] != "auto":
        return c["regime"]
    if float(c.get("flow_F", 0)) > 0:
        reg = "ASF"
    elif float(c["dn_nm"]) > 0:
        reg = "ARF"
    else:
        reg = "AAF" if c["start"].get("kind") in ("aaf", "run") else "NAF"
    return reg + "_3D" if int(c.get("dim", 2)) == 3 else reg


ATT_MODELS = {"LJ2n": "n", "SW": "width", "YUK": "kappa"}


def att_of(c):
    """Settings of the extra attraction models (paper 2), or None for LJ / AO / none."""
    m = c["potential"]
    if m not in ATT_MODELS:
        return None
    return {"model": m, "eps": float(c["eps"]), ATT_MODELS[m]: float(c[ATT_MODELS[m]])}


def radii_of(c, seed):
    """Per-particle radii for polydisperse runs (Gaussian, relative width c['polydispersity'], cut at +-2 sigma),
    drawn from the run seed so a repeat gives the same particles; None = monodisperse."""
    pd = float(c.get("polydispersity", 0) or 0)
    if pd <= 0:
        return None
    rng = np.random.default_rng(int(seed) + 5000)
    z = np.clip(rng.standard_normal(4096), -2, 2)
    return bd.A_P * (1 + pd * z)


def run_folder_name(c, seed):
    ao = (float(c["q"]), float(c["phi_p"])) if c["potential"] == "AO" else None
    reg = regime_of(c)
    name = bd.run_name(float(c["A"]) if c["potential"] != "none" else 0.0, float(c["T_tauB"]), float(c["dn_nm"]),
                       ao, seed, int(c["steps_per_tauB"]), reg.split("_")[0], float(c.get("flow_F", 0)),
                       att_of(c), float(c.get("polydispersity", 0) or 0))
    tag = c.get("tag", "").strip().strip("_")
    if int(c.get("dim", 2)) == 3:
        H = c.get("H")
        t3 = f"3D_H{float(H):g}" if H not in (None, "", 0) else "3D_open"
        t3 += f"_drho{float(c.get('drho', 0)):g}"
        tag = f"{t3}_{tag}" if tag else t3
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


_ARF3D_CACHE = {}


def _arf3d_force(c):
    """GridForce from c['arf3d_file'] (COMSOL export, see arf3d_field.py), scaled to the 2D fit at c['dn_nm'].
    Settings: arf3d_freq (Hz, needed if no velocity columns), arf3d_rho_p / arf3d_c_p (particle; default PS 1050 kg/m3,
    2350 m/s), arf3d_offset_um (x, y, z of the BD origin in the COMSOL frame), arf3d_match (default true)."""
    import arf3d_field as af
    key = (c["arf3d_file"], float(c["dn_nm"]), str(c.get("arf3d_offset_um")), c.get("arf3d_rho_p"), c.get("arf3d_c_p"))
    if key not in _ARF3D_CACHE:
        if str(c["arf3d_file"]).lower().endswith(".npz"):     # precomputed force grid (acoustic3d / save_force)
            axes, F = af.load_force(c["arf3d_file"])
        else:
            axes, p, v = af.load_grid(c["arf3d_file"])
            f1, f2 = af.contrast_factors(float(c.get("arf3d_rho_p", 1050.0)), float(c.get("arf3d_c_p", 2350.0)))
            F = af.force_grid(axes, p, v, f1=f1, f2=f2, freq=c.get("arf3d_freq"))
        gf = af.GridForce(axes, F, offset_um=tuple(c.get("arf3d_offset_um", (0.0, 0.0, 0.0))))
        if c.get("arf3d_match", True):
            af.match_F0(gf, float(c["dn_nm"]), float(c.get("arf_F0_ref", bd.ARF_F0_REF)))
        _ARF3D_CACHE[key] = gf
    return _ARF3D_CACHE[key]


_AUTO_CACHE = {}


def _field_auto(c):
    """Radiation force and streaming of one solved chamber field (streaming2d.compute), cached in <out_root>/_fields.
    c['field_auto'] = {"drive": "side" | "idt", "lid": "glass" | "pdms" | "absorb", "freq_MHz": 20}; both scaled to the
    2D fit at c['dn_nm'] (one acoustic amplitude). c['field_auto_streaming']: factor on the streaming (1 = as
    computed, 0 = off); c['radiation_off']: true = streaming only (Regime IV-like). Returns (arf3d callable, flow3d)."""
    import streaming2d as s2
    fa = c["field_auto"]
    drive, lid, f = fa.get("drive", "side"), fa.get("lid", "glass"), float(fa.get("freq_MHz", 20.0))
    H, bh = float(c["H"]), float(c.get("box_half", bd.BOX))
    fluid = fa.get("fluid", "legacy")
    path = os.path.join(c.get("out_root", "."), "_fields",
                        f"auto_{drive}{f:g}_H{H:g}_box{2 * bh:g}_{lid}" + ("" if fluid == "legacy" else f"_{fluid}") + ".npz")
    if path not in _AUTO_CACHE:
        if not os.path.isfile(path):
            res = s2.compute(drive, H_um=H, box_half_um=bh, freq_MHz=f, lid=lid, fluid=fluid)
            s2.save(path, res, fa)
        _AUTO_CACHE[path] = np.load(path)
    d = _AUTO_CACHE[path]
    s = (float(c["dn_nm"]) / 4000.0) ** 2
    ys, zs = d["ys"] * 1e6, d["zs"] * 1e6
    Fy, Fz = d["Fy"] * s, d["Fz"] * s
    sf = float(c.get("field_auto_streaming", 1.0)) * s
    U, W = d["U"] * sf, d["W"] * sf

    def interp(G, P):
        fy = np.clip((P[:, 1] - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.000001)
        fz = np.clip((P[:, 2] - zs[0]) / (zs[1] - zs[0]), 0, len(zs) - 1.000001)
        i, j = fy.astype(int), fz.astype(int); ty, tz = fy - i, fz - j
        return ((1 - ty) * (1 - tz) * G[i, j] + ty * (1 - tz) * G[i + 1, j] + (1 - ty) * tz * G[i, j + 1]
                + ty * tz * G[i + 1, j + 1])

    def arf3d(P):
        out = np.zeros_like(P)
        if not c.get("radiation_off"):
            out[:, 1], out[:, 2] = interp(Fy, P), interp(Fz, P)
        return out

    def flow3d(P):
        out = np.zeros_like(P)
        if sf:
            out[:, 1], out[:, 2] = interp(U, P), interp(W, P)
        return out
    return arf3d, (flow3d if sf else None)


def _streaming3d(c):
    """Rayleigh streaming rolls (3D, needs H): c['rayleigh_A'] in um/s (0 = off), c['rayleigh_sign'] (-1 = physical
    direction for a pressure node at y = 0), wavenumber of the pressure field c['rayleigh_k'] (default arf_k / 2)."""
    A = float(c.get("rayleigh_A", 0) or 0)
    H = c.get("H")
    if not A or H in (None, "", 0) or int(c.get("dim", 2)) != 3:
        return None
    return {"A": A, "k": float(c.get("rayleigh_k", float(c.get("arf_k", 0.124)) / 2)), "H": float(H),
            "sign": float(c.get("rayleigh_sign", -1.0))}


def _secondary(c):
    """Secondary (particle-particle) radiation force: c['secondary_E0'] acoustic energy density in kBT/um^3
    (0 = off; ~390 nominal, 1663 upper bound at d_n = 4000 nm), monopole-free dipole term (Silva & Bruus 2014)."""
    E0 = float(c.get("secondary_E0", 0) or 0)
    if not E0:
        return None
    return {"E0": E0, "f2": float(c.get("secondary_f2", -0.079)), "k": float(c.get("arf_k", 0.124)) / 2,
            "rcut": float(c.get("secondary_rcut", 6.0))}


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
    if int(c.get("dim", 2)) != 3 and P0.shape[1] == 3:
        P0 = P0[:, :2]                       # a 2D run started from a frame of a 3D run: drop the heights
    ao = (float(c["q"]), float(c["phi_p"])) if c["potential"] == "AO" else None
    att = att_of(c)
    radii = radii_of(c, seed)
    if radii is not None:
        radii = radii[:len(P0)]
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
        dim = int(c.get("dim", 2))
        use_nl = c.get("engine", "dense") == "nl" or dim == 3
        if use_nl:
            try:
                import bd_engine as be
            except ImportError:                      # no scipy: fall back to the dense pair search (same physics)
                use_nl = False
        if use_nl:
            H = c.get("H")
            arf3d = None
            flow3d = None
            if dim == 3 and c.get("field_auto"):      # solved chamber field: radiation force + computed streaming
                arf3d, flow3d = _field_auto(c)
            elif dim == 3 and c.get("arf3d_file"):    # COMSOL 3D pressure field -> Gor'kov force (arf3d_field.py)
                arf3d = _arf3d_force(c)
            flowF = float(c.get("flow_F", 0))
            if dim == 3 and flowF > 0 and c.get("flow_hele_shaw") and H not in (None, "", 0):
                K = 2 * np.pi / float(c.get("flow_vortex_um", 50.0))
                flowF *= K ** 2 / (K ** 2 + 12.0 / float(H) ** 2)
            fr, t = be.simulate(A, float(c["dn_nm"]), float(c["T_tauB"]), ao=ao, att=att, radii=radii,
                                steps_per_tauB=int(c["steps_per_tauB"]), out_per_tauB=int(c["out_per_tauB"]),
                                seed=seed, core=c["core"] if c["core"] != "none" else "off", start=P0,
                                progress=progress, flow_F=flowF, dim=dim,
                                drho=float(c.get("drho", 0.0)), H=float(H) if H not in (None, "", 0) else None,
                                arf3d=arf3d, flow_profile=c.get("flow_profile", "uniform"),
                                streaming3d=_streaming3d(c), secondary=_secondary(c), flow3d=flow3d)
            x, y = fr[:, :, 0], fr[:, :, 1]
            if dim == 3:
                be.write_csv(folder, fr, t)              # datax/datay/dataz
        else:
          x, y, t = bd.simulate(A, float(c["dn_nm"]), float(c["T_tauB"]), ao=ao,
                              steps_per_tauB=int(c["steps_per_tauB"]), out_per_tauB=int(c["out_per_tauB"]),
                              seed=seed, core=c["core"] if c["core"] != "none" else "off", verbose=False,
                              start=P0, progress=progress, flow_F=float(c.get("flow_F", 0)),
                              **({"att": att} if att else {}), **({"radii": radii} if radii is not None else {}))
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


CSV_NUMERIC = {"rayleigh_A", "rayleigh_sign", "secondary_E0", "dim", "drho", "H", "eps", "n", "width", "kappa", "polydispersity", "A", "q", "phi_p", "dn_nm", "flow_F", "T_tauB", "steps_per_tauB", "out_per_tauB", "core_Z",
               "lj_cutoff_sigma"}
REGIME_START = {"NAF": {"kind": "grid", "n_side": 12, "spacing": 4.0},
                "ARF": {"kind": "grid", "n_side": 12, "spacing": 4.0},
                "AAF": {"kind": "aaf"}, "ASF": {"kind": "aaf"}}


def load_csv_batch(path, priorities=None):
    """Batch config from a CSV file (one row per parameter set). priorities: keep only these rows."""
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    delim = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    batch = []
    for r in rows:
        r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
        if not any(r.values()) or r.get("regime", "").startswith("#"):
            continue
        if priorities and r.get("priority") and int(float(r["priority"].replace(",", "."))) not in priorities:
            continue
        it = {}
        for k, v in r.items():
            if v == "" or k in ("priority", "runs", "est_core_hours", "purpose", "est_hours_6_cores"):
                continue
            if k == "seeds":
                it["seeds"] = [int(x) for x in v.replace(",", " ").replace(";", " ").split()]
            elif k in CSV_NUMERIC:
                x = float(v.replace(",", "."))
                it[k] = int(x) if k in ("steps_per_tauB", "out_per_tauB", "core_Z") else x
            else:
                it[k] = v
        reg = it.get("regime", "auto").upper()
        it["regime"] = reg
        if reg in REGIME_START and "start" not in it:
            it["start"] = REGIME_START[reg]
        if reg in ("NAF", "AAF", "ASF"):
            it.setdefault("dn_nm", 0.0)        # no radiation-force field unless the row asks for one
        if reg != "ASF":
            it.setdefault("flow_F", 0.0)
        batch.append(it)
    return {"batch": batch, "batch_source": os.path.abspath(path)}


def main(cfg_path, workers=None, priorities=None, out=None):
    job_dir = os.path.dirname(os.path.abspath(cfg_path))
    if cfg_path.lower().endswith(".csv"):
        c = {**DEFAULTS, **load_csv_batch(cfg_path, priorities)}
        here = os.path.dirname(os.path.abspath(cfg_path))
        if os.path.basename(here) == "_batches":
            c["out_root"] = os.path.dirname(here)
        if not c["batch"]:
            print("no rows to run (check the --priority filter)")
            return
    else:
        with open(cfg_path) as f:
            c = {**DEFAULTS, **json.load(f)}
    if out:
        c["out_root"] = out
    if workers:
        c["workers"] = int(workers)
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
        print(f"finished {len(jobs)} run(s)")
        return True
    except Exception as ex:
        log.write(traceback.format_exc())
        if isinstance(ex, MemoryError) or "BrokenProcessPool" in type(ex).__name__ or "Memory" in type(ex).__name__:
            log.write("\nOUT OF MEMORY (or a run process was killed): the computer ran out of RAM. Start it again with "
                      "fewer 'runs at the same time', and close other simulations first. Finished runs are kept and "
                      "skipped on the next start.\n")
        open(os.path.join(job_dir, "failed"), "w").close()
        print(f"FAILED - see {os.path.join(job_dir, 'log.txt')}")
        return False


def run_queue(qdir, workers=None, priorities=None, out=None, watch=False, poll_s=30):
    """Run every .json / .csv in qdir (name order), move each to done/ or failed/ afterwards, pick up new files."""
    os.makedirs(os.path.join(qdir, "done"), exist_ok=True)
    os.makedirs(os.path.join(qdir, "failed"), exist_ok=True)
    print(f"queue: {qdir}  (add .json/.csv files any time{'; Ctrl+C to stop' if watch else ''})")
    while True:
        todo = sorted(f for f in os.listdir(qdir) if f.lower().endswith((".json", ".csv"))
                      and os.path.isfile(os.path.join(qdir, f)))
        if not todo:
            if not watch:
                print("queue empty - done")
                return
            time.sleep(poll_s)
            continue
        f = todo[0]
        src = os.path.join(qdir, f)
        print(f"{time.strftime('%H:%M')}  starting {f}  ({len(todo) - 1} more waiting)")
        # a queue file without an output folder writes next to the queue's _batches folder (the Data folder)
        dflt_out = out or (os.path.dirname(os.path.dirname(os.path.abspath(qdir)))
                           if os.path.basename(os.path.dirname(os.path.abspath(qdir))) == "_batches" else None)
        w = int(workers or 4)
        wf = os.path.join(qdir, "WORKERS")          # a file with a number overrides --workers (read every file)
        if os.path.isfile(wf):
            try:
                w = int(open(wf).read().strip())
            except ValueError:
                pass
        # out of memory: retry (finished runs are skipped), slowly going down to one run at a time
        tries = [w, w] + [x for x in range(w - 1, 0, -1)] + [1] * 6
        ok = False
        for k, wk in enumerate(tries):
            if k:
                print(f"{time.strftime('%H:%M')}  retrying {f} with {wk} at once (finished runs are skipped)")
                time.sleep(120)
            try:
                ok = main(src, wk, priorities, dflt_out) is not False
            except Exception:
                traceback.print_exc()
                ok = False
            if ok:
                break
        dest = os.path.join(qdir, "done" if ok else "failed", time.strftime("%Y%m%d-%H%M_") + f)
        os.replace(src, dest)
        print(f"{time.strftime('%H:%M')}  {'finished' if ok else 'FAILED'} {f} -> {os.path.relpath(dest, qdir)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run airpods_bd simulations from settings / batch files (.json or .csv)")
    ap.add_argument("files", nargs="*", help="one or more files, run one after the other")
    ap.add_argument("--queue", help="queue folder: run every file in it, in name order")
    ap.add_argument("--watch", action="store_true", help="with --queue: keep waiting for new files")
    ap.add_argument("--workers", type=int, help="runs at the same time (default: from the file, else 1)")
    ap.add_argument("--priority", help="CSV only: run only rows with these priorities, e.g. 1 or 1,2")
    ap.add_argument("--out", help="data folder for the results (default: from the file / the folder above _batches)")
    a = ap.parse_args()
    pr = {int(x) for x in a.priority.split(",")} if a.priority else None
    if a.queue:
        run_queue(a.queue, a.workers, pr, a.out, a.watch)
    for f in a.files:
        main(f, a.workers, pr, a.out)
    if not a.queue and not a.files:
        ap.print_help()

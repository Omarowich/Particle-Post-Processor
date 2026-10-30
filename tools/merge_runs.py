"""merge_runs.py - copy COMSOL and Python (BD) runs into one Data folder with source tags.

    python merge_runs.py --comsol "D:/Work/AIRPODS Paper 2627/raw csv data" \
                         --bd "D:/Work/AIRPODS Paper 2627/Simulations/Python simulations/BD runs v2" \
                         --archive-bd "D:/Work/AIRPODS Paper 2627/Simulations/Python simulations/BD runs" \
                         --out "D:/Work/AIRPODS Paper 2627/Data"

Layout of --out:
    ARF/ NAF/ AAF/ ASF/        all runs; COMSOL names end in _CS, Python names contain _BD
    _archive/BD runs v1/       first Python version (soft core, AO too sticky) - not for analysis
    _archive/duplicates/       COMSOL exports that are byte-identical to another run
    index.csv                  one row per run (source, parameters, true length, notes, original path)
    README.md                  naming scheme

After adding runs to --out directly (Run simulation page, airpods_bd.py --out Data, COMSOL exports
named <...>_CS), refresh the index with:   python merge_runs.py --out "<Data>" --reindex

Only copies; never deletes or changes the source folders. Re-running skips files that are already
there (same size), so new runs can be added later. --budget stops after that many seconds (resume
by running again).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import time

# same run-name rule as the analyser (regime_state_model.parse_run_name); kept here so this
# script needs nothing but Python itself
_NAME_RE = re.compile(r"^([\d.]+)Tkb_([\d.]+)TauB(?:_(\d+(?:\.\d+)?)(nm|nn|F))?((?:_[A-Za-z]+[\d.]*)*)$", re.I)


def _tag(tags, key, cast):
    m = re.search(rf"_{key}([\d.]+)(?=_|$)", tags or "")
    return cast(m.group(1)) if m else None


def parse_name(name):
    m = _NAME_RE.match(name)
    if not m:
        return {}
    unit, val, tags = (m.group(4) or "").lower(), (float(m.group(3)) if m.group(3) else None), m.group(5) or ""
    return {"A": float(m.group(1)), "T_name": float(m.group(2)),
            "dn": val if unit in ("nm", "nn") else None, "F": val if unit == "f" else None,
            "potential": "AO" if re.search(r"(^|_)AO(_|$)", tags) else "LJ",
            "q": _tag(tags, "q", float), "phi_p": _tag(tags, "phi", float), "seed": _tag(tags, "seed", int)}

REGIMES = ("ARF", "NAF", "AAF", "ASF")
TAUB_S = 13.514
SKIP = {".metric_cache", "_sim_jobs", "_paper_figures_cache.pkl", "__pycache__"}

README = """# AIRPODS run data (COMSOL + Python)

All runs, one folder per regime:

| Folder | Regime (paper) |
|---|---|
| NAF | Regime I: Brownian aggregation, no acoustic force |
| ARF | Regime II: aggregation under the acoustic radiation force |
| AAF | Regime III: post-acoustic relaxation (field switched off, start from an assembled band) |
| ASF | Regime IV: destabilization by acoustic streaming flow |

## Run names

`<A>Tkb_<T>TauB[_<d_n>nm | _<F>F]` + tags

| Tag | Meaning |
|---|---|
| `_CS` | COMSOL run (the old folders, unchanged data) |
| `_BD` | Python Brownian-dynamics run (`airpods_bd.py`) |
| `_AO_q<q>_phi<φp>` | Asakura–Oosawa attraction instead of LJ; A is then the LJ depth it is matched to by B2* |
| `_dt<N>` | time step τ_B/N (Python default τ_B/4000) |
| `_seed<N>` | random seed of a Python run |
| other letters+digits | extra tag given in the Run simulation page |

Examples: `3Tkb_1TauB_4000nm_CS`, `3Tkb_10TauB_4000nm_BD_seed2`,
`3Tkb_20TauB_BD_AO_q0.2_phi0.633_dt32000_seed1`.

## Things to know

- All COMSOL runs used the same random numbers ("Generate unique"), so COMSOL runs are not
  independent repeats. Python runs have real seeds.
- `index.csv` lists every run with its true length; some COMSOL runs are longer than their name says.
- `_archive/` holds data that should not be analysed: the first Python version and duplicate exports.
"""


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_info(folder):
    """Particles, frames and true length from datax.csv (reads the time rows only)."""
    with open(os.path.join(folder, "datax.csv")) as f:
        header = f.readline()
        n = len(header.split(","))
        last_t, frames = 0.0, 0
        for i, line in enumerate(f):
            if i % 2 == 1:
                last_t = float(line.split(",", 1)[0])
                frames += 1
    return n, frames, last_t / TAUB_S


def copy_run(src, dst, deadline):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f in SKIP or f.startswith("."):
            continue
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.isdir(s):
            continue
        if os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        if time.time() > deadline:
            return False
        shutil.copy2(s, d)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comsol", default=None, help="old COMSOL csv folder (raw csv data)")
    ap.add_argument("--bd", default=[], action="append", help="Python runs root (can be given several times)")
    ap.add_argument("--reindex", action="store_true",
                    help="only rebuild index.csv from what is in --out (after adding new runs there)")
    ap.add_argument("--archive-bd", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds")
    a = ap.parse_args()
    deadline = time.time() + a.budget
    out = a.out
    os.makedirs(out, exist_ok=True)
    rows, done = [], True

    if a.reindex:
        rows = reindex(out)
        write_outputs(out, rows)
        print(f"index.csv rebuilt: {len(rows)} runs")
        return

    # COMSOL
    for reg in REGIMES:
        if not a.comsol:
            break
        base = os.path.join(a.comsol, reg)
        if not os.path.isdir(base):
            continue
        seen = {}
        for n in sorted(os.listdir(base)):
            src = os.path.join(base, n)
            if not os.path.isfile(os.path.join(src, "datax.csv")):
                continue
            notes = []
            new = n
            if re.search(r"\dnn$", new):
                new = new[:-2] + "nm"
                notes.append(f"renamed from {n} (typo)")
            new += "_CS"
            h = md5(os.path.join(src, "datax.csv"))
            dest_reg = reg
            if h in seen:
                notes.append(f"identical to {seen[h]} (duplicate export) - archived")
                dst = os.path.join(out, "_archive", "duplicates", reg, new)
                dest_reg = "_archive/duplicates/" + reg
            else:
                seen[h] = new
                dst = os.path.join(out, reg, new)
            npart, frames, T = run_info(src)
            p = parse_name(new)
            Tn = p.get("T_name", float("nan"))
            if Tn and abs(T - Tn) > 0.05 * Tn:
                notes.append(f"runs {T:.2f} tau_B, name says {Tn:g}")
            if reg == "NAF" and p.get("A") == 0:
                notes.append("COMSOL NAF with A = 0: no repulsive core in this run")
            done &= copy_run(src, dst, deadline)
            rows.append(dict(regime=dest_reg, name=new, source="COMSOL", A=p.get("A"), T_name_tauB=Tn,
                             T_actual_tauB=round(T, 3), dn_nm=p.get("dn"), F=p.get("F"), potential="LJ",
                             q="", phi_p="", steps_per_tauB="", seed="", n_particles=npart, n_frames=frames,
                             original=src, notes="; ".join(notes)))

    # Python runs
    for root in a.bd:
        for reg in REGIMES:
            base = os.path.join(root, reg)
            if not os.path.isdir(base):
                continue
            for n in sorted(os.listdir(base)):
                src = os.path.join(base, n)
                if not os.path.isfile(os.path.join(src, "datax.csv")):
                    continue
                p = parse_name(n)
                m = re.search(r"_dt(\d+)", n)
                npart, frames, T = run_info(src)
                done &= copy_run(src, os.path.join(out, reg, n), deadline)
                rows.append(dict(regime=reg, name=n, source="BD", A=p.get("A"), T_name_tauB=p.get("T_name"),
                                 T_actual_tauB=round(T, 3), dn_nm=p.get("dn"), F=p.get("F"),
                                 potential=p.get("potential"), q=p.get("q"), phi_p=p.get("phi_p"),
                                 steps_per_tauB=int(m.group(1)) if m else 4000, seed=p.get("seed"),
                                 n_particles=npart, n_frames=frames, original=src, notes=""))

    # archive of the first Python version
    if a.archive_bd and os.path.isdir(a.archive_bd):
        for reg in REGIMES:
            base = os.path.join(a.archive_bd, reg)
            if not os.path.isdir(base):
                continue
            for n in sorted(os.listdir(base)):
                src = os.path.join(base, n)
                if os.path.isfile(os.path.join(src, "datax.csv")):
                    done &= copy_run(src, os.path.join(out, "_archive", "BD runs v1", reg, n), deadline)

    old = {}
    if os.path.isfile(os.path.join(out, "index.csv")):   # keep rows (and notes) of runs already indexed
        with open(os.path.join(out, "index.csv"), encoding="utf-8") as f:
            old = {(r["regime"], r["name"]): r for r in csv.DictReader(f)}
    new = {(r["regime"], r["name"]) for r in rows}
    rows += [r for k, r in old.items() if k not in new]
    write_outputs(out, rows)
    print(f"{'DONE' if done else 'PARTIAL (time budget) - run again to continue'}: {len(rows)} runs in index.csv")


def reindex(out):
    """Index of everything in the Data folder; keeps notes and original paths of known runs."""
    old = {}
    if os.path.isfile(os.path.join(out, "index.csv")):
        with open(os.path.join(out, "index.csv"), encoding="utf-8") as f:
            old = {(r["regime"], r["name"]): r for r in csv.DictReader(f)}
    rows = [r for k, r in old.items() if k[0].startswith("_archive")]
    for reg in REGIMES:
        base = os.path.join(out, reg)
        if not os.path.isdir(base):
            continue
        for n in sorted(os.listdir(base)):
            rd = os.path.join(base, n)
            if not os.path.isfile(os.path.join(rd, "datax.csv")):
                continue
            p = parse_name(n)
            npart, frames, T = run_info(rd)
            bd = bool(re.search(r"(^|_)BD(_|$)", n))
            m = re.search(r"_dt(\d+)", n)
            prev = old.get((reg, n), {})
            notes = [prev.get("notes", "")] if prev.get("notes") else []
            Tn = p.get("T_name")
            if Tn and abs(T - Tn) > 0.05 * Tn and not any("tau_B, name says" in x for x in notes):
                notes.append(f"runs {T:.2f} tau_B, name says {Tn:g}")
            rows.append(dict(regime=reg, name=n, source="BD" if bd else "COMSOL", A=p.get("A"), T_name_tauB=Tn,
                             T_actual_tauB=round(T, 3), dn_nm=p.get("dn"), F=p.get("F"),
                             potential=p.get("potential"), q=p.get("q"), phi_p=p.get("phi_p"),
                             steps_per_tauB=(int(m.group(1)) if m else 4000) if bd else "",
                             seed=p.get("seed") if bd else "", n_particles=npart, n_frames=frames,
                             original=prev.get("original", rd), notes="; ".join(notes)))
    return rows


def write_outputs(out, rows):
    rows = [{k: ("" if v is None or (isinstance(v, float) and v != v) else v) for k, v in r.items()} for r in rows]
    rows.sort(key=lambda r: (r["regime"], r["source"], r["name"]))
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)


if __name__ == "__main__":
    main()

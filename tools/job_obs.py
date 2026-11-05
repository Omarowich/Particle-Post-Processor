"""Observables for every run of one or more _sim_jobs folders (numpy only), at given times.
Usage: python job_obs.py <out csv> <times tauB, comma-separated> <job dir> [<job dir> ...]
Columns: projected z, psi6, S_max, band (fraction within 5 um of the node, RMS distance), and for 3D runs f_on_top,
f_stacked, z3, mean height."""
import sys, os, re, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hscan_2d_vs_3d import obs   # noqa: E402  (reuses the same definitions)
out, times = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
rows = []
for job in sys.argv[3:]:
    for st in sorted(glob.glob(os.path.join(job, "status*.json"))):
        s = json.load(open(st)); name = s["name"]
        sub = s["folder"].replace("\\", "/").split("/Data/")[-1]
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", sub)
        dim = 3 if "_3D_" in name else 2
        m = re.search(r"_H(\d+(?:\.\d+)?)_", name)
        box = 100.0 if ("idt" in name or "box100" in name) else 50.0
        try:
            for o in obs(folder, dim, times, box=box):
                rows.append(dict(job=os.path.basename(job), name=name, H=float(m.group(1)) if m else 0, **o))
        except Exception as e:
            print("skip", name, e)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out, "w") as fh:
    fh.write(",".join(keys) + "\n")
    for r in rows: fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
print("rows", len(rows))

"""2D vs full 3D (batches 17a-f): observables of the 3D runs at several chamber heights and of the matching 2D runs.
Numpy only. Usage: python hscan_2d_vs_3d.py <Data folder> <out csv>
For every run: projected (x, y) coordination z and psi6 (same definitions as obs_np.frame), 3D stacking
(f_on_top, f_stacked from obs3d.frame), and for Regime II the band: fraction of particles within 5 um of the node
and RMS distance from the node."""
import sys, os, re, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from obs_np import frame as frame2d
from obs3d import frame as frame3d
TAUB = 13.51288926834258

def load(folder, dim):
    cols = []
    for nm in ("datax.csv", "datay.csv") + (("dataz.csv",) if dim == 3 else ()):
        a = np.loadtxt(os.path.join(folder, nm), delimiter=",", skiprows=1)
        cols.append(a[0::2]); t = a[1::2, 0]
    return cols, t / TAUB

def node_offset(y, box=50.0):
    # the 1D standing wave has its node at y = 0 in the model frame; frames are stored either centred or in [0, box]
    return y - box / 2 if y.min() >= -1e-6 else y

def obs(folder, dim, times, box=50.0):
    cols, t = load(folder, dim); rows = []
    for T in times:
        k = min(np.searchsorted(t, T - 1e-6), len(t) - 1)
        P2 = np.c_[cols[0][k], cols[1][k]]; o = frame2d(P2, box=box)
        yy = node_offset(cols[1][k], box)
        o.update(t=round(float(t[k]), 2), band5=round(float((np.abs(yy) < 5).mean()), 3), y_rms=round(float(np.sqrt((yy ** 2).mean())), 2))
        if dim == 3:
            o3 = frame3d(np.c_[cols[0][k], cols[1][k], cols[2][k]], box=box)
            o.update(f_on_top=o3["f_on_top"], f_stacked=o3["f_stacked"], z3=o3["z3"], h_mean=o3["h_mean"])
        rows.append(o)
    return rows

def parse(name):
    A = float(re.match(r"(\d+(?:\.\d+)?)Tkb", name).group(1))
    m = re.search(r"_(\d+)nm_", name); dn = float(m.group(1)) if m else 0.0
    m = re.search(r"_H(\d+(?:\.\d+)?)_", name); H = float(m.group(1)) if m else np.nan
    seed = int(re.search(r"seed(\d+)", name).group(1))
    return A, dn, H, seed

def main(D, OUT):
    rows = []
    def add(regime, folder, dim, times):
        name = os.path.basename(folder.rstrip("/\\"))
        if not os.path.exists(os.path.join(folder, "datax.csv")): return
        A, dn, H, seed = parse(name)
        for o in obs(folder, dim, times):
            rows.append(dict(regime=regime, dim=dim, H=H if dim == 3 else 0, A=A, dn=dn, seed=seed, name=name, **o))
        print(regime, name, rows[-1]["t"], rows[-1]["z"], rows[-1]["psi6"], flush=True)

    pat = r"^(1|2|3|5)Tkb_10TauB_(core_)?BD_AO_q0\.05_phi[0-9.]+_dt256000_3D_H(2\.2|3|4|6|10|20)_drho52_seed\d$"
    for reg, sub, times in (("I", "NAF_3D", (5, 10)), ("III", "AAF_3D", (5, 10))):
        for f in sorted(glob.glob(os.path.join(D, sub, "*"))):
            if re.match(pat, os.path.basename(f)): add(reg, f, 3, times)
    for f in sorted(glob.glob(os.path.join(D, "ARF_3D", "3Tkb_5TauB_*nm_BD_AO_q0.05_phi0.182_dt256000_3D_H*_drho52_seed*"))):
        if re.search(r"_H(2\.2|3|4|6|10|20)_drho52_seed\d$", f): add("II", f, 3, (1, 2, 5))
    # 2D references
    for f in sorted(glob.glob(os.path.join(D, "ARF", "3Tkb_10TauB_*nm_BD_AO_q0.05_phi0.182_dt256000_seed*"))):
        add("II", f, 2, (1, 2, 5))
    json.dump(rows, open(OUT.replace(".csv", ".json"), "w"))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows: fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print("rows", len(rows))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Impact test: secondary radiation force and Rayleigh streaming. Prints one CSV line per run."""
import sys, json, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation"); sys.path.insert(0, "/home/claude/particle-post-processor")
import airpods_bd as bd, bd_engine as be, viewer3d as v3
k = 0.062
E = {"nom": 390.0, "worst": 1663.0}           # kBT/um^3 at d_n = 4000 nm (c0 = 1850 / 2030 m/s, PS)
A = {"nom": 0.54, "worst": 2.1}                # Rayleigh streaming amplitude, um/s
case, seed = sys.argv[1], int(sys.argv[2])
ao = (0.05, 0.182)
kw = dict(ao=ao, steps_per_tauB=256000, out_per_tauB=100, seed=seed)
sec = st = None
tag = case.split("_")
if "sec" in tag: sec = {"E0": E[tag[-1]], "f2": -0.079, "k": k, "rcut": 6.0}
H = None
if case.startswith("2D"):
    fr, t = be.simulate(0.0, 4000.0, 2.0, dim=2, secondary=sec, **kw)
    sys.path.insert(0, "/mnt/user-data/uploads/AIRPODS Paper 2627/Data/_tools"); import obs_np as o
    for T in (1.0, 2.0):
        kk = np.searchsorted(t, T * bd.TAUB - 1e-6); r = o.frame(fr[kk])
        print(f"{case},{seed},{T},{r['z']},{r['Smax']},{r['psi6']},,,", flush=True)
else:
    H = float(tag[0][1:])                      # e.g. H4_...
    if "st+" in tag or "st-" in tag:
        st = {"A": A[tag[-1]], "k": k, "H": H, "sign": 1.0 if "st+" in tag else -1.0}
    fr, t = be.simulate(0.0, 4000.0, 5.0, dim=3, drho=52.0, H=H, secondary=sec, streaming3d=st, **kw)
    for T in (1.0, 5.0):
        kk = min(np.searchsorted(t, T * bd.TAUB - 1e-6), len(t) - 1); s, z = v3.frame_stats(fr[kk])
        print(f"{case},{seed},{T},{s['z3']:.3f},{s['Smax']:.3f},,{s['f_on_top']:.3f},{s['h_mean']:.3f},{s['h_max']:.3f}", flush=True)

"""3D Regime II stacking with the SSAW (IDT at the floor) field. One CSV line per run."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation"); sys.path.insert(0, "/home/claude/particle-post-processor")
import bd_engine as be, arf3d_field as af, viewer3d as v3
H, top, seed = float(sys.argv[1]), sys.argv[2], int(sys.argv[3])
axes, F = af.load_force(f"/home/claude/sim/ssaw_H{H:g}_{top}.npz")
gf = af.GridForce(axes, F)
fr, t = be.simulate(0.0, 4000.0, 5.0, ao=(0.05, 0.182), dim=3, drho=52.0, H=H, arf3d=gf,
                    steps_per_tauB=256000, out_per_tauB=100, seed=seed)
for T in (1.0, 5.0):
    k = min(np.searchsorted(t, T * 13.51288926834258 - 1e-6), len(t) - 1); s, _ = v3.frame_stats(fr[k])
    print(f"{H:g},{top},{seed},{T},{s['z3']:.3f},{s['Smax']:.3f},{s['f_on_top']:.3f},{s['h_mean']:.3f},{s['h_max']:.3f}", flush=True)

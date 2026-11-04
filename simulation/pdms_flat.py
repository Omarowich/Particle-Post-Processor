"""Why a PDMS lid flattens the side-wall field in a thin chamber. Side-wall drive (COMSOL set-up, 20 MHz, c0 = 2030,
50 um wide), field uniform in x -> thin y-z slab. Saves pressure and force for H = 3..40 um, hard vs PDMS (Z = 1 MRayl)."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import acoustic3d as a3, arf3d_field as af
out = {}
for H in (3, 4, 6, 10, 20, 40):
    for lid, kw in (("hard", dict(top="hard")), ("pdms", dict(top="impedance", Z_top=1.0e6))):
        axes, p = a3.solve_box(Lx=2e-6, Ly=50e-6, H=H * 1e-6, h=0.5e-6 if H <= 6 else 1e-6, freq=20e6, c0=2030.0,
                               rho0=1180.0, actuate="y", **kw)
        f1, f2 = af.contrast_factors(1050.0, 2350.0, 1180.0, 2030.0)
        F = af.force_grid(axes, p, None, f1=f1, f2=f2, rho0=1180.0, c0=2030.0, freq=20e6)
        ix = p.shape[0] // 2
        out[f"{lid}_H{H}"] = dict(ys=axes[1] * 1e6, zs=axes[2] * 1e6, p=p[ix], F=F[ix])
np.save("/home/claude/sim/pdms_flat.npy", out, allow_pickle=True)
print("ok", list(out))

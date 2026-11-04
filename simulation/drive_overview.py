"""Pressure and force for every drive x lid combination used in the 3D runs, H = 20 um (thin y-z slab, uniform in x).
Side-wall drive: 50 um wide, c0 = 2030 (COMSOL set-up). IDT: 20 MHz standing SAW on LiNbO3, 100 um wide, c0 = 1850."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import acoustic3d as a3, arf3d_field as af
out = {}
lids = {"glass": dict(top="hard"), "pdms": dict(top="impedance", Z_top=1.0e6)}
for drive, (Ly, c0, kw) in {"side": (50e-6, 2030.0, dict(actuate="y")),
                            "idt": (100e-6, 1850.0, dict(actuate="ssaw", k_saw=2 * np.pi * 20e6 / 3990))}.items():
    for lid, lk in list(lids.items()) + [("absorb", dict(top="impedance", Z_top=1180 * c0))]:
        ax_, p = a3.solve_box(Lx=2e-6, Ly=Ly, H=20e-6, h=0.5e-6, freq=20e6, c0=c0, rho0=1180.0, **kw, **lk)
        f1, f2 = af.contrast_factors(1050.0, 2350.0, 1180.0, c0)
        F = af.force_grid(ax_, p, None, f1=f1, f2=f2, rho0=1180.0, c0=c0, freq=20e6)
        i = p.shape[0] // 2
        out[f"{drive}_{lid}"] = dict(ys=ax_[1] * 1e6, zs=ax_[2] * 1e6, p=p[i], F=F[i])
np.save("/home/claude/sim/drive_overview.npy", out, allow_pickle=True); print("ok")

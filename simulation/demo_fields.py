"""Fields for the schematic figure: side-wall drive (BAW, 20 MHz, 50 um wide, H = 20 um, glass lid) and IDT drive
(standing SAW, 20 MHz, LiNbO3, 299 um wide so the hard side walls sit at antinodes, tall 150 um chamber, absorbing lid
so the leaky beams are visible)."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import acoustic3d as a3, arf3d_field as af
out = {}
ax_, p = a3.solve_box(Lx=2e-6, Ly=50e-6, H=20e-6, h=0.5e-6, freq=20e6, c0=2030.0, actuate="y", top="hard")
out["baw"] = dict(ys=ax_[1] * 1e6, zs=ax_[2] * 1e6, p=p[p.shape[0] // 2])
k = 2 * np.pi * 20e6 / 3990
for lid, kw in (("absorb", dict(top="impedance", Z_top=1180 * 1850)), ("hard", dict(top="hard"))):
    ax_, p = a3.solve_box(Lx=3e-6, Ly=299e-6, H=150e-6, h=1.5e-6, freq=20e6, c0=1850.0, actuate="ssaw", k_saw=k, **kw)
    out[f"saw_{lid}"] = dict(ys=ax_[1] * 1e6, zs=ax_[2] * 1e6, p=p[p.shape[0] // 2])
np.save("/home/claude/sim/demo_fields.npy", out, allow_pickle=True); print("ok")
ax_, p = a3.solve_box(Lx=3e-6, Ly=299e-6, H=150e-6, h=1.5e-6, freq=20e6, c0=1850.0, actuate="ssaw", k_saw=k, saw="travel",
                      top="impedance", Z_top=1180 * 1850)
out["saw_travel"] = dict(ys=ax_[1] * 1e6, zs=ax_[2] * 1e6, p=p[p.shape[0] // 2])
np.save("/home/claude/sim/demo_fields.npy", out, allow_pickle=True); print("ok2")

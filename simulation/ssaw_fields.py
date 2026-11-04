import sys, time, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import acoustic3d as a3, arf3d_field as af
f, c0, rho0, cs = 39.9e6, 1850.0, 1180.0, 3990.0
print(f"Rayleigh angle {np.degrees(np.arcsin(c0 / cs)):.1f} deg, lambda_z {c0 / f / np.cos(np.arcsin(c0 / cs)) * 1e6:.1f} um")
out = {}
for H in (6e-6, 20e-6, 60e-6):
    for top, Z in (("hard", None), ("pdms", 1.0e6), ("absorb", rho0 * c0)):
        t0 = time.time()
        kw = dict(top="hard") if top == "hard" else dict(top="impedance", Z_top=Z)
        gf, axes, p = a3.chamber_force(H=H, h=1e-6, freq=f, c0=c0, rho0=rho0, actuate="ssaw", match_2d=True, **kw)
        xs, ys, zs = [x * 1e6 for x in axes]
        F = gf.F                                   # (nx, ny, nz, 3) kBT/um, scaled to the 2D fit at mid-height
        ix = len(xs) // 2
        Fy, Fz = F[ix, :, :, 1], F[ix, :, :, 2]
        band = np.abs(ys) < 3
        # height of the band: where Fy restoring is strongest; z-force on the band near floor / mid / lid
        zi = [1, len(zs) // 2, len(zs) - 2]
        fz_band = [float(Fz[band][:, k].mean()) for k in zi]
        out[(H, top)] = dict(Fy_mid=float(np.abs(Fy[:, len(zs) // 2]).max()),
                             Fy_floor=float(np.abs(Fy[:, 1]).max()), Fy_top=float(np.abs(Fy[:, -2]).max()),
                             Fz_band=fz_band, Fz_max=float(np.abs(Fz).max()))
        np.savez_compressed(f"/home/claude/sim/ssaw_H{H*1e6:g}_{top}.npz", xs=axes[0], ys=axes[1], zs=axes[2], F=F)
        print(f"H={H*1e6:g} {top:6s} |Fy| floor/mid/top {out[(H,top)]['Fy_floor']:.1f}/{out[(H,top)]['Fy_mid']:.1f}/{out[(H,top)]['Fy_top']:.1f}"
              f"  Fz at band floor/mid/top {fz_band[0]:+.2f}/{fz_band[1]:+.2f}/{fz_band[2]:+.2f}  |Fz|max {out[(H,top)]['Fz_max']:.1f}  ({time.time()-t0:.0f}s)", flush=True)

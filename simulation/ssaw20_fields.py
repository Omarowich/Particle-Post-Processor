"""IDT (standing SAW) chamber fields at 20 MHz: LiNbO3 c_SAW = 3990 m/s -> pressure-node spacing 99.75 um, so the
box is 100 um wide (hard y walls at pressure antinodes). The field is uniform in x: solved on a thin y-z slab and
extruded to x = +-50 um. Gor'kov force for PS, scaled to the 2D fit (F0 = 18.4 kBT/um at d_n = 4000 nm) at mid-height
by sim_runner (arf3d_match). Writes Data/_fields/field_SSAW20_H<H>_<lid>_PS.npz."""
import sys, numpy as np
sys.path.insert(0, "/home/claude/particle-post-processor/simulation")
import acoustic3d as a3, arf3d_field as af
f, c0, rho0, cs, Ly = 20e6, 1850.0, 1180.0, 3990.0, 100e-6
k = 2 * np.pi * f / cs
print(f"k_SAW = {k*1e-6:.4f} /um, node spacing {np.pi/k*1e6:.1f} um, Rayleigh angle {np.degrees(np.arcsin(c0/cs)):.1f} deg, "
      f"vertical wavelength {2*np.pi/np.sqrt((2*np.pi*f/c0)**2-k**2)*1e6:.0f} um")
out = sys.argv[1]
for H in (6e-6, 20e-6):
    for lid, kw in (("hard", dict(top="hard")), ("pdms", dict(top="impedance", Z_top=1.0e6)),
                    ("absorb", dict(top="impedance", Z_top=rho0 * c0))):
        axes, p = a3.solve_box(Lx=2e-6, Ly=Ly, H=H, h=1e-6, freq=f, c0=c0, rho0=rho0, actuate="ssaw", k_saw=k, **kw)
        f1, f2 = af.contrast_factors(1050.0, 2350.0, rho0, c0)
        F = af.force_grid(axes, p, None, f1=f1, f2=f2, rho0=rho0, c0=c0, freq=f)
        Fx = F[1:2]                                      # middle x slice
        xs = np.linspace(-50e-6, 50e-6, 21)
        Fe = np.repeat(Fx, len(xs), axis=0); Fe[..., 0] = 0.0
        ys, zs = axes[1], axes[2]
        iz = len(zs) // 2; s = 18.4 / np.abs(Fe[10, :, iz, 1]).max()
        G = Fe * s
        band = np.abs(ys * 1e6) < 3
        print(f"H={H*1e6:g} {lid:6s}  |Fy| floor/mid/lid {np.abs(G[10,:,1,1]).max():.1f}/{np.abs(G[10,:,iz,1]).max():.1f}/{np.abs(G[10,:,-2,1]).max():.1f}"
              f"   Fz at band floor/mid/lid {G[10,band,1,2].mean():+.2f}/{G[10,band,iz,2].mean():+.2f}/{G[10,band,-2,2].mean():+.2f}")
        af.save_force(f"{out}/field_SSAW20_H{H*1e6:g}_{lid}_PS.npz", (xs, ys, zs), Fe)

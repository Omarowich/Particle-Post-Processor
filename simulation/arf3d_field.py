"""arf3d_field.py - 3D acoustic radiation force from a COMSOL pressure field (Gor'kov), for bd_engine.simulate(arf3d=...).

Input: a CSV exported from a COMSOL 3D Pressure Acoustics (frequency domain) study on a regular grid:
    x, y, z, p_re, p_im, vx_re, vx_im, vy_re, vy_im, vz_re, vz_im      (SI units: m, Pa, m/s; header line allowed)
Velocity columns are optional: without them the velocity is computed from the pressure gradient,
v = grad(p) / (i omega rho0) (needs the frequency).

Gor'kov potential of a small sphere (radius a << wavelength):
    U = V * ( f1 <p^2> / (2 rho0 c0^2)  -  (3/4) f2 rho0 <v^2> ),   F = -grad U
    f1 = 1 - kappa_p / kappa_0,   f2 = 2 (rho_p - rho0) / (2 rho_p + rho0),   <x^2> = |x|^2 / 2
The force grid (kBT/um) is precomputed once and trilinearly interpolated during the run.

Matching to the 2D model: with `match_F0` the field is scaled so that the peak y-force at the chamber mid-plane equals
the 2D fit (F0 = 18.4 kBT/um at d_n = 4000 nm, scaled with (d_n/4000)^2), so 2D and 3D runs use the same amplitude.
"""
from __future__ import annotations

import numpy as np

KB, T_K = 1.380649e-23, 293.0


def contrast_factors(rho_p, c_p, rho0=1180.0, c0=1700.0):
    """f1, f2 for a sphere (density rho_p, sound speed c_p) in a fluid (rho0, c0). Defaults: water-glycerol ~ 60 %."""
    kp, k0 = 1.0 / (rho_p * c_p ** 2), 1.0 / (rho0 * c0 ** 2)
    return 1.0 - kp / k0, 2.0 * (rho_p - rho0) / (2.0 * rho_p + rho0)


def load_grid(csv_path):
    """Regular grid from a COMSOL export -> axes (xs, ys, zs) in m and complex arrays p, (vx, vy, vz) or None."""
    a = np.loadtxt(csv_path, delimiter=",", comments=("%", "#"), skiprows=0)
    xs, ys, zs = (np.unique(np.round(a[:, k], 12)) for k in range(3))
    shape = (len(xs), len(ys), len(zs))
    ix = np.searchsorted(xs, np.round(a[:, 0], 12)); iy = np.searchsorted(ys, np.round(a[:, 1], 12))
    iz = np.searchsorted(zs, np.round(a[:, 2], 12))

    def grid(col_re, col_im):
        G = np.full(shape, np.nan, complex)
        G[ix, iy, iz] = a[:, col_re] + 1j * a[:, col_im]
        return G
    p = grid(3, 4)
    v = tuple(grid(5 + 2 * k, 6 + 2 * k) for k in range(3)) if a.shape[1] >= 11 else None
    return (xs, ys, zs), p, v


def force_grid(axes, p, v=None, *, a_um=1.0, f1, f2, rho0=1180.0, c0=1700.0, freq=None):
    """Radiation force on the grid, kBT/um, shape (nx, ny, nz, 3)."""
    xs, ys, zs = axes
    if v is None:
        if freq is None:
            raise ValueError("velocity not exported: give the frequency to compute it from grad p")
        om = 2 * np.pi * freq
        g = np.gradient(p, xs, ys, zs)
        v = tuple(gk / (1j * om * rho0) for gk in g)
    V = 4.0 / 3.0 * np.pi * (a_um * 1e-6) ** 3
    p2 = 0.5 * np.abs(p) ** 2
    v2 = 0.5 * sum(np.abs(vk) ** 2 for vk in v)
    U = V * (f1 * p2 / (2 * rho0 * c0 ** 2) - 0.75 * f2 * rho0 * v2)          # J
    gx, gy, gz = np.gradient(U, xs, ys, zs)                                     # J/m
    return -np.stack([gx, gy, gz], axis=-1) / (KB * T_K) * 1e-6                 # kBT/um


class GridForce:
    """Callable F(P) for bd_engine: P in um (centred chamber coordinates), returns (N, 3) kBT/um."""

    def __init__(self, axes_m, F, offset_um=(0.0, 0.0, 0.0), scale=1.0):
        self.ax = [np.asarray(x) * 1e6 - o for x, o in zip(axes_m, offset_um)]   # um, shifted to the BD frame
        self.F = np.nan_to_num(F) * scale

    def __call__(self, P):
        idx, w = [], []
        for k in range(3):
            g = self.ax[k]
            f = np.clip((P[:, k] - g[0]) / (g[1] - g[0]), 0, len(g) - 1.000001)
            i = f.astype(int)
            idx.append(i); w.append(f - i)
        (i, j, l), (tx, ty, tz) = idx, w
        out = np.zeros((len(P), 3))
        for di, wx in ((0, 1 - tx), (1, tx)):
            for dj, wy in ((0, 1 - ty), (1, ty)):
                for dl, wz in ((0, 1 - tz), (1, tz)):
                    out += (wx * wy * wz)[:, None] * self.F[i + di, j + dj, l + dl]
        return out


def match_F0(gf: GridForce, dn_nm=4000.0, F0_ref=18.4, dn_ref=4000.0, z_mid_um=None):
    """Scale gf so that the largest |F_y| on the mid-plane equals the 2D fit F0_ref (dn/dn_ref)^2. Returns the scale."""
    zs = gf.ax[2]
    zm = (zs[0] + zs[-1]) / 2 if z_mid_um is None else z_mid_um
    ys = np.linspace(gf.ax[1][0], gf.ax[1][-1], 400)
    P = np.c_[np.zeros_like(ys), ys, np.full_like(ys, zm)]
    peak = np.abs(gf(P)[:, 1]).max()
    s = F0_ref * (dn_nm / dn_ref) ** 2 / peak if peak > 0 else 1.0
    gf.F *= s
    return s

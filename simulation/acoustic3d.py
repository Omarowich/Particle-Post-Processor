"""acoustic3d.py - 3D acoustic pressure field of the chamber, solved in Python (finite differences), for 3D BD runs.

Same set-up as the COMSOL 2D model (Pressure Acoustics, Frequency Domain, 20 MHz): a box Lx x Ly x H, hard walls,
and an inward normal displacement d_n on the two walls normal to y (y = +-Ly/2), here extended in z.
Top (z = H) can be hard (glass) or an impedance (PDMS); bottom (z = 0) hard or actuated.

    (lap + k^2 (1 + i delta)) p = 0,      k = omega / c0
    hard wall:          dp/dn = 0
    normal displacement: dp/dn = rho0 omega^2 d_n   (inward displacement d_n of the wall; n = outward normal)
    impedance wall:     dp/dn = -i omega rho0 p / Z

Second-order finite differences with ghost nodes, sparse direct solve (scipy). Grid spacing ~1 um is enough
(wavelength ~ 100 um). The result feeds arf3d_field.force_grid -> GridForce -> bd_engine.simulate(arf3d=...).

c0 = 2030 m/s makes the y-wavelength match the 2D fit (force period 2 pi / 0.124 um^-1 = 50.7 um at 20 MHz).
"""
from __future__ import annotations

import numpy as np


def solve_box(Lx=50e-6, Ly=50e-6, H=20e-6, h=1e-6, freq=20e6, c0=2030.0, rho0=1180.0, d_n=4000e-9,
              delta=1e-3, top="hard", Z_top=1.0e6 * 1.0, bottom="hard", actuate="y", k_saw=None, saw="standing"):
    """Pressure p (complex, Pa) on a node grid; returns (xs, ys, zs) in m (box centred in x, y; z from 0) and p.
    top / bottom: 'hard' or 'impedance' (Z_top in Pa s/m, PDMS ~ 1.0e6). actuate: 'y' (walls y = +-Ly/2 displaced by d_n
    along y in phase, i.e. one inwards, one outwards: pressure node at y = 0, where the COMSOL band forms), 'bottom' (floor moving
    with amplitude d_n everywhere) or 'ssaw' (standing surface acoustic wave from IDTs on the substrate at z = 0: floor normal
    displacement d_n sin(k_saw y), pressure node at y = 0; k_saw = 2 pi f / c_SAW, default pi / Ly so the hard y walls sit at
    pressure antinodes; the wave enters the liquid at the Rayleigh angle asin(k_saw c0 / omega))."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    nx, ny, nz = (int(round(L / h)) + 1 for L in (Lx, Ly, H))
    xs = np.linspace(-Lx / 2, Lx / 2, nx); ys = np.linspace(-Ly / 2, Ly / 2, ny); zs = np.linspace(0, H, nz)
    hx, hy, hz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]
    om = 2 * np.pi * freq
    k2 = (om / c0) ** 2 * (1 + 1j * delta)
    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)
    rows, cols, vals = [], [], []
    b = np.zeros(N, complex)

    def add(r, c, v):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(np.broadcast_to(v, r.shape).ravel().astype(complex))

    I = idx
    add(I, I, k2 - 2 / hx ** 2 - 2 / hy ** 2 - 2 / hz ** 2)
    # neighbours along each axis; at a wall the ghost node gives p_ghost = p_inner + 2 h g  (g = dp/dn)
    for ax, hh, n in ((0, hx, nx), (1, hy, ny), (2, hz, nz)):
        for side in (-1, 1):
            sl_src = [slice(None)] * 3
            sl_dst = [slice(None)] * 3
            if side == 1:
                sl_src[ax] = slice(0, n - 1); sl_dst[ax] = slice(1, n)
            else:
                sl_src[ax] = slice(1, n); sl_dst[ax] = slice(0, n - 1)
            add(I[tuple(sl_src)], I[tuple(sl_dst)], 1 / hh ** 2)
            # boundary plane: missing neighbour -> mirrored inner neighbour + flux term
            wall = [slice(None)] * 3
            inner = [slice(None)] * 3
            wall[ax] = n - 1 if side == 1 else 0
            inner[ax] = n - 2 if side == 1 else 1
            Iw, Ii = I[tuple(wall)], I[tuple(inner)]
            add(Iw, Ii, 1 / hh ** 2)                       # mirror (hard part)
            # flux g = dp/dn (outward) at this wall
            if ax == 1 and actuate == "y":       # walls move together along y (one in, one out): node at y = 0
                b[Iw.ravel()] -= 2 * hh * (side * rho0 * om ** 2 * d_n) / hh ** 2
            if ax == 2 and side == -1 and actuate == "bottom":
                b[Iw.ravel()] -= 2 * hh * (rho0 * om ** 2 * d_n) / hh ** 2
            if ax == 2 and side == -1 and actuate == "ssaw":
                ks = np.pi / Ly if k_saw is None else k_saw
                prof = (np.sin(ks * ys) if saw == "standing" else np.exp(1j * ks * ys))[None, :] * np.ones((nx, 1))
                b[Iw.ravel()] -= 2 * hh * (rho0 * om ** 2 * d_n * prof.ravel()) / hh ** 2
            if ax == 2 and side == 1 and top == "impedance":
                add(Iw, Iw, 2 * hh * (-1j * om * rho0 / Z_top) / hh ** 2)
            if ax == 2 and side == -1 and bottom == "impedance":
                add(Iw, Iw, 2 * hh * (-1j * om * rho0 / Z_top) / hh ** 2)
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsc()
    p = spsolve(A, b).reshape(nx, ny, nz)
    return (xs, ys, zs), p


def chamber_force(H=20e-6, *, top="hard", h=1e-6, d_n=4000e-9, freq=20e6, c0=2030.0, rho0=1180.0,
                  rho_p=1050.0, c_p=2350.0, match_2d=True, **kw):
    """GridForce (kBT/um) for bd_engine: the solved box field -> Gor'kov force. BD frame: x, y centred, z from 0.
    match_2d: scale to the 2D fit F0 = 18.4 kBT/um (d_n / 4000 nm)^2 at mid-height (absolute scale otherwise)."""
    import arf3d_field as af
    axes, p = solve_box(H=H, h=h, d_n=d_n, freq=freq, c0=c0, rho0=rho0, top=top, **kw)
    f1, f2 = af.contrast_factors(rho_p, c_p, rho0, c0)
    F = af.force_grid(axes, p, None, f1=f1, f2=f2, rho0=rho0, c0=c0, freq=freq)
    gf = af.GridForce(axes, F)
    gf.scale_abs = 1.0
    if match_2d:
        gf.scale_abs = af.match_F0(gf, d_n * 1e9)
    return gf, axes, p

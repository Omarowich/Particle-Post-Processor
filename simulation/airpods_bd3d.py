"""Quasi-3D version of the BD model: particles may leave the plane.

Same pair forces, radiation force (along y, node at y = 0), walls in x and y and time step as airpods_bd,
plus a z coordinate:
  - floor: particle centres stay at z >= a (the particles sit on the chamber bottom at the start),
  - gravity: buoyant weight pulls down (density difference drho, kg/m^3),
  - optional ceiling at height H (centres stay at z <= H - a); H=None = open top,
  - Brownian motion in z as in x and y.
No hydrodynamic wall effects. Used to check whether the squeeze at the band centre makes particles stack.
"""
import os
import time

import numpy as np

import airpods_bd as bd

G = 9.81


def weight_kbt_per_um(drho, a_um=bd.A_P):
    return 4.0 / 3.0 * np.pi * (a_um * 1e-6) ** 3 * drho * G / (bd.KB * bd.T_K / 1e-6)


def simulate3d(A=0.0, dn_nm=4000.0, T_tauB=1.0, *, ao=None, steps_per_tauB=4000, out_per_tauB=100, seed=1,
               core=None, start=None, drho=52.0, H=None, progress=None):
    """Return x, y, z (frames x N, µm) and t (s). start: (N, 2) in-plane start positions (default 12 x 12 grid),
    all on the floor (z = a)."""
    rng = np.random.default_rng(seed)
    if start is None:
        gx, gy = np.meshgrid(bd.GRID, bd.GRID, indexing="ij")
        start = np.c_[gx.ravel(), gy.ravel()]
    P = np.c_[np.asarray(start, float), np.full(len(start), bd.A_P)]
    dt = bd.TAUB / steps_per_tauB
    n_steps = int(round(T_tauB * steps_per_tauB))
    every = steps_per_tauB // out_per_tauB
    noise = np.sqrt(2 * bd.D * dt)
    fg = weight_kbt_per_um(drho)
    zmin, zmax = bd.A_P, (H - bd.A_P) if H else np.inf
    frames, times = [P.copy()], [0.0]
    for s in range(1, n_steps + 1):
        F = bd.pair_forces(P, A, ao, core)
        F[:, 1] += bd.arf_force(P[:, 1], dn_nm)
        F[:, 2] -= fg
        move = bd.D * F * dt
        mag = np.linalg.norm(move, axis=1)
        big = mag > bd.MAX_STEP_UM
        if big.any():
            move[big] *= (bd.MAX_STEP_UM / mag[big])[:, None]
        P += move + noise * rng.standard_normal(P.shape)
        for k, lo, hi in ((0, -bd.BOX, bd.BOX), (1, -bd.BOX, bd.BOX), (2, zmin, zmax)):
            c = P[:, k]
            c = np.where(c < lo, 2 * lo - c, c)
            if np.isfinite(hi):
                c = np.where(c > hi, 2 * hi - c, c)
            P[:, k] = c
        if s % every == 0:
            frames.append(P.copy())
            times.append(s * dt)
            if progress is not None:
                progress(s, n_steps)
    F = np.array(frames)
    return F[:, :, 0], F[:, :, 1], F[:, :, 2], np.array(times)


def save(folder, x, y, z, t):
    """Like the 2D runs (datax / datay, readable by the analyser) plus dataz.csv."""
    bd.write_comsol_csv(folder, x, y, t)
    n = z.shape[1]
    rows = np.empty((2 * len(t), n))
    rows[0::2] = z
    rows[1::2] = t[:, None]
    np.savetxt(os.path.join(folder, "dataz.csv"), rows, delimiter=",",
               header=",".join(str(i) for i in range(1, n + 1)), comments="", fmt="%.10g")

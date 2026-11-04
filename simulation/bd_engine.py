"""bd_engine.py - Brownian dynamics with Verlet neighbour lists, 2D or 3D (paper 2, section 2.7 / 5.2).

Same physics and the same force functions as airpods_bd.py (radial_force, att_force, arf_force, asf_flow),
so a 2D run here is the same model as a 2D run there; only the pair search differs:
  * airpods_bd: all N^2 pairs every step (fine for N = 144)
  * here: a Verlet list (cKDTree, cutoff + skin), rebuilt when any particle has moved more than skin/2
    since the last build -> cost ~ N instead of N^2, which makes N = 1000-5000 and 3D possible.

3D box: walls in x and y at +-box_half (centres), floor at z = a, optional ceiling at z = H - a,
buoyant weight (drho), radiation force of the 1D standing wave along y (as in 2D; a COMSOL 3D Gor'kov field
can be passed as `arf3d(P) -> (N,3)` later). No hydrodynamic interactions or wall hydrodynamics.

Output: datax/datay(/dataz).csv in the COMSOL export layout (the Particle Post Processor reads them).
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy.spatial import cKDTree

import airpods_bd as bd

G = 9.81


def interaction_range(A=0.0, ao=None, att=None, lj_cut_sigma=3.0):
    """Largest centre distance at which any pair force acts (um)."""
    r = 2.0 * bd.A_P                                      # core
    if att is not None:
        m = att["model"]
        if m == "LJ2n":
            r = max(r, bd.SIGMA * 4000.0 ** (1 / float(att["n"])))
        elif m == "SW":
            w = float(att["width"]); r = max(r, 2 * bd.A_P + w + 12 * w * float(att.get("soft", 0.25)))
        elif m == "YUK":
            r = max(r, 2 * bd.A_P + np.log(1e4) / float(att["kappa"]))
    elif ao is not None:
        r = max(r, 2 * (bd.A_P + ao[0] * bd.A_P))
    elif A > 0:
        r = max(r, lj_cut_sigma * bd.SIGMA)
    return r


def weight_kbt_per_um(drho, a_um=None):
    a_um = bd.A_P if a_um is None else a_um
    return 4.0 / 3.0 * np.pi * (a_um * 1e-6) ** 3 * drho * G / (bd.KB * bd.T_K / 1e-6)


class NeighbourList:
    def __init__(self, rcut, skin=0.3):
        self.rcut, self.skin = rcut, skin
        self.P0 = None
        self.i = self.j = None
        self.builds = 0

    def pairs(self, P):
        if self.P0 is None or np.max(np.sum((P - self.P0) ** 2, axis=1)) > (0.5 * self.skin) ** 2:
            b = cKDTree(P).query_pairs(self.rcut + self.skin, output_type="ndarray")
            self.i, self.j = b[:, 0], b[:, 1]
            self.P0 = P.copy()
            self.builds += 1
        return self.i, self.j


def pair_forces_nl(P, nl, A, ao=None, core=None, att=None, radii=None):
    """(N, dim) pair forces in kBT/um from the neighbour list; same force law as airpods_bd.pair_forces."""
    i, j = nl.pairs(P)
    n, dim = P.shape
    F = np.zeros((n, dim))
    if len(i) == 0:
        return F
    d = P[i] - P[j]
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    m = r < nl.rcut
    i, j, d, r = i[m], j[m], d[m], r[m]
    re = r if radii is None else np.maximum(r - (radii[i] + radii[j] - 2 * bd.A_P), 1e-3)
    if att is None:
        fr = bd.radial_force(re, A, ao, core)
    else:
        fr = bd.radial_force(re, 0.0, None, core) + bd.att_force(re, att)
    f = (fr / r)[:, None] * d                             # force on i from j (+ = repulsive, along i - j)
    for k in range(dim):
        F[:, k] = np.bincount(i, f[:, k], n) - np.bincount(j, f[:, k], n)
    return F


def secondary_forces(P, nl2, sec, radii=None):
    """Secondary (particle-particle) acoustic radiation force at short range, dipole (density) term only
    (Silva & Bruus, Phys. Rev. E 2014; Weiser, Apfel & Neppiras 1984). The monopole term vanishes at the pressure node
    and is left out. Standing wave along y with the pressure node at y = 0, so the local kinetic energy density is
    E_kin(y) = E0 cos^2(k y). Pair potential (k r << 1):
        U = -pi E0 cos^2(k y_mid) f2^2 a_i^3 a_j^3 (1 - 3 cos^2 theta) / r^3,   theta = angle between r and y
    -> attractive side by side in the nodal plane (x, z), repulsive along the field axis y.
    sec = {"E0": kBT/um^3, "f2": dipole factor, "k": pressure wave number 1/um, "rcut": um}. Returns (N, dim) kBT/um."""
    i, j = nl2.pairs(P)
    n, dim = P.shape
    F = np.zeros((n, dim))
    if len(i) == 0:
        return F
    d = P[i] - P[j]
    r2 = np.einsum("ij,ij->i", d, d)
    m = r2 < sec["rcut"] ** 2
    i, j, d, r2 = i[m], j[m], d[m], r2[m]
    r = np.sqrt(r2)
    ai3 = (radii[i] ** 3 if radii is not None else bd.A_P ** 3)
    aj3 = (radii[j] ** 3 if radii is not None else bd.A_P ** 3)
    ym = 0.5 * (P[i, 1] + P[j, 1])
    C = np.pi * sec["E0"] * np.cos(sec["k"] * ym) ** 2 * sec["f2"] ** 2 * ai3 * aj3   # kBT um^3
    # U = -C g, g = (1 - 3 dy^2/r^2) / r^3 ; force on i = -dU/dr_i = C grad g
    dy = d[:, 1]
    gr = (-3.0 / r ** 5 + 15.0 * dy ** 2 / r ** 7)[:, None] * d       # radial part of grad g
    gr[:, 1] += -6.0 * dy / r ** 5                                      # from the dy^2 term
    f = C[:, None] * gr
    for k in range(dim):
        F[:, k] = np.bincount(i, f[:, k], n) - np.bincount(j, f[:, k], n)
    return F


def rayleigh_streaming(P, rs):
    """Boundary-driven (Rayleigh) streaming between the floor (z = 0) and a hard lid (z = H), standing wave along y
    (pressure node at y = 0), outer solution for H << wavelength (Rayleigh 1884; Muller et al., Lab Chip 2012):
        u_y =  s A sin(2 k y) (1 - 3 zeta^2),     zeta = 2 z / H - 1
        u_z = -s A k H cos(2 k y) (zeta - zeta^3)   (from div u = 0)
    A = 3 v0^2 / (16 c0) (um/s), s = +-1 picks the roll direction. rs = {"A", "k", "H", "sign"}. Returns (N, 3) um/s."""
    A, k, H, s = rs["A"], rs["k"], rs["H"], rs.get("sign", 1.0)
    zeta = 2.0 * P[:, 2] / H - 1.0
    u = np.zeros_like(P)
    u[:, 1] = s * A * np.sin(2 * k * P[:, 1]) * (1 - 3 * zeta ** 2)
    u[:, 2] = -s * A * k * H * np.cos(2 * k * P[:, 1]) * (zeta - zeta ** 3)
    return u


def acoustic_energy(F0, Phi, k, a=None):
    """Acoustic energy density E0 (kBT/um^3) of the standing wave that gives the primary force amplitude F0 (kBT/um):
    F = 4 pi Phi k a^3 E0 sin(2 k y) (Bruus, Lab Chip 2012)."""
    a = bd.A_P if a is None else a
    return F0 / (4 * np.pi * Phi * k * a ** 3)


def simulate(A=0.0, dn_nm=4000.0, T_tauB=1.0, *, ao=None, att=None, radii=None, steps_per_tauB=4000,
             out_per_tauB=100, seed=1, core=None, start=None, dim=2, drho=0.0, H=None, flow_F=0.0,
             arf3d=None, skin=0.3, progress=None, verbose=False, secondary=None, streaming3d=None):
    """Return positions (frames, N, dim) and t (s). dim = 2: the airpods_bd model with neighbour lists.
    dim = 3: start (N,2) or (N,3); a 2D start is put on the floor (z = a)."""
    rng = np.random.default_rng(seed)
    if start is None:
        gx, gy = np.meshgrid(bd.GRID, bd.GRID, indexing="ij")
        start = np.c_[gx.ravel(), gy.ravel()]
    P = np.asarray(start, float).copy()
    if dim == 3 and P.shape[1] == 2:
        P = np.c_[P, np.full(len(P), bd.A_P)]
    n = len(P)
    dt = bd.TAUB / steps_per_tauB
    n_steps = int(round(T_tauB * steps_per_tauB))
    every = steps_per_tauB // out_per_tauB
    if radii is not None:
        radii = np.asarray(radii, float)
        Dp = (bd.D * bd.A_P / radii)[:, None]
        arf_scale = (radii / bd.A_P) ** 3
    else:
        Dp, arf_scale = bd.D, 1.0
    noise = np.sqrt(2 * Dp * dt)
    nl = NeighbourList(interaction_range(A, ao, att), skin)
    fg = weight_kbt_per_um(drho) * (1.0 if radii is None else (radii / bd.A_P) ** 3) if dim == 3 else 0.0
    nl2 = NeighbourList(float(secondary["rcut"]), skin) if secondary else None
    zmin = bd.A_P
    zmax = (H - bd.A_P) if (dim == 3 and H) else np.inf
    frames, times = [P.copy()], [0.0]
    clipped = 0
    t0 = time.time()
    for s in range(1, n_steps + 1):
        F = pair_forces_nl(P, nl, A, ao, core, att, radii)
        if secondary:
            F += secondary_forces(P, nl2, secondary, radii)
        if arf3d is not None:
            F += arf_scale[:, None] * arf3d(P) if radii is not None else arf3d(P)
        else:
            F[:, 1] += arf_scale * bd.arf_force(P[:, 1], dn_nm)
        if dim == 3:
            F[:, 2] -= fg
        move = Dp * F * dt
        mag = np.linalg.norm(move, axis=1)
        big = mag > bd.MAX_STEP_UM
        if big.any():
            clipped += int(big.sum())
            move[big] *= (bd.MAX_STEP_UM / mag[big])[:, None]
        if streaming3d and dim == 3:
            move += rayleigh_streaming(P, streaming3d) * dt
        if flow_F:
            move[:, :2] += bd.asf_flow(P[:, :2], flow_F) * dt
        P += move + noise * rng.standard_normal(P.shape)
        P[:, :2] = np.where(P[:, :2] > bd.BOX, 2 * bd.BOX - P[:, :2], P[:, :2])
        P[:, :2] = np.where(P[:, :2] < -bd.BOX, -2 * bd.BOX - P[:, :2], P[:, :2])
        if dim == 3:
            P[:, 2] = np.where(P[:, 2] < zmin, 2 * zmin - P[:, 2], P[:, 2])
            if np.isfinite(zmax):
                P[:, 2] = np.where(P[:, 2] > zmax, 2 * zmax - P[:, 2], P[:, 2])
        if s % every == 0:
            frames.append(P.copy())
            times.append(s * dt)
            if progress is not None:
                progress(s, n_steps, clipped)
    if verbose:
        el = time.time() - t0
        print(f"   {n_steps} steps, N = {n}, {dim}D: {el:.1f} s ({1e3 * el / n_steps:.3f} ms/step), "
              f"{nl.builds} list builds, force cap {clipped} times")
    return np.array(frames), np.array(times)


def write_csv(folder, frames, t):
    """COMSOL export layout: datax/datay(/dataz).csv, header 1..N, position row / time row pairs."""
    os.makedirs(folder, exist_ok=True)
    n = frames.shape[1]
    header = ",".join(str(i) for i in range(1, n + 1))
    for k, name in enumerate(("datax.csv", "datay.csv", "dataz.csv")[:frames.shape[2]]):
        rows = np.empty((2 * len(t), n))
        rows[0::2] = frames[:, :, k]
        rows[1::2] = t[:, None]
        np.savetxt(os.path.join(folder, name), rows, delimiter=",", header=header, comments="", fmt="%.10g")


def grid_start(n_side, spacing, dim=2, layers=1, dz=None):
    """Square grid of n_side^2 particles (centred); for dim = 3, `layers` stacked layers dz apart, on the floor."""
    g = (np.arange(n_side) - (n_side - 1) / 2) * spacing
    gx, gy = np.meshgrid(g, g, indexing="ij")
    P = np.c_[gx.ravel(), gy.ravel()]
    if dim == 2:
        return P
    dz = spacing if dz is None else dz
    return np.vstack([np.c_[P, np.full(len(P), bd.A_P + k * dz)] for k in range(layers)])

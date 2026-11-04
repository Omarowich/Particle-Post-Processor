"""streaming2d.py - acoustic streaming computed from the solved acoustic field (paper 2, 3D runs).

The first-order field p comes from acoustic3d.solve_box (side-wall or IDT drive, any lid) on a y-z cross-section
(the fields are uniform in x). The steady streaming is then computed in two steps, the standard "slip velocity"
method of acoustofluidics:

1. Boundary layers. Every wall along which the liquid oscillates has a viscous boundary layer of thickness
   delta = sqrt(2 nu / omega) (~0.2 um here). Its time-averaged stress acts on the bulk as an effective slip velocity
   at the wall (Bach & Bruus, J. Acoust. Soc. Am. 2018, Eqs. 58 and 64; Rayleigh 1884 for the standing-wave limit
   3/8 v^2/c sin 2kx). For a flat wall with tangential coordinate s, inward normal n, bulk acoustic velocity
   v1 = (v_t, v_n) and wall normal velocity V (= v_n at the wall):
       u_slip,t = -(1/2w) Re{ 1/2 v_t* d_s v_t - i V* d_n v_t + [ (2-i)/2 d_s v_t* + i d_n v_n* ] v_t }
       u_slip,n =  (1/2w) Re{ -i v_t d_s V* + i V* d_n v_n }
   (time convention exp(-i w t); the solver's opposite convention is converted by complex conjugation).
2. Bulk flow. The liquid outside the boundary layers obeys the Stokes equations (Re << 1) with those slip velocities
   as boundary conditions; in 2D this is the biharmonic equation for the stream function psi
   (u_y = d psi / d z, u_z = -d psi / d y), solved by finite differences.
Bulk (Eckart) streaming from sound absorption is neglected: its attenuation length at 20 MHz in water-glycerol is
~25 cm, against a 20-100 um chamber.

The streaming scales like the radiation force (both ~ |p|^2), so it is scaled with the same factor that matches the
radiation force to the 2D fit (F0 = 18.4 kBT/um at d_n = 4000 nm): one acoustic amplitude for both.
"""
from __future__ import annotations

import os

import numpy as np


def first_order(drive="side", H_um=20.0, box_half_um=25.0, freq_MHz=20.0, lid="glass", c0=None, rho0=1180.0,
                c_saw=3990.0, h_um=None, Z_pdms=1.0e6):
    """y-z cross-section of the first-order field: ys, zs (m), p (Pa, exp(-i w t) convention), omega, rho0, c0."""
    import acoustic3d as a3
    c0 = (2030.0 if drive == "side" else 1850.0) if c0 is None else c0
    f = freq_MHz * 1e6
    h = (h_um or (0.25 if H_um <= 6 else 0.5)) * 1e-6
    kw = dict(top="hard") if lid == "glass" else dict(top="impedance", Z_top=Z_pdms if lid == "pdms" else rho0 * c0)
    if drive == "idt":
        kw.update(actuate="ssaw", k_saw=2 * np.pi * f / c_saw)
    else:
        kw.update(actuate="y")
    axes, p = a3.solve_box(Lx=2 * h, Ly=2 * box_half_um * 1e-6, H=H_um * 1e-6, h=h, freq=f, c0=c0, rho0=rho0, **kw)
    p = np.conj(p[p.shape[0] // 2])          # solver uses exp(+i w t): conjugate -> exp(-i w t)
    return axes[1], axes[2], p, 2 * np.pi * f, rho0, c0


def acoustic_velocity(ys, zs, p, omega, rho0):
    gy, gz = np.gradient(p, ys, zs, edge_order=2)
    return -1j * gy / (omega * rho0), -1j * gz / (omega * rho0)


def _wall_slip(vt, vn, ds_vt, dn_vt, dn_vn, ds_V, V, omega):
    ut = -(1 / (2 * omega)) * np.real(0.5 * np.conj(vt) * ds_vt - 1j * np.conj(V) * dn_vt
                                      + ((2 - 1j) / 2 * np.conj(ds_vt) + 1j * np.conj(dn_vn)) * vt)
    un = (1 / (2 * omega)) * np.real(-1j * vt * np.conj(ds_V) + 1j * np.conj(V) * dn_vn)
    return ut, un


def slip_velocities(ys, zs, vy, vz, omega):
    """Slip velocities (m/s) on the four walls: dict with floor/lid (u_y, u_z along y) and left/right (u_y, u_z along z)."""
    dy_vy, dz_vy = np.gradient(vy, ys, zs, edge_order=2)
    dy_vz, dz_vz = np.gradient(vz, ys, zs, edge_order=2)
    out = {}
    # floor z = 0: s = +y, n = +z
    vt, vn = vy[:, 0], vz[:, 0]
    ut, un = _wall_slip(vt, vn, dy_vy[:, 0], dz_vy[:, 0], dz_vz[:, 0], np.gradient(vn, ys), vn, omega)
    out["floor"] = (ut, un)
    # lid z = H: s = +y, n = -z
    vt, vn = vy[:, -1], -vz[:, -1]
    ut, un = _wall_slip(vt, vn, dy_vy[:, -1], -dz_vy[:, -1], dz_vz[:, -1], np.gradient(vn, ys), vn, omega)
    out["lid"] = (ut, -un)
    # left y = y0: s = +z, n = +y
    vt, vn = vz[0, :], vy[0, :]
    ut, un = _wall_slip(vt, vn, dz_vz[0, :], dy_vz[0, :], dy_vy[0, :], np.gradient(vn, zs), vn, omega)
    out["left"] = (un, ut)
    # right y = y1: s = +z, n = -y
    vt, vn = vz[-1, :], -vy[-1, :]
    ut, un = _wall_slip(vt, vn, dz_vz[-1, :], -dy_vz[-1, :], dy_vy[-1, :], np.gradient(vn, zs), vn, omega)
    out["right"] = (-un, ut)
    return out


def stokes_streamfunction(ys, zs, slip):
    """Solve the biharmonic equation for psi with the slip velocities as boundary conditions; returns U, W (m/s)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    M, N = len(ys) - 1, len(zs) - 1
    h = ys[1] - ys[0]
    assert abs((zs[1] - zs[0]) - h) < 1e-12 * max(1.0, h), "needs equal grid spacing in y and z"
    # boundary values of psi from the normal slip, integrated around the perimeter (net flux removed)
    uf_y, uf_z = slip["floor"]; ul_y, ul_z = slip["lid"]; uL_y, uL_z = slip["left"]; uR_y, uR_z = slip["right"]
    flux = (np.trapezoid(uf_z, ys) - np.trapezoid(ul_z, ys) + np.trapezoid(uL_y, zs) - np.trapezoid(uR_y, zs))
    perim = 2 * (ys[-1] - ys[0]) + 2 * (zs[-1] - zs[0])
    corr = flux / perim                       # spread the (small) leftover inflow evenly as outflow
    uf_z = uf_z - corr; ul_z = ul_z + corr; uL_y = uL_y - corr; uR_y = uR_y + corr
    cum = lambda u, x: np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(x))])
    psi_floor = -cum(uf_z, ys)                                   # d psi / dy = -u_z along the floor
    psi_right = psi_floor[-1] + cum(uR_y, zs)                    # d psi / dz =  u_y up the right wall
    # lid from right to left: d psi / dy = -u_z  ->  psi(y) = psi(y1) + int_y^{y1} u_z dy
    tail = np.concatenate([np.cumsum((0.5 * (ul_z[1:] + ul_z[:-1]) * np.diff(ys))[::-1])[::-1], [0.0]])
    psi_lid = psi_right[-1] + tail
    # left wall from top to bottom: d psi / dz = u_y -> psi(z) = psi(H) - int_z^H u_y dz
    tailL = np.concatenate([np.cumsum((0.5 * (uL_y[1:] + uL_y[:-1]) * np.diff(zs))[::-1])[::-1], [0.0]])
    psi_left = psi_lid[0] - tailL
    nY, nZ = M + 3, N + 3
    idx = lambda i, j: (i + 1) * nZ + (j + 1)
    rows, cols, vals, b = [], [], [], np.zeros(nY * nZ)

    def eq(r, entries, rhs=0.0):
        for (i, j), v in entries:
            rows.append(r); cols.append(idx(i, j)); vals.append(v)
        b[r] = rhs
    done = np.zeros(nY * nZ, bool)
    for i in range(1, M):
        for j in range(1, N):
            r = idx(i, j)
            eq(r, [((i, j), 20.0), ((i + 1, j), -8.0), ((i - 1, j), -8.0), ((i, j + 1), -8.0), ((i, j - 1), -8.0),
                   ((i + 1, j + 1), 2.0), ((i - 1, j + 1), 2.0), ((i + 1, j - 1), 2.0), ((i - 1, j - 1), 2.0),
                   ((i + 2, j), 1.0), ((i - 2, j), 1.0), ((i, j + 2), 1.0), ((i, j - 2), 1.0)])
            done[r] = True
    for i in range(M + 1):                                      # floor and lid: psi and its z-derivative
        for j, psi_b, u_t, sgn in ((0, psi_floor[i], uf_y[i], -1), (N, psi_lid[i], ul_y[i], +1)):
            r = idx(i, j); eq(r, [((i, j), 1.0)], psi_b); done[r] = True
            g = idx(i, j + sgn)                                 # ghost node
            eq(g, [((i, j + 1), 1.0), ((i, j - 1), -1.0)], 2 * h * u_t); done[g] = True
    for j in range(1, N):                                       # side walls: psi and its y-derivative
        for i, psi_b, u_t, sgn in ((0, psi_left[j], uL_z[j], -1), (M, psi_right[j], uR_z[j], +1)):
            r = idx(i, j); eq(r, [((i, j), 1.0)], psi_b); done[r] = True
            g = idx(i + sgn, j)
            eq(g, [((i + 1, j), 1.0), ((i - 1, j), -1.0)], -2 * h * u_t); done[g] = True
    for j in (0, N):                                            # side-wall ghosts at the corner rows
        for i, u_t, sgn in ((0, uL_z[j], -1), (M, uR_z[j], +1)):
            g = idx(i + sgn, j)
            if not done[g]:
                eq(g, [((i + 1, j), 1.0), ((i - 1, j), -1.0)], -2 * h * u_t); done[g] = True
    for r in np.where(~done)[0]:                                # unused corner ghosts
        rows.append(r); cols.append(r); vals.append(1.0)
    A = coo_matrix((vals, (rows, cols)), shape=(nY * nZ, nY * nZ)).tocsc()
    psi = spsolve(A, b).reshape(nY, nZ)[1:-1, 1:-1]
    gy, gz = np.gradient(psi, ys, zs, edge_order=2)
    return gz, -gy, psi


def compute(drive="side", H_um=20.0, box_half_um=25.0, freq_MHz=20.0, lid="glass", dn_nm=4000.0,
            F0_ref=18.4, rho_p=1050.0, c_p=2350.0, **kw):
    """Streaming (um/s) and radiation force (kBT/um) of one chamber, on the y-z grid, with one acoustic amplitude:
    scaled so that the peak F_y at mid-height equals F0_ref (d_n / 4000 nm)^2."""
    import arf3d_field as af
    ys, zs, p, omega, rho0, c0 = first_order(drive, H_um, box_half_um, freq_MHz, lid, **kw)
    vy, vz = acoustic_velocity(ys, zs, p, omega, rho0)
    U, W, _ = stokes_streamfunction(ys, zs, slip_velocities(ys, zs, vy, vz, omega))
    f1, f2 = af.contrast_factors(rho_p, c_p, rho0, c0)
    V = 4.0 / 3.0 * np.pi * (1e-6) ** 3
    Ug = V * (f1 * 0.5 * np.abs(p) ** 2 / (2 * rho0 * c0 ** 2) - 0.75 * f2 * rho0 * 0.5 * (np.abs(vy) ** 2 + np.abs(vz) ** 2))
    gy, gz = np.gradient(Ug, ys, zs, edge_order=2)
    kBT = 1.380649e-23 * 293.0
    Fy, Fz = -gy / kBT * 1e-6, -gz / kBT * 1e-6
    s = F0_ref * (dn_nm / 4000.0) ** 2 / np.abs(Fy[:, len(zs) // 2]).max()
    return dict(ys=ys, zs=zs, U=U * 1e6 * s, W=W * 1e6 * s, Fy=Fy * s, Fz=Fz * s, scale=s, omega=omega, c0=c0,
                v1_max=np.sqrt(s) * np.abs(vy).max())


def save(path, res, meta=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, ys=res["ys"], zs=res["zs"], U=res["U"], W=res["W"], Fy=res["Fy"], Fz=res["Fz"],
                        meta=str(meta or {}))


class GridFlow:
    """Callable u(P) for bd_engine (P in um, BD frame: y centred, z from the floor), from a saved streaming file.
    Uniform in x; scaled by (d_n / 4000 nm)^2 and an extra factor."""

    def __init__(self, path, dn_nm=4000.0, factor=1.0):
        d = np.load(path)
        self.ys, self.zs = d["ys"] * 1e6, d["zs"] * 1e6
        s = factor * (dn_nm / 4000.0) ** 2
        self.U, self.W = d["U"] * s, d["W"] * s

    def __call__(self, P):
        fy = np.clip((P[:, 1] - self.ys[0]) / (self.ys[1] - self.ys[0]), 0, len(self.ys) - 1.000001)
        fz = np.clip((P[:, 2] - self.zs[0]) / (self.zs[1] - self.zs[0]), 0, len(self.zs) - 1.000001)
        i, j = fy.astype(int), fz.astype(int)
        ty, tz = fy - i, fz - j
        out = np.zeros_like(P)
        for G, k in ((self.U, 1), (self.W, 2)):
            out[:, k] = ((1 - ty) * (1 - tz) * G[i, j] + ty * (1 - tz) * G[i + 1, j]
                         + (1 - ty) * tz * G[i, j + 1] + ty * tz * G[i + 1, j + 1])
        return out

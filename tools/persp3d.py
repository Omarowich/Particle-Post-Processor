"""Shared particle views for the paper figures.
persp: 3D perspective view (box outline, shaded floor), particles drawn at their true size.
topview: view from above, full circles of the true particle size coloured by the local lattice orientation (phase of psi6),
grey for particles with fewer than two neighbours - the same style as the 2D snapshots."""
import numpy as np
from scipy.spatial import cKDTree
XT, OT = "#a3222b", "#9fb3c8"
def _sphere_size(ax, x0, x1, zoom):
    fig = ax.figure; w = ax.get_position().width * fig.get_figwidth() * 72          # axes width in points
    d = 2.0 * w / (x1 - x0) * 0.62 * zoom / 1.3                                       # particle diameter (2 um) in points
    return d ** 2
def persp(ax, P, fl, lim, title, s=None, elev=24, azim=-62, note=None, zoom=1.3, color=None):
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_box_aspect((x1 - x0, y1 - y0, (z1 - z0)), zoom=zoom); ax.set_axis_off()
    xx, yy = np.meshgrid([x0, x1], [y0, y1]); ax.plot_surface(xx, yy, 0 * xx + z0, color="#dfe5ec", alpha=0.25, shade=False, zorder=0)
    for zz, kw in ((z0, dict(color="#8a96a3", lw=0.6)), (z1, dict(color="#8a96a3", lw=0.5, ls=(0, (2, 2))))):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zz] * 5, **kw)
    for xv, yv in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)): ax.plot([xv, xv], [yv, yv], [z0, z1], color="#8a96a3", lw=0.4, ls=(0, (2, 2)))
    o = np.argsort(P[:, 1] - P[:, 0] * 0.3)
    c = (np.where(fl[o], XT, OT) if color is None else np.asarray(color)[o])
    ss = _sphere_size(ax, x0, x1, zoom) if s is None else s
    ax.scatter(P[o, 0], P[o, 1], P[o, 2], s=ss, c=c, edgecolor="#444", linewidth=0.2, depthshade=False, alpha=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1); ax.view_init(elev=elev, azim=azim)
    ax.text2D(0.03, 0.97, title, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")
    if note: ax.text2D(0.03, 0.06, note, transform=ax.transAxes, fontsize=5.8)
def zcolor(P, zmax=6.0):
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis")(np.clip((P[:, 2] - 1) / (zmax - 1), 0, 1))
def orientation_colors(XY, rc=2.9):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap("twilight"); n = len(XY)
    b = cKDTree(XY).query_pairs(rc, output_type="ndarray"); z = np.bincount(b.ravel(), minlength=n)
    d = XY[b[:, 1]] - XY[b[:, 0]]; e = np.exp(6j * np.arctan2(d[:, 1], d[:, 0])); psi = np.zeros(n, complex)
    np.add.at(psi, b[:, 0], e); np.add.at(psi, b[:, 1], e)
    col = [cm((np.angle(p) % (2 * np.pi)) / (2 * np.pi)) if zz >= 2 else (0.75, 0.75, 0.75, 1) for p, zz in zip(psi, z)]
    ok = z >= 2; p6 = float(np.abs(psi[ok] / z[ok]).mean()) if ok.any() else np.nan
    return col, p6
def topview(ax, P, title, xlim=(-25, 25), ylim=(-25, 25), note=None):
    """Seen from above; particles drawn floor first so the top layer covers it. Colour from in-plane neighbours."""
    from matplotlib.collections import EllipseCollection
    o = np.argsort(P[:, 2]); P = P[o]
    col, p6 = orientation_colors(P[:, :2])
    ax.add_collection(EllipseCollection(np.full(len(P), 2.0), np.full(len(P), 2.0), 0, units="xy", offsets=P[:, :2],
                                        transOffset=ax.transData, facecolors=col, edgecolors="none"))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999"); s.set_linewidth(0.5)
    ax.set_title(title, loc="left", fontsize=7, fontweight="bold")
    if note: ax.set_xlabel(note, fontsize=6.0)
    return p6
def top2d(ax, P, xlim, ylim, title, note=None):
    """topview with explicit limits; '{psi6}' in note is replaced by the in-plane psi6 of the shown configuration."""
    p6 = topview(ax, P, title, xlim=xlim, ylim=ylim)
    if note: ax.set_xlabel(note.replace("{psi6}", f"{p6:.2f}"), fontsize=6.0)
    return p6

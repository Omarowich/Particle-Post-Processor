"""Shared 3D perspective view for the paper figures (box outline, shaded floor, crystal-like particles dark red)."""
import numpy as np
XT, OT = "#a3222b", "#9fb3c8"
def persp(ax, P, fl, lim, title, s=9, elev=24, azim=-62, note=None, zoom=1.3, color=None):
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_box_aspect((x1 - x0, y1 - y0, (z1 - z0)), zoom=zoom); ax.set_axis_off()
    xx, yy = np.meshgrid([x0, x1], [y0, y1]); ax.plot_surface(xx, yy, 0 * xx + z0, color="#dfe5ec", alpha=0.5, shade=False, zorder=0)
    for zz, kw in ((z0, dict(color="#8a96a3", lw=0.6)), (z1, dict(color="#8a96a3", lw=0.5, ls=(0, (2, 2))))):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zz] * 5, **kw)
    for xv, yv in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)): ax.plot([xv, xv], [yv, yv], [z0, z1], color="#8a96a3", lw=0.4, ls=(0, (2, 2)))
    o = np.argsort(P[:, 1] - P[:, 0] * 0.3)
    c = (np.where(fl[o], XT, OT) if color is None else np.asarray(color)[o])
    ax.scatter(P[o, 0], P[o, 1], P[o, 2], s=s, c=c, edgecolor="white", linewidth=0.15, depthshade=True, alpha=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1); ax.view_init(elev=elev, azim=azim)
    ax.text2D(0.03, 0.97, title, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")
    if note: ax.text2D(0.03, 0.06, note, transform=ax.transAxes, fontsize=5.8)
def zcolor(P, zmax=6.0):
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis")(np.clip((P[:, 2] - 1) / (zmax - 1), 0, 1))

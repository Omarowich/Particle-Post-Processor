# phase_snapshot_fields.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform

from data_reader_csv import read_particle_data_csv


# ─────────────────────────────────────────────
# 1) Local density snapshot (Voronoi area → ρ)
# ─────────────────────────────────────────────

def snapshot_local_density(
    coordDynX,
    coordDynY,
    timestep,
    skip=1,
    normY=1,
    ax=None,
    box_size_um=50.0,
    particle_radius_um=1.0,  # radius in data units (µm)
):
    """
    Plot local density per particle, estimated as 1 / Voronoi cell area.
    Particles are drawn as disks with radius = particle_radius_um in data units,
    so they "touch" when spaced ~2R apart.
    """
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_frames = X.shape[0]
    if timestep is None:
        timestep = n_frames - 1
    timestep = max(0, min(n_frames - 1, timestep))

    positions = np.vstack((X[timestep], Y[timestep])).T

    if positions.shape[0] < 3:
        print(f"Not enough points for Voronoi at timestep {timestep}.")
        return

    vor = Voronoi(positions)

    densities = np.zeros(len(positions))
    for i in range(len(positions)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if not region or -1 in region:
            densities[i] = 0.0
            continue

        poly = vor.vertices[region]
        x = poly[:, 0]
        y = poly[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        densities[i] = 0.0 if area <= 0 else 1.0 / area

    if densities.max() > 0:
        densities_vis = densities / densities.max()
    else:
        densities_vis = densities

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        plt.sca(ax)

    # Set box limits first so circles are sized in correct data units
    if normY == 1:
        half_box = box_size_um / 2.0
        ax.set_xlim(-half_box, half_box)
        ax.set_ylim(-half_box, half_box)
        ax.set_aspect("equal", adjustable="box")

    cmap = plt.get_cmap("viridis")
    colors = cmap(densities_vis)

    # Draw real disks with radius=particle_radius_um in data coordinates
    for (x, y), c in zip(positions, colors):
        circ = plt.Circle(
            (x, y),
            particle_radius_um,
            facecolor=c,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(circ)

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Local density at t={timestep*13.513:.2f}s")

    if created_fig:
        # Colorbar via a dummy mappable
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        plt.colorbar(mappable, ax=ax, label="Normalized local density")
        plt.tight_layout()
        plt.show()




# ─────────────────────────────────────────────
# 3) Displacement vector field (t vs t-1)
# ─────────────────────────────────────────────

def snapshot_displacement_vectors(
    coordDynX,
    coordDynY,
    timestep,
    skip=1,
    normY=1,
    ax=None,
    box_size_um=50.0,
    particle_radius_um=1.0,
):
    """
    Dense vector field of particle displacements between t-1 and t.

    - One arrow PER PARTICLE.
    - Direction = physical displacement direction.
    - Length is a softened power-law of |Δr|:
          L_i ~ (mag_i / p95)^gamma
      capped so the biggest arrows are only a small fraction of the box.

    Parameters
    ----------
    coordDynX, coordDynY : str
        Paths to CSV files.
    timestep : int or None
        Frame index; if None → last frame.
    skip : int
        Row skipping like in your other functions.
    normY : int (0/1)
        If 1, use [-box_size_um/2, box_size_um/2] for x,y limits.
    box_size_um : float
        Physical box size used for plotting limits and scaling.
    """

    # --- load data (same convention as other functions) ---
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_frames = X.shape[0]
    if n_frames < 2:
        print("Not enough frames for displacement vectors.")
        return

    # need both t and t-1
    if timestep is None:
        timestep = n_frames - 1
    timestep = max(1, min(n_frames - 1, timestep))

    x_t = X[timestep]
    y_t = Y[timestep]
    x_prev = X[timestep - 1]
    y_prev = Y[timestep - 1]

    dx = x_t - x_prev
    dy = y_t - y_prev

    # --- magnitudes + robust stats ---
    mag = np.sqrt(dx**2 + dy**2)
    positive = mag[mag > 0]
    if positive.size == 0:
        print("All displacements are zero at this step.")
        return

    p95 = np.percentile(positive, 95)

    # unit vectors
    ux = np.zeros_like(dx)
    uy = np.zeros_like(dy)
    nonzero = mag > 0
    ux[nonzero] = dx[nonzero] / mag[nonzero]
    uy[nonzero] = dy[nonzero] / mag[nonzero]

    # --- scaling parameters ---
    # soften the emphasis on large motions
    gamma = 1.3          # 1.0 = linear, 2.0 = strong boost; 1.3 is mild

    if normY == 1:
        box_len = box_size_um
        L_max = 0.06 * box_len   # biggest arrows ~6% of box width
    else:
        # fallback: just use a physical scale if normY is off
        L_max = 0.06 * positive.max()

    # normalized displacement (relative to 95th percentile)
    l_norm = mag / p95 if p95 > 0 else mag

    # power-law mapping with cap at 1
    l_scaled = np.clip(l_norm**gamma, 0, 1) * L_max

    # final arrow components
    dx_plot = ux * l_scaled
    dy_plot = uy * l_scaled

    # --- plotting ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        plt.sca(ax)

    if normY == 1:
        half_box = box_size_um / 2.0
        ax.set_xlim(-half_box, half_box)
        ax.set_ylim(-half_box, half_box)
        ax.set_aspect("equal", adjustable="box")

    ax.quiver(
        x_prev,
        y_prev,
        dx_plot,
        dy_plot,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        alpha=0.9,
    )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Displacement field t-Δt → t (t={timestep*13.513:.2f}s)")

    if created_fig:
        plt.tight_layout()
        plt.show()








# ─────────────────────────────────────────────
# 5) Alignment field snapshot (local polar order)
# ─────────────────────────────────────────────

def snapshot_alignment_field(
    coordDynX,
    coordDynY,
    timestep,
    skip=1,
    normY=1,
    ax=None,
    box_size_um=50.0,
    particle_radius_um=1.0,
    neighbor_cutoff=3.5,
):
    """
    Local alignment of displacement directions:
    for each particle i, compute displacement angle θ_i between t-1 and t,
    then ψ_i = |mean_j exp(i θ_j)| over neighbors within neighbor_cutoff.
    ψ_i ∈ [0,1]: 1 = perfectly aligned, 0 = random.
    Particles are drawn as disks with radius=particle_radius_um.
    """
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_frames = X.shape[0]
    if n_frames < 2:
        print("Not enough frames for alignment field.")
        return

    if timestep is None:
        timestep = n_frames - 1
    timestep = max(1, min(n_frames - 1, timestep))  # need t-1

    x_t = X[timestep]
    y_t = Y[timestep]
    x_prev = X[timestep - 1]
    y_prev = Y[timestep - 1]

    dx = x_t - x_prev
    dy = y_t - y_prev

    theta = np.arctan2(dy, dx)
    positions = np.vstack((x_t, y_t)).T
    D = squareform(pdist(positions))

    psi = np.zeros(len(positions))
    for i in range(len(positions)):
        neighbors = np.where(D[i] <= neighbor_cutoff)[0]
        if len(neighbors) <= 1:
            psi[i] = 0.0
            continue
        exp_i_theta = np.exp(1j * theta[neighbors])
        psi[i] = np.abs(exp_i_theta.mean())

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        plt.sca(ax)

    if normY == 1:
        half_box = box_size_um / 2.0
        ax.set_xlim(-half_box, half_box)
        ax.set_ylim(-half_box, half_box)
        ax.set_aspect("equal", adjustable="box")

    cmap = plt.get_cmap("plasma")
    colors = cmap(psi)

    for (x, y), c in zip(positions, colors):
        circ = plt.Circle(
            (x, y),
            particle_radius_um,
            facecolor=c,
            edgecolor="none",
            linewidth=0,
        )
        ax.add_patch(circ)

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Alignment field ψ at t={timestep*13.513:.2f}s")

    if created_fig:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        plt.colorbar(mappable, ax=ax, label="|ψ| (local alignment)")
        plt.tight_layout()
        plt.show()




# ─────────────────────────────────────────────
# 6) Displacement vector field (Voronoi-binned)
# ─────────────────────────────────────────────

def snapshot_displacement_vectors_voronoi(
    coordDynX,
    coordDynY,
    timestep,
    skip=1,
    normY=1,
    ax=None,
    box_size_um=50.0,
    particle_radius_um=1.0,
    draw_cells=False,
):
    """
    Displacement vector field binned into Voronoi cells.

    Steps:
    - Build Voronoi diagram from positions at t-1.
    - For each finite cell, compute its polygon centroid.
    - Use the particle's displacement between t-1 and t to define
      an arrow at that centroid.
    - Arrow length is scaled with a soft power law of |Δr| and capped
      as a small fraction of the box size.
    """

    # --- load data ---
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_frames = X.shape[0]
    if n_frames < 2:
        print("Not enough frames for displacement vectors (Voronoi).")
        return

    # need both t and t-1
    if timestep is None:
        timestep = n_frames - 1
    timestep = max(1, min(n_frames - 1, timestep))

    x_t = X[timestep]
    y_t = Y[timestep]
    x_prev = X[timestep - 1]
    y_prev = Y[timestep - 1]

    dx = x_t - x_prev
    dy = y_t - y_prev

    positions_prev = np.vstack((x_prev, y_prev)).T

    # --- Voronoi tessellation at t-1 ---
    vor = Voronoi(positions_prev)

    centroids_x = []
    centroids_y = []
    disp_x = []
    disp_y = []
    mags = []

    for i, point in enumerate(positions_prev):
        reg_idx = vor.point_region[i]
        region = vor.regions[reg_idx]
        # skip empty or infinite regions
        if not region or -1 in region:
            continue

        verts = vor.vertices[region]
        # centroid of polygon
        cx = verts[:, 0].mean()
        cy = verts[:, 1].mean()

        centroids_x.append(cx)
        centroids_y.append(cy)

        disp_x.append(dx[i])
        disp_y.append(dy[i])
        mags.append(np.hypot(dx[i], dy[i]))

    centroids_x = np.array(centroids_x)
    centroids_y = np.array(centroids_y)
    disp_x = np.array(disp_x)
    disp_y = np.array(disp_y)
    mags = np.array(mags)

    if len(mags) == 0:
        print("No finite Voronoi cells for vector field.")
        return

    # --- robust magnitude stats ---
    positive = mags[mags > 0]
    if positive.size == 0:
        print("All Voronoi-binned displacements are zero.")
        return

    p95 = np.percentile(positive, 95)

    # unit directions
    ux = np.zeros_like(disp_x)
    uy = np.zeros_like(disp_y)
    nz = mags > 0
    ux[nz] = disp_x[nz] / mags[nz]
    uy[nz] = disp_y[nz] / mags[nz]

    # --- scaling: soft power law, capped ---
    gamma = 1.3                       # mild nonlinearity
    if normY == 1:
        box_len = box_size_um
        L_max = 0.06 * box_len        # max arrow ≈ 6% of box
    else:
        L_max = 0.06 * positive.max()

    l_norm = mags / p95 if p95 > 0 else mags
    l_scaled = np.clip(l_norm**gamma, 0, 1) * L_max

    dx_plot = ux * l_scaled
    dy_plot = uy * l_scaled

    # --- plot ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        plt.sca(ax)

    if normY == 1:
        half_box = box_size_um / 2.0
        ax.set_xlim(-half_box, half_box)
        ax.set_ylim(-half_box, half_box)
        ax.set_aspect("equal", adjustable="box")

    if draw_cells:
        # very light Voronoi edges
        for region in vor.regions:
            if not region or -1 in region:
                continue
            poly = vor.vertices[region]
            ax.plot(poly[:, 0], poly[:, 1], color="lightgray", linewidth=0.5)

    ax.quiver(
        centroids_x,
        centroids_y,
        dx_plot,
        dy_plot,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        alpha=0.9,
    )

    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title(f"Voronoi-binned displacement field (t={timestep*13.513:.2f}s)")

    if created_fig:
        plt.tight_layout()
        plt.show()
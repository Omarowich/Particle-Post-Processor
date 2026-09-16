import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from shapely.geometry import Point
from shapely.ops import unary_union

from data_reader_csv import read_particle_data_csv


# ----------------------------------------------------------------------
# Cluster region from particle disks + gap closing
# ----------------------------------------------------------------------
def cluster_region_with_holes(
    cluster_points,
    particle_radius=1.0,
    close_gap=1.5,
    circle_resolution=32
):
    """
    Create a cluster region that:
    - follows the outer cluster shape finely
    - includes small gaps between nearby particles
    - excludes large internal holes from the area

    Method:
    1. Create disks around particle centers.
    2. Union all disks.
    3. Buffer outward to close small gaps.
    4. Buffer inward again to return close to the original boundary.

    close_gap controls the size of gaps that will be closed.

    Approximate meaning:
        gaps smaller than about 2 * close_gap are filled.
        gaps larger than about 2 * close_gap can remain as holes.
    """

    if len(cluster_points) == 0:
        return None

    disks = [
        Point(x, y).buffer(particle_radius, resolution=circle_resolution)
        for x, y in cluster_points
    ]

    particle_union = unary_union(disks).buffer(0)

    if particle_union.is_empty:
        return None

    region_poly = (
        particle_union
        .buffer(close_gap, resolution=circle_resolution)
        .buffer(-close_gap, resolution=circle_resolution)
        .buffer(0)
    )

    if region_poly.is_empty:
        return None

    return region_poly


# ----------------------------------------------------------------------
# Visualization helper
# ----------------------------------------------------------------------
def _visualize_cluster_frame(
    positions,
    labels,
    cluster_shapes,
    title="Clusters"
):
    """
    Visualize particle clusters.

    cluster_shapes is a list of dictionaries:
        {
            "lab": cluster label,
            "points": particle center coordinates,
            "poly": shapely Polygon or MultiPolygon
        }

    Outer outlines are solid.
    Internal holes are dashed.
    """

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Plot noise points
    if -1 in labels:
        noise_pts = positions[labels == -1]

        if len(noise_pts) > 0:
            ax.scatter(
                noise_pts[:, 0],
                noise_pts[:, 1],
                c="lightgray",
                s=10,
                label="noise"
            )

    unique_labels = sorted(set(labels) - {-1})
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_labels))))
    color_map = {lab: col for lab, col in zip(unique_labels, colors)}

    for shape in cluster_shapes:
        lab = shape["lab"]
        cluster_points = shape["points"]
        region_poly = shape["poly"]

        col = color_map.get(lab, "k")

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[col],
            s=15,
            label=f"cluster {lab}"
        )

        if region_poly is None or region_poly.is_empty:
            continue

        # Handle Polygon and MultiPolygon
        if region_poly.geom_type == "Polygon":
            polygons = [region_poly]
        elif region_poly.geom_type == "MultiPolygon":
            polygons = list(region_poly.geoms)
        else:
            continue

        for poly in polygons:
            # Outer boundary
            exterior = np.array(poly.exterior.coords)[:, :2]

            ax.plot(
                exterior[:, 0],
                exterior[:, 1],
                "-",
                linewidth=1.5,
                c=col
            )

            # Internal holes
            for interior in poly.interiors:
                hole = np.array(interior.coords)[:, :2]

                ax.plot(
                    hole[:, 0],
                    hole[:, 1],
                    "--",
                    linewidth=1.2,
                    c=col
                )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main area fraction function
# ----------------------------------------------------------------------
def area_fraction_over_time(
    coordDynX,
    coordDynY,
    particle_radius=1.0,
    TauB=25,
    skip=0,
    normY=0,

    cluster_mode="per_cluster",      # "global" or "per_cluster"

    eps=3.5,
    min_samples=3,
    min_cluster_size=3,

    close_gap=1.5,
    circle_resolution=32,

    visualize_clusters=False,
    visualize_timesteps=("first", "middle", "last"),
    **_extra_kwargs,
):
    """
    Calculate particle area fraction over time.

    cluster_mode = "global":
        Uses one convex hull around all particles.
        This is simple, but does not follow the real cluster boundary.

    cluster_mode = "per_cluster":
        1. DBSCAN detects clusters.
        2. Each cluster is converted to a particle-disk region.
        3. Small gaps between particles are closed.
        4. Large holes remain excluded from area.
        5. Area fraction is calculated as:

            occupancy = 100 * particle_area / cluster_region_area

    Important:
        region_poly.area automatically subtracts internal holes.
    """

    # Load data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)

    occupancy_percentages = []

    # Choose timesteps to visualize
    vis_indices = set()

    if visualize_clusters and cluster_mode == "per_cluster" and num_timesteps > 0:
        if "first" in visualize_timesteps:
            vis_indices.add(20)

        if "middle" in visualize_timesteps:
            vis_indices.add(num_timesteps // 2)

        if "last" in visualize_timesteps:
            vis_indices.add(num_timesteps - 1)

        for v in visualize_timesteps:
            if isinstance(v, int) and 0 <= v < num_timesteps:
                vis_indices.add(v)

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]

        positions = np.column_stack((x_coords, y_coords))

        if len(positions) < 3:
            occupancy_percentages.append(0.0)
            continue

        # --------------------------------------------------------------
        # Global convex hull mode
        # --------------------------------------------------------------
        if cluster_mode == "global":
            hull = ConvexHull(positions)
            area_hull = float(hull.volume)

            area_particles = len(positions) * np.pi * particle_radius**2

            if area_hull > 0:
                occupancy = 100.0 * area_particles / area_hull
            else:
                occupancy = 0.0

        # --------------------------------------------------------------
        # Per-cluster mode with small-gap closing and large-hole exclusion
        # --------------------------------------------------------------
        elif cluster_mode == "per_cluster":
            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples
            ).fit_predict(positions)

            unique_labels = [lab for lab in set(labels) if lab != -1]

            total_particle_area = 0.0
            total_region_area = 0.0
            cluster_shapes = []

            for lab in unique_labels:
                cluster_points = positions[labels == lab]

                if len(cluster_points) < min_cluster_size:
                    continue

                region_poly = cluster_region_with_holes(
                    cluster_points,
                    particle_radius=particle_radius,
                    close_gap=close_gap,
                    circle_resolution=circle_resolution
                )

                if region_poly is None or region_poly.is_empty:
                    continue

                region_area = float(region_poly.area)

                if region_area <= 0:
                    continue

                n_particles = len(cluster_points)
                particle_area = n_particles * np.pi * particle_radius**2

                total_particle_area += particle_area
                total_region_area += region_area

                cluster_shapes.append({
                    "lab": lab,
                    "points": cluster_points,
                    "poly": region_poly
                })

            if total_region_area > 0:
                occupancy = 100.0 * total_particle_area / total_region_area

                # Avoid numerical artifacts only.
                occupancy = max(0.0, min(100.0, occupancy))
            else:
                occupancy = 0.0

            if visualize_clusters and t in vis_indices:
                _visualize_cluster_frame(
                    positions,
                    labels,
                    cluster_shapes,
                    title=f"Clusters at t = {time_axis[t]:.2f} τB"
                )

        else:
            raise ValueError(
                'cluster_mode must be either "global" or "per_cluster"'
            )

        occupancy_percentages.append(float(occupancy))

    # --------------------------------------------------------------
    # Plot result over time
    # --------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, occupancy_percentages)

    plt.xlabel("Time (τB)")
    plt.ylabel("Area Occupied by Particles (%)")

    if cluster_mode == "global":
        mode_str = "Global convex hull"
    else:
        mode_str = "Per-cluster region with large-hole exclusion"

    plt.title(f"Particle Area Occupancy Over Time ({mode_str})")
    plt.grid(True)
    plt.tight_layout()

    if normY == 1:
        plt.ylim(0, 100)

    plt.show()

    return time_axis, occupancy_percentages


# ----------------------------------------------------------------------
# Visualize one timestep only
# ----------------------------------------------------------------------
def visualize_area_fraction_clusters_at_timestep(
    coordDynX,
    coordDynY,
    timestep=None,
    particle_radius=1.0,
    skip=0,
    cluster_eps=3.5,
    cluster_min_samples=3,
    min_cluster_size=3,
    close_gap=1.5,
    circle_resolution=32,
):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_frames = len(coordDynX)

    if n_frames == 0:
        print("No frames.")
        return

    if timestep is None:
        timestep = n_frames - 1

    timestep = int(max(0, min(n_frames - 1, timestep)))

    x_coords = coordDynX[timestep]
    y_coords = coordDynY[timestep]

    positions = np.column_stack((x_coords, y_coords))

    if len(positions) < 3:
        print("Not enough points for clustering.")
        return

    labels = DBSCAN(
        eps=cluster_eps,
        min_samples=cluster_min_samples
    ).fit_predict(positions)

    unique_labels = [lab for lab in set(labels) if lab != -1]

    cluster_shapes = []

    for lab in unique_labels:
        cluster_points = positions[labels == lab]

        if len(cluster_points) < min_cluster_size:
            continue

        region_poly = cluster_region_with_holes(
            cluster_points,
            particle_radius=particle_radius,
            close_gap=close_gap,
            circle_resolution=circle_resolution
        )

        if region_poly is None or region_poly.is_empty:
            continue

        cluster_shapes.append({
            "lab": lab,
            "points": cluster_points,
            "poly": region_poly
        })

    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(7, 7))

    # Set aspect and limits FIRST so radius=1 == 1 data unit
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_labels))))
    color_map = {lab: col for lab, col in zip(unique_labels, colors)}

    # ------------------------------------------------------------------
    # Noise particles — drawn with ax.add_patch so radius is in data units
    # ------------------------------------------------------------------
    if -1 in labels:
        noise_pts = positions[labels == -1]
        for x, y in noise_pts:
            ax.add_patch(Circle(
                (x, y),
                radius=particle_radius,
                facecolor="lightgray",
                edgecolor="gray",
                linewidth=0.5,
                alpha=0.5,
                zorder=1
            ))

    # ------------------------------------------------------------------
    # Clustered particles and boundaries
    # ------------------------------------------------------------------
    for shape in cluster_shapes:
        lab = shape["lab"]
        cluster_points = shape["points"]
        region_poly = shape["poly"]

        col = color_map.get(lab, "k")

        # Each circle added individually — guarantees data-unit radius
        for x, y in cluster_points:
            ax.add_patch(Circle(
                (x, y),
                radius=particle_radius,
                facecolor=col,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.45,
                zorder=2
            ))

        # Tiny center markers
        ax.plot(
            cluster_points[:, 0],
            cluster_points[:, 1],
            ".",
            color="black",
            markersize=1.5,
            zorder=3
        )

        if region_poly is None or region_poly.is_empty:
            continue

        if region_poly.geom_type == "Polygon":
            polygons = [region_poly]
        elif region_poly.geom_type == "MultiPolygon":
            polygons = list(region_poly.geoms)
        else:
            polygons = []

        for poly in polygons:
            # Outer cluster boundary
            exterior = np.array(poly.exterior.coords)[:, :2]
            ax.plot(
                exterior[:, 0],
                exterior[:, 1],
                "-",
                linewidth=1.8,
                c=col,
                zorder=4
            )

            # Internal holes
            for interior in poly.interiors:
                hole = np.array(interior.coords)[:, :2]
                ax.plot(
                    hole[:, 0],
                    hole[:, 1],
                    "--",
                    linewidth=1.4,
                    c=col,
                    zorder=4
                )

    # ------------------------------------------------------------------
    # Simulation box
    # ------------------------------------------------------------------
    ax.add_patch(Rectangle(
        (-25, -25),
        50,
        50,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        zorder=5
    ))

    ax.set_xticks(np.arange(-25, 26, 5))
    ax.set_yticks(np.arange(-25, 26, 5))
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"Area-fraction clusters at frame {timestep} | particle radius = {particle_radius}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw data\NAF\50Tkb_50TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw data\NAF\50Tkb_50TauB\datay.csv"

    TauB = 5
    skip = 1

    # example call
    area_fraction_over_time(
        coordDynX,
        coordDynY,
        TauB=TauB,
        skip=skip,
        particle_radius=1.0,

        cluster_mode="per_cluster",

        eps=2.5,
        min_samples=3,
        min_cluster_size=3,

        close_gap=1.5,
        circle_resolution=32,

        visualize_clusters=True,
        visualize_timesteps=("first", "middle", "last"),
        normY=1,
    )


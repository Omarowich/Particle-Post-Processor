import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN

from shapely.geometry import Polygon
from shapely.ops import unary_union

from data_reader_csv import read_particle_data_csv


# ----------------------------------------------------------------------
# Concave hull (alpha shape) around centers
# ----------------------------------------------------------------------
def alpha_shape_shapely(points, alpha=1.0):
    """
    Compute a concave hull (alpha-shape) of a 2D point cloud.

    alpha:
        smaller  -> tighter hull (follows canyons more)
        larger   -> looser hull (more convex)

    Returns
    -------
    poly : shapely Polygon
    """
    pts = np.asarray(points)

    # Too few points -> convex hull only
    if len(pts) < 4:
        return Polygon(pts).convex_hull

    tri = Delaunay(pts)
    triangles = pts[tri.simplices]

    def edge_length(a, b):
        return np.linalg.norm(a - b)

    kept_polys = []

    for tri_pts in triangles:
        a, b, c = tri_pts
        ab = edge_length(a, b)
        bc = edge_length(b, c)
        ca = edge_length(c, a)

        s = 0.5 * (ab + bc + ca)
        area_sq = s * (s - ab) * (s - bc) * (s - ca)
        if area_sq <= 0:
            continue
        area = np.sqrt(area_sq)

        # circumradius
        R = (ab * bc * ca) / (4.0 * area)

        # small circumradius => keep this triangle
        if R < 1.0 / alpha:
            kept_polys.append(Polygon(tri_pts))

    # nothing survived → convex hull
    if not kept_polys:
        return Polygon(pts).convex_hull

    concave = unary_union(kept_polys)
    if concave.geom_type == "MultiPolygon":
        concave = max(concave.geoms, key=lambda p: p.area)

    return concave


# ----------------------------------------------------------------------
# Visualization helper
# ----------------------------------------------------------------------
def _visualize_cluster_frame(positions, labels, cluster_shapes, title="Clusters"):
    """
    cluster_shapes: list of (lab, cluster_points, outline) where outline
                    is an (M,2) np.array of boundary coords (dilated hull).
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # noise
    if -1 in labels:
        noise_pts = positions[labels == -1]
        if len(noise_pts) > 0:
            ax.scatter(noise_pts[:, 0], noise_pts[:, 1],
                       c="lightgray", s=10, label="noise")

    unique_labels = sorted(set(labels) - {-1})
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique_labels))))
    color_map = {lab: col for lab, col in zip(unique_labels, colors)}

    for lab, cluster_points, outline in cluster_shapes:
        col = color_map.get(lab, "k")
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   c=[col], s=15, label=f"cluster {lab}")

        if outline is not None and len(outline) >= 3:
            closed = np.vstack([outline, outline[0]])
            ax.plot(closed[:, 0], closed[:, 1], "-", linewidth=1.5, c=col)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def area_fraction_over_time(
    coordDynX,
    coordDynY,
    particle_radius=1.0,
    TauB=25,
    skip=0,
    normY=0,
    cluster_mode="per_cluster",          # "global" or "per_cluster"
    cluster_eps=3.5,
    cluster_min_samples=3,
    tight_alpha=1.0,                # concavity of center-hull
    visualize_clusters=False,
    visualize_timesteps=("first", "middle", "last"),
):
    """
    cluster_mode = "global":
        - Convex hull over all particles (centers).

    cluster_mode = "per_cluster":
        - DBSCAN clusters
        - Concave hull (alpha-shape) around centers in each cluster
        - That hull is then dilated by `particle_radius` to ensure
          all disks lie inside the region.
        - area_region = sum area(dilated hulls)
        - area_particles = sum N_cluster * π r²
        - occupancy = 100 * area_particles / area_region
          (now naturally ≲ 100 %)
    """

    # load data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    occupancy_percentages = []

    # what to visualize
    vis_indices = set()
    if visualize_clusters and cluster_mode == "per_cluster" and num_timesteps > 0:
        if "first" in visualize_timesteps:
            vis_indices.add(0)
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

        if cluster_mode == "global":
            # single global hull
            hull = ConvexHull(positions)
            area_hull = float(hull.volume)
            area_particles = len(positions) * np.pi * particle_radius**2
            occupancy = 100.0 * area_particles / area_hull if area_hull > 0 else 0.0

        elif cluster_mode == "per_cluster":
            labels = DBSCAN(eps=cluster_eps,
                            min_samples=cluster_min_samples).fit_predict(positions)
            unique_labels = [lab for lab in set(labels) if lab != -1]

            total_particle_area = 0.0
            total_region_area = 0.0
            cluster_shapes = []

            for lab in unique_labels:
                cluster_points = positions[labels == lab]
                if len(cluster_points) < 3:
                    continue

                # 1) concave hull of centers
                center_poly = alpha_shape_shapely(cluster_points, alpha=tight_alpha)
                center_poly = center_poly.buffer(0)  # clean topology

                # 2) dilate by particle radius so all disks fit inside
                region_poly = center_poly.buffer(particle_radius)

                region_area = float(region_poly.area)
                if region_area <= 0:
                    continue

                n_particles = len(cluster_points)
                particle_area = n_particles * np.pi * particle_radius**2

                total_particle_area += particle_area
                total_region_area += region_area

                outline = np.array(region_poly.exterior.coords)[:, :2]
                cluster_shapes.append((lab, cluster_points, outline))

            if total_region_area > 0:
                occupancy = 100.0 * total_particle_area / total_region_area
                # tiny numerical overshoots
                if occupancy < 0:
                    occupancy = 0.0
                if occupancy > 100:
                    occupancy = 100.0
            else:
                occupancy = 0.0

            if visualize_clusters and t in vis_indices:
                _visualize_cluster_frame(
                    positions, labels, cluster_shapes,
                    title=f"Clusters at t = {time_axis[t]:.2f} (τB)"
                )
        else:
            raise ValueError('cluster_mode must be either "global" or "per_cluster"')

        occupancy_percentages.append(float(occupancy))

    # time-series plot
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, occupancy_percentages)
    plt.xlabel("Time (τB)")
    plt.ylabel("Area Occupied by Particles (%)")
    mode_str = "Global hull" if cluster_mode == "global" else "Per-cluster (concave+dilated hull)"
    plt.title(f"Particle Area Occupancy Over Time ({mode_str})")
    plt.grid(True)
    plt.tight_layout()
    if normY == 1:
        plt.ylim(0, 100)
    plt.show()

    return time_axis, occupancy_percentages


def visualize_area_fraction_clusters_at_timestep(
    coordDynX,
    coordDynY,
    timestep=None,                 # None = last
    particle_radius=1.0,
    skip=0,
    normY=0,
    cluster_eps=3.5,
    cluster_min_samples=3,
    tight_alpha=1.0,
):
    """
    Visualize per-cluster concave+dilated hulls for ONE frame.
    Uses the same logic as area_fraction_over_time(cluster_mode="per_cluster").
    """

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
        print("Not enough points for clustering/hulls.")
        return

    labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit_predict(positions)
    unique_labels = [lab for lab in set(labels) if lab != -1]

    cluster_shapes = []
    for lab in unique_labels:
        cluster_points = positions[labels == lab]
        if len(cluster_points) < 3:
            continue

        center_poly = alpha_shape_shapely(cluster_points, alpha=tight_alpha)
        center_poly = center_poly.buffer(0)

        region_poly = center_poly.buffer(particle_radius)
        if region_poly.is_empty:
            continue

        outline = np.array(region_poly.exterior.coords)[:, :2]
        cluster_shapes.append((lab, cluster_points, outline))

    _visualize_cluster_frame(
        positions, labels, cluster_shapes,
        title=f"Area-fraction clusters at frame {timestep}"
    )



if __name__ == "__main__":
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\3Tkb_50TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\3Tkb_50TauB\datay.csv"

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
        cluster_eps=2.5,
        cluster_min_samples=3,
        tight_alpha=0.3,
        visualize_clusters=True,
        visualize_timesteps=("middle", "last"),
        normY=1,
    )


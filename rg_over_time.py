import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from data_reader_csv import read_particle_data_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components


def get_clusters_dbscan(positions, eps=3.5, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # skip noise
        cluster_indices = np.where(labels == label)[0]
        clusters.append(cluster_indices)
    return clusters

def average_rg_over_time(coordDynX, coordDynY, eps=10,min_samples=3,min_cluster_size=3, cluster_mode="per cluster", TauB=2.5, skip=1, normY=0, **_extra_kwargs):
    # Read particle data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]
    num_timesteps = len(coordDynX)

    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    rg_list = []

    for t in range(num_timesteps):
        positions = np.column_stack((coordDynX[t], coordDynY[t]))
        com = np.mean(positions, axis=0)  # Center of mass
        rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))  # Radius of gyration
        rg_list.append(rg)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(time_axis, rg_list, '-', color='darkgreen')
    plt.xlabel('Time')
    plt.ylabel('Rg (radius of gyration)')
    plt.title('Radius of Gyration Rg of All Particles over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        labels=[
            f'RG ({round(rg_list[-1], 4)})',
        ],
        loc='lower right', fontsize=10
    )
    if normY == 1:
        plt.ylim(14, 21)
    plt.show()

    return time_axis, rg_list


# def average_rg_over_time(coordDynX, coordDynY, eps=10, TauB=2.5, skip=1, normY=0):
#     coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
#     coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]
#     num_timesteps = len(coordDynX)
#
#     time_axis = np.arange(num_timesteps) * TauB
#     avg_rg_list = []
#
#     for t in range(num_timesteps):
#         positions = np.column_stack((coordDynX[t], coordDynY[t]))
#         clusters = get_clusters_dbscan(positions, eps)
#         Rg_list = []
#         # print(f"t = {t}, num clusters = {len(clusters)}")
#
#         for cluster in clusters:
#             if len(cluster) > 1:
#                 cluster_pos = positions[cluster]
#                 com = np.mean(cluster_pos, axis=0)
#                 Rg = np.sqrt(np.mean(np.sum((cluster_pos - com) ** 2, axis=1)))
#                 Rg_list.append(Rg)
#         avg_rg = np.mean(Rg_list) if Rg_list else 0
#         avg_rg_list.append(avg_rg)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(time_axis, avg_rg_list, '-', color='darkgreen')
#     plt.xlabel('Time')
#     plt.ylabel('⟨Rg⟩ (mean radius of gyration)')
#     plt.title('Average Radius of Gyration ⟨Rg⟩ over Time')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.legend(
#         labels=[
#             f'RG ({round(avg_rg_list[-1], 4)})',
#         ],
#         loc='lower right', fontsize=10
#     )
#     if normY == 1: plt.ylim(14, 18)
#     plt.show()
#
#     return time_axis, avg_rg_list


# def rg_one_cluster_over_time(coordDynX, coordDynY, eps=3.5, TauB=2.5, skip=1):
#     coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
#     coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]
#     num_timesteps = len(coordDynX)
#
#     time_axis = np.arange(num_timesteps) * TauB
#     rg_list = []
#
#     for t in range(num_timesteps):
#         positions = np.column_stack((coordDynX[t], coordDynY[t]))
#         com = np.mean(positions, axis=0)
#         rg = np.sqrt(np.mean(np.sum((positions - com) ** 2, axis=1)))
#         rg_list.append(rg)
#
#     plt.figure(figsize=(8, 5))
#     plt.plot(time_axis, rg_list, '-', color='darkred')
#     plt.xlabel('Time')
#     plt.ylabel('Rg (entire structure)')
#     plt.title('Global Radius of Gyration Over Time')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.legend(
#         labels=[
#             f'Rg ({round(rg_list[-1], 4)})',
#         ],
#         loc='lower right', fontsize=10
#     )
#     plt.show()



def get_clusters_dbscan(positions, eps=3.5, min_samples=3):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # skip noise
        cluster_indices = np.where(labels == label)[0]
        clusters.append(cluster_indices)
    return clusters, labels

def _rg(points):
    """Radius of gyration of points (N x D)."""
    if points.shape[0] == 0:
        return np.nan
    com = points.mean(axis=0)
    return np.sqrt(np.mean(np.sum((points - com) ** 2, axis=1)))

def average_rg_over_time_clustered(
    coordDynX_path,
    coordDynY_path,
    eps=10,
    min_samples=3,
    TauB=2.5,
    skip=1,
    normY=0,
    aggregate="weighted",      # 'weighted' | 'mean' | 'median'
    plot_per_cluster=False,    # True to also plot per-cluster Rg lines
):
    """
    Reads particle trajectories, clusters particles at each timestep with DBSCAN,
    computes per-cluster Rg, and aggregates to an overall Rg per timestep.

    aggregate:
      - 'weighted': size-weighted mean of cluster Rg's (default, recommended)
      - 'mean': unweighted mean of cluster Rg's
      - 'median': median of cluster Rg's
    """
    # You already have this reader; keeping your slicing choices:
    coordDynX = read_particle_data_csv(coordDynX_path)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY_path)[skip::2, 1:][1:, :]
    num_timesteps = len(coordDynX)

    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)

    overall_rg = []
    all_cluster_rgs = []  # list of lists: per timestep list of cluster Rg's
    n_clusters_over_time = []

    for t in range(num_timesteps):
        positions = np.column_stack((coordDynX[t], coordDynY[t]))

        clusters, labels = get_clusters_dbscan(positions, eps=eps, min_samples=min_samples)

        # Compute per-cluster Rg
        rgs = []
        sizes = []
        for idxs in clusters:
            pts = positions[idxs]
            rgs.append(_rg(pts))
            sizes.append(len(idxs))

        n_clusters_over_time.append(len(rgs))
        all_cluster_rgs.append(rgs)

        # Aggregate to "overall" Rg that doesn't include inter-cluster spacing
        if len(rgs) == 0:
            overall_rg.append(np.nan)
        else:
            rgs_arr = np.array(rgs, dtype=float)
            if aggregate == "weighted":
                w = np.array(sizes, dtype=float)
                overall_rg.append(np.sum(w * rgs_arr) / np.sum(w))
            elif aggregate == "mean":
                overall_rg.append(np.mean(rgs_arr))
            elif aggregate == "median":
                overall_rg.append(np.median(rgs_arr))
            else:
                raise ValueError("aggregate must be 'weighted', 'mean', or 'median'")

    # Plot
    plt.figure(figsize=(9, 5.5))
    if plot_per_cluster:
        # Draw faint lines per cluster (variable number per timestep, so we plot as scatter)
        for t, rgs in enumerate(all_cluster_rgs):
            if len(rgs) > 0:
                plt.scatter([time_axis[t]] * len(rgs), rgs, s=10, alpha=0.25)

    plt.plot(time_axis, overall_rg, '-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Rg (radius of gyration)')
    aggr_txt = {"weighted": "size-weighted mean", "mean": "mean", "median": "median"}[aggregate]
    plt.title(f'Clustered Radius of Gyration over Time ({aggr_txt})')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        labels=[f'Overall Rg ({aggr_txt}); final={np.round(overall_rg[-1], 4)}'],
        loc='lower right',
        fontsize=10
    )
    if normY == 1:
        plt.ylim(14, 21)
    plt.show()

    return time_axis, overall_rg, all_cluster_rgs, n_clusters_over_time





def _rg(points):
    """Radius of gyration of points (N x D)."""
    if points.shape[0] == 0:
        return np.nan
    com = points.mean(axis=0)
    return np.sqrt(np.mean(np.sum((points - com) ** 2, axis=1)))

def _clusters_by_cutoff(positions, cutoff, min_size=3):
    """
    Build an undirected graph where edges connect particles closer than `cutoff`,
    then return connected components (clusters) of size >= min_size.
    """
    # Sparse adjacency (no self-loops); make it symmetric and unweighted
    G = radius_neighbors_graph(positions, radius=cutoff, mode='connectivity', include_self=False)
    # Connected components -> labels
    n_comp, labels = connected_components(G, directed=False)
    clusters = []
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if idx.size >= min_size:
            clusters.append(idx)
    return clusters, labels

def average_rg_over_time_clustered_cutoff(
    coordDynX_path,
    coordDynY_path,
    R=1,                         # particle radius (same units as coords)
    eps=0.05,                # cutoff factor: pair linked if distance < alpha * R
    min_size=3,                # min particles in a cluster to keep
    TauB=2.5,
    skip=1,
    normY=0,
    aggregate="weighted",      # 'weighted' | 'mean' | 'median'
    plot_per_cluster=False,    # scatter all cluster Rg’s on top of the overall line
):
    """
    Cluster by physical cutoff: particles i,j connected if ||ri - rj|| < alpha * R.
    Clusters = connected components of that graph. Then compute per-cluster Rg and
    an overall Rg per timestep (default: size-weighted mean of cluster Rg’s).
    """
    # Load your trajectories (matches your prior slicing)
    coordDynX = read_particle_data_csv(coordDynX_path)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY_path)[skip::2, 1:][1:, :]
    num_timesteps = len(coordDynX)

    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    overall_rg = []
    all_cluster_rgs = []
    n_clusters_over_time = []

    cutoff = eps * R

    for t in range(num_timesteps):
        positions = np.column_stack((coordDynX[t], coordDynY[t]))

        clusters, labels = _clusters_by_cutoff(positions, cutoff=cutoff, min_size=min_size)

        # Per-cluster Rg’s
        rgs = []
        sizes = []
        for idxs in clusters:
            pts = positions[idxs]
            rgs.append(_rg(pts))
            sizes.append(len(idxs))

        n_clusters_over_time.append(len(rgs))
        all_cluster_rgs.append(rgs)

        # Aggregate
        if len(rgs) == 0:
            overall_rg.append(np.nan)
        else:
            rgs_arr = np.array(rgs, dtype=float)
            if aggregate == "weighted":
                w = np.array(sizes, dtype=float)
                overall_rg.append(np.sum(w * rgs_arr) / np.sum(w))
            elif aggregate == "mean":
                overall_rg.append(np.mean(rgs_arr))
            elif aggregate == "median":
                overall_rg.append(np.median(rgs_arr))
            else:
                raise ValueError("aggregate must be 'weighted', 'mean', or 'median'")

    # Plot
    plt.figure(figsize=(9, 5.5))
    if plot_per_cluster:
        for t, rgs in enumerate(all_cluster_rgs):
            if len(rgs) > 0:
                plt.scatter([time_axis[t]] * len(rgs), rgs, s=10, alpha=0.25)

    plt.plot(time_axis, overall_rg, '-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Rg (radius of gyration)')
    aggr_txt = {"weighted": "size-weighted mean", "mean": "mean", "median": "median"}[aggregate]
    final_val = np.round(overall_rg[-1], 4) if len(overall_rg) else np.nan
    plt.title(f'Clustered Radius of Gyration over Time (cutoff: {eps}·R, {aggr_txt})')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        labels=[f'Overall Rg ({aggr_txt}); final={final_val}'],
        loc='lower right',
        fontsize=10
    )
    if normY == 1:
        plt.ylim(14, 21)
    plt.show()

    return time_axis, overall_rg, all_cluster_rgs, n_clusters_over_time

if __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ARF\5Tkb_2TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ARF\5Tkb_2TauB\datax.csv"

    TauB = 5
    skip = 1

    eps = 1

    #cluster_mass_distribution_over_time(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip)
    average_rg_over_time_clustered_cutoff(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip)
    #rg_one_cluster_over_time(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip)
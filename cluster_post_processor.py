import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from data_reader_csv import read_particle_data_csv






def plot_clusters(x_coords, y_coords, cluster_labels):
    plt.figure(figsize=(8, 6))
    unique_clusters = set(cluster_labels) - {-1}  # Exclude noise points

    # Plot noise points (if any)
    noise_mask = cluster_labels == -1
    if np.any(noise_mask):
        plt.scatter(x_coords[noise_mask], y_coords[noise_mask], c='gray', label='Noise', alpha=0.5)

    # Plot clusters and their convex hulls
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_points = np.column_stack((x_coords[cluster_mask], y_coords[cluster_mask]))

        # Plot particles in the cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')

        # Compute and plot convex hull
        if len(cluster_points) >= 3:  # Convex hull requires at least 3 points
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k--', lw=1)
            plt.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], alpha=0.1)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Clusters with Convex Hulls')
    plt.legend()
    plt.show()





def analyze_clusters_over_time(
    coordDynX,
    coordDynY,
    TauB=25,
    skip=1,
    eps=3.5,
    min_samples=3,
    min_cluster_size=3,
    normY=1,
    plot=False,
):
    """
    Detect DBSCAN clusters per frame (same clustering logic as in area_fraction_over_time),
    then compute:
      - number of clusters over time
      - average cluster size over time (mean particles per cluster, excluding noise)

    Notes:
      - Uses DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples)
      - Excludes noise label (-1)
      - Filters clusters by min_cluster_size (default 3)

    Returns
    -------
    time_axis : (T,) array
    n_clusters : (T,) int array
    avg_cluster_size : (T,) float array (0 if no clusters)
    cluster_sizes_per_t : list of lists (each inner list = sizes of clusters at t)
    """

    # load data (same slicing pattern you started)
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(X)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)

    n_clusters = np.zeros(num_timesteps, dtype=int)
    avg_cluster_size = np.zeros(num_timesteps, dtype=float)
    cluster_sizes_per_t = []

    for t in range(num_timesteps):
        x_coords = X[t]
        y_coords = Y[t]
        positions = np.column_stack((x_coords, y_coords))

        # Not enough points to form clusters reliably
        if positions.shape[0] < max(1, min_samples):
            cluster_sizes_per_t.append([])
            continue

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(positions)

        # collect cluster sizes excluding noise
        sizes = []
        for lab in (set(labels) - {-1}):
            sz = int(np.sum(labels == lab))
            if sz >= int(min_cluster_size):
                sizes.append(sz)

        cluster_sizes_per_t.append(sizes)
        n_clusters[t] = len(sizes)
        avg_cluster_size[t] = float(np.mean(sizes)) if sizes else 0.0

    if plot:
        # plot: number of clusters
        plt.figure(figsize=(8, 4))
        plt.plot(time_axis, n_clusters)
        plt.xlabel("Time (τB)")
        plt.ylabel("Number of clusters")
        plt.title("Number of clusters over time (DBSCAN)")
        plt.grid(True)
        plt.tight_layout()
        if normY == 1:
            plt.ylim(bottom=0)
        plt.show()

        # plot: average cluster size
        plt.figure(figsize=(8, 4))
        plt.plot(time_axis, avg_cluster_size)
        plt.xlabel("Time (τB)")
        plt.ylabel("Avg cluster size (particles)")
        plt.title("Average cluster size over time (DBSCAN)")
        plt.grid(True)
        plt.tight_layout()
        if normY == 1:
            plt.ylim(bottom=0)
        plt.show()

    return time_axis, n_clusters, avg_cluster_size, cluster_sizes_per_t


def num_clusters_over_time(coordDynX, coordDynY, eps=3.5, min_samples=3, min_cluster_size=3, TauB=25, skip=1, normY=0, **_extra_kwargs):
    time_axis, n_clusters, avg_cluster_sizes, cluster_sizes = analyze_clusters_over_time(
        coordDynX, coordDynY,
        eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size,
        TauB=TauB, skip=skip,
        plot=False
    )
    return time_axis, np.asarray(n_clusters, float)


def avg_cluster_size_over_time(coordDynX, coordDynY, eps=3.5, min_samples=3, min_cluster_size=3, TauB=25, skip=1, normY=0, **_extra_kwargs):
    time_axis, n_clusters, avg_cluster_sizes, cluster_sizes = analyze_clusters_over_time(
        coordDynX, coordDynY,
        eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size,
        TauB=TauB, skip=skip,
        plot=False
    )
    return time_axis, np.asarray(avg_cluster_sizes, float)



def plot_cluster_metrics(cluster_sizes, num_clusters,avg_cluster_sizes,time_axis):
    """
    Plots the number of clusters and their sizes over time.
    """

    # Plot number of clusters
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, num_clusters, marker='', linestyle='-', label="Number of Clusters")
    plt.xlabel("Time(s)")
    plt.ylabel("Number of Clusters")
    plt.title("Cluster Count Over Time")
    plt.legend()
    plt.show()

    # Plot average cluster size over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, avg_cluster_sizes, marker='', linestyle='-', label="Avg Cluster Size")
    plt.xlabel("Time(s)")
    plt.ylabel("Average Cluster Size")
    plt.title("Cluster Size Over Time")
    plt.legend()
    plt.show()






if  __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\10Tkb_1app. py0TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\10Tkb_1app. py0TauB\datay.csv"

    # Returns: time_axis, n_clusters, avg_cluster_size, cluster_sizes_per_t
    time_axis, num_clusters, avg_cluster_sizes, cluster_sizes = analyze_clusters_over_time(coordDynX, coordDynY, eps=3.5, min_samples=3)
    plot_cluster_metrics(cluster_sizes, num_clusters, avg_cluster_sizes, time_axis)

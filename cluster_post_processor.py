import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

from data_reader_csv import read_particle_data_csv


def detect_clusters_DBSCAN(x_coords, y_coords, eps=1.0, min_samples=3):
    x_coords = np.array(x_coords, dtype=float)
    y_coords = np.array(y_coords, dtype=float)
    positions = np.column_stack((x_coords, y_coords))  # Combine x and y into one array
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    #print(clustering.labels_)
    return clustering.labels_  # Returns a list of cluster IDs (-1 means noise)

# def detect_clusters(x_coords, y_coords, num_clusters=3):
#     positions = np.column_stack((x_coords, y_coords))
#     kmeans = KMeans(n_clusters=num_clusters).fit(positions)
#     return kmeans.labels_






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




def analyze_clusters_over_time(coordDynX, coordDynY, eps=10, min_samples=2,TauB=25,skip=0):

    coordDynX = read_particle_data_csv(coordDynX)[skip::2,1:][1:,:]  # Read x-coordinates
    coordDynY = read_particle_data_csv(coordDynY)[skip::2,1:][1:,:]  # Read y-coordinates

    num_timesteps = len(coordDynX)
    cluster_sizes = {}
    num_clusters = []
    avg_cluster_sizes = []
    time_axis = []

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        cluster_labels = detect_clusters_DBSCAN(x_coords, y_coords, eps, min_samples)

        unique_clusters = set(label for label in cluster_labels if label != -1)
        num_clusters.append(len(unique_clusters))

        sizes = []

        for cluster_id in unique_clusters:
            cluster_points = np.array([(x, y) for x, y, lbl in zip(x_coords, y_coords, cluster_labels) if lbl == cluster_id])
            sizes.append(len(cluster_points))

        cluster_sizes[t] = sizes

        # Calculate average cluster size for this timestep
        avg_cluster_size = np.mean(sizes) if sizes else 0
        avg_cluster_sizes.append(avg_cluster_size)
        time_axis.append(t)


    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps)


    # Smooth the average cluster sizes using a Gaussian filter
    #avg_cluster_sizes = gaussian_filter1d(avg_cluster_sizes, sigma=2)

    return cluster_sizes, num_clusters,avg_cluster_sizes,time_axis





def plot_cluster_metrics(cluster_sizes, num_clusters,avg_cluster_sizes,time_axis):
    """
    Plots the number of clusters and their sizes over time.
    """
    time_steps = list(cluster_sizes.keys())

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
    coordDynX = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\10TkB_25TauB\datax.csv"
    coordDynY = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\10TkB_25TauB\datay.csv"

    # Run cluster detection over all timesteps
    cluster_sizes, num_clusters,avg_cluster_sizes,time_axis = analyze_clusters_over_time(coordDynX, coordDynY, eps=3.043, min_samples=3)
    # Plot number of clusters and their sizes
    plot_cluster_metrics(cluster_sizes, num_clusters,avg_cluster_sizes,time_axis)


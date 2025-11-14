import matplotlib.pyplot as plt
import numpy as np
from Classify_Crystals import classify_crystal_structure
from Mat_Data_Reader import Mat_read_particle_data
from Nearest_Partners import find_nearest_neighbors
from Particle_Angels import angles_between_neighbors
from Visualise_Crystals import visualize_crystals
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from data_reader_csv import read_particle_data_csv


def detect_clusters_DBSCAN(x_coords, y_coords, eps=1.0, min_samples=3):
    """
    Detect clusters using DBSCAN.
    """
    positions = np.column_stack((x_coords, y_coords))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_

def match_clusters(prev_centroids, current_centroids):
    """
    Match clusters between two consecutive time steps using the Hungarian algorithm.
    """
    if not prev_centroids or not current_centroids:
        return {}

    # Compute the cost matrix (distance between centroids)
    cost_matrix = cdist(prev_centroids, current_centroids, metric='euclidean')

    # Find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a dictionary of matches
    matches = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    return matches




def analyze_clusters_sizes_over_time(coordDynX, coordDynY, eps=10, min_samples=2,TauB=25,skip=0):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2,1:][1:,:]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2,1:][1:,:]

    num_timesteps = len(coordDynX)
    cluster_sizes = {}  # Stores sizes of clusters at each timestep
    cluster_centroids = {}  # Stores centroids at each timestep
    num_clusters = []  # Stores the number of clusters at each timestep
    time_axis = []

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        cluster_labels = detect_clusters_DBSCAN(x_coords, y_coords, eps, min_samples)

        unique_clusters = set(label for label in cluster_labels if label != -1)
        num_clusters.append(len(unique_clusters))

        sizes = []
        centroids = []

        for cluster_id in unique_clusters:
            cluster_points = np.array([(x, y) for x, y, lbl in zip(x_coords, y_coords, cluster_labels) if lbl == cluster_id])
            sizes.append(len(cluster_points))  # Size of the cluster
            centroids.append(cluster_points.mean(axis=0))  # Centroid of the cluster

        cluster_sizes[t] = sizes
        cluster_centroids[t] = centroids
        time_axis.append(t)
    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps-1)

    return cluster_sizes, cluster_centroids, num_clusters,time_axis


def calculate_formation_speed(cluster_sizes, time_step_interval=1):
    """
    Calculate the cluster formation speed over time.
    """
    time_steps = sorted(cluster_sizes.keys())
    formation_speed = []

    for t in range(1, len(time_steps)):
        prev_sizes = cluster_sizes[time_steps[t - 1]]
        current_sizes = cluster_sizes[time_steps[t]]

        # Calculate the total number of particles in clusters at each timestep
        total_particles_prev = sum(prev_sizes)
        total_particles_current = sum(current_sizes)

        # Calculate the rate of change in the total number of particles in clusters
        speed = (total_particles_current - total_particles_prev) / time_step_interval
        formation_speed.append(speed)

    return formation_speed

def plot_formation_speed(formation_speed, time_axis):
    """
    Plot the cluster formation speed over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, formation_speed, marker='', linestyle='-', label="Cluster Formation Speed")
    plt.xlabel("Time(s)")
    plt.ylabel("Formation Speed (particles per unit time)")
    plt.title("Cluster Formation Speed Over Time")
    plt.legend()
    plt.show()







#coordDynX = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\20TkB_50TauB\no acpr\datax_NO_acpr.csv"
#coordDynY = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\20TkB_50TauB\no acpr\datay_NO_acpr.csv"


if  __name__ == '__main__':
    coordDynX = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\10TkB_25TauB\datax.csv"
    coordDynY = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs\10TkB_25TauB\datay.csv"


    # Analyze clusters over time
    cluster_sizes, cluster_centroids, num_clusters,time_axis = analyze_clusters_sizes_over_time(coordDynX, coordDynY, eps=3.043, min_samples=5)

    # Calculate formation speed
    formation_speed = calculate_formation_speed(cluster_sizes)

    # Plot formation speed
    plot_formation_speed(formation_speed,time_axis)
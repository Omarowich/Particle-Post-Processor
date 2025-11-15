from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

from data_reader_csv import read_particle_data_csv


def detect_clusters_DBSCAN(x_coords, y_coords, eps=1.0, min_samples=3):
    x_coords = np.array(x_coords, dtype=float)
    y_coords = np.array(y_coords, dtype=float)
    positions = np.column_stack((x_coords, y_coords))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_

def match_clusters(prev_centroids, current_centroids):
    """
    Match clusters between two consecutive time steps using the Hungarian algorithm.
    """
    if not prev_centroids or not current_centroids:
        return {}

    prev_ids = list(prev_centroids.keys())
    current_ids = list(current_centroids.keys())
    cost_matrix = np.zeros((len(prev_ids), len(current_ids)))

    for i, prev_id in enumerate(prev_ids):
        for j, current_id in enumerate(current_ids):
            cost_matrix[i, j] = np.linalg.norm(prev_centroids[prev_id] - current_centroids[current_id])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = {prev_ids[row]: current_ids[col] for row, col in zip(row_ind, col_ind)}
    return matches

def track_cluster_centroids(coordDynX, coordDynY, eps=10, min_samples=2, TauB=25,skip=0,normY=0):
    """
    Tracks the centroids of clusters over time and calculates the median movement of clusters.
    """
    coordDynX = read_particle_data_csv(coordDynX)[skip::2,1:][1:,:]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2,1:][1:,:]

    num_timesteps = len(coordDynX)
    time_axis = []
    cluster_centroids = {}
    cluster_labels_history = {}
    cluster_movements = []
    cluster_id_counter = 0
    cluster_id_map = {}  # Maps DBSCAN labels to unique cluster IDs

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        cluster_labels = detect_clusters_DBSCAN(x_coords, y_coords, eps, min_samples)

        unique_clusters = set(label for label in cluster_labels if label != -1)
        current_centroids = {}

        for cluster_label in unique_clusters:
            cluster_points = np.array(
                [(x, y) for x, y, lbl in zip(x_coords, y_coords, cluster_labels) if lbl == cluster_label])
            centroid = cluster_points.mean(axis=0)
            current_centroids[cluster_label] = centroid

        if t == 0:
            # Assign unique IDs to clusters in the first timestep
            for cluster_label in unique_clusters:
                cluster_id_map[cluster_label] = cluster_id_counter
                cluster_id_counter += 1
        else:
            # Match clusters between current and previous timesteps
            matches = match_clusters(cluster_centroids[t - 1], current_centroids)
            for prev_label, current_label in matches.items():
                if prev_label in cluster_id_map:  # Ensure prev_label exists in cluster_id_map
                    cluster_id_map[current_label] = cluster_id_map[prev_label]
                else:
                    # If prev_label doesn't exist, assign a new ID
                    cluster_id_map[current_label] = cluster_id_counter
                    cluster_id_counter += 1

            # Assign new IDs to new clusters (those without a match)
            for current_label in unique_clusters:
                if current_label not in matches.values():
                    cluster_id_map[current_label] = cluster_id_counter
                    cluster_id_counter += 1

        # Update centroids with unique IDs
        cluster_centroids[t] = {cluster_id_map[label]: centroid for label, centroid in current_centroids.items()}
        cluster_labels_history[t] = cluster_labels

        # Calculate cluster movements
        if t > 0:
            movements = []
            for cluster_id, centroid in cluster_centroids[t].items():
                if cluster_id in cluster_centroids[t - 1]:
                    prev_centroid = cluster_centroids[t - 1][cluster_id]
                    movement = np.linalg.norm(centroid - prev_centroid)
                    movements.append(movement)
            if movements:
                cluster_movements.append(np.average(movements))
            else:
                cluster_movements.append(0)
        else:
            cluster_movements.append(0)

        time_axis.append(t)

    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps)
    pprint(time_axis.shape)
    pprint(len(coordDynX))
    # Smooth the movement data
    cluster_movements_smoothed = gaussian_filter1d(cluster_movements, sigma=1.5)

    min_movement = min(cluster_movements)
    max_movement = max(cluster_movements)

    # Plot cluster movement over time
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, cluster_movements_smoothed, marker='', linestyle='-', label="Cluster Movement (Smoothed)")
    plt.plot(time_axis, cluster_movements, marker='', linestyle='-', label="Cluster Movement")
    plt.xlabel('Time (s)')
    plt.ylabel("Total Cluster Movement")
    plt.title("Cluster Movement Over Time")
    plt.legend()

    if normY == 1: plt.ylim(0, 0.2)


    #plt.ylim(min_movement, max_movement)
    #plt.xlim(10, 20)

    # Plot the vertical line if a flattening point was found
    flatten_timestep = None
    for i in range(1, len(cluster_movements_smoothed)):
        if abs(cluster_movements_smoothed[i]) < 1 and flatten_timestep is None:  # Example threshold
            flatten_timestep = i
            break

    if flatten_timestep is not None:
        flatten_time = time_axis[flatten_timestep]
        plt.axvline(x=flatten_time, color='r', linestyle='--', label='Flattening Point')
        plt.text(flatten_timestep + 3, max(cluster_movements_smoothed) * 0.9, f'Time: {flatten_time:.2f} s',
                 color='red', ha='left', va='center')

    plt.show()

    return time_axis, cluster_movements_smoothed

###########################################################################################

    # # Compute the Fourier transform of the cluster movements
    # fft_values = np.fft.fft(cluster_movements_smoothed)
    # frequencies = np.fft.fftfreq(len(cluster_movements_smoothed))
    #
    # # Plot the Fourier transform (magnitude spectrum)
    # plt.figure(figsize=(10, 5))
    # plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_values[:len(fft_values) // 2]),
    #          label="FFT of Cluster Movements")
    # plt.xlabel("Frequency")
    # plt.ylabel("Magnitude")
    # plt.title("Fourier Transform of Cluster Movements")
    # plt.legend()
    # plt.grid()
    # plt.show()



    return cluster_centroids, cluster_labels_history, cluster_movements_smoothed




if  __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\2.85Tkb_20TauB_after_acoustics\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\2.85Tkb_20TauB_after_acoustics\datay.csv"
    pprint(coordDynX)
    cluster_centroids, cluster_labels_history, cluster_movements = track_cluster_centroids(coordDynX, coordDynY, eps=3.043, min_samples=5, skip=1, normY=1)
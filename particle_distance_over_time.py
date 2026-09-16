import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from data_reader_csv import read_particle_data_csv


def particle_distance_over_time(coordDynX, coordDynY, eps=10, min_samples=2, TauB=25, skip=0, normY=0, **_extra_kwargs):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    avg_nn_distances = []

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        positions = np.column_stack((x_coords, y_coords))

        # Build KDTree for neighbor search
        tree = cKDTree(positions)
        dists, _ = tree.query(positions, k=7)  # k+1 because first neighbor is the point itself

        # Exclude self-distance (zero) from each row
        avg_dists_per_particle = np.mean(dists[:, 1:], axis=1)
        avg_nn_distance = np.mean(avg_dists_per_particle)
        avg_nn_distances.append(avg_nn_distance)


    plt.plot(time_axis, avg_nn_distances)
    plt.xlabel("Time (τB)")
    plt.ylabel("Distance (µm)")
    plt.title("Particle Distance Over Time")
    if normY == 1: plt.ylim(0, 6)
    plt.show()

    return time_axis, avg_nn_distances


if  __name__ == '__main__':
    coordDynX = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datax.csv"
    coordDynY = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datay.csv"

    skip=1
    TauB=5

    minsamples=5
    eps=3.043

    particle_distance_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples, TauB=TauB, skip=skip)
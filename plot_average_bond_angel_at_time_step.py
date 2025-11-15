import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from data_reader_csv import read_particle_data_csv


def plot_average_bond_angle_at_timestep(coordDynX, coordDynY, timestep, neighbor_cutoff=3.5, skip=0, degrees=True):
    # Load data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    # Extract positions
    x_coords = coordDynX[timestep]
    y_coords = coordDynY[timestep]
    positions = np.column_stack((x_coords, y_coords))

    # KDTree for neighbors
    tree = cKDTree(positions)
    neighbor_lists = tree.query_ball_tree(tree, r=neighbor_cutoff)

    avg_angles = []

    for i, neighbors in enumerate(neighbor_lists):
        neighbors = [j for j in neighbors if j != i]
        if not neighbors:
            avg_angles.append(np.nan)
            continue

        dx = x_coords[neighbors] - x_coords[i]
        dy = y_coords[neighbors] - y_coords[i]
        angles = np.arctan2(dy, dx)

        if degrees:
            angles = np.degrees(angles)

        mean_angle = np.mean(angles)
        avg_angles.append(mean_angle)

    avg_angles = np.array(avg_angles)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, c=avg_angles, cmap='hsv', s=250, edgecolors='black')
    cbar = plt.colorbar(scatter)
    unit = "°" if degrees else "rad"
    cbar.set_label(f"Average Bond Angle ({unit})")
    plt.title(f"Average Bond Angle per Particle at Timestep {timestep*13.513}s")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(False)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    coordDynX = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datax.csv"
    coordDynY = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datay.csv"

    skip = 1
    timestep= 1

    plot_average_bond_angle_at_timestep(coordDynX, coordDynY, timestep=timestep, neighbor_cutoff=3.5, skip=1)

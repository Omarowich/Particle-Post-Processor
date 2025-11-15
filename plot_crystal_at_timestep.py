import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform

from data_reader_csv import read_particle_data_csv


def plot_crystals_at_timestep(coordDynX, coordDynY, timestep, eps=3.043, min_samples=5, skip=1, normY=1):
    """Plots particles and their crystal types at a given time step with percentage in legend."""


    # Load particle data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    if timestep == None:
        timestep = len(coordDynX)-1

    # Extract positions for the given timestep
    positions = np.vstack((coordDynX[timestep], coordDynY[timestep])).T

    # Calculate pairwise distances between all particles at this timestep
    pairwise_distances = squareform(pdist(positions))

    # Filter particles that are within the given distance_limit from each other
    valid_distances = pairwise_distances < eps
    valid_particles = np.sum(valid_distances,
                             axis=1) > 1  # Consider a particle as part of a crystal if it has neighbors

    # Filter the positions of particles that are part of a crystal
    positions = positions[valid_particles]

    if len(positions) == 0:
        print(f"No crystalline particles found at timestep {timestep}.")
        return

    # Create Voronoi diagram
    vor = Voronoi(positions)

    # Function to classify crystal type based on Voronoi neighbor count
    def classify_crystal_type(neighbor_count):
        if neighbor_count == 6:
            return 'Hexagonal'
        elif neighbor_count == 4:
            return 'Square'
        elif neighbor_count == 3:
            return 'Triangular'
        else:
            return 'Other'

    # Count number of neighbors per Voronoi cell
    neighbor_counts = np.array([
        len(vor.regions[vor.point_region[i]])
        if -1 not in vor.regions[vor.point_region[i]] and vor.regions[vor.point_region[i]] else 0
        for i in range(len(positions))
    ])

    # Assign crystal type
    crystalline_types = [classify_crystal_type(count) for count in neighbor_counts]

    # Count occurrences for percentage calculation
    from collections import Counter
    type_counts = Counter(crystalline_types)
    total_particles = len(crystalline_types)

    # Plot
    plt.figure(figsize=(8, 6))

    # Color map
    color_map = {'Hexagonal': 'blue', 'Square': 'green', 'Triangular': 'red', 'Other': 'gray'}

    # Plot particles and prepare legend info
    legend_entries = {}
    for i, crystal_type in enumerate(crystalline_types):
        plt.scatter(positions[i, 0], positions[i, 1], c=color_map[crystal_type], label=crystal_type, marker='o')

    # Plot Voronoi
    voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=0.5, ax=plt.gca())

    # Compose legend with percentages
    unique_labels = []
    unique_handles = []
    for crystal_type in type_counts:
        percentage = (type_counts[crystal_type] / total_particles) * 100
        label = f"{crystal_type} ({percentage:.1f}%)"
        color = color_map.get(crystal_type, 'gray')
        handle = plt.Line2D([], [], color=color, marker='o', linestyle='None', label=label)
        unique_labels.append(label)
        unique_handles.append(handle)

    plt.legend(unique_handles, unique_labels, loc='upper right')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Crystals at Timestep {timestep*13.513}s")
    plt.grid()
    plt.tight_layout()
    if normY == 1:
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


# Usage example
if __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\2.85Tkb_20TauB_after_acoustics\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\2.85Tkb_20TauB_after_acoustics\datay.csv"
    # Example of plotting at timestep 50
    plot_crystals_at_timestep(coordDynX, coordDynY, timestep=None, eps=3.043, min_samples=5, skip=1,normY=1)

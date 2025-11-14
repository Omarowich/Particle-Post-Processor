import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freud
from data_reader_csv import read_particle_data_csv
from scipy.spatial import Voronoi


def detect_crystals_over_time(coordDynX, coordDynY, eps=3.043, min_samples=5, TauB=1, skip=1,normY=0):
    """Detect crystalline particles (hexagonal, square, triangular) over time and plot the fraction of particles in crystals."""

    # Load particle data
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    hexagonal_fractions = []
    square_fractions = []
    triangular_fractions = []
    crystalline_fractions = []
    other_fractions = []

    hexagonal_counts = []
    square_counts = []
    triangular_counts = []
    other_counts = []

    # Define a function to classify crystal types based on Voronoi cell neighbors
    def classify_crystal_type(neighbor_count):
        if neighbor_count == 6:
            return 'Hexagonal'
        elif neighbor_count == 4:
            return 'Square'
        elif neighbor_count == 3:
            return 'Triangular'
        else:
            return 'Other'

    for t in range(num_timesteps):
        positions = np.vstack((coordDynX[t], coordDynY[t])).T
        vor = Voronoi(positions)

        # Identify crystalline particles (Hexagonal, Square, Triangular)
        neighbor_counts = np.array([len(vor.regions[vor.point_region[i]])
                                    if -1 not in vor.regions[vor.point_region[i]] else 0
                                    for i in range(len(positions))])

        crystalline_types = [classify_crystal_type(count) for count in neighbor_counts]

        # Calculate the fraction of crystalline particles for each type
        hexagonal_fraction = (np.array(crystalline_types) == 'Hexagonal').sum() / len(positions)
        square_fraction = (np.array(crystalline_types) == 'Square').sum() / len(positions)
        triangular_fraction = (np.array(crystalline_types) == 'Triangular').sum() / len(positions)
        crystalline_fraction = (np.array(crystalline_types) != 'Other').sum() / len(positions)
        other_fraction = (np.array(crystalline_types) == 'Other').sum() / len(positions)

        hexagonal_fractions.append(hexagonal_fraction)
        square_fractions.append(square_fraction)
        triangular_fractions.append(triangular_fraction)
        crystalline_fractions.append(crystalline_fraction)
        other_fractions.append(other_fraction)

        # Optional: You can also track the counts of specific crystal types
        hexagonal_count = crystalline_types.count('Hexagonal')
        square_count = crystalline_types.count('Square')
        triangular_count = crystalline_types.count('Triangular')
        other_count = crystalline_types.count('Other')

        # Append counts for plotting
        hexagonal_counts.append(hexagonal_count)
        square_counts.append(square_count)
        triangular_counts.append(triangular_count)
        other_counts.append(other_count)

    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps)

    # Plot crystalline fraction over time (Hexagonal, Square, Triangular, All)
    plt.figure(figsize=(8, 5))
    line1, = plt.plot(time_axis, hexagonal_fractions, marker='', linestyle=':', label='Hexagonal Crystals')
    line2, = plt.plot(time_axis, square_fractions, marker='', linestyle='--', label='Square Crystals')
    line3, = plt.plot(time_axis, triangular_fractions, marker='', linestyle='-.', label='Triangular Crystals')
    line4, = plt.plot(time_axis, crystalline_fractions, marker='', linestyle='-', label='All Crystalline Particles')
    line5, = plt.plot(time_axis, other_fractions, marker='', linestyle='-', label='No Crystals')



    plt.xlabel("Time(s)")
    plt.ylabel("Fraction of Crystalline Particles")
    plt.title("Crystalline Fraction Over Time")
    # Create a custom legend with the final values
    # Here, we use the last values in the plot for each curve
    plt.legend(
        handles=[line1,
                 line2,
                 line3,
                 line4,
                 line5],
        labels=[
            f'Hexagonal ({hexagonal_fractions[-1]:.2f})',
            f'Square ({square_fractions[-1]:.2f})',
            f'Triangular ({triangular_fractions[-1]:.2f})',
            f'All Crystals ({crystalline_fractions[-1]:.2f})',
            f'No Crystals ({other_fractions[-1]:.2f})'
        ],
        loc='lower right', fontsize=10
    )
    plt.grid()
    if normY == 1: plt.ylim(0, 1)
    plt.show()

    return time_axis, hexagonal_fractions, other_fractions



    # # Plot the counts of crystal types over time (Hexagonal, Square, Triangular, Other)
    # plt.figure(figsize=(8, 5))
    # line6, = plt.plot(time_axis, hexagonal_counts, marker='', linestyle='-', label='Hexagonal')
    # line7, = plt.plot(time_axis, square_counts, marker='', linestyle='-', label='Square')
    # line8, = plt.plot(time_axis, triangular_counts, marker='', linestyle='-', label='Triangular')
    # line9, = plt.plot(time_axis, other_counts, marker='', linestyle='-', label='Other')
    #
    # plt.xlabel("Time(s)")
    # plt.ylabel("Count of Particles")
    # plt.title("Counts of Crystal Types Over Time")
    # plt.legend(
    #     handles=[line6, line7, line8, line9],
    #     labels=[
    #         f'Hexagonal ({hexagonal_counts[-1]})',
    #         f'Square ({square_counts[-1]})',
    #         f'Triangular ({triangular_counts[-1]})',
    #         f'Other ({other_counts[-1]})'
    #     ],
    #     loc='lower right', fontsize=10
    # )
    # plt.grid()
    # plt.show()

if  __name__ == '__main__':
    coordDynX = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datax.csv"
    coordDynY = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datay.csv"


    detect_crystals_over_time(coordDynX, coordDynY, eps=3.043, min_samples=5, TauB=1, skip=1)
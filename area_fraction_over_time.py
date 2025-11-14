import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from data_reader_csv import read_particle_data_csv  # make sure this is accessible in your script


def area_fraction_over_time(coordDynX, coordDynY, particle_radius=1.0, TauB=25, skip=0, normY=0):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    occupancy_percentages = []

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        positions = np.column_stack((x_coords, y_coords))

        if len(positions) < 3:
            occupancy_percentages.append(0)
            continue

        try:
            hull = ConvexHull(positions)
            area_hull = hull.volume  # In 2D, this is the area

            area_particles = len(positions) * np.pi * particle_radius**2
            occupancy = (area_particles / area_hull) * 100 if area_hull > 0 else 0
        except:
            occupancy = 0

        occupancy_percentages.append(occupancy)

    # Plotting
    plt.plot(time_axis, occupancy_percentages)
    plt.xlabel("Time (τB)")
    plt.ylabel("Area Occupied by Particles (%)")
    plt.title("Particle Area Occupancy Over Time")
    plt.grid(True)
    plt.tight_layout()
    if normY == 1: plt.ylim(0, 100)
    plt.show()

    return time_axis, occupancy_percentages


if __name__ == '__main__':
    coordDynX = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datax.csv"
    coordDynY = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\1Tkb_5TauB_after_acoustics\datay.csv"

    TauB = 5
    skip = 1

    area_fraction_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip)

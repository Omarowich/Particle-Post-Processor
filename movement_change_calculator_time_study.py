import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_reader_csv import read_particle_data_csv


# from Classify_Crystals import classify_crystal_structure
# from Mat_Data_Reader import Mat_read_particle_data
# from Nearest_Partners import find_nearest_neighbors
# from Particle_Angels import angles_between_neighbors
# from Visualise_Crystals import visualize_crystals


def particle_displacement_over_time(coordDynX, coordDynY, TauB=25,skip=0, normY=0):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2,1:][1:,:]  # Read x-coordinates
    coordDynY = read_particle_data_csv(coordDynY)[skip::2,1:][1:,:]  # Read y-coordinates

    num_particles = len(coordDynX[0])  # Assuming each row is a time step, each column is a particle
    num_timesteps = len(coordDynX)

    median_displacements = []
    time_axis = []

    # Iterate through time steps (excluding the last step)
    for t in range(num_timesteps-1):
        displacements = []

        for p in range(num_particles):
            x0, y0 = coordDynX[t][p], coordDynY[t][p]
            x1, y1 = coordDynX[t + 1][p], coordDynY[t + 1][p]

            # Calculate Euclidean distance
            displacement = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            displacements.append(displacement)

        # Compute median displacement for this time step
        median_displacements.append(np.mean(displacements))
        time_axis.append(t)

    # Scale the time axis to go from 0 to TauB * 13.513
    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps-1)

    # Smooth the median displacement over time
    smoothed_median = pd.Series(median_displacements).ewm(span=10, adjust=False).mean()

    # Find the time step where the displacement starts to flatten out
    diff_smoothed = np.diff(smoothed_median)
    flat_threshold = 0.0001 # Threshold to determine flattening (can adjust based on data)
    flatten_timestep = None

    for i in range(1, len(diff_smoothed)):
        if abs(diff_smoothed[i]) < flat_threshold and flatten_timestep is None:
            flatten_timestep = i
            break

    # Plot the median displacement over time
    plt.plot(time_axis, median_displacements, marker='', linestyle='-', label='Median Displacement')
    plt.plot(time_axis, smoothed_median, marker='', linestyle='-', label='Smoothed Median Displacement')

    if flatten_timestep is not None:
        flatten_time = time_axis[flatten_timestep]
        plt.axvline(x=flatten_time, color='r', linestyle='--', label='Flattening Point')
        # Add the time label near the vertical line
        plt.text(flatten_time + 3, max(smoothed_median) * 0.9, f'Time: {flatten_time:.2f} s',
                 color='red', ha='left', va='center')

    plt.xlabel('Time (s)')
    plt.ylabel('Median Displacement')
    plt.title('Median Particle Displacement Over Time')
    plt.legend()
    if normY == 1: plt.ylim(0, 0.2)
    plt.show()

    return time_axis, median_displacements


###########################################################################################


    # # Compute the Fourier transform of the cluster movements
    # fft_values = np.fft.fft(median_displacements)
    # frequencies = np.fft.fftfreq(len(median_displacements))
    #
    # # Plot the Fourier transform (magnitude spectrum)
    # plt.figure(figsize=(10, 5))
    # plt.plot(frequencies[:len(frequencies) // 2], np.abs(fft_values[:len(fft_values) // 2]),
    #          label="FFT of Particle Movements")
    # plt.xlabel("Frequency")
    # plt.ylabel("Magnitude")
    # plt.title("Fourier Transform of Particle Movements")
    # plt.legend()
    # plt.grid()
    # plt.show()

def particle_displacement_from_inintal_position_over_time(coordDynX, coordDynY, TauB=25, skip=0, normY=False):
    # Load the X/Y arrays exactly as before
    x = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_timesteps, n_particles = x.shape

    # Array to hold running total distance for each particle
    total_dist = np.zeros(n_particles)

    med_totals = []
    # Build time axis scaled to real seconds
    time_axis = np.linspace(0, TauB * 13.513, n_timesteps - 1)

    for t in range(n_timesteps-1):
        # Compute per-step distances
        dx = x[t] - x[0]
        dy = y[t] - y[0]
        step_dists = np.sqrt(dx**2 + dy**2)


        # Record the median of cumulative totals
        med_totals.append(np.median(step_dists))

    # Plot
    plt.figure()
    plt.plot(time_axis, med_totals, label='Median Distance from Position(t=0)')
    plt.xlabel('Time (s)')
    plt.ylabel('Median Distance from Position(t=0)')
    if normY == 1: plt.ylim(0, 20)
    plt.title('Median of Particle Distance from Initial Position Over Time')
    plt.legend()
    plt.show()

    return time_axis, np.array(med_totals)



def median_total_path_distance_over_time(coordDynX, coordDynY, TauB=25, skip=0, normY=False):
    # Load the X/Y arrays exactly as before
    x = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_timesteps, n_particles = x.shape

    # Array to hold running total distance for each particle
    total_dist = np.zeros(n_particles)

    med_totals = []
    # Build time axis scaled to real seconds
    time_axis = np.linspace(0, TauB * 13.513, n_timesteps - 1)

    for t in range(n_timesteps - 1):
        # Compute per-step distances
        dx = x[t + 1] - x[t]
        dy = y[t + 1] - y[t]
        step_dists = np.sqrt(dx**2 + dy**2)

        # Accumulate scalar distance
        total_dist += step_dists

        # Record the median of cumulative totals
        med_totals.append(np.median(total_dist))

    # Plot
    plt.figure()
    plt.plot(time_axis, med_totals, label='Median Total Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Median Cumulative Distance')
    if normY == 1: plt.ylim(0, 50)
    plt.title('Median of Particle Total Distance Traveled Over Time')
    plt.legend()
    plt.show()

    return time_axis, np.array(med_totals)


def movement_heatmap_over_time(coordDynX, coordDynY, bins=(100,100), TauB=25, skip=0):
    """
    Build and plot a heatmap of cumulative particle movement.

    coordDynX, coordDynY : file paths to your CSVs
    bins                 : tuple (nx, ny) for the 2D heatmap resolution
    TauB                 : used only to rescale time if you want timestamps
    skip                 : number of initial rows to skip (and then every other row)
    """
    # --- load positions just like your other funcs ---
    x = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    n_timesteps, _ = x.shape

    # spatial bin edges
    x_edges = np.linspace(x.min(), x.max(), bins[0] + 1)
    y_edges = np.linspace(y.min(), y.max(), bins[1] + 1)

    # accumulator for total distance per bin
    heat = np.zeros((bins[0], bins[1]))

    # loop over all steps
    for t in range(n_timesteps - 1):
        # step‐by‐step displacements
        dx = x[t+1] - x[t]
        dy = y[t+1] - y[t]
        step_dist = np.sqrt(dx**2 + dy**2)

        # midpoints of each step
        mid_x = 0.5 * (x[t]   + x[t+1])
        mid_y = 0.5 * (y[t]   + y[t+1])

        # bin them, weighted by movement
        H, _, _ = np.histogram2d(
            mid_x, mid_y,
            bins=[x_edges, y_edges],
            weights=step_dist
        )
        heat += H

    # plot the heatmap
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    plt.figure()
    plt.imshow(heat.T, origin='lower', extent=extent, aspect='auto')
    plt.colorbar(label='Total Distance Traveled')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Heatmap of Cumulative Movement Over Time')
    plt.show()



if  __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\0Tkb_20TauB_after_acoustics\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\0Tkb_20TauB_after_acoustics\datay.csv"


    particle_displacement_over_time(coordDynX, coordDynY, TauB=25,skip=0)

    particle_displacement_from_inintal_position_over_time(coordDynX, coordDynY, TauB=20, skip=1, normY=False)

    median_total_path_distance_over_time(coordDynX, coordDynY, TauB=20, skip=1, normY=False)

    movement_heatmap_over_time(coordDynX, coordDynY, bins=(50, 50), TauB=20, skip=1)


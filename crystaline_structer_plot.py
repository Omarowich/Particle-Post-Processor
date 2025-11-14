import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import os
import numpy as np
from pprint import pprint
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree

from Data_Reader import read_particle_data
from Nearest_Partners import find_nearest_neighbors
from Particle_Angels import angles_between_neighbors
from Classify_Crystals import classify_crystal_structure
from New_Crystal_Classificator_only_center_particle_coloured import crystal_classifier
from Visualise_Crystals import visualize_crystals
from Mat_Data_Reader import Mat_read_particle_data
from data_reader_csv import read_particle_data_csv
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from pyxtal import pyxtal
from pyxtal.symmetry import Group


coordDynX = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs 100K Pascal\20TkB_1TauB\datax.csv"
coordDynY = r"Z:\MML MS BS students\Bachelor Students\Omar Elsabbagh\50x50+ARF\acoustic outputs 100K Pascal\20TkB_1TauB\datay.csv"

coordDynX = read_particle_data_csv(coordDynX)[1::2, 1:][1:, :]
coordDynY = read_particle_data_csv(coordDynY)[1::2, 1:][1:, :]

num_timesteps = coordDynX.shape


# Function to detect hexagonal ordering
def is_hexagonal(x, y):
    try:
        structure = pyxtal()
        structure.from_array([coordDynX, coordDynY, np.zeros_like(x)], 2, "P6")  # P6 is a common hexagonal space group
        return True  # If PyXtal successfully forms P6, it's hexagonal
    except:
        return False


# Plot hexagonal crystals over time
# for t in range(num_timesteps):
#     coordDynX = coordDynX[t]
#     coordDynY = coordDynY[t]
#     hex_indices = detect_hexagonal_crystals(x, y)
#
#     plt.figure(figsize=(6, 6))
#     plt.scatter(x, y, color='gray', alpha=0.3, label="Other Particles")
#     plt.scatter(x[hex_indices], y[hex_indices], color='blue', label="Hexagonal Crystals")
#
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.title(f"Hexagonal Crystals at Timestep {t}")
#     plt.legend()
#     plt.show()


    coordDynX = coordDynX[50]
    coordDynY = coordDynY[50]


    hexagonal = is_hexagonal(coordDynX, coordDynY)

    # Plot the snapshot
    plt.figure(figsize=(6, 6))
    plt.scatter(coordDynX, coordDynY, color='blue' if hexagonal else 'gray', alpha=0.5,
                label="Hexagonal Crystals" if hexagonal else "Non-Hexagonal")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Hexagonal Crystals in Snapshot" if hexagonal else "No Hexagonal Structure Detected")
    plt.legend()
    plt.show()
import os
import numpy as np
from pprint import pprint
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree
from data_reader_csv import read_particle_data_csv
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from track_cluster_movement import track_cluster_centroids
from cluster_post_processor import plot_cluster_metrics
from cluster_post_processor import analyze_clusters_over_time
from movement_change_calculator_time_study import particle_displacement_over_time, particle_displacement_from_inintal_position_over_time, median_total_path_distance_over_time,movement_heatmap_over_time
from cluster_formation_speed import calculate_formation_speed, plot_formation_speed, analyze_clusters_sizes_over_time
from track_crystal_formation_over_time import detect_crystals_over_time
from particle_distance_over_time import particle_distance_over_time
from plot_crystal_at_timestep import plot_crystals_at_timestep
from area_fraction_over_time import area_fraction_over_time
from bond_orientation_over_time import bond_orientational_order_over_time
from plot_bond_orientation_at_time_step import plot_bond_orientational_order_at_timestep
from rg_over_time import average_rg_over_time

coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\2Tkb_20TauB_ASF\datax.csv"
coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\2Tkb_20TauB_ASF\datay.csv"

skip=1
TauB=1

minsamples=5
eps=3.043

# Track cluster centroids over time
track_cluster_centroids(coordDynX, coordDynY, eps=eps, min_samples=minsamples,TauB=TauB, skip=skip)

# Run cluster detection over all timesteps
cluster_sizes, num_clusters,avg_cluster_sizes,time_axis = analyze_clusters_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples,TauB=TauB,skip=skip)
plot_cluster_metrics(cluster_sizes, num_clusters,avg_cluster_sizes,time_axis)

# Plot particle displacement over time
particle_displacement_over_time(coordDynX, coordDynY, TauB=TauB,skip=skip)
particle_distance_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples, TauB=TauB, skip=skip)

# Calculate formation speed
#cluster_sizes, cluster_centroids, num_clusters,time_axis=analyze_clusters_sizes_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples,TauB=TauB,skip=skip)
#formation_speed = calculate_formation_speed(cluster_sizes)
#plot_formation_speed(formation_speed, time_axis)

# Detect crystals over time
detect_crystals_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples, TauB=TauB, skip=skip)
plot_crystals_at_timestep(coordDynX, coordDynY, timestep=None, eps=eps, min_samples=minsamples, skip=skip)

# Plot bond orientational order
plot_bond_orientational_order_at_timestep(coordDynX, coordDynY, timestep=None, n=6, neighbor_cutoff=3.5, skip=skip)
bond_orientational_order_over_time(coordDynX, coordDynY, n=6, neighbor_cutoff=3.5, TauB=TauB, skip=skip)

area_fraction_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip)
average_rg_over_time(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip)

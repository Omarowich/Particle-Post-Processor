# top_level_runner.py

import matplotlib
# 1) switch to a file-only backend (no GUI)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# 2) make plt.show() a no-op so it won’t clear your figures
plt.show = lambda *args, **kwargs: None


import numpy as np
import os
import io
import contextlib
import imageio
from matplotlib.patches import Circle
from scipy.spatial import KDTree
import math
from sklearn.cluster import DBSCAN, KMeans
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

# Project-specific modules
from data_reader_csv import read_particle_data_csv
from track_cluster_movement import track_cluster_centroids
from cluster_post_processor import plot_cluster_metrics, analyze_clusters_over_time
from movement_change_calculator_time_study import particle_displacement_over_time, particle_displacement_from_inintal_position_over_time, median_total_path_distance_over_time,movement_heatmap_over_time
from cluster_formation_speed import calculate_formation_speed, plot_formation_speed, analyze_clusters_sizes_over_time
from track_crystal_formation_over_time import detect_crystals_over_time
from particle_distance_over_time import particle_distance_over_time
from plot_crystal_at_timestep import plot_crystals_at_timestep
from area_fraction_over_time import area_fraction_over_time
from bond_orientation_over_time import bond_orientational_order_over_time
from plot_bond_orientation_at_time_step import plot_bond_orientational_order_at_timestep
from rg_over_time import average_rg_over_time
from force_analysis_over_time import plot_average_forces_over_time


def compute_average_slope(time, y):
    # fit a line y = m⋅t + b
    m, b = np.polyfit(time, y, 1)
    return m

def run_all_analyses(coordDynX, coordDynY, drag_data_path, lj_data_path, skip, TauB, minsamples, eps, timestep, normY, output_dir):
    """
    Run the full suite of analyses and save plots to the specified output directory.
    Suppresses console printing from individual analysis functions.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Redirect stdout to suppress prints
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        # 1. Track cluster centroids (data only)
       track_cluster_centroids(coordDynX, coordDynY, eps=eps, min_samples=minsamples, TauB=TauB, skip=skip, normY=normY)

        #2. Particle displacement

       particle_displacement_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip, normY=normY)
       time1, y_values1 = particle_displacement_from_inintal_position_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'particle_displacement_over_time.png'))
       plt.close(fig)
       slope1 = compute_average_slope(time1, y_values1)

       time2, y_values2 = median_total_path_distance_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'particle_total_path_over_time.png'))
       plt.close(fig)
       slope2 = compute_average_slope(time2, y_values2)


       # 3. Particle distance
       time3, y_values3 = particle_distance_over_time(coordDynX, coordDynY, eps=eps, min_samples=minsamples, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'particle_distance_over_time.png'))
       plt.close(fig)
       slope3 = compute_average_slope(time3, y_values3)


       # 4. Crystal detection
       detect_crystals_over_time(coordDynX, coordDynY, eps=eps,min_samples=minsamples, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'crystal_detection_over_time.png'))
       plt.close(fig)

       # 5. Plot crystals at a timestep
       plot_crystals_at_timestep(coordDynX, coordDynY, timestep=timestep, eps=eps,min_samples=minsamples, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'crystals_at_timestep.png'))
       plt.close(fig)

       # 6. Bond orientational order at a timestep
       plot_bond_orientational_order_at_timestep(coordDynX, coordDynY, timestep=timestep,n=6, neighbor_cutoff=3.5, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'bond_order_at_timestep.png'))
       plt.close(fig)

       # 7. Bond orientational order over time
       bond_orientational_order_over_time(coordDynX, coordDynY, n=6, neighbor_cutoff=3.5, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'bond_order_over_time.png'))
       plt.close(fig)

       # 8. Area fraction over time
       area_fraction_over_time(coordDynX, coordDynY, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'area_fraction_over_time.png'))
       plt.close(fig)

       # 9. Radius of gyration over time
       time9, y_values9 = average_rg_over_time(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip, normY=normY)
       fig = plt.gcf()
       fig.savefig(os.path.join(output_dir, 'radius_of_gyration_over_time.png'))
       plt.close(fig)
       slope9 = compute_average_slope(time9, y_values9)




       # plot_average_forces_over_time(drag_data_path=drag_data_path, lj_data_path=lj_data_path, TauB=TauB, skip=skip, normY=normY)
       # fig = plt.gcf()
       # fig.savefig(os.path.join(output_dir, 'average_forces_over_time.png'))
       # plt.close(fig)





def batch_process(base_dir, skip, TauB, minsamples, eps, timestep, normY):
    """
    Walk through base_dir, find datax.csv and datay.csv in subfolders,
    and run analyses saving to 'normalised analysis/<relative_path>'.
    """

    for root, dirs, files in os.walk(base_dir):
        if 'datax.csv' in files and 'datay.csv' in files:
            coordDynX = os.path.join(root, 'datax.csv')
            coordDynY = os.path.join(root, 'datay.csv')
            drag_data_path = os.path.join(root, 'datad.csv')
            lj_data_path = os.path.join(root, 'dataflj.csv')
            rel_path = os.path.relpath(root, base_dir)
            output_dir = os.path.join(base_dir, 'normalised analysis', rel_path)
            run_all_analyses(coordDynX, coordDynY, drag_data_path, lj_data_path, skip, TauB, minsamples, eps, timestep, normY, output_dir)


if __name__ == '__main__':
    base_dir = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF"
    skip = 1
    TauB = 2
    minsamples = 5
    eps = 3.043
    timestep = None
    normY = 1



    batch_process(base_dir, skip, TauB, minsamples, eps, timestep, normY)

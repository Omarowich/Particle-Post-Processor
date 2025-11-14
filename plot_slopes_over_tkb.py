import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

##########
from movement_change_calculator_time_study import particle_displacement_over_time, particle_displacement_from_inintal_position_over_time, median_total_path_distance_over_time,movement_heatmap_over_time
from cluster_formation_speed import calculate_formation_speed, plot_formation_speed, analyze_clusters_sizes_over_time
from track_crystal_formation_over_time import detect_crystals_over_time
from particle_distance_over_time import particle_distance_over_time
from plot_crystal_at_timestep import plot_crystals_at_timestep
from area_fraction_over_time import area_fraction_over_time
from bond_orientation_over_time import bond_orientational_order_over_time
from plot_bond_orientation_at_time_step import plot_bond_orientational_order_at_timestep
from rg_over_time import average_rg_over_time
from track_cluster_movement import track_cluster_centroids
from cluster_post_processor import plot_cluster_metrics
from cluster_post_processor import analyze_clusters_over_time

plt.ioff()
def compute_slope(t, y):
    slope, _ = np.polyfit(t, y, 1)
    return slope

def plot_slopes_vs_tkb(base_dir,
                       metric_func,
                       metric_kwargs,
                       x_name='datax.csv',
                       y_name='datay.csv',
                       TauB=20,
                       skip=0):

    folder_re = re.compile(r'^([0-9]+(?:\.[0-9]+)?)Tkb_')

    tkbs, slopes = [], []

    for fname in sorted(os.listdir(base_dir)):
        m = folder_re.match(fname)
        if not m:
            continue
        tkb = float(m.group(1))
        folder = os.path.join(base_dir, fname)
        fx = os.path.join(folder, x_name)
        fy = os.path.join(folder, y_name)
        if not (os.path.isfile(fx) and os.path.isfile(fy)):
            print(f"Skipping {fname}: missing data files")
            continue


        # get median-distance curve
        t, med = metric_func(
            fx, fy, **metric_kwargs
        )
        plt.close('all')
        print(f"  • Tkb={tkb:4.2f} →  med[:5] = {med[:5]}, slope = {compute_slope(t, med):.5f}")

        slopes.append(compute_slope(t, med))
        tkbs.append(tkb)

    # sort and plot
    tkbs = np.array(tkbs)
    slopes = np.array(slopes)
    order = np.argsort(tkbs)
    tkbs, slopes = tkbs[order], slopes[order]

    plt.figure()
    plt.plot(tkbs, slopes, 'o-')
    plt.xlabel('Tkb')
    plt.ylabel('Dispersion rate (slope of median distance) [units/s]')
    plt.title(f'Dispersion rate vs Tkb (TauB={TauB})')
    plt.grid(True)
    plt.show()

    return tkbs, slopes




if __name__ == "__main__":
    base = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF"

    kwargs = dict(
        #eps=3.043,
        TauB=20,
        skip=1,
        normY=1
    )
    tkb_vals, slope_vals = plot_slopes_vs_tkb(
        base_dir=base,
        x_name='datax.csv',
        y_name='datay.csv',
        metric_kwargs= kwargs,
        metric_func=average_rg_over_time
    )

    # average_rg_over_time
    # area_fraction_over_time
    # bond_orientational_order_over_time
    # detect_crystals_over_time
    # particle_distance_over_time
    # median_total_path_distance_over_time
    # particle_displacement_from_inintal_position_over_time
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


def fourier_of_metric(coordDynX,
                      coordDynY,
                      metric_func,
                      metric_kwargs=None,
                      window=None,
                      normalize=True,
                      title=None):

    metric_kwargs = metric_kwargs or {}

    # 1) Generate the time-series
    t, y = metric_func(coordDynX, coordDynY, **metric_kwargs)
    t = np.asarray(t)
    y = np.asarray(y)
    N = len(y)

    # 2) Sampling
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # 3) Windowing
    if window is not None:
        w = window(N)
        yw = y * w
    else:
        yw = y

    # 4) FFT
    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(N, dt)
    amps = np.abs(Y)

    # 5) Normalize & one‐sided scale
    if normalize:
        amps = amps / N
    if N % 2 == 0:
        amps[1:-1] *= 2
    else:
        amps[1:] *= 2

    # 6) Plot
    plt.figure()
    plt.plot(freqs, amps, lw=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title or f'FFT of {metric_func.__name__}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return freqs, amps


if __name__ == '__main__':

    # 1) Basic FFT of your radial‐displacement metric
    freqs, amps = fourier_of_metric(
        coordDynX=r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\1Tkb_20TauB_ASF\datax.csv",
        coordDynY=r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\1Tkb_20TauB_ASF\datay.csv",
        metric_func=particle_displacement_from_inintal_position_over_time,
        metric_kwargs={'TauB': 20, 'skip': 1},
        window=None,
        normalize=True,
        title='Displacement FFT'
    )


    # average_rg_over_time
    # area_fraction_over_time
    # bond_orientational_order_over_time
    # detect_crystals_over_time
    # particle_distance_over_time
    # median_total_path_distance_over_time
    # particle_displacement_from_inintal_position_over_time
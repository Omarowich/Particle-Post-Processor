import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from data_reader_csv import read_particle_data_csv


def bond_orientational_order_over_time(coordDynX, coordDynY, n=6, neighbor_cutoff=3.5, TauB=25, skip=0, normY=0, **_extra_kwargs):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    time_axis = np.linspace(0, TauB * 13.513, num_timesteps)
    avg_psi_n_list = []

    for t in range(num_timesteps):
        x_coords = coordDynX[t]
        y_coords = coordDynY[t]
        positions = np.column_stack((x_coords, y_coords))

        # Remove NaNs (e.g. from dropped particles)
        positions = positions[~np.isnan(positions).any(axis=1)]

        if len(positions) == 0:
            avg_psi_n_list.append(0)
            continue

        tree = cKDTree(positions)
        neighbors = tree.query_ball_tree(tree, r=neighbor_cutoff)

        psi_n_values = []

        for i, nbrs in enumerate(neighbors):
            nbrs = [j for j in nbrs if j != i]
            if len(nbrs) == 0:
                continue

            angles = np.arctan2(
                positions[nbrs, 1] - positions[i, 1],
                positions[nbrs, 0] - positions[i, 0]
            )
            psi_n_i = np.sum(np.exp(1j * n * angles)) / len(nbrs)
            psi_n_values.append(np.abs(psi_n_i))

        avg_psi_n = np.mean(psi_n_values) if psi_n_values else 0
        avg_psi_n_list.append(avg_psi_n)

    # Plot
    plt.plot(time_axis, avg_psi_n_list, marker='')
    plt.xlabel("Time (τB)")
    plt.ylabel(f"Average |ψ{n}|")
    plt.title(f"Bond-Orientational Order Parameter (n={n}) Over Time")
    plt.grid()
    plt.tight_layout()
    if normY == 1: plt.ylim(0.4, 1)
    plt.show()

    return time_axis, avg_psi_n_list


if __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\0Tkb_20TauB_after_acoustics\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Finding Tkb after acoustics\0Tkb_20TauB_after_acoustics\datay.csv"

    TauB = 20
    skip = 1

    bond_orientational_order_over_time(coordDynX, coordDynY, n=6, neighbor_cutoff=3.5, TauB=TauB, skip=skip, normY=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freud
from data_reader_csv import read_particle_data_csv
from scipy.spatial import cKDTree


def detect_crystals_over_time(coordDynX, coordDynY, eps=3.043, min_samples=5,
                              TauB=1, skip=1, normY=0, plot=False,
                              plot_types=None, type="Hexagonal"):
    """Detect crystalline particles over time using eps as bond cutoff + local orientational order."""

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

    def classify_crystal_type(psi6, psi4, psi3, z):
        if z < 3:
            return "Other"

        if psi6 > 0.65:
            return "Hexagonal"
        elif psi4 > 0.65:
            return "Square"
        elif psi3 > 0.65:
            return "Triangular"
        else:
            return "Other"

    for t in range(num_timesteps):
        positions = np.vstack((coordDynX[t], coordDynY[t])).T

        tree = cKDTree(positions)
        neighbor_lists = tree.query_ball_tree(tree, r=eps)

        crystalline_types = []

        for i, neighbors in enumerate(neighbor_lists):
            neighbors = [j for j in neighbors if j != i]
            z = len(neighbors)

            if z < 3:
                crystalline_types.append("Other")
                continue

            dx = positions[neighbors, 0] - positions[i, 0]
            dy = positions[neighbors, 1] - positions[i, 1]
            angles = np.arctan2(dy, dx)

            psi6 = np.abs(np.mean(np.exp(1j * 6 * angles)))
            psi4 = np.abs(np.mean(np.exp(1j * 4 * angles)))
            psi3 = np.abs(np.mean(np.exp(1j * 3 * angles)))

            crystalline_types.append(classify_crystal_type(psi6, psi4, psi3, z))

        crystalline_types = np.array(crystalline_types)

        hexagonal_fraction = np.sum(crystalline_types == "Hexagonal") / len(positions)
        square_fraction = np.sum(crystalline_types == "Square") / len(positions)
        triangular_fraction = np.sum(crystalline_types == "Triangular") / len(positions)
        crystalline_fraction = np.sum(crystalline_types != "Other") / len(positions)
        other_fraction = np.sum(crystalline_types == "Other") / len(positions)

        hexagonal_fractions.append(hexagonal_fraction * 100)
        square_fractions.append(square_fraction * 100)
        triangular_fractions.append(triangular_fraction * 100)
        crystalline_fractions.append(crystalline_fraction * 100)
        other_fractions.append(other_fraction * 100)

        hexagonal_counts.append(np.sum(crystalline_types == "Hexagonal"))
        square_counts.append(np.sum(crystalline_types == "Square"))
        triangular_counts.append(np.sum(crystalline_types == "Triangular"))
        other_counts.append(np.sum(crystalline_types == "Other"))

    max_time = TauB * 13.513
    time_axis = np.linspace(0, max_time, num_timesteps)

    series = {
        "Hexagonal": np.array(hexagonal_fractions),
        "Square": np.array(square_fractions),
        "Triangular": np.array(triangular_fractions),
        "All": np.array(crystalline_fractions),
        "None": np.array(other_fractions),
    }

    if plot:
        if plot_types is None:
            plot_types = ["Hexagonal", "Square", "Triangular", "All", "None"]

        plt.figure(figsize=(8, 5))
        for k in plot_types:
            if k in series:
                plt.plot(time_axis, series[k], label=f"{k} ({series[k][-1]:.2f})")

        plt.xlabel("Time (s)")
        plt.ylabel("Fraction of Crystalline Particles")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid()

        if normY == 1:
            plt.ylim(0, 100)

        plt.show()

    y = series.get(type, series["Hexagonal"])
    return time_axis, y


def crystals_hex_over_time(fx, fy, **kwargs):
    return detect_crystals_over_time(fx, fy, type="Hexagonal", **kwargs)


def crystals_square_over_time(fx, fy, **kwargs):
    return detect_crystals_over_time(fx, fy, type="Square", **kwargs)


def crystals_tri_over_time(fx, fy, **kwargs):
    return detect_crystals_over_time(fx, fy, type="Triangular", **kwargs)


def crystals_all_over_time(fx, fy, **kwargs):
    return detect_crystals_over_time(fx, fy, type="All", **kwargs)


def crystals_none_over_time(fx, fy, **kwargs):
    return detect_crystals_over_time(fx, fy, type="None", **kwargs)


if __name__ == "__main__":
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\AAF\5Tkb_20TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\AAF\5Tkb_20TauB\datay.csv"

    detect_crystals_over_time(
        coordDynX,
        coordDynY,
        eps=3.5,
        min_samples=5,
        TauB=1,
        skip=1,
        plot=True,
        plot_types=["Hexagonal", "All", "None"],
        type="Hexagonal"
    )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freud
from data_reader_csv import read_particle_data_csv
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import freud
from data_reader_csv import read_particle_data_csv
from scipy.spatial import cKDTree




def rdf_over_time(
    coordDynX,
    coordDynY,
    r_max=10.0,
    dr=0.1,
    TauB=25,
    skip=0,
    normY=0,
    box_size=None,
    timestep=None,
    **kwargs,
):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)

    if num_timesteps == 0:
        bins = np.arange(0, r_max + dr, dr)
        r_centers = 0.5 * (bins[:-1] + bins[1:])
        return r_centers, np.zeros(len(r_centers))

    # If no timestep is selected, use the last timestep
    if timestep is None:
        timestep = num_timesteps - 1

    timestep = int(timestep)

    if timestep < 0 or timestep >= num_timesteps:
        raise ValueError(
            f"Invalid timestep {timestep}. Valid range is 0 to {num_timesteps - 1}."
        )

    bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    positions = np.column_stack((coordDynX[timestep], coordDynY[timestep]))
    N = len(positions)

    if N < 2:
        return r_centers, np.zeros(len(r_centers))

    counts = np.zeros(len(r_centers))

    for i in range(N):
        diff = positions - positions[i]

        if box_size is not None:
            diff = diff - box_size * np.round(diff / box_size)

        dist = np.sqrt((diff ** 2).sum(axis=1))
        dist = dist[dist > 0]

        counts += np.histogram(dist, bins=bins)[0]

    if box_size is not None:
        area = box_size ** 2
    else:
        from scipy.spatial import ConvexHull

        try:
            area = ConvexHull(positions).volume
        except Exception:
            area = 1.0

    rho = N / area

    shell_area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    ideal = rho * shell_area * N

    g_r = counts / np.maximum(ideal, 1e-12)

    return r_centers, g_r


if __name__ == "__main__":
    cx1 = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\ARF\0Tkb_0TauB\datax.csv"
    cy1 = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\ARF\0Tkb_0TauB\datay.csv"

    cx2 = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\ARF\0Tkb_2TauB\datax.csv"
    cy2 = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\Raw Data\ARF\0Tkb_2TauB\datay.csv"

    r1, g1 = rdf_over_time(cx1, cy1, skip=1)
    r2, g2 = rdf_over_time(cx2, cy2, skip=1)

    plt.figure()
    plt.plot(r1, g1, color='lightgrey')
    plt.plot(r2, g2, color='k')
    # plt.xlim(0, 10)
    # plt.ylim(0, 6)
    plt.xlabel("", fontsize=16)
    plt.ylabel("", fontsize=16)
    plt.title("", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.show()
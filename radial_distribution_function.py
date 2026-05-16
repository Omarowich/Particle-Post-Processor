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
    **kwargs,
):
    coordDynX = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    coordDynY = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    num_timesteps = len(coordDynX)
    bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    all_gr = []

    for t in range(num_timesteps):
        positions = np.column_stack((coordDynX[t], coordDynY[t]))
        N = len(positions)
        if N < 2:
            continue

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
        all_gr.append(g_r)

    if not all_gr:
        return r_centers, np.zeros(len(r_centers))

    g_r_mean = np.mean(np.array(all_gr), axis=0)

    return r_centers, g_r_mean
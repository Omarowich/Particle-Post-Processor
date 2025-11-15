import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

from data_reader_csv import read_particle_data_csv


def plot_bond_orientational_order_at_timestep(
    coordDynX, coordDynY, timestep,
    n=6, neighbor_cutoff=3.5, skip=0, normY=1,
    box_size_um=50.0,           # total box width/height in µm
    particle_radius_um=1.0,     # <-- physical particle radius in µm
    edgecolor="black", linewidth=0.35,
    ax=None,
    **kwargs                    # swallow extras like TauB safely
):
    # --- load data (your exact slicing) ---
    X = read_particle_data_csv(coordDynX)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY)[skip::2, 1:][1:, :]

    # choose frame
    if timestep is None:
        timestep = X.shape[0] - 1
    x_coords = X[timestep]
    y_coords = Y[timestep]

    # --- compute local ψ_n magnitudes ---
    positions = np.column_stack((x_coords, y_coords))
    tree = cKDTree(positions)
    neighbor_lists = tree.query_ball_tree(tree, r=neighbor_cutoff)

    psi_n_values = []
    for i, nbrs in enumerate(neighbor_lists):
        nbrs = [j for j in nbrs if j != i]
        if not nbrs:
            psi_n_values.append(0.0)
            continue
        dx = x_coords[nbrs] - x_coords[i]
        dy = y_coords[nbrs] - y_coords[i]
        angles = np.arctan2(dy, dx)
        psi_n = np.sum(np.exp(1j * n * angles)) / len(nbrs)
        psi_n_values.append(np.abs(psi_n))
    psi_n_values = np.asarray(psi_n_values)

    # --- plotting (true circles in data units) ---
    created = False
    if ax is None:
        created = True
        fig, ax = plt.subplots(figsize=(6, 6))

    # set physical axes: center at 0 with half-extent box_size/2
    if normY == 1:
        half = box_size_um / 2.0
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    # build circles (radius in µm) and add as PatchCollection
    patches = [Circle((xi, yi), radius=particle_radius_um) for xi, yi in zip(x_coords, y_coords)]

    cmap = LinearSegmentedColormap.from_list("RedGreen", ["red", "green"])
    coll = PatchCollection(
        patches,
        array=psi_n_values, cmap=cmap,
        edgecolor=edgecolor, linewidth=linewidth,
        antialiased=True
    )
    ax.add_collection(coll)

    # keep limits fixed (adding a collection can auto-rescale)
    if normY == 1:
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)

    # optional: remove per-cell labels for panel use
    ax.set_xlabel("")
    ax.set_ylabel("")

    if created:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\100Tkb_20TauB_ASF\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\100Tkb_20TauB_ASF\datay.csv"


    skip = 0
    timestep= None

    plot_bond_orientational_order_at_timestep(coordDynX, coordDynY, timestep=timestep, n=6, neighbor_cutoff=3.5, skip=1)

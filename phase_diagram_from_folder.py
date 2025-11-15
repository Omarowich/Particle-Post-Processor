import os
import re

import matplotlib.pyplot as plt
import numpy as np

from data_reader_csv import read_particle_data_csv
##########
from plot_bond_orientation_at_time_step import plot_bond_orientational_order_at_timestep


def make_snapshot_phase_panel(
    base_dir,
    tkb_rows,                       # rows, e.g. (1,3,4,5,10,20)
    column_times_s,                 # columns in SECONDS, e.g. (1, 5, 10, 20, 26)
    taub,                 # total simulated time in seconds (e.g. 2*TauB ≈ 27.0)
    x_name="datax.csv",
    y_name="datay.csv",
    figure_size=(12, 14),           # size in inches
    dpi=200,                        # dpi for on-screen display (higher = crisper)
    plot_func=None,                 # e.g. plot_bond_orientational_order_at_timestep
    plot_kwargs=None,               # forwarded to plot_func (skip, normY, etc.)
    show_xy_labels=False            # remove X/Y labels in the panel
):
    """
    Builds a grid (rows=TkB, cols=given times IN SECONDS).
    For each cell we call:
        plot_func(fx, fy, timestep=<frame_idx>, ax=ax, **plot_kwargs)
    `timestep` is a frame index derived from requested time and run length.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if plot_func is None:
        raise ValueError("plot_func must be provided")

    total_time_sec = taub*13.514

    # find Tkb folders
    folder_re = re.compile(r"^([0-9]+(?:\.[0-9]+)?)Tkb_")
    tkb_rows = [float(t) for t in tkb_rows]
    tkb_to_folder = {}
    for fname in sorted(os.listdir(base_dir)):
        m = folder_re.match(fname)
        if m:
            tkb = float(m.group(1))
            if tkb in tkb_rows:
                tkb_to_folder[tkb] = os.path.join(base_dir, fname)

    nrows, ncols = len(tkb_rows), len(column_times_s)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figure_size,
        constrained_layout=True,
        dpi=dpi
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for r, tkb in enumerate(tkb_rows):
        folder = tkb_to_folder.get(tkb)
        for c, t_req in enumerate(column_times_s):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])

            if not folder:
                ax.axis("off")
                continue

            fx = os.path.join(folder, x_name)
            fy = os.path.join(folder, y_name)

            # number of frames after your slicing convention
            skip = int(plot_kwargs.get("skip", 0))
            X = read_particle_data_csv(fx)[skip::2, 1:][1:, :]
            n_frames = X.shape[0]

            # map requested time (seconds) -> nearest frame index
            t_sec = max(0.0, min(float(total_time_sec), float(t_req)))
            dt = float(total_time_sec) / max(n_frames - 1, 1)
            frame_idx = int(round(t_sec / dt))
            frame_idx = max(0, min(n_frames - 1, frame_idx))

            # draw the cell
            plot_func(fx, fy, timestep=frame_idx, ax=ax, **plot_kwargs)

            # strip per-cell X/Y labels if desired
            if not show_xy_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")

            # column titles in SECONDS
            if r == 0:
                ax.set_title(f"{t_sec:.1f} s", fontsize=14)

            # row label: T_kB
            if c == 0:
                ax.set_ylabel(rf"{int(tkb) if tkb.is_integer() else tkb}$T_{{kB}}$",
                              fontsize=14)

    return fig, axes











if __name__ == "__main__":
    base = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF"

    # inside make_snapshot_phase_panel call:
    plot_kwargs = dict(skip=1, normY=1, box_size_um=50.0, particle_radius_um=1.0)

    fig, axes = make_snapshot_phase_panel(
        base_dir=base,
        tkb_rows=(0, 5, 10, 20, 50),
        column_times_s=(1, 5, 10, 20, 26),  # in seconds
        taub=2,
        plot_func=plot_bond_orientational_order_at_timestep,
        plot_kwargs=dict(skip=1, normY=1, box_size_um=50.0, particle_radius_um=1.0),
        dpi=300,  # crisp in PyCharm viewer
        show_xy_labels=False
    )
    plt.show()

    # tkb_rows=(0, 1, 2, 3, 4, 5, 10, 20, 50, 100)

    # average_rg_over_time
    # area_fraction_over_time
    # bond_orientational_order_over_time
    # detect_crystals_over_time
    # particle_distance_over_time
    # median_total_path_distance_over_time
    # particle_displacement_from_inintal_position_over_time
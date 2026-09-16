"""Build the multi-row/multi-column snapshot phase-diagram panel figure
(rows = Tkb, columns = requested times), reused by the Phase diagram mode.

Extracted from app.py (Phase 1 de-spaghetti pass).
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np

from data_reader_csv import read_particle_data_csv


def make_snapshot_phase_panel(
    base_dir,
    tkb_rows,
    column_times_s,
    taub,
    x_name="datax.csv",
    y_name="datay.csv",
    figure_size=(12, 14),
    dpi=200,
    plot_func=None,
    plot_kwargs=None,
    show_xy_labels=False,
    colorbar_cmap=None,   # NEW
    colorbar_label=None,  # NEW
):

    if plot_kwargs is None:
        plot_kwargs = {}
    if plot_func is None:
        raise ValueError("plot_func must be provided")

    total_time_sec = taub * 13.514

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

            skip = int(plot_kwargs.get("skip", 0))
            X = read_particle_data_csv(fx)[skip::2, 1:][1:, :]
            n_frames = X.shape[0]

            t_sec = max(0.0, min(float(total_time_sec), float(t_req)))
            dt = float(total_time_sec) / max(n_frames - 1, 1)
            frame_idx = int(round(t_sec / dt))
            frame_idx = max(0, min(n_frames - 1, frame_idx))

            plot_func(fx, fy, timestep=frame_idx, ax=ax, **plot_kwargs)

            if not show_xy_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")

            if r == 0:
                ax.set_title(f"{t_sec:.1f} s", fontsize=14)

            if c == 0:
                label = rf"{int(tkb) if tkb.is_integer() else tkb}$T_{{kB}}$"
                ax.set_ylabel(label, fontsize=14)


    # --- optional shared colorbar on the side ---
    if colorbar_cmap is not None:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                            cmap=plt.get_cmap(colorbar_cmap))
        sm.set_array([])

        fig.colorbar(
            sm,
            ax=axes.ravel().tolist(),
            label=colorbar_label if colorbar_label else "",
            fraction=0.03,
            pad=0.02,
        )

    return fig, axes

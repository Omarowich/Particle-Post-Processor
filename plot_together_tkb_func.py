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


BROWNIAN_SECONDS = 13.513  # 1 T_kB ≈ 13.513 s


def plots_together(
    base_dir,
    metric_func,
    metric_kwargs,
    plot_title,
    x_name="datax.csv",
    y_name="datay.csv",
    TkBs=(1, 2, 3, 4, 5),
    color_mode="default",      # "default" or "gradient"
    color_start="#000000",     # used only if color_mode == "gradient"
    color_end="#CCCCCC",
    x_axis_mode="seconds",     # "seconds" or "tkb"
):
    """
    Loop over Tkb_* folders under base_dir, call metric_func(fx, fy, **metric_kwargs),
    and plot all curves in one figure.

    metric_func must return either:
        (t, values) or (t, values1, values2)

    color_mode:
      - "default": Matplotlib default color cycle
      - "gradient": colors interpolated between color_start and color_end over TkB

    x_axis_mode:
      - "seconds": use t as returned by metric_func (assumed in seconds)
      - "tkb":     plot t / 13.513, labeled as Brownian time T_kB
    """
    folder_re = re.compile(r"^([0-9]+(?:\.[0-9]+)?)Tkb_")
    TkBs = set(float(t) for t in TkBs)

    curves = []   # store (tkb, values) or (tkb, values1, values2)
    t_refs = []
    max_len = 0

    # --- collect data from all matching Tkb_* folders ---
    for fname in sorted(os.listdir(base_dir)):
        m = folder_re.match(fname)
        if not m:
            continue

        tkb = float(m.group(1))
        if tkb not in TkBs:
            continue

        folder = os.path.join(base_dir, fname)
        fx = os.path.join(folder, x_name)
        fy = os.path.join(folder, y_name)

        if not (os.path.isfile(fx) and os.path.isfile(fy)):
            print(f"Skipping {fname}: missing {x_name} or {y_name}")
            continue

        result = metric_func(fx, fy, **metric_kwargs)

        if len(result) == 2:
            t, values = result
            curves.append((tkb, np.asarray(values)))
        elif len(result) == 3:
            t, values1, values2 = result
            curves.append((tkb, np.asarray(values1), np.asarray(values2)))
        else:
            print(f"Skipping Tkb={tkb}: unexpected number of return values from metric_func")
            continue

        t_refs.append(np.asarray(t))
        max_len = max(max_len, len(t))
        print(f"Collected data for Tkb = {tkb}")

    if not curves:
        print("No data found for the given TkBs.")
        return

    # --- Use the longest time vector as reference ---
    longest_idx = np.argmax([len(t) for t in t_refs])
    t_ref = t_refs[longest_idx]

    # Optionally rescale time axis
    if x_axis_mode.lower() == "tkb":
        x_vals_ref = t_ref / BROWNIAN_SECONDS
        x_label = r"Time $T_{kB}$"
    else:
        x_vals_ref = t_ref
        x_label = "Time (s)"

    # Pad shorter time series with NaNs
    for i, entry in enumerate(curves):
        if len(entry) == 2:
            tkb, values = entry
            padded_values = np.pad(
                values, (0, max_len - len(values)), constant_values=np.nan
            )
            curves[i] = (tkb, padded_values)
        elif len(entry) == 3:
            tkb, values1, values2 = entry
            padded_values1 = np.pad(
                values1, (0, max_len - len(values1)), constant_values=np.nan
            )
            padded_values2 = np.pad(
                values2, (0, max_len - len(values2)), constant_values=np.nan
            )
            curves[i] = (tkb, padded_values1, padded_values2)

    # --- Color handling: default vs custom 2-color gradient over TkB ---
    use_gradient = (color_mode == "gradient")
    if curves and use_gradient:
        all_tkbs = [entry[0] for entry in curves]
        vmin, vmax = min(all_tkbs), max(all_tkbs)
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        start_rgb = np.array(matplotlib.colors.to_rgb(color_start))
        end_rgb = np.array(matplotlib.colors.to_rgb(color_end))
    else:
        norm = None
        start_rgb = None
        end_rgb = None

    plt.figure()

    for entry in sorted(curves, key=lambda e: e[0]):  # sort by TkB
        if len(entry) == 2:
            tkb, values = entry
            if use_gradient and norm is not None:
                alpha = norm(tkb)  # 0..1
                rgb = start_rgb + alpha * (end_rgb - start_rgb)
                color = rgb
            else:
                color = None

            plt.plot(x_vals_ref, values, label=f"Tkb = {tkb}", color=color)

        elif len(entry) == 3:
            tkb, values1, values2 = entry
            if use_gradient and norm is not None:
                alpha = norm(tkb)
                rgb = start_rgb + alpha * (end_rgb - start_rgb)
                color = rgb
                color2 = rgb
            else:
                color = None
                color2 = None

            plt.plot(x_vals_ref, values1, label=f"Tkb = {tkb} – hex", color=color)
            plt.plot(x_vals_ref, values2, label=f"Tkb = {tkb} – non-hex", color=color2)

    plt.xlabel(x_label)
    plt.ylabel(plot_title)
    plt.grid(True)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    # no plt.show(); Streamlit will render via st.pyplot


if __name__ == "__main__":
    base = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF"

    kwargs = dict(
        TauB=2,
        skip=1,
        normY=1
    )

    plots_together(
        base_dir=base,
        x_name='datax.csv',
        y_name='datay.csv',
        metric_kwargs=kwargs,
        metric_func=average_rg_over_time,
        plot_title="Radius of gyration",
        TkBs=(1, 5, 10, 100),
        color_mode="gradient",
        color_start="#000000",
        color_end="#CCCCCC",
        x_axis_mode="seconds",  # or "tkb"
    )

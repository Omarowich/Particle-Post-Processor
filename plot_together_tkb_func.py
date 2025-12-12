import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rg_over_time import average_rg_over_time

BROWNIAN_SECONDS = 13.513  # 1 T_kB ≈ 13.513 s


def plots_together(
    base_dir,
    metric_func,
    metric_kwargs,
    plot_title,
    x_name="datax.csv",
    y_name="datay.csv",
    runs=None,               # list of (folder_path, tkb, tauB)
    color_mode="default",    # "default" or "gradient"
    color_start="#000000",   # used only if color_mode == "gradient"
    color_end="#CCCCCC",
    x_axis_mode="seconds",   # "seconds" or "tkb"
):
    """
    Plot multiple simulation runs in one figure.

    runs: list of tuples (folder_path, tkb, tauB)
          This is passed directly from Streamlit's selection.
    metric_func must return either:
        (t, values) or (t, values1, values2)
    """
    if not runs:
        print("[plots_together] No runs provided; nothing to plot.")
        return

    # Normalise run structure: ensure we always have folder_path, tkb, tau
    norm_runs = []
    for folder_path, tkb, tau in runs:
        norm_runs.append((folder_path, float(tkb), float(tau)))

    curves = []   # entries: (tkb, tau, values) or (tkb, tau, values1, values2)
    t_refs = []
    max_len = 0

    # --- collect data from all selected runs ---
    for folder_path, tkb, tau in norm_runs:
        fx = os.path.join(folder_path, x_name)
        fy = os.path.join(folder_path, y_name)

        if not (os.path.isfile(fx) and os.path.isfile(fy)):
            print(f"Skipping folder {folder_path}: missing {x_name} or {y_name}")
            continue

        result = metric_func(fx, fy, **metric_kwargs)

        if len(result) == 2:
            t, values = result
            curves.append((tkb, tau, np.asarray(values)))
        elif len(result) == 3:
            t, values1, values2 = result
            curves.append((tkb, tau, np.asarray(values1), np.asarray(values2)))
        else:
            print(
                f"Skipping TkB={tkb}, TauB={tau}: "
                f"unexpected number of return values from metric_func"
            )
            continue

        t_arr = np.asarray(t)
        t_refs.append(t_arr)
        max_len = max(max_len, len(t_arr))
        print(f"Collected data for Tkb = {tkb}, TauB = {tau}")

    if not curves:
        print("[plots_together] No curves to plot after reading data.")
        return

    # --- Use the longest time vector as reference ---
    longest_idx = np.argmax([len(t) for t in t_refs])
    t_ref = t_refs[longest_idx]

    # Optionally rescale time axis
    if x_axis_mode.lower() == "tkb":
        x_vals_ref = t_ref / BROWNIAN_SECONDS
        x_label = r"Time $\tau_B$"
    else:
        x_vals_ref = t_ref
        x_label = "Time (s)"

    # Pad shorter time series with NaNs so all lines share x_vals_ref
    for i, entry in enumerate(curves):
        if len(entry) == 3:
            tkb, tau, values = entry
            padded_values = np.pad(
                values, (0, max_len - len(values)), constant_values=np.nan
            )
            curves[i] = (tkb, tau, padded_values)
        elif len(entry) == 4:
            tkb, tau, values1, values2 = entry
            padded_values1 = np.pad(
                values1, (0, max_len - len(values1)), constant_values=np.nan
            )
            padded_values2 = np.pad(
                values2, (0, max_len - len(values2)), constant_values=np.nan
            )
            curves[i] = (tkb, tau, padded_values1, padded_values2)

    # --- Color handling: default vs custom 2-color gradient over TkB ---
    use_gradient = (color_mode == "gradient")
    if curves and use_gradient:
        all_tkbs = [entry[0] for entry in curves]  # tkb is entry[0]
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

    # sort by TkB then TauB for plotting
    # sort by TkB then TauB for plotting
    for entry in sorted(curves, key=lambda e: (e[0], e[1])):
        if len(entry) == 3:
            tkb, tau, values = entry
            if use_gradient and norm is not None:
                alpha = norm(tkb)  # 0..1
                rgb = start_rgb + alpha * (end_rgb - start_rgb)
                color = rgb
            else:
                color = None

            # legend: e.g. "10 T_{kb}"
            label = rf"${tkb:g}\,T_{{kb}}$"
            plt.plot(x_vals_ref, values, label=label, color=color)

        elif len(entry) == 4:
            tkb, tau, values1, values2 = entry
            if use_gradient and norm is not None:
                alpha = norm(tkb)
                rgb = start_rgb + alpha * (end_rgb - start_rgb)
                color = rgb
                color2 = rgb
            else:
                color = None
                color2 = None

            # legends: e.g. "10 T_{kb} – hex", "10 T_{kb} – non-hex"
            base_label = rf"${tkb:g}\,T_{{kb}}$"
            label1 = base_label + " – hex"
            label2 = base_label + " – non-hex"

            plt.plot(x_vals_ref, values1, label=label1, color=color)
            plt.plot(x_vals_ref, values2, label=label2, color=color2)

    plt.xlabel(x_label)
    plt.ylabel(plot_title)
    plt.grid(True)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    # Streamlit will render via st.pyplot

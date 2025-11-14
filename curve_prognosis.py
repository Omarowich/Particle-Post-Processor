import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import statsmodels.api as sm



def analyze_rg_curve(TkB, RG, resolution=500, plot=True, remove_outliers=True):
    TkB = np.array(TkB)
    RG = np.array(RG)


    # Interpolation only within data bounds
    TkB_fine = np.linspace(min(TkB), max(TkB), resolution)
    cs = CubicSpline(TkB, RG, extrapolate=False)
    RG_fine = cs(TkB_fine)


    # Find extrema
    minima_idx = argrelextrema(RG_fine, np.less)[0]
    maxima_idx = argrelextrema(RG_fine, np.greater)[0]
    minima = list(zip(TkB_fine[minima_idx], RG_fine[minima_idx]))
    maxima = list(zip(TkB_fine[maxima_idx], RG_fine[maxima_idx]))

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(TkB, RG, 'o', label='Original Data', color='black')
        plt.plot(TkB_fine, RG_fine, label='Cubic Spline Fit', color='blue')
        if minima:
            plt.scatter(*zip(*minima), color='green', label='Minima')
        if maxima:
            plt.scatter(*zip(*maxima), color='red', label='Maxima')
        plt.xlabel('TkB')
        plt.ylabel('Radius of Gyration (Rg)')
        plt.title('Interpolated Rg vs TkB with Extrema')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':


    values = r"C:\Users\omare\Documents\HIWI - Paper - Proposal\tkbrg.xlsx"

    # Load Excel file, skip first column, drop NaNs
    Tkb, RG = pd.read_excel(values).dropna().values.T

    # # Convert to NumPy and clean
    # Tkb = np.array(Tkb, dtype=np.float64)
    # RG = np.array(RG, dtype=np.float64)
    #
    # # Keep only valid range (to exclude phantom points)
    # valid = (Tkb >= 2.5) & (Tkb <= 5)
    # Tkb = Tkb[valid]
    # RG = RG[valid]
    #
    # # Sort and remove duplicate Tkb entries
    # sort_idx = np.argsort(Tkb)
    # Tkb = Tkb[sort_idx]
    # RG = RG[sort_idx]
    # Tkb, unique_idx = np.unique(Tkb, return_index=True)
    # RG = RG[unique_idx]

    print(Tkb,RG)
    plt.plot(Tkb, RG, 'o', label='Original Data', color='black')
    plt.plot(Tkb, RG, '-')
    plt.show()


    analyze_rg_curve(Tkb, RG)
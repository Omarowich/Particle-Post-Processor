import matplotlib.pyplot as plt
import numpy as np

##########
from movement_change_calculator_time_study import particle_displacement_from_inintal_position_over_time


def fourier_of_metric(coordDynX,
                      coordDynY,
                      metric_func,
                      metric_kwargs=None,
                      window=None,
                      normalize=True,
                      title=None):

    metric_kwargs = metric_kwargs or {}

    # 1) Generate the time-series
    t, y = metric_func(coordDynX, coordDynY, **metric_kwargs)
    t = np.asarray(t)
    y = np.asarray(y)
    N = len(y)

    # 2) Sampling
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    # 3) Windowing
    if window is not None:
        w = window(N)
        yw = y * w
    else:
        yw = y

    # 4) FFT
    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(N, dt)
    amps = np.abs(Y)

    # 5) Normalize & one‐sided scale
    if normalize:
        amps = amps / N
    if N % 2 == 0:
        amps[1:-1] *= 2
    else:
        amps[1:] *= 2

    # 6) Plot
    plt.figure()
    plt.plot(freqs, amps, lw=1.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title or f'FFT of {metric_func.__name__}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return freqs, amps


if __name__ == '__main__':

    # 1) Basic FFT of your radial‐displacement metric
    freqs, amps = fourier_of_metric(
        coordDynX=r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\1Tkb_20TauB_ASF\datax.csv",
        coordDynY=r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\ASF\1Tkb_20TauB_ASF\datay.csv",
        metric_func=particle_displacement_from_inintal_position_over_time,
        metric_kwargs={'TauB': 20, 'skip': 1},
        window=None,
        normalize=True,
        title='Displacement FFT'
    )


    # average_rg_over_time
    # area_fraction_over_time
    # bond_orientational_order_over_time
    # detect_crystals_over_time
    # particle_distance_over_time
    # median_total_path_distance_over_time
    # particle_displacement_from_inintal_position_over_time
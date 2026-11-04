"""Cost per step vs particle number: dense all-pairs (airpods_bd) vs neighbour lists (bd_engine), 2D and 3D.
Fixed area fraction 0.18 (2D) by scaling the box with N; 3D: same footprint, particles in `layers` layers
(volume fraction ~ 0.06 in a 10 um tall box). One process, nothing else running. Writes bench_scaling.csv."""
import sys, time, csv, numpy as np
import airpods_bd as bd, bd_engine as be

CASES = {"LJ": dict(A=3.0, steps=4000), "AO q=0.05": dict(A=0.0, ao=(0.05, 0.182), steps=256000)}
rows = []
def timed(fn, n_steps):
    t0 = time.perf_counter(); fn(); return (time.perf_counter() - t0) / n_steps * 1e3

for name, c in CASES.items():
    for n_side in (12, 24, 36, 48, 72):
        N = n_side ** 2
        bd.BOX = 2.0 * n_side + 1.0                     # 4 um spacing -> same area fraction as 144 in 50 um
        P0 = be.grid_start(n_side, 4.0)
        nst = max(200, int(4e5 / N))
        T = nst / c["steps"]
        kw = dict(steps_per_tauB=c["steps"], out_per_tauB=100 if c["steps"] == 4000 else 1000)
        T = max(T, 1 / kw["out_per_tauB"])
        n_steps = int(round(T * c["steps"]))
        if N <= 1296:
            ms = timed(lambda: bd.simulate(c["A"], 0.0, T, ao=c.get("ao"), seed=1, start=P0, verbose=False, **kw), n_steps)
            rows.append((name, "dense 2D", N, round(ms, 4))); print(rows[-1], flush=True)
        ms = timed(lambda: be.simulate(c["A"], 0.0, T, ao=c.get("ao"), seed=1, start=P0, dim=2, **kw), n_steps)
        rows.append((name, "neighbour list 2D", N, round(ms, 4))); print(rows[-1], flush=True)
        P3 = be.grid_start(n_side, 4.0, dim=3, layers=1)
        ms = timed(lambda: be.simulate(c["A"], 0.0, T, ao=c.get("ao"), seed=1, start=P3, dim=3, drho=52.0, H=10.0, **kw), n_steps)
        rows.append((name, "neighbour list 3D", N, round(ms, 4))); print(rows[-1], flush=True)
bd.BOX = 25.0
with open(sys.argv[1] if len(sys.argv) > 1 else "bench_scaling.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["potential", "method", "N", "ms_per_step"]); w.writerows(rows)

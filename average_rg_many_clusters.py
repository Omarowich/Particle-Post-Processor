import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

from data_reader_csv import read_particle_data_csv


def cluster_rg_over_time(
    coordDynX_path,
    coordDynY_path,
    *,
    # ----- clustering cutoff -----
    alpha=None, R=None,         # fixed cutoff: cutoff = alpha * R (if BOTH provided)
    k=1.5,                      # else adaptive: cutoff = k × median NN distance
    min_size=3,                 # ignore clusters smaller than this
    # ----- aggregation/plot -----
    aggregate="median",       # "weighted" | "mean" | "median" | None
    TauB=2,
    skip=1,
    smooth_window=1,            # >=2 -> rolling median on aggregate; 1 disables
    plot=True,
    show_per_cluster=False,      # scatter of each cluster's Rg over time
    normY=False,                # keep legacy y-lims if needed
):
    """
    Compute *per-cluster* radius of gyration (Rg) over time and an aggregate across clusters.

    Each timestep:
      1) Build a proximity graph with distance cutoff.
         - Fixed: cutoff = alpha * R (if both given).
         - Adaptive: cutoff = k * median nearest-neighbor distance.
      2) Get connected components (clusters); drop clusters with size < min_size.
      3) Compute Rg for every cluster, collect sizes.
      4) Aggregate across clusters (optional).

    Returns
    -------
    time_axis : (T,)
    agg_rg    : (T,) aggregate Rg (or np.nan if no clusters that step, or None if aggregate=None)
    per_cluster_rgs : list of lists; per_cluster_rgs[t] = list of Rg values for clusters at t
    per_cluster_sizes : list of lists; per_cluster_sizes[t] = list of sizes for clusters at t
    """
    # ---------- helpers ----------
    def _rg(points):
        if points.size == 0:
            return np.nan
        com = points.mean(axis=0)
        return np.sqrt(np.mean(np.sum((points - com) ** 2, axis=1)))

    def _median_nn_distance(positions):
        n = positions.shape[0]
        if n < 2:
            return np.nan
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(positions)
        dists, _ = nbrs.kneighbors(positions)
        return np.median(dists[:, 1])

    def _clusters_by_cutoff(positions, cutoff, min_size):
        if positions.shape[0] == 0 or not np.isfinite(cutoff) or cutoff <= 0:
            return []
        G = radius_neighbors_graph(positions, radius=float(cutoff),
                                   mode="connectivity", include_self=False)
        n_comp, labels = connected_components(G, directed=False)
        if n_comp == 0:
            return []
        sizes = np.bincount(labels, minlength=n_comp)
        valid_labels = [lbl for lbl, s in enumerate(sizes) if s >= min_size]
        return [np.where(labels == lbl)[0] for lbl in valid_labels]

    # ---------- load ----------
    X = read_particle_data_csv(coordDynX_path)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY_path)[skip::2, 1:][1:, :]
    T = len(X)
    time_axis = np.linspace(0, TauB * 13.513, T) if T > 0 else np.array([])

    per_cluster_rgs = []
    per_cluster_sizes = []
    agg_rg = np.full(T, np.nan, dtype=float) if aggregate else None

    fixed_cutoff = (alpha is not None) and (R is not None)

    # ---------- main loop ----------
    for t in range(T):
        pos = np.column_stack((X[t], Y[t]))
        cutoff = (float(alpha) * float(R)) if fixed_cutoff else (k * _median_nn_distance(pos))

        clusters = _clusters_by_cutoff(pos, cutoff, min_size=min_size)

        rg_list = []
        size_list = []
        for idx in clusters:
            pts = pos[idx]
            rg_list.append(_rg(pts))
            size_list.append(len(idx))

        per_cluster_rgs.append(rg_list)
        per_cluster_sizes.append(size_list)

        if aggregate:
            if len(rg_list) == 0:
                agg_rg[t] = np.nan
            elif aggregate == "weighted":
                w = np.asarray(size_list, float)
                r = np.asarray(rg_list, float)
                agg_rg[t] = np.nansum(w * r) / np.nansum(w)
            elif aggregate == "mean":
                agg_rg[t] = np.nanmean(np.asarray(rg_list, float))
            elif aggregate == "median":
                agg_rg[t] = np.nanmedian(np.asarray(rg_list, float))
            else:
                raise ValueError("aggregate must be 'weighted', 'mean', 'median', or None")

    # ---------- optional smoothing on aggregate ----------
    if aggregate and smooth_window and smooth_window > 1:
        w = int(smooth_window)
        pad = w // 2
        padded = np.pad(agg_rg, (pad, pad), constant_values=np.nan)
        smoothed = np.empty_like(agg_rg)
        for i in range(T):
            smoothed[i] = np.nanmedian(padded[i:i+w])
        agg_rg = smoothed

    # ---------- plotting ----------
    if plot and T > 0:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        if show_per_cluster:
            # scatter all clusters; add slight x jitter so overlapping points are visible
            jitter = (time_axis[1] - time_axis[0]) * 0.05 if T > 1 else 0.0
            xs = []
            ys = []
            for tt, rgs in enumerate(per_cluster_rgs):
                if not rgs:
                    continue
                tt_arr = np.full(len(rgs), time_axis[tt])
                if jitter:
                    tt_arr = tt_arr + np.random.uniform(-jitter, jitter, size=len(rgs))
                xs.extend(tt_arr.tolist())
                ys.extend(rgs)
            if xs:
                ax.scatter(xs, ys, s=12, alpha=0.45, label="per-cluster Rg")

        if aggregate:
            ax.plot(time_axis, agg_rg, lw=2, label=f'aggregate ({aggregate})')

        title_cutoff = (f"fixed cutoff = {alpha}·R"
                        if fixed_cutoff else f"adaptive cutoff = {k}× median NN dist")
        ax.set_title(f'Per-cluster Rg over time ({title_cutoff})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Rg (radius of gyration)')
        ax.grid(True)
        if normY:
            ax.set_ylim(14, 21)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

    return time_axis, agg_rg, per_cluster_rgs, per_cluster_sizes

def dbscan_avg_cluster_rg_over_time(
    coordDynX_path: str,
    coordDynY_path: str,
    *,
    eps: float = 1.1,           # DBSCAN neighborhood radius in your coordinate units
    min_samples: int = 3,       # min points for a cluster
    TauB: float = 2.5,          # for building the time axis (scale factor)
    skip: int = 1,              # keep your current decimation pattern
    plot: bool = True
):
    """
    DBSCAN per-timestep clustering -> Rg per cluster -> average of cluster Rg's -> time plot.

    Returns
    -------
    time_axis : (T,) np.ndarray
    avg_rg    : (T,) np.ndarray, mean Rg of clusters each timestep (NaN if no clusters)
    per_cluster_rgs : list[list[float]], per_cluster_rgs[t] = list of Rg values for clusters at t
    n_clusters_over_time : (T,) np.ndarray, number of clusters (labels >= 0) per timestep
    """

    # ---------- helpers ----------
    def _rg(points: np.ndarray) -> float:
        """Radius of gyration for a set of 2D points (N x 2)."""
        if points.shape[0] == 0:
            return np.nan
        com = points.mean(axis=0)
        return np.sqrt(np.mean(np.sum((points - com) ** 2, axis=1)))

    # ---------- load data ----------
    # Keep your established slicing pattern to match other scripts
    X = read_particle_data_csv(coordDynX_path)[skip::2, 1:][1:, :]
    Y = read_particle_data_csv(coordDynY_path)[skip::2, 1:][1:, :]

    T = len(X)
    if T == 0:
        return np.array([]), np.array([]), [], np.array([])

    time_axis = np.linspace(0, TauB * 13.513, T)

    # ---------- main loop ----------
    avg_rg = np.full(T, np.nan, dtype=float)
    n_clusters_over_time = np.zeros(T, dtype=int)
    per_cluster_rgs = []

    for t in range(T):
        pos = np.column_stack((X[t], Y[t]))
        if pos.shape[0] == 0:
            per_cluster_rgs.append([])
            continue

        # DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree")
        labels = db.fit_predict(pos)

        # clusters are labels >= 0
        cluster_ids = np.unique(labels[labels >= 0])
        n_clusters_over_time[t] = len(cluster_ids)

        rgs = []
        for cid in cluster_ids:
            pts = pos[labels == cid]
            rgs.append(_rg(pts))

        per_cluster_rgs.append(rgs)

        # simple mean of cluster Rg's (your request)
        if len(rgs) > 0:
            avg_rg[t] = float(np.nanmean(rgs))
        else:
            avg_rg[t] = np.nan

    # ---------- plot ----------
    if plot:
        plt.figure(figsize=(9, 5.2))
        plt.plot(time_axis, avg_rg, lw=2, label="Mean Rg across clusters")
        plt.xlabel("Time")
        plt.ylabel("Mean cluster Rg")
        plt.title(f"DBSCAN clusters (eps={eps}, min_samples={min_samples}) — mean Rg vs time")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    return time_axis, avg_rg, per_cluster_rgs, n_clusters_over_time

if __name__ == '__main__':
    coordDynX = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\10Tkb_2TauB\datax.csv"
    coordDynY = r"\\nas.ads.mwn.de\tuei\mml\MML MS BS students\Bachelor Students\Omar Elsabbagh\HIWI\NAF\10Tkb_2TauB\datax.csv"

    TauB = 2
    skip = 1

    eps = 1.5


    #cluster_rg_over_time(coordDynX, coordDynY, alpha=eps, TauB=TauB, skip=skip)

    dbscan_avg_cluster_rg_over_time(coordDynX, coordDynY, eps=eps, TauB=TauB, skip=skip)

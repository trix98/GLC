import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks, peak_widths
from importlib import resources
from . import GGM, FeatDicts


class EdgeDf:
    """Generate and filter an edge DataFrame from a Gaussian Graphical Model (GGM).

    When instantiated:
      1. Computes the raw ``edge_df_raw`` from the adjacency matrix.
      2. Applies filtering based on ``rt_delta`` and whether to keep only positive partial correlations.
    """

    def __init__(
        self,
        ggm: GGM,
        feat_dicts: FeatDicts,
        rt_delta: float = 2.0,
        only_pos_pcor: bool = False
    ) -> None:
        """Initialize EdgeDf.

        Args:
            ggm: Gaussian Graphical Model object containing adjacency matrix and feature labels.
            feat_dicts: Object containing feature metadata (m/z and RT lookups).
            rt_delta: Maximum RT difference to keep an edge. IMPORTANT: Note that this will be half the interval either side e.g. for 'rt_delta=2' means one second before and one after. 
            only_pos_pcor: If True, only keep edges with positive partial correlations.
        """
        self.ggm = ggm
        self.feat_dicts = feat_dicts
        self.rt_delta = rt_delta
        self.only_pos_pcor = only_pos_pcor

        self.edge_df_raw = self._build_edge_df()
        self.edge_df_filtered = self._filter_edge_df()

    @staticmethod
    def _get_edges_arr(adj_mx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract upper-triangular edge indices and values.

        Args:
            adj_mx: Square adjacency matrix.

        Returns:
            Tuple of:
                - indices of upper-triangular entries (rows, cols)
                - corresponding partial correlation values
        """
        triu_indices = np.triu_indices(adj_mx.shape[0], k=1)
        triu_values = adj_mx[triu_indices]
        return triu_indices, triu_values

    def _build_edge_df(self) -> pd.DataFrame:
        """Construct raw edge DataFrame containing mz/rt deltas and partial correlations.

        Returns:
            Raw edge DataFrame.
        """
        ggm = self.ggm
        feat_dicts = self.feat_dicts

        triu_indices, triu_values = self._get_edges_arr(ggm.adj_mx)
        row_idx, col_idx = triu_indices

        feat_mz = [feat_dicts.mz[f] for f in ggm.feat_labels]
        feat_rt = [feat_dicts.rt[f] for f in ggm.feat_labels]

        results = []

        for r, c, pcor in zip(row_idx, col_idx, triu_values):
            if pcor == 0:
                continue

            mz_delta_raw = feat_mz[r] - feat_mz[c]
            rt_delta = abs(feat_rt[r] - feat_rt[c])

            if mz_delta_raw < 0:
                out_idx, in_idx = r, c
            else:
                out_idx, in_idx = c, r

            results.append(
                {
                    "feature_out": ggm.feat_labels[out_idx],
                    "feature_in": ggm.feat_labels[in_idx],
                    "mz_delta": abs(mz_delta_raw),
                    "rt_delta": rt_delta,
                    "out_mz": feat_mz[out_idx],
                    "in_mz": feat_mz[in_idx],
                    "out_rt": feat_rt[out_idx],
                    "in_rt": feat_rt[in_idx],
                    "pcor": pcor,
                }
            )

        return pd.DataFrame(results)

    def _filter_edge_df(self) -> pd.DataFrame:
        """Apply filtering to `edge_df_raw` based on RT threshold and pcor sign.

        Returns:
            Filtered edge DataFrame.
        """
        df = self.edge_df_raw
        mask = df["rt_delta"] <= self.rt_delta

        if self.only_pos_pcor:
            mask &= df["pcor"] > 0

        return df[mask].reset_index(drop=True)


class PickPeaks(EdgeDf):
    """Peak-picking on mass difference distributions derived from GGM edges."""

    def __init__(
        self,
        ggm: GGM,
        feat_dicts: FeatDicts,
        rt_delta: float = 2.0,
        only_pos_pcor: bool = False,
        bw: float = 0.005,
        height_thresh: Optional[float] = None
    ) -> None:
        """Initialize PickPeaks.

        Args:
            ggm: Gaussian Graphical Model.
            feat_dicts: Feature metadata object.
            rt_delta: Maximum RT delta allowed.
            only_pos_pcor: If True, keep only positive partial correlations.
            bw: KDE bandwidth.
            height_thresh: Peak height threshold; if None, set automatically.
        """
        super().__init__(ggm, feat_dicts, rt_delta, only_pos_pcor)
        self.bw = bw
        self.mz = self.edge_df_filtered["mz_delta"].sort_values().values.reshape(-1, 1)

        if height_thresh is None:
            self.height_thresh = np.median(np.exp(self.log_density)) * 1.5
        else:
            self.height_thresh = height_thresh

        self.peaks, self.peak_heights, self.fwhms = self._pick()

    def _fit_kde(self) -> KernelDensity:
        """Fit KDE to m/z deltas.

        Returns:
            Fitted `KernelDensity` object.
        """
        kde = KernelDensity(bandwidth=self.bw, kernel="gaussian")
        kde.fit(self.mz)
        return kde

    @property
    def log_density(self) -> np.ndarray:
        """Compute log-density of m/z deltas.

        Returns:
            Log density values.
        """
        kde = self._fit_kde()
        return kde.score_samples(self.mz)

    def _find_peaks(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks in KDE density.

        Returns:
            Tuple of:
                - peak indices
                - peak heights
        """
        peaks, info = find_peaks(np.exp(self.log_density), height=self.height_thresh)
        return peaks, info["peak_heights"]

    def _fwhm(self, peaks: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Compute full width at half maximum (FWHM).

        Args:
            peaks: Array of mz difference peak indices.

        Returns:
            Peak width metrics from ``scipy.signal.peak_widths``.
        """
        return peak_widths(np.exp(self.log_density), peaks, rel_height=0.5)

    def _pick(self) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
        """Pick peaks and compute heights and FWHM.

        Returns:
            peaks, peak_heights, fwhms
        """
        peaks, peak_heights = self._find_peaks()
        fwhms = self._fwhm(peaks)
        return peaks, peak_heights, fwhms

    def plot(
        self,
        ymax: float = 1,
        xmin: float = 0,
        xmax: Optional[float] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot KDE density.

        Args:
            ymax: Max y-axis value.
            xmin: Min x-axis.
            xmax: Max x-axis; if None, uses max of m/z delta.
            ax: Optional matplotlib axis.
        """
        if xmax is None:
            xmax = np.max(self.mz.squeeze())

        if ax is None:
            _, ax = plt.subplots()

        ax.fill_between(self.mz.squeeze(), np.exp(self.log_density))
        ax.set_xlabel("m/z delta")
        ax.set_ylabel("Density")
        ax.set_ylim([0, ymax])
        ax.set_xlim([xmin, xmax])

    def plot_detailed(
        self,
        ymax: float = 1,
        xmin: float = 0,
        xmax: Optional[float] = None,
        ax: Optional[plt.Axes] = None
    ) -> None:
        """Plot detailed KDE density with peaks and FWHM markers.

        Args:
            ymax: Max y-axis value.
            xmin: Min x-axis.
            xmax: Max x-axis; if None, uses max m/z delta.
            ax: Optional matplotlib axis.
        """
        mz = self.mz.squeeze()
        if xmax is None:
            xmax = np.max(mz)

        if ax is None:
            _, ax = plt.subplots()

        ax.fill_between(mz, np.exp(self.log_density), alpha=0.2, color="grey")
        ax.set_xlabel("m/z delta")
        ax.set_ylabel("Density")
        ax.set_ylim([0, ymax])
        ax.set_xlim([xmin, xmax])

        mz_subset = mz[(mz >= xmin) & (mz < xmax)]
        indices = np.where((mz >= xmin) & (mz < xmax))
        log_subset = self.log_density[indices]

        peaks = [idx for idx, mz_idx in enumerate(self.peaks) if mz[mz_idx] in mz_subset]

        left_mz = self.fwhms[2][peaks]
        right_mz = self.fwhms[3][peaks]

        fwhm_indices = [int(x) for pair in zip(left_mz, right_mz) for x in pair]
        fwhm_mz = mz[fwhm_indices]
        fwhm_height = self.fwhms[1][peaks]

        ax.vlines(
            fwhm_mz,
            ymin=0,
            ymax=[item for item in fwhm_height for _ in range(2)],
            colors="r",
            linestyles="dashed",
            alpha=0.35,
        )

        sns.scatterplot(x=mz_subset, y=np.exp(log_subset), ax=ax, marker="x", legend=False)

    def get_peak_df(self, n_thresh: int = 20) -> pd.DataFrame:
        """Return a DataFrame of peaks and metadata.

        Args:
            n_thresh: Minimum number of edges required to keep a peak.

        Returns:
            DataFrame describing peaks.
        """
        left_indices = np.ceil(self.fwhms[2]).astype(int)
        right_indices = np.floor(self.fwhms[3]).astype(int)
        n_samples = (right_indices - left_indices + 1).astype(int)

        results = []
        for n, mz_idx in enumerate(self.peaks):
            if n_samples[n] >= n_thresh:
                mzmin = self.mz[left_indices[n]]
                mzmax = self.mz[right_indices[n]]
                fwhm = mzmax - self.mz[mz_idx]
                sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

                results.append(
                    {
                        "mz": self.mz[mz_idx][0],
                        "n_edges": n_samples[n],
                        "width": (mzmax - mzmin)[0],
                        "mzmin": mzmin[0],
                        "mzmax": mzmax[0],
                        "sigma": sigma[0],
                        "density": self.peak_heights[n],
                    }
                )

        df = pd.DataFrame(results)
        df["mz_peak_id"] = df.index
        return df
    
    def classify_edges(self) -> pd.DataFrame:
        """Classify edges by assigning them to detected m/z-difference peaks.

        Uses:
          * Peak intervals from :meth:`get_peak_df`
          * Edge information from :attr:`edge_df_filtered`

        Returns:
            pandas.DataFrame: A copy of ``edge_df_filtered`` with an added``mz_peak_id`` column indicating the matched peak. Edges that do not fall inside any peak interval are discarded.
        """
        # Get peak metadata produced by get_peak_df()
        mz_peak_df = self.get_peak_df()
        edge_df = self.edge_df_filtered.copy()

        if mz_peak_df.empty or edge_df.empty:
            edge_df["mz_peak_id"] = np.nan
            return edge_df

        mz_peak_df = mz_peak_df.sort_values(by="mz")

        # Create interval index
        intervals = pd.IntervalIndex.from_arrays(
            mz_peak_df["mzmin"], mz_peak_df["mzmax"], closed="both"
        )

        # Map interval position → peak_id
        idx_map = {
            idx: int(peak_id) for idx, peak_id in enumerate(mz_peak_df["mz_peak_id"])
        }

        edge_peak_ids: List[Optional[int]] = []

        for mz_delta in edge_df["mz_delta"]:
            try:
                loc = intervals.get_loc(mz_delta)

                # Case 1: A single interval index (np.int64)
                if isinstance(loc, np.int64):
                    peak_id = idx_map[int(loc)]

                # Case 2: A slice (overlapping intervals) → choose closest
                elif isinstance(loc, slice):
                    indices = range(*loc.indices(len(intervals)))
                    closest_interval_idx = min(
                        indices,
                        key=lambda i: min(
                            abs(intervals[i].left - mz_delta),
                            abs(intervals[i].right - mz_delta),
                        ),
                    )
                    peak_id = idx_map[closest_interval_idx]

            except KeyError:
                # No interval matched
                peak_id = None

            edge_peak_ids.append(peak_id)

        # Add classification results
        edge_df["mz_peak_id"] = edge_peak_ids

        # Keep only classified edges
        edge_df = edge_df.dropna(subset=["mz_peak_id"])
        edge_df["mz_peak_id"] = edge_df["mz_peak_id"].astype(int)

        return edge_df


class MzPeakLookup:
    """
    A class for matching observed m/z peak intervals to m/z differences that were observed 
    in the paper by Nash et al. https://doi.org/10.1021/acs.analchem.4c00966
    These are 271 m/z differences frequently observed in ESI-MS data from 142 studies using 
    a very similar method. 
    """
    def __init__(self):
        from . import data
        self._data = data

        # Load Dunn rule definitions
        filename = 'esi_mass_differences.parquet'
        self.rule_df = pd.read_parquet(resources.files(self._data) / filename)

    def _find_matching_intervals(
        self, mz_theoretical: float, intervals: pd.IntervalIndex
    ) -> List[pd.Interval]:
        """Find intervals that contain a given theoretical m/z.

        Args:
            mz_theoretical (float): The m/z value to search for.
            intervals (pd.IntervalIndex): IntervalIndex representing peak bounds.

        Returns:
            List[pd.Interval]: Intervals that contain the value.
        """
        return intervals[intervals.contains(mz_theoretical)]

    def _match_intervals(
        self, mz_peak_df: pd.DataFrame, rule_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Match m/z peak intervals to paper m/z differences.

        Args:
            mz_peak_df (pd.DataFrame):
                DataFrame containing mz peak data with columns:
                - 'mzmin', 'mzmax', 'mz_peak_id', 'mz'
            rule_df (pd.DataFrame):
                Dunn rule DataFrame with column:
                - 'm/z difference (experimental)'

        Returns:
            pd.DataFrame: A merged DataFrame containing matched intervals.
        """
        intervals = pd.IntervalIndex.from_arrays(
            mz_peak_df["mzmin"], mz_peak_df["mzmax"], closed="both"
        )

        mz_peak_df = mz_peak_df.copy()
        mz_peak_df["interval"] = intervals

        rule_df = rule_df.copy()
        rule_df["matching_intervals"] = rule_df[
            "m/z difference (experimental)"
        ].astype(float).apply(lambda x: list(self._find_matching_intervals(x, intervals)))

        rule_df["n_matches"] = rule_df["matching_intervals"].apply(len)
        rule_df = rule_df.explode("matching_intervals")

        merged = pd.merge(
            rule_df,
            mz_peak_df,
            left_on="matching_intervals",
            right_on="interval",
            how="left"

        )
        return merged

    def identify_mz_diffs(
        self, mz_peak_df: pd.DataFrame) -> pd.DataFrame:
        """Identify m/z differences between observed m/z peaks and the paper

        Args:
            mz_peak_df (pd.DataFrame):
                Peak dataframe from PickPeaks with columns:
                - 'mzmin', 'mzmax', 'mz_peak_id', 'mz', 'width'

        Returns:
            pd.DataFrame:
                A DataFrame containing matched differences with columns:
                ['Annotation', 'Annotation class', 
                 'm/z difference (theoretical)', 'm/z difference (experimental)',
                 'n_edges', 'dunn_rank', 'mz_peak_id', 'mz_centre',
                 'mzmin', 'mzmax', 'width']
        """
        df = self._match_intervals(mz_peak_df, self.rule_df)

        df["paper_rank"] = df["Rank"]
        df["mz_centre"] = df["mz"]

        keep_cols = [
            "mz_peak_id",
            "Annotation",
            "Annotation class",
            "m/z difference (theoretical)",
            "mz_centre", 
            "mzmin", 
            "mzmax", 
            "width",
            'paper_rank', 
            'n_edges'  
        ]

        df = df[keep_cols]
        return (
            df.dropna(subset=["mz_peak_id"])
            .sort_values(by="n_edges", ascending=False)
            .reset_index(drop=True)
        )
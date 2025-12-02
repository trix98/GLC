import pandas as pd
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple, Set, Optional, Any

from molmass import Formula

from itertools import combinations, product
from collections import defaultdict, Counter


@dataclass
class ESISpecies:
    """Represents an ESI chemical species.

    Attributes:
        name (str): The species name or formula.
        mass (float): The monoisotopic mass of the species.
    """
    name: str
    mass: float


@dataclass
class ESIData:
    """Base class for handling ESI species with masses and mode-specific adjustments.

    Attributes:
        ion_mode (str): Ionization mode, "pos" or "neg".
        df (pd.DataFrame): Table defining species by formula and type codes. e.g. for [M+H]+ the row would be {'formula': 'H', 'type_code': 'charged'}
        CA_species (List[ESISpecies]): Charged adduct species.
        ISF_species (List[ESISpecies]): In-source fragment species.
        NA_species (List[ESISpecies]): Neutral adduct species.

    Note that type code must be one of 'charged', 'isf', or 'neutral_adduct'
    """
    ion_mode: str
    df: pd.DataFrame
    CA_species: List[ESISpecies] = field(default_factory=list)
    ISF_species: List[ESISpecies] = field(default_factory=list)
    NA_species: List[ESISpecies] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initializes species lists and attaches species as attributes."""
        for _, row in self.df.iterrows():
            formula: str = row["formula"]
            type_code: str = row["type_code"]

            mass: float = Formula(formula).isotope.mass
            if formula == "H":
                mass = mass if self.ion_mode == "pos" else -mass

            species = ESISpecies(name=formula, mass=mass)
            setattr(self, formula, species)

            if type_code == "charged":
                self.CA_species.append(species)
            elif type_code == "isf":
                self.ISF_species.append(species)
            elif type_code == "neutral_adduct":
                self.NA_species.append(species)


class ESISearchPatterns(ESIData):
    """Generates all valid ESI-related theoretical mass-difference patterns."""

    def __init__(self, ion_mode: str, species_df: pd.DataFrame) -> None:
        """
        Args:
            ion_mode (str): Ionization mode, "pos" or "neg".
            species_df (pd.DataFrame): Species definition table.
        """
        super().__init__(ion_mode, species_df)
        self.rule_df: pd.DataFrame = self.get_rule_df()

    def _single(self, type_code: str) -> pd.DataFrame:
        """Computes theoretical m/z values for single ISF or NA species.

        Args:
            type_code (str): Either "isf" or "neutral_adduct".

        Returns:
            pd.DataFrame: Table with (adduct1, adduct2, mz_theoretical).
        """
        if type_code == "isf":
            masses = {s.name: s.mass for s in self.ISF_species}
        else:
            masses = {s.name: s.mass for s in self.NA_species}

        rows = [[name, np.nan, mass] for name, mass in masses.items()]
        return pd.DataFrame(rows, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _charged_cross_charged(self) -> pd.DataFrame:
        """All pairwise charged–charged differences."""
        masses = {s.name: s.mass for s in self.CA_species}
        data = [
            [a, b, abs(masses[a] - masses[b])]
            for a, b in combinations(masses, 2)
        ]
        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _isf_cross_charged(self) -> pd.DataFrame:
        """ISF + charged combinations."""
        isf = {s.name: s.mass for s in self.ISF_species}
        ca = {s.name: s.mass for s in self.CA_species if s.name != "H"}

        data = []
        for i, c in product(isf, ca):
            diff = abs((isf[i] - self.H.mass) + ca[c])
            data.append([i, c, diff])

        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _na_cross_charged(self) -> pd.DataFrame:
        """Neutral adduct + charged combinations."""
        na = {s.name: s.mass for s in self.NA_species}
        ca = {s.name: s.mass for s in self.CA_species if s.name != "H"}

        data = []
        for n, c in product(na, ca):
            diff = abs((-na[n] - self.H.mass) - (-ca[c]))
            data.append([n, c, diff])

        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _isf_cross_na(self) -> pd.DataFrame:
        """ISF + Neutral adduct combinations."""
        isf = {s.name: s.mass for s in self.ISF_species}
        na = {s.name: s.mass for s in self.NA_species}

        data = []
        for i, n in product(isf, na):
            diff = abs(isf[i] + na[n])
            data.append([i, n, diff])

        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _isf_cross_isf(self) -> pd.DataFrame:
        """ISF-ISF pairwise differences."""
        isf = {s.name: s.mass for s in self.ISF_species}
        data = [
            [a, b, abs(isf[a] - isf[b])]
            for a, b in combinations(isf, 2)
        ]
        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def _na_cross_na(self) -> pd.DataFrame:
        """Neutral adduct-neutral adduct pairwise differences."""
        na = {s.name: s.mass for s in self.NA_species}
        data = [
            [a, b, abs(na[a] - na[b])]
            for a, b in combinations(na, 2)
        ]
        return pd.DataFrame(data, columns=["adduct1", "adduct2", "mz_theoretical"])

    def get_rule_df(self) -> pd.DataFrame:
        """Generates a concatenated table of all valid ESI mass-difference rules.

        Returns:
            pd.DataFrame: Combined rule table.
        """
        dfs = [
            self._single("isf"),
            self._single("neutral_adduct"),
            self._charged_cross_charged(),
            self._isf_cross_charged(),
            self._na_cross_charged(),
            self._isf_cross_na(),
            self._isf_cross_isf(),
            self._na_cross_na(),
        ]
        return pd.concat(dfs, axis=0).reset_index(drop=True)


class FeatureESI(ESISearchPatterns):
    """Links ESI species to observed m/z peak differences.

    This impportance of using the m/z peaks is from the min and max of the peak we have a defined interval to search later on. 

    Attributes:
        mz_peak_df (pd.DataFrame): Peak table with `mzmin`/`mzmax`. Should correspond to the output of ```glc.PickPeaks.get_peak_df()```
        edges_clf_df (pd.DataFrame): Feature-peak mapping table. Should correspond to the output of ```glc.PickPeaks.classify_edges()```
        feat2mz_peak (Dict[str, Set[int]]): Mapping of feature - associated peak IDs.
    """

    def __init__(
        self,
        ion_mode: str,
        species_df: pd.DataFrame,
        mz_peak_df: pd.DataFrame,
        edges_clf_df: pd.DataFrame,
    ) -> None:
        super().__init__(ion_mode, species_df)

        self.mz_peak_df = mz_peak_df
        self.edges_clf_df = edges_clf_df

        self.feat2mz_peak: Dict[str, Set[int]] = self._process_edges_clf()
        self.df: pd.DataFrame = self._harmonise_tables()

        self.mz_peak_set: Set[int] = set(self.df["mz_peak_id"])
        self.mzpeak2adduct: Dict[int, List[str]] = {
            row["mz_peak_id"]: [row["adduct1"], row["adduct2"]]
            for _, row in self.df.iterrows()
        }

    def find_matching_intervals(
        self, mz_theoretical: float, intervals: pd.IntervalIndex
    ) -> pd.IntervalIndex:
        """Finds all interval bins that contain a theoretical m/z.

        Args:
            mz_theoretical (float): The theoretical mass difference.
            intervals (pd.IntervalIndex): Mass-window intervals.

        Returns:
            pd.IntervalIndex: The intervals containing the mz value.
        """
        return intervals[intervals.contains(mz_theoretical)]

    def _harmonise_tables(self) -> pd.DataFrame:
        """Matches theoretical m/z rules to observed m/z peaks.

        Returns:
            pd.DataFrame: Merged rule-peak table.
        """
        peak_df = self.mz_peak_df.copy()
        rule_df = self.rule_df.copy()

        intervals = pd.IntervalIndex.from_arrays(
            peak_df["mzmin"], peak_df["mzmax"], closed="both"
        )
        peak_df["interval"] = intervals

        rule_df["matching_intervals"] = rule_df["mz_theoretical"].apply(
            lambda mz: list(self.find_matching_intervals(mz, intervals))
        )

        rule_df = rule_df.explode("matching_intervals")

        merged = pd.merge(
            rule_df,
            peak_df,
            left_on="matching_intervals",
            right_on="interval",
            how="left",
        )

        merged = merged[
            ["adduct1", "adduct2", "mz_theoretical", "mz_peak_id", "mzmin", "mzmax"]
        ]
        return merged.dropna(subset=["mz_peak_id"])

    def _process_edges_clf(self) -> Dict[str, Set[int]]:
        """Builds feature - mz_peak_id mapping.

        Returns:
            Dict[str, Set[int]]: Feature to associated peak IDs.
        """
        mapping: Dict[str, Set[int]] = defaultdict(set)

        for f, pid in zip(self.edges_clf_df["feature_in"], self.edges_clf_df["mz_peak_id"]):
            mapping[f].add(pid)
        for f, pid in zip(self.edges_clf_df["feature_out"], self.edges_clf_df["mz_peak_id"]):
            mapping[f].add(pid)

        return mapping

    def esi_search_feature(
        self, feature: str, count_thresh: int = 1
    ) -> List[str]:
        """Finds ESI types associated with a given feature.

        Args:
            feature (str): Feature identifier.
            count_thresh (int): Minimum required count for a peak.

        Returns:
            List[str]: Unique list of ESI adduct / fragment types.
        """
        counter = Counter(self.feat2mz_peak[feature])

        counter = {k: v for k, v in counter.items() if v >= count_thresh and k in self.mz_peak_set}

        esi_types = []
        for peak_id in counter.keys():
            esi_types.extend([a for a in self.mzpeak2adduct[peak_id] if a is not np.nan])

        return list(set(esi_types))


class FeatMapper(FeatureESI):
    """Maps features to possible neutral masses based on ESI species rules and ppm tolerances.

    The method ```compute_multiple``` populates the `results` attribute with mappings for each feature.

    Attributes:
        feat_dicts (glc.FeatDicts): Holds m/z values and metadata.
        ppm_interval (int): PPM window for mass tolerance.
        results (Dict[str, Dict[str, Dict[str, Any]]]): Output mass mapping.
    """

    def __init__(
        self,
        ion_mode: str,
        species_df: pd.DataFrame,
        mz_peak_df: pd.DataFrame,
        edges_clf_df: pd.DataFrame,
        feat_dicts: Any,
        ppm_interval: int = 10,
    ) -> None:
        super().__init__(ion_mode, species_df, mz_peak_df, edges_clf_df)

        self.feat_dicts = feat_dicts
        self.ppm_interval = ppm_interval

        self.ca_list: Set[str] = {s.name for s in self.CA_species if s.name != "H"}
        self.isf_list: List[str] = [s.name for s in self.ISF_species]
        self.na_list: List[str] = [s.name for s in self.NA_species]

    def calculate_ppm_range(self, mass: float) -> Tuple[float, float]:
        """Computes a ±ppm mass window.

        Args:
            mass (float): Mass value.

        Returns:
            Tuple[float, float]: (lower_bound, upper_bound).
        """
        ppm_val = (self.ppm_interval / 1_000_000) * mass
        return (mass - ppm_val, mass + ppm_val)

    def calculate_mass(self, feature_id: str) -> Dict[str, Dict[str, Union[float, Tuple[float, float]]]]:
        """Computes all possible neutral masses for a feature.

        Args:
            feature_id (str): Feature identifier.

        Returns:
            Dict[str, Dict]: Mapping ion_type → mass and ppm range.
        """
        esi_types = self.esi_search_feature(feature_id)
        mz_value = self.feat_dicts.mz[feature_id]

        results: Dict[str, float] = {}
        results["H"] = mz_value - self.H.mass

        for esi in esi_types:
            if esi in self.ca_list:
                results[esi] = mz_value - getattr(self, esi).mass
            elif esi in self.isf_list:
                results[esi] = mz_value + getattr(self, esi).mass - self.H.mass
            elif esi in self.na_list:
                results[esi] = mz_value - getattr(self, esi).mass - self.H.mass

        return {
            ion: {"mass": mass, "ppm_range": self.calculate_ppm_range(mass)}
            for ion, mass in results.items()
        }

    def compute_multiple(self, features: List[str]) -> None:
        """Computes mass mappings for multiple features.

        Args:
            features (List[str]): Feature IDs to process.
        """
        self.results: Dict[str, Any] = {}

        for feature in tqdm(features):
            try:
                self.results[feature] = self.calculate_mass(feature)
            except Exception:
                continue

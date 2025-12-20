import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple, Set, Optional, Any
from molmass import Formula
from itertools import combinations, product
from collections import defaultdict, Counter
from copy import deepcopy
from . import FeatDicts, filter_by_logp


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
    """Base class for handling ESI species

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
        feat_dicts (glc.FeatDicts): Holds lookups including feat_id to m/z value.
        ppm_interval (int): PPM window for mass tolerance.
        results (Dict[str, Dict[str, Dict[str, Any]]]): Output mass mapping.
    """

    def __init__(
        self,
        ion_mode: str,
        species_df: pd.DataFrame,
        mz_peak_df: pd.DataFrame,
        edges_clf_df: pd.DataFrame,
        feat_dicts: FeatDicts,
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

    def compute_multiple(self, features: List[int]) -> None:
        """Computes mass mappings for multiple features.

        Args:
            features (List[int]): Feature IDs to process.
        """
        self.results: Dict[str, Any] = {}

        for feature in tqdm(features):
            try:
                self.results[feature] = self.calculate_mass(feature)
            except Exception:
                continue


class AdjustForC13:
    """Adjust metabolite feature-LMID mappings based on C13 isotope components.

    This class identifies GGM edges corresponding to the C13 isotope mass shift. 
    These edges are defined by the ```glc.PickPeaks.get_mz_rule_df``` output.
    The code finds the M0 feature and its structural assignments are updated to M1 .. Mn features.
    

    Attributes:
        feat2lmids (Dict[int, List[Any]]):
            Mapping from feature ID to a list of LIPID MAP IDs.
        mz_rule_df (pd.DataFrame):
            DataFrame defining m/z intervals from ```glc.PickPeaks.get_mz_rule_df```. Must include
            'mzmin', 'mzmax', 'n_edges', and 'mz_peak_id'.
        edges_clf (pd.DataFrame):
            Edge-level classification table containing isotope relationships.
            Must include 'feature_out', 'feature_in', 'mz_peak_id', 'out_mz'.
        iso_delta (float):
            The theoretical mass difference between 13C and 12C.
    """

    def __init__(
        self,
        feat2lmids: Dict[int, List[Any]],
        mz_rule_df: pd.DataFrame,
        edges_clf: pd.DataFrame
    ) -> None:
        """
        Initialize AdjustForC13.

        Args:
            feat2lmids: Mapping from feature ID to LIPID MAP IDs.
            mz_rule_df: DataFrame defining the m/z rule intervals.
            edges_clf: DataFrame of classified edges between features.
        """
        self.feat2lmids = deepcopy(feat2lmids)
        self.mz_rule_df = mz_rule_df
        self.edges_clf = edges_clf
        self.iso_delta = (
            Formula("13C").isotope.mass - Formula("12C").isotope.mass
        )

    def _get_isotope_mz_peak_id(self) -> int:
        """Return the mz_peak_id corresponding to the C13 isotope mass shift.

        Returns:
            The `mz_peak_id` associated with the interval that contains the
            calculated 13C-12C mass difference.

        Raises:
            KeyError: If no interval in the DataFrame contains the isotope delta.
        """
        intervals = pd.IntervalIndex.from_arrays(
            self.mz_rule_df["mzmin"],
            self.mz_rule_df["mzmax"],
            closed="both"
        )
        matches = self.mz_rule_df.loc[intervals.get_loc(self.iso_delta)]

        if isinstance(matches, pd.DataFrame):
            return int(matches.loc[matches["n_edges"].idxmax(), "mz_peak_id"])
        else:
            return int(matches["mz_peak_id"])

    def _get_components(self, df: pd.DataFrame) -> List[Set[int]]:
        """Build a network of features connected by C13 isotope edges.
            Make use of network components to find subgraphs (putative isotope groups).

        Args:
            df: DataFrame containing 'feature_out' and 'feature_in' columns.

        Returns:
            A list of connected components, each represented as a set of feature IDs.
        """
        G = nx.from_pandas_edgelist(df, "feature_out", "feature_in")
        return list(nx.connected_components(G))

    def run(self) -> Dict[int, List[Any]]:
        """Perform C13 adjustment on the feature-to-LM ID mapping.

        For each C13-related connected component, identify the feature with
        the lowest m/z and extend all other features' LM ID lists with its LM IDs.

        Returns:
            Updated feature-to-LM ID mapping (`feat2lmids`).
        """
        mz_peak_id = self._get_isotope_mz_peak_id()
        iso_edges = self.edges_clf[self.edges_clf["mz_peak_id"] == mz_peak_id]

        components = self._get_components(iso_edges)

        for comp in tqdm(components, desc="Adjusting for C13 isotope components"):
            comp_df = iso_edges[
                iso_edges["feature_out"].isin(comp)
                | iso_edges["feature_in"].isin(comp)
            ]

            idx = comp_df["out_mz"].idxmin()
            primary_row = comp_df.loc[idx]

            primary_feat = int(primary_row["feature_out"])
            primary_struct = self.feat2lmids[primary_feat]

            for feat in comp:
                if feat != primary_feat:
                    # Extend so original are not overwritten as this is putative and the original may be correct. 
                    self.feat2lmids[feat].extend(primary_struct)

        return self.feat2lmids


def map_ungrouped_feature_table(
        db: pd.DataFrame,
        feat_dicts: FeatDicts,
        ion_mode: str,
        ppm_interval: int = 10,
        proton_only: bool = True,
        esi_df: Optional[pd.DataFrame] = None,
        mz_peak_picker_obj: Optional[bool] = None,
        c13_adjustment: Optional[bool] = False,
        logp_filter_thresh: Optional[float] = None
) -> Dict[str, List[str]]:
    """
    Map ungrouped features to candidate lipid IDs based on accurate mass matching, with optional
    accounting for ion species, C13 adjustment, and logP-based filtering.

    This function calculates ppm search windows around each feature m/z value,
    matches database exact masses to those windows, and returns a mapping from
    feature identifiers to lipid IDs (LM_ID values). Optional steps allow the use
    of ion species, C13 correction, and filtering by estimated logP values.

    Args:
        db (pd.DataFrame):
            Lipid database containingthe columns:
            - ``LM_ID``: lipid identifier
            - ``EXACT_MASS``: monoisotopic exact mass
            This data frame is intended from ``glc.load_lm_database()``.

        feat_dicts (FeatDicts):
            Object containing feature information, specifically the feature - m/z mapping.

        ion_mode (str):
            Ionization mode ("pos" or "neg").

        ppm_interval (int, optional):
            PPM window size for matching exact masses. Defaults to ``10``.

        proton_only (bool, optional):
            If ``True``, only protonated/deprotonated ions are considered.
            If ``False``, ion types can be defined with ``esi_df`` arg and masses are computed using ``FeatMapper``.
            Defaults to ``True``.

        esi_df (pd.DataFrame, optional):
            Electrospray ionization species table. Should contain columns:
            - ``formula``: chemical formula of the species
            - ``type_code``: one of 'charged', 'isf', or 'neutral_adduct'
            Required when ``proton_only`` is ``False``.

        mz_peak_picker_obj (optional):
            Object providing peak information via:
            - ``get_peak_df()``
            - ``classify_edges()``.
            Intended to be from ```glc.PickPeaks```.
            Required when ``esi_df`` is provided. 

        c13_adjustment (bool, optional):
            Whether to run isotopic correction using ``AdjustForC13``.
            Requires ``mz_peak_picker_obj``. Defaults to ``False``.

        logp_filter_thresh (float, optional):
            If provided, filters mapped lipid candidates based on predicted
            logP values, retaining only those within ``n_stds`` standard deviations of the monotonic regression of LogP and rt.

    Returns:
        Dict[str, List[str]]:
            A mapping from feature identifiers to lists of matching ``LM_ID`` entries.

    Raises:
        ValueError:
            If inconsistent or insufficient inputs are provided (e.g., ``esi_df``
            without ``mz_peak_picker_obj``).

    """
    if esi_df is None and mz_peak_picker_obj is not None:
        raise ValueError("esi_df must be provided for mz_peak_picker_obj to be used.")
    if c13_adjustment is True and mz_peak_picker_obj is None:
        raise ValueError("mz_peak_picker_obj must be provided for c13_adjustment to be used.")
    if esi_df is not None and mz_peak_picker_obj is None:
        raise ValueError("The esi_df can only be used together with mz_peak_picker_obj.")
    
    if proton_only:
        proton_offset = Formula('H').monoisotopic_mass if ion_mode == 'pos' else Formula('H').monoisotopic_mass * -1
        features_data = []
        for feat, mz in feat_dicts.mz.items():
            ppm_value = (ppm_interval / 1_000_000) * mz
            m = mz - proton_offset
            ppm_min, ppm_max = (m - ppm_value, m + ppm_value)

            features_data.append({
                'feat': feat,
                'ppm_min': ppm_min,
                'ppm_max': ppm_max
            })

        features_df = pd.DataFrame(features_data)
    else:
        feat_mapper = FeatMapper(
            ion_mode=ion_mode,
            species_df=esi_df,
            mz_peak_df=mz_peak_picker_obj.get_peak_df(n_thresh=20),
            edges_clf_df=mz_peak_picker_obj.classify_edges(),
            feat_dicts=feat_dicts,
            ppm_interval=ppm_interval
        )
        feat_mapper.compute_multiple(feat_dicts.mz.keys())

        features_data = []
        for feat, data in feat_mapper.results.items():
            for adduct, mass_data in data.items():
                features_data.append({
                    'feat': feat,
                    'ppm_min': mass_data['ppm_range'][0],
                    'ppm_max': mass_data['ppm_range'][1]
                })
        features_df = pd.DataFrame(features_data)

    lm_ids = db['LM_ID'].values
    exact_masses = db['EXACT_MASS'].values

    feat2lmids = defaultdict(list)
    for lm_id, exact_mass in tqdm(zip(lm_ids, exact_masses), total=len(lm_ids)):
        mask = (features_df['ppm_min'] <= exact_mass) & (exact_mass <= features_df['ppm_max'])
        matching_feats = features_df.loc[mask, 'feat'].values
        for feat in matching_feats:
            feat2lmids[feat].append(lm_id)

    if c13_adjustment:
        adjuster = AdjustForC13(
            feat2lmids=feat2lmids,
            mz_rule_df=mz_peak_picker_obj.get_peak_df(n_thresh=20),
            edges_clf=mz_peak_picker_obj.classify_edges(),
        )
        feat2lmids = adjuster.run()
    
    if logp_filter_thresh is not None:
        feat2lmids = filter_by_logp(
            feat2lmids=feat2lmids,
            lm_df=db,
            n_stds=logp_filter_thresh,
            feat_dicts=feat_dicts,
            return_df=False
        )

    return feat2lmids


def map_grouped_feature_table(
        db: pd.DataFrame,
        feat_dicts: FeatDicts,
        ppm_interval: int = 10,
        logp_filter_thresh: Optional[float] = None
) -> Dict[str, List[str]]:
    
    # TODO add docstring
    
    features_data = []
    for feat, mz in feat_dicts.mz.items():
        ppm_value = (ppm_interval / 1_000_000) * mz
        ppm_min, ppm_max = (mz - ppm_value, mz+ ppm_value)

        features_data.append({
            'feat': feat,
            'ppm_min': ppm_min,
            'ppm_max': ppm_max
        })

    # Create a DataFrame from the features data
    features_df = pd.DataFrame(features_data)

    # Extract LM_IDs and EXACT_MASS values for vectorized operations
    lm_ids = db['LM_ID'].values
    exact_masses = db['EXACT_MASS'].values

    # Iterate over each lipid mass
    feat2lmids = defaultdict(list)
    for lm_id, exact_mass in tqdm(zip(lm_ids, exact_masses), total=len(lm_ids)):
        # Vectorized range check for the current exact_mass
        mask = (features_df['ppm_min'] <= exact_mass) & (exact_mass <= features_df['ppm_max'])

        # Get matching features
        matching_feats = features_df.loc[mask, 'feat'].values

        # Append lm_id to each matching feature
        for feat in matching_feats:
            feat2lmids[feat].append(lm_id)

    if logp_filter_thresh is not None:
        feat2lmids = filter_by_logp(
            feat2lmids=feat2lmids,
            lm_df=db,
            n_stds=logp_filter_thresh,
            feat_dicts=feat_dicts,
            return_df=False
        )

    return feat2lmids


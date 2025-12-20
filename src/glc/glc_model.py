from typing import Dict, List, Tuple, Iterable
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm
from functools import lru_cache
from . import GGM, FeatDicts


class GLCModel:
    """Graph-based lipid classifier (GLC).

    This class predicts lipid classes using a Gaussian Graphical Model (GGM),
    feature metadata, and accurate mass matches to database (LMSD). Scoring uses first-hop and
    second-hop neighbor information, weighted by partial correlations and with
    retention-time filtering.
    """

    def __init__(
        self,
        ggm: GGM,
        feat_dicts: FeatDicts,
        node_ids: Dict[int, List[str]],
        db_df: pd.DataFrame,
        db_id_col: str = "LM_ID",
        class_level: str = "SUB_CLASS",
        rt_thresh: float = 50.0,
        feat_weight: float = 5.0,
        label_propogation: bool = False,
    ) -> None:
        """Initialize the GLC predictor.

        Args:
            ggm: Object containing the graph (`G`) and partial correlations (`pcor_dict`). Object intended to be from ```glc.GGM```
            feat_dicts: Object providing metadata (e.g., retention times via `feat_dicts.rt`). Intended to be from ```glc.FeatDicts```
            node_ids: Mapping of feature/node IDs to lists of database IDs.
            db_df: Database dataframe containing lipid annotations.
            db_id_col: Column in `db_df` containing the database ID (e.g., "LM_ID").
            class_level: Level of lipid classification to predict (e.g., "SUB_CLASS").
            rt_thresh: Maximum retention-time difference allowed for neighbor filtering.
            feat_weight: Weight multiplier for the feature itself when scoring.
            label_propogation: Whether to use harmonic label propagation for nodes with
                no assigned class. (Default: ``False``). A very small handful of nodes in the subgraph
                may have no annotated neighbors; enabling this option assigns them classes
                based on the labels of other nodes in the graph. Typically, less than 10 nodes in a dataset. 
        """
        self.ggm = ggm
        self.feat_dicts = feat_dicts
        self.node_ids = node_ids
        self.db_df = db_df
        self.db_id_col = db_id_col
        self.class_level = class_level
        self.rt_thresh = rt_thresh
        self.feat_weight = feat_weight
        self.label_propogation = label_propogation
        self.lmid2class = db_df.set_index(db_id_col)[class_level].to_dict()

    @lru_cache(maxsize=None)
    def _get_neighbors(self, feature_id: int) -> List[int]:
        """Return immediate neighbors of a node in the GGM graph.

        Args:
            feature_id: Node/feature ID.

        Returns:
            List of neighboring node IDs.
        """
        return list(nx.neighbors(self.ggm.G, feature_id))

    def _filter_neighbors(
        self, feature_id: int, neighbors: List[int]
    ) -> Tuple[Iterable[int], Iterable[float]]:
        """Filter neighbors by retention time & compute absolute partial correlations.

        Args:
            feature_id: Feature ID whose neighbors are being filtered.
            neighbors: List of immediate neighbors.

        Returns:
            A tuple `(filtered_neighbor_ids, weights)` where:
                - filtered_neighbor_ids: Iterable of node IDs.
                - weights: Iterable of absolute partial correlation coefficients.
        """
        rt = self.feat_dicts.rt[feature_id]
        filtered = [
            (node, self.ggm.pcor_dict[f"{feature_id}::{node}"])
            for node in neighbors
            if abs(self.feat_dicts.rt[node] - rt) <= self.rt_thresh
        ]
        return zip(*filtered) if filtered else ([], [])

    def score_node(self, feature_id: int) -> List[Tuple[str, float]]:
        """Score lipid classes for a single feature ID.

        Scoring is based on:
            - filtered first-hop neighbors,
            - second-hop neighbors,
            - partial correlations as weights,
            - retention-time constraints.
            - feature itself weighted by `feat_weight`.

        Args:
            feature_id: Node/feature ID to classify.

        Returns:
            A list of `(class_name, score)` tuples sorted by score (top 10).
        """
        try:
            neighbors = self._get_neighbors(feature_id)
            neighbors, weights = self._filter_neighbors(feature_id, neighbors)

            neighbors = [feature_id] + list(neighbors)
            weights = [max(weights, default=0) * self.feat_weight] + list(weights)

            counter = Counter()
            for node, weight in zip(neighbors, weights):
                categories = {self.lmid2class[lmid] for lmid in self.node_ids[node]}
                for category in categories:
                    counter[category] += weight

            counter = self._second_hop_score(neighbors[1:], weights[1:], counter)
            return counter.most_common(10)

        except KeyError as e:
            raise KeyError(
                f"Feature ID {feature_id} not in GGM subgraph. Missing feature: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error scoring feature {feature_id}: {e}")

    def _second_hop_score(
        self,
        neighbors: List[int],
        weights: List[float],
        counter: Counter,
    ) -> Counter:
        """Compute scores from second-hop neighbors.

        Uses only maximal scoring path for each second-hop feature to avoid
        double counting.

        Args:
            neighbors: First-hop neighbor IDs.
            weights: Weights for each first-hop neighbor.
            counter: Score accumulator.

        Returns:
            Counter updated with second-hop class contributions.
        """
        max_weights: Dict[int, float] = {}

        for feature_id, weight_1hop in zip(neighbors, weights):
            second_neighbors = self._get_neighbors(feature_id)
            second_neighbors, weights_2hop = self._filter_neighbors(
                feature_id, second_neighbors
            )

            for node, weight_2hop in zip(second_neighbors, weights_2hop):
                score = weight_1hop * weight_2hop
                if node not in max_weights or score > max_weights[node]:
                    max_weights[node] = score

        for node, score in max_weights.items():
            categories = {self.lmid2class[lmid] for lmid in self.node_ids[node]}
            for category in categories:
                counter[category] += score

        return counter

    def predict_all(self) -> Dict[int, List[Tuple[str, float]]]:
        """Predict classes for all features in the main GGM subgraph.

        Returns:
            A dict mapping feature IDs → list of `(class, score)` predictions.
        """
        predictions: Dict[int, List[Tuple[str, float]]] = {}
        missing_feats: List[int] = []

        for feature_id in tqdm(self.ggm.G.nodes, desc="Predicting feature classes"):
            try:
                predictions[feature_id] = self.score_node(feature_id)
                if predictions[feature_id] == []:
                    missing_feats.append(feature_id)
            except Exception:
                missing_feats.append(feature_id)

        if not self.label_propogation:
            return predictions

        nx.set_node_attributes(
            self.ggm.G,
            {node: preds[0][0] for node, preds in predictions.items() if preds},
            "label",
        )

        prop_labels = nx.algorithms.node_classification.harmonic_function(self.ggm.G)

        for feature_id in missing_feats:
            predictions[feature_id] = [(prop_labels[feature_id], np.nan)]

        return predictions

    def convert2mainclass(
        self, predictions: Dict[int, List[Tuple[str, float]]]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Convert predicted SUB_CLASS values to MAIN_CLASS values.

        Args:
            predictions: Output from `predict_all()` using SUB_CLASS predictions.

        Returns:
            A dict mapping feature IDs -> list of `(MAIN_CLASS, score)` pairs.
        """
        mainclass_converter_dict = dict(
            zip(self.db_df["SUB_CLASS"], self.db_df["MAIN_CLASS"])
        )

        output_dict: Dict[int, List[Tuple[str, float]]] = {}

        for key, value_list in predictions.items():
            unique_classes = set()
            output_dict[key] = [
                (mainclass_converter_dict[item], score)
                for item, score in value_list
                if mainclass_converter_dict.get(item)
                and mainclass_converter_dict[item] not in unique_classes
                and not unique_classes.add(mainclass_converter_dict[item])
            ]

        return output_dict


def build_prediction_dataframe(
    subclass_predictions: Dict[int, List[Tuple[str, float]]],
    mainclass_predictions: Dict[int, List[Tuple[str, float]]],
    quality_score_df: pd.DataFrame,
    feat_dicts: FeatDicts
) -> pd.DataFrame:
    """
    Build a tidy DataFrame summarizing GLC subclass and main class predictions
    together with feature metadata and quality scores.

    Args:
        subclass_predictions:
            Dictionary mapping peak_id to a list of subclass predictions.
            Each entry is expected to be a ranked list, where the top prediction
            is accessed as subclass_predictions'[peak_id][0][0]'.
        mainclass_predictions:
            Dictionary mapping peak_id to a list of main class predictions,
            structured analogously to subclass_predictions.
        quality_score_df:
            DataFrame indexed by peak_id containing quality score columns
             'lsi_score', 'pcor_score', and 'product_score'.
        feat_dicts:
            A glc.FeatDicts object providing dictionary-like access to
            feature m/z and retention time via feat_dicts.mz and feat_dicts.rt.

    Returns:
        pd.DataFrame:
            A DataFrame with one row per peak_id containing feature metadata,
            predicted subclass and main class, and associated quality scores.
    """
    results = []

    for peak_id in subclass_predictions.keys():
        if not subclass_predictions[peak_id]:
            scl_p = np.nan
            mcl_p = np.nan
        else:
            scl_p = subclass_predictions[peak_id][0][0]
            mcl_p = mainclass_predictions[peak_id][0][0]

        results.append({
            'peak_id': peak_id,
            'mz': feat_dicts.mz[peak_id],
            'rt': feat_dicts.rt[peak_id],
            'subclass': scl_p,
            'mainclass': mcl_p,
            'lsi_score': quality_score_df['lsi_score'].get(peak_id, np.nan),
            'pcor_score': quality_score_df['pcor_score'].get(peak_id, np.nan),
            'product_score': quality_score_df['product_score'].get(peak_id, np.nan),
        })

    return pd.DataFrame(results)

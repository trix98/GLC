# TODO check this and add docstring and typing 

from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from . import UMAPEmbedder

class QualityScorer:

    """
    Compute quality scores for features based on prediction confidence and
    local neighborhood consistency in UMAP space.

    The following scores are computed:

    - PCOR score:
        GLC score of the top ranking prediction over the sum of the scores of the top 10 ranking.
        Noramlized to [0, 1] via quantile transformation.
        Called the PCOR score (partial correlation score) as GLC scoring is based on partial correlations. 
        Higher GLC scores for a feature tend to results from a combination of i.) more database matches and ii.) stronger partial correlations 
        For the PCOR score, we are seeking to capture how dominant the top prediction is compared to other predictions for that feature.
    
    - LSI score:
        Local Simpson's Index measuring the diversity of subclass predictions
        among k-nearest neighbors in UMAP space of the GGM.
        Based on the assumption that GGM structure encodes lipid class. We expect that nearest features have the same lipid class. 

    - Product score:
        Product of PCOR and LSI scores, quantile-scaled to [0, 1].

    - The quality scores should be interpreted as the higher the score, the higher the GLC prediction confidence for that feature. 
    - However, note of caution, these quality scores are dataset specific and cannot be directly compared across datasets.

    """

    def __init__(
        self,
        prediction_dict: Dict[int, List[Tuple[str, float]]],
        embedder_obj: UMAPEmbedder,
        k: int = 5,
    ) -> None:

        """
        Initialize the QualityScorer and compute all quality metrics.

        Upon initialization, the following steps are performed:
        - Encode subclass labels
        - Compute k-nearest neighbors in UMAP space
        - Calculate LSI, PCOR, and product quality scores
        - Assemble the results into a single DataFrame

        The primary output of this class is the `df` attribute, which contains the quality scores for each feature.

        Args:
            prediction_dict:
                GLC predictions output. A mapping from feature to subclass prediction.
            embedder_obj:
                Fitted UMAPEmbedder object of the GGM structure.
            k:
                Number of neighbors used for local diversity calculations.

        Attributes:
            df (pd.DataFrame):
                DataFrame with one row per feature and the following columns:
                - feature: Feature identifier
                - subclass: Top predicted subclass
                - lsi_score: Local Simpson's Index score
                - pcor_score: Quantile-scaled PCOR score
                - product_score: Combined quality score in [0, 1]
        """

        self.prediction_dict = {}
        for key, val in prediction_dict.items():
            if val == []: # unless using label propagation in the GLC model, a very very small number of features may have no predictions
                self.prediction_dict[key] = [(np.nan, np.nan)]
            else:
                self.prediction_dict[key] = val
            

        self.emb = embedder_obj.umap_embedding
        self.k = k + 1 # +1, because the first one is the feature itself
        self.node_names = embedder_obj.node_names # node names are features/peak_id

        self.labels_encoded, self.label2code = self._encode_labels()
        self.distances, self.indices = self._knn()
        self.df = self._create_df()

    def _encode_labels(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Encode top predicted subclass labels as integers.

        Returns:
            Tuple containing:
            - Array of encoded subclass labels aligned with node order.
            - Mapping from subclass name to encoded code
        """

        labels = [self.prediction_dict[feat][0][0] for feat in self.node_names]
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        label2code = {label: i for i, label in enumerate(label_encoder.classes_)}
        return labels_encoded, label2code

    def _knn(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform k-nearest neighbors search on UMAP embedding of the GGM. 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors.
        """
        scaled_data = StandardScaler().fit_transform(self.emb)
        knn = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(scaled_data)
        distances, indices = knn.kneighbors(scaled_data)
        return distances, indices

    def _score_all_lsi(self) -> List[float]:
        """
        Compute Local Simpson's Index (LSI) for all features.

        LSI measures the concentration of subclass labels among local neighbors.
        Higher values indicate lower local diversity and higher confidence in prediction. 

        Returns:
            List of LSI scores, one per feature.
        """
        lsi_scores = []
        for neighbor_idx in self.indices:
            neighbor_labels = self.labels_encoded[neighbor_idx]
            label_counts = np.bincount(neighbor_labels)
            simpson_index = np.sum((label_counts / self.k) ** 2)
            lsi_scores.append(simpson_index)
        return lsi_scores

    def _get_weighted_pcor_scores(
        self, k: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Normalize partial-correlation scores for the top-k predictions.

        Args:
            k:
                Number of top subclass predictions to consider per feature.

        Returns:
            Dictionary mapping feature identifier to a list of
            (subclass, normalized_score) tuples.
        """
        result = {}
        for peak, preds in self.prediction_dict.items():
            # Only keep the first k tuples
            preds_k = preds[:k]

            total = sum(score for _, score in preds_k)
            if total == 0:
                # If sum is zero, assign np.nan for all subclasses in the first k
                result[peak] = [(subclass, np.nan) for subclass, _ in preds_k]
            else:
                result[peak] = [(subclass, score / total) for subclass, score in preds_k]
        return result


    def _scale_pcor_scores(self) -> np.ndarray:
        """
        Extract and quantile-scale the PCOR score for eachfeature.

        Returns:
            Array of PCOR scores scaled to a uniform [0, 1] distribution.
        """
        pcor_score_dict = self._get_weighted_pcor_scores(k=10)
        scores = [pcor_score_dict[feat][0][1] for feat in self.node_names]
        scores = np.array(scores).reshape(-1, 1)
        quantile_scores = QuantileTransformer(output_distribution='uniform').fit_transform(scores)
        return quantile_scores.flatten()


    def _create_df(self) -> pd.DataFrame:
        """
        Assemble the final quality score DataFrame.

        Returns:
            DataFrame with columns:
            - feature
            - subclass
            - lsi_score
            - pcor_score
            - product_score
        """
        lsi_score = self._score_all_lsi()
        pcor_norm_score = self._scale_pcor_scores()
        product_score = np.array(lsi_score) * pcor_norm_score
        product_score = QuantileTransformer(output_distribution='uniform').fit_transform(product_score.reshape(-1, 1)).flatten()
        return pd.DataFrame(
            {
                'feature': self.node_names,
                'subclass': [self.prediction_dict[feat][0][0] for feat in self.node_names],
                'lsi_score': lsi_score,
                'pcor_score': pcor_norm_score,
                'product_score': product_score
            }
        )



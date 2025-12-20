import networkx as nx
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Tuple, List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as cc
from . import FeatDicts
import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm


class UMAPEmbedder:
    """Compute a UMAP embedding from a NetworkX graph adjacency matrix.

    This class converts a weighted NetworkX graph into an adjacency matrix,
    optionally removes outliers using IsolationForest, scales the matrix,
    and computes a UMAP embedding. The results (embedding and node names)
    are stored as properties.

    Attributes:
        ndim (int): Number of UMAP embedding dimensions.
        remove_outliers (bool | None): Whether outlier removal is applied.
        adj_mx (np.ndarray): Raw adjacency matrix derived from the graph.
        node_names (list[str]): List of node identifiers from the graph.
        umap_embedding (np.ndarray): The resulting UMAP coordinates after fitting.
    """

    def __init__(
        self,
        G: nx.Graph,
        remove_outliers: bool = False,
        ndim: int = 2
    ):
        """Initialize UMAPEmbedder and compute the UMAP embedding.

        Args:
            G (nx.Graph):
                A NetworkX graph. Edge weights are interpreted as partial
                correlations (assumed to already be absolute values).
            remove_outliers (bool):
                If True, detects and removes outlier nodes using 
                IsolationForest before UMAP embedding. 
                Defaults to False. (no outlier removal).
            ndim (int, optional):
                Number of UMAP dimensions to compute. Defaults to 2.
        """
        self.ndim = ndim
        self.remove_outliers = remove_outliers
        self.adj_mx = nx.to_numpy_array(G, weight="weight")
        self.node_names = list(G.nodes())

        self._compute_embedding()

    def _compute_embedding(self):
        """Compute the scaled adjacency matrix and apply UMAP.

        Steps:
            1. MinMax-scale the adjacency matrix.
            2. Optionally remove outliers via IsolationForest.
            3. Compute a UMAP embedding using cosine distance and
               dynamic neighborhood size (2% of node count).
        """
        # MinMax scale the adjacency matrix
        scaled_adj_mx = MinMaxScaler().fit_transform(self.adj_mx)

        # Optionally remove outliers
        if self.remove_outliers:
            isf = IsolationForest(random_state=0)
            inliers = isf.fit_predict(scaled_adj_mx) == 1
            scaled_adj_mx = scaled_adj_mx[inliers]

            print(f"Removed {len(self.node_names) - scaled_adj_mx.shape[0]} outliers.")
            self.node_names = [
                name for i, name in enumerate(self.node_names) if inliers[i]
            ]

        # UMAP neighbors = 2% of nodes
        n_neighbors = int(scaled_adj_mx.shape[0] * 0.02)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*n_jobs value 1 overridden to 1 by setting random_state.*",
                category=UserWarning
            )
            self.umap_embedding = umap.UMAP(
                n_components=self.ndim,
                n_neighbors=n_neighbors,
                metric="cosine",
                min_dist=1,
                random_state=0,
            ).fit_transform(scaled_adj_mx)


class UMAPPlots:
    """Utility class for visualizing UMAP embeddings and related annotations.

    This class wraps a UMAPEmbedder instance and provides multiple plotting
    functions for analyzing feature attributes, annotations, and predictions
    on top of UMAP coordinates.
    
    Attributes:
        umap_embedding (np.ndarray): UMAP coordinates (n_samples, 2).
        node_names (list[str]): Ordered node names corresponding to points in
            the embedding.
    """

    def __init__(self, umap_embedder_obj: UMAPEmbedder):
        """Initializes the UMAPPlots visualization wrapper.

        Args:
            umap_embedder_obj (UMAPEmbedder): 
                A fitted UMAPEmbedder instance containing `umap_embedding`
                (array of shape (n_samples, 2)) and `node_names` (list of feature
                identifiers). These values are stored for use in all plotting
                methods.
        """
        self.umap_embedding = umap_embedder_obj.umap_embedding
        self.node_names = umap_embedder_obj.node_names

    def _plot_umap(
        self,
        color_continuous=None,
        ax: Optional[plt.Axes] = None,
        title: str = ''
    ) -> None:
        """Plots the base UMAP embedding.

        Args:
            color_continuous (array-like, optional): 
                Continuous values used for point coloring (e.g., m/z or RT). 
                If None, points are colored light grey.
            ax (plt.Axes, optional): 
                Matplotlib axis to draw on. If None, a new figure and axis 
                are created.
            title (str, optional): 
                Title of the plot.

        Returns:
            None
        """
        df = pd.DataFrame(self.umap_embedding, columns=['Emb 1', 'Emb 2'])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5.5))

        if color_continuous is None:
            sns.scatterplot(
                data=df, x='Emb 1', y='Emb 2', ax=ax, s=10, color='lightgrey'
            )
        else:
            sns.scatterplot(
                data=df, x='Emb 1', y='Emb 2',
                c=color_continuous, ax=ax, s=10
            )

        ax.set_title(title)

    @staticmethod
    def _extract_codes(code: str) -> Tuple[str, str, str]:
        """Parses hierarchical LipidMaps codes into components.

        LipidMaps class codes may be 4 or 6 characters long. This function
        splits them into category, main, and subclass codes.

        Args:
            code (str): 
                LipidMaps class code, e.g., 'FA01', 'FA0102'.

        Returns:
            Tuple[str, str, str]: 
                A tuple containing:
                - category_code (str)
                - main_code (str)
                - sub_code (str)
                Returns (None, None, None) for unexpected code lengths.
        """
        if len(code) == 6:
            return code[:2], code[2:4], code[4:6]
        elif len(code) == 4:
            return code[:2], code[2:4], '00'
        return None, None, None
    
    def plot_feature_attribute(
            self, 
            feat_dicts: FeatDicts, 
            attribute: str = 'mz', 
            ax: Optional[plt.Axes] = None, 
            vmax=None, 
            colorbar=True
    ) -> plt.Axes:
        """Plots UMAP colored by a continuous feature attribute (e.g., RT or m/z).

        Args:
            feat_dicts (FeatDicts): 
                Object holding lookup dicts for feature to `mz` or `rt`.
            attribute (str, optional): 
                Attribute to visualize. Must be a key in `feat_dicts`. 
                Defaults to 'mz'.
            ax (plt.Axes, optional): 
                Matplotlib axis on which to draw. If None, a new one is created.
            vmax (float, optional): 
                Upper color normalization bound. If None, uses the max value 
                across all nodes.
            colorbar (bool, optional): 
                Whether to display a colorbar. Defaults to True.

        Returns:
            plt.Axes: The axis with the plotted embedding.
        """
        
        attribute_dict = getattr(feat_dicts, attribute)
        node_vals = [attribute_dict[feat] for feat in self.node_names]
        
        if vmax is None:
            vmax = max(node_vals)
        
        # Set the color normalization using vmin and vmax
        norm = plt.Normalize(vmin=min(node_vals), vmax=vmax)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)

        if ax is None:
            fig, ax = plt.subplots()

        # Set colorbar label based on attribute
        if attribute == 'mz':
            label = 'm/z'
        elif attribute == 'rt':
            label = 'Retention Time (s)'
        else:
            label = attribute

        # Add colorbar
        if colorbar:
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(label)

        colors = sm.to_rgba(node_vals)

        # Plot UMAP with continuous color based on the normalized attribute
        self._plot_umap(ax=ax, color_continuous=colors)
        return ax

    def _colors_for_annotations(
        self,
        annot_df: pd.DataFrame,
        class_level: str = 'lm_subclass'
    ) -> Tuple[np.ndarray, List, List[plt.Line2D]]:
        """Builds color mappings and legend components for annotation overlays.

        Args:
            annot_df (pd.DataFrame): 
                Annotation table containing `peak_id` and a column specifying 
                the class label (e.g., 'lm_subclass').
            class_level (str, optional): 
                Column name in `annot_df` specifying the the level of annotation class to plot
                Defaults to 'lm_subclass'.

        Returns:
            Tuple:
                coords (np.ndarray): 
                    Array of UMAP coordinates for annotated nodes.
                colors (list): 
                    List of RGB tuples corresponding to annotation colors.
                legend_handles (list[plt.Line2D]): 
                    Line2D objects for constructing legends.
        """
        annot_features = annot_df['peak_id'].tolist()

        # Collect coordinates and labels for annotated nodes
        coords, labels = [], []
        for idx, feature in enumerate(self.node_names):
            if feature in annot_features:
                labels.append(annot_df.loc[annot_df['peak_id'] == feature, class_level].iloc[0])
                coords.append(self.umap_embedding[idx, :])

        coords = np.array(coords)
        labels = np.array(labels)

        # Generate color mapping
        cats = np.unique(labels)
        palette =  sns.color_palette(cc.glasbey, cats.shape[0])
        color_mapping = {category: palette[i] for i, category in enumerate(cats)}
        colors = [color_mapping[label] for label in labels]

        # Order classes by extracted LM codes
        class_df = pd.DataFrame({'class': cats})
        class_df['code'] = class_df['class'].str.extract(r'\[(.*?)\]')
        class_df[['category_code', 'main_code', 'sub_code']] = class_df['code'].apply(lambda x: pd.Series(self._extract_codes(x)))
        class_df = class_df.sort_values(by=['category_code', 'main_code', 'sub_code'])
        sorted_cats = class_df['class'].tolist()

        # Create legend handles
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=category,
                                     markerfacecolor=color_mapping[category], markersize=10)
                          for category in sorted_cats]

        return coords, colors, legend_handles


    def plot_ground_truth(
            self, 
            annot_df: pd.DataFrame, 
            class_level: str = 'lm_subclass', 
            ax: Optional[plt.Axes] = None, 
            legend: bool = True
    ) -> plt.Axes:
        """Overlays ground-truth annotations onto the UMAP embedding.

        Args:
            annot_df (pd.DataFrame): 
                Annotation table with class labels and `peak_id` values.
            class_level (str, optional): 
                Annotation level to display (e.g., 'lm_category', 
                'lm_mainclass', 'lm_subclass'). Defaults to 'lm_subclass'.
            ax (plt.Axes, optional): 
                Axis on which to draw. If None, creates a new one.
            legend (bool, optional): 
                Whether to draw a legend. Defaults to True.

        Returns:
            plt.Axes: The axis containing the annotation overlay.
        """
        if class_level == 'lm_subclass':
            legend_title = 'Subclass'
        elif class_level == 'lm_mainclass':
            legend_title = 'Main class'
        elif class_level == 'lm_category':
            legend_title = 'Category'
        else:
            legend_title = class_level

        coords, colors, legend_handles = self._colors_for_annotations(annot_df, class_level)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6.4, 6.4))

        self._plot_umap(ax=ax)
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], c=colors, ax=ax, s=45, marker='o')
        if legend:
            ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title=legend_title)
        return ax

    def plot_predictions(
            self, 
            predictions: Dict[int, List[Tuple[str, float]]], 
            top_n: int = 25, 
            ax: Optional[plt.Axes] = None, 
            fig_title='', 
            class_level: str = ''
        ) -> plt.Axes:
        """Plots predicted annotation classes on the UMAP embedding.

        Args:
            predictions (dict): 
                GLC output dictionary mapping feature indices to lists of 
                (predicted_class, score) tuples. Only the top prediction per 
                feature is used.
            top_n (int, optional): 
                Minimum count threshold for a class to be included in the plot. 
                Defaults to 25.
            ax (plt.Axes, optional): 
                Axis to draw on. If None, a new figure is created.
            fig_title (str, optional): 
                Title for the output figure.
            class_level (str, optional): 
                Name of the class level being predicted, used for the legend title.

        Returns:
            plt.Axes: The axis containing the prediction visualization.
        """
        # Get the top prediction for each feature
        class_labels = np.array([predictions[i][0][0] for i in self.node_names])
        tup = np.unique(class_labels, return_counts=True)

        # Get the top_n most common classes
        class_dict = {i:j for i, j in zip(tup[0], tup[1]) if j>=top_n}
        class_dict.pop('nan', None)
        class_dict.pop(np.nan, None)
        class_dict = dict(sorted(class_dict.items(), key=lambda x: x[1], reverse=True))

        # disqualify classes that are not in the top_n
        # and get umap coordinates for remaining predictions
        class_labels_filt = []
        coords = []
        rmv_count = 0
        for idx, pred_class in enumerate(class_labels):
            if pred_class in class_dict:
                coords.append(self.umap_embedding[idx, :])
                class_labels_filt.append(pred_class)
            else:
                rmv_count += 1
        print(f"Removed {rmv_count} instances with less than {top_n} instances")
        coords = np.array(coords)

        # get class colors and legend
        cats = np.array(list(class_dict.keys()))
        palette = sns.color_palette(cc.glasbey_category10, cats.shape[0])
        color_mapping = {category: palette[i] for i, category in enumerate(cats)}
        colors = [color_mapping[label] for label in class_labels_filt]

        legend_handles, legend_labels = [], []
        for category in cats:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=category, 
                                            markerfacecolor=color_mapping[category], markersize=15))
            legend_labels.append(category)

        if ax is None:
            fig, ax = plt.subplots()

        with sns.plotting_context():
            sns.set_style('whitegrid')
            sns.scatterplot(x=coords[:, 0], y=coords[:, 1], ax=ax, c=colors, s=15, edgecolor='none')
            ax.set_xlabel('Emb 1')
            ax.set_ylabel('Emb 2')
            ax.set_title(fig_title)
            ax.legend(handles=legend_handles, labels=legend_labels, bbox_to_anchor=(1.05, 1), ncols=2, title=class_level)

        return ax



class EmbStats:
    """
    Base class to compute embedding statistics based on nearest neighbors.
    Test if nearest-neighbor matches in a UMAP embedding are enriched
    for shared class labels beyond random expectation.
    Can be used to evaluate just annotations, or for GLC predictions. 
    Based on the method outlined by Chen et al. https://doi.org/10.1038/s41467-024-46089-y
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize EmbStats with embedding coordinates and class labels.
        Args:
            X (np.ndarray): Embedding coordinates (samples x features).
            y (np.ndarray): Class labels for each sample.
        """
        self.X = X
        self.y = y
        self.labels = np.unique(self.y)
        self.n = self.y.shape[0]

        self.class_counts = self._class_counts()
        self.nn_portions = self._nn_mean()
        self.ses = self._se()

    # ---- internal helpers ----

    def _class_counts(self) -> Dict[Any, float]:
        """
        Compute the proportion of each class in the dataset.

        Returns:
            Dict[Any, float]: Mapping from class label to class proportion.
        """
        labels, counts = np.unique(self.y, return_counts=True)
        return {label: count / self.n for label, count in zip(labels, counts)}

    def _get_knn(self) -> np.ndarray:
        """
        Compute the nearest neighbors for each annotation/prediction.

        Returns:
            np.ndarray: Indices of nearest neighbors for each annotation/prediction.
        """
        nn_model = NearestNeighbors(n_neighbors=2)
        nn_model.fit(self.X)
        distances, indices = nn_model.kneighbors(self.X)
        return indices[:, 1:]    # skip self

    def _nn_mean(self) -> Dict[Any, float]:
        """
        Compute the mean proportion of nearest neighbors matching the same class.

        Returns:
            Dict[Any, float]: Mapping from class label to nearest neighbor match proportion.
        """
        nn_indices = self._get_knn()
        nn_portion = {}

        for label, class_portion in self.class_counts.items():
            mask = (self.y == label)
            predicted = self.y[nn_indices[mask]]
            n_match = np.sum(predicted == label)
            nn_portion[label] = n_match / (class_portion * self.n)

        return nn_portion

    def _se(self) -> Dict[Any, float]:
        """
        Compute the standard error of nearest neighbor proportions for each class.

        Returns:
            Dict[Any, float]: Mapping from class label to standard error.
        """
        ses = {}
        n = self.y.shape[0]

        for label in self.labels:
            ses[label] = (
                np.sqrt(self.nn_portions[label] *
                        (1 - self.nn_portions[label]) / n)
                + 1e-16
            )

        return ses

class UMAPEmbStats(EmbStats):
    """
    Test if nearest-neighbor matches in a UMAP embedding are enriched
    for shared class labels beyond random expectation.
    Can be used to evaluate just annotations, or for GLC predictions. 
    Based on the method outlined by Chen et al. https://doi.org/10.1038/s41467-024-46089-y
    """

    @staticmethod
    def get_inputs_for_emb_stats(
        embedder: UMAPEmbedder, ground_truth: pd.DataFrame, class_level: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embedding coordinates and class labels for statistics.

        Args:
            embedder (UMAPEmbedder): Embedder object with 'node_names' and 'umap_embedding'.
            ground_truth (pd.DataFrame): DataFrame with 'peak_id' and class annotations.
            class_level (str): Column name for class annotations in ground_truth df.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Embedding coordinates (X) and class labels (y).
        """
        coords, y = [], []
        annot_feats = ground_truth['peak_id'].tolist()

        for idx, feature in enumerate(embedder.node_names):
            if feature in annot_feats:
                y.append(
                    ground_truth.loc[
                        ground_truth['peak_id'] == feature, class_level
                    ].iloc[0]
                )
                coords.append(embedder.umap_embedding[idx, :])

        return np.array(coords), np.array(y)

    def __init__(self, embedder: Any, ground_truth: pd.DataFrame, class_level: str):
        """
        Initialize EmbStatsWithInputs with an embedder and ground truth labels.

        Args:
            embedder (Any): Embedder object with 'node_names' and 'umap_embedding'.
            ground_truth (pd.DataFrame): DataFrame with 'peak_id' and class annotations.
            class_level (str): Column name for class annotations in ground_truth.
        """
        X, y = self.get_inputs_for_emb_stats(embedder, ground_truth, class_level)
        super().__init__(X, y)

    def get_stats(self, min_samples: int = 3, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compute z-scores and p-values (with adjustment) for nearest neighbor enrichment per class.

        Args:
            min_samples (int, optional): Minimum number of samples required to include a class. Defaults to 3.
            save_path (Optional[str], optional): Path to save the results as CSV. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing subclass, counts, z-scores, p-values, adjusted p-values, and significance.
        """
        results = []

        for label in self.labels:
            z = (self.nn_portions[label] - self.class_counts[label]) / self.ses[label]
            p = 1 - norm.cdf(z)

            n = round(self.class_counts[label] * self.n)
            if n > min_samples:
                results.append({
                    'Sub class': label,
                    'n': n,
                    'z score': z,
                    'p': p
                })

        df = pd.DataFrame(results)
        df = df.sort_values(by='n', ascending=False)

        # Bonferroni
        df['p_adj'] = (df['p'] * df.shape[0]).clip(upper=1)
        df['is_sig'] = df['p_adj'] < 0.05
        df = df.reset_index(drop=True)

        if save_path is not None:
            df.to_csv(save_path)

        return df
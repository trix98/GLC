from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import umap
import networkx as nx
from typing import Optional

import networkx as nx
import umap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from typing import Optional


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
        remove_outliers: Optional[bool] = None,
        ndim: int = 2
    ):
        """Initialize UMAPEmbedder and compute the UMAP embedding.

        Args:
            G (nx.Graph):
                A NetworkX graph. Edge weights are interpreted as partial
                correlations (assumed to already be absolute values).
            remove_outliers (bool, optional):
                If True, detects and removes outlier nodes using 
                IsolationForest before UMAP embedding. Defaults to None 
                (no outlier removal).
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

        self.umap_embedding = umap.UMAP(
            n_components=self.ndim,
            n_neighbors=n_neighbors,
            metric="cosine",
            min_dist=1,   # As specified in your original code
            random_state=0
        ).fit_transform(scaled_adj_mx)

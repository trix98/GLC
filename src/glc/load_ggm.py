import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Union


class GGM:
    """Container for GGM results and graph computation.

    This class loads a Gaussian graphical model (GGM) adjacency matrix from
    a CSV file or a pandas DataFrame, constructs a NetworkX graph, extracts 
    the largest connected component, and builds a lookup of partial correlations.

    Args:
        ggm_source (str | pd.DataFrame):
            Either a file path to a CSV containing the GGM adjacency matrix, 
            or a pandas DataFrame containing the adjacency matrix. The DataFrame
            must have feature labels as its index.

    Attributes:
        _ggm_df (pd.DataFrame): DataFrame containing the adjacency matrix.
        feat_labels (list[str]): List of feature names corresponding to graph nodes.
        adj_mx (np.ndarray): Raw adjacency matrix representing partial correlations.
        G (nx.Graph): Main connected subgraph extracted from the adjacency matrix.
        pcor_dict (Dict[str, float]): Dictionary mapping "<feature1>::<feature2>" to
            absolute partial correlation values.
    """

    def __init__(self, ggm_source: Union[str, pd.DataFrame]):
        """Initialize the GGM container.

        Args:
            ggm_source (str | pd.DataFrame):
                Path to the GGM adjacency matrix CSV file, or a pandas DataFrame 
                containing the adjacency matrix with feature labels as the index.

        Raises:
            TypeError: If `ggm_source` is not a string or DataFrame.
            ValueError: If a DataFrame is provided without an index.
        """

        # Load CSV or validate DataFrame
        if isinstance(ggm_source, str):
            self._ggm_df = pd.read_csv(ggm_source, index_col=0)

        elif isinstance(ggm_source, pd.DataFrame):
            if ggm_source.index is None:
                raise ValueError("DataFrame must have an index of feature labels.")
            self._ggm_df = ggm_source.copy()

        else:
            raise TypeError(
                "ggm_source must be either a file path (str) or a pandas DataFrame."
            )

        # Extract metadata
        self.feat_labels = self._ggm_df.index.tolist()
        self.adj_mx = self._ggm_df.values

        # Build graph + dictionary
        self.G = self._to_graph()
        self.pcor_dict = self._get_pcor_dict()

    def _get_main_subgraph(self, G: nx.Graph) -> nx.Graph:
        """Extract the largest connected component from a graph.

        Args:
            G (nx.Graph): Input graph.

        Returns:
            nx.Graph: The largest connected component as a subgraph.
        """
        connected_components = list(nx.connected_components(G))
        main_component = max(connected_components, key=len)
        main_subgraph = G.subgraph(main_component)
        print(len(list(main_subgraph.nodes())))
        return main_subgraph

    def _to_graph(self) -> nx.Graph:
        """Convert the adjacency matrix to a graph and return its main subgraph.

        Node identifiers in the output graph correspond to feature labels
        (e.g., peak IDs).

        Returns:
            nx.Graph: The main subgraph created from the adjacency matrix.
        """
        G = nx.from_numpy_array(np.abs(self.adj_mx), create_using=nx.Graph())
        mapping = {i: name for i, name in enumerate(self.feat_labels)}

        print(len(list(G.nodes())))
        G = nx.relabel_nodes(G, mapping)

        print("All nodes")
        G = self._get_main_subgraph(G)
        return G

    def _get_pcor_dict(self) -> Dict[str, float]:
        """Construct a dictionary of absolute partial correlations.

        Keys follow the format:
            "<feature1>::<feature2>"

        Returns:
            Dict[str, float]: Mapping of feature-pair identifiers to
            absolute partial correlation values.
        """
        pcor_dict = {}
        for i in range(len(self.adj_mx)):
            for j in range(len(self.adj_mx)):
                if abs(self.adj_mx[i, j]) > 0:
                    key = f"{self.feat_labels[i]}::{self.feat_labels[j]}"
                    pcor_dict[key] = abs(self.adj_mx[i, j])
        return pcor_dict

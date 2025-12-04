from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
from importlib import resources


@dataclass(frozen=True)
class FeatDicts:
    """Encapsulates dictionaries for quick lookup of m/z and retention time by feature ID.

    This class expects a DataFrame containg the columns 'peak_id', 'mz', and 'rt'. 

    Attributes:
        df (pd.DataFrame): DataFrame containing feature information, including
            'peak_id', 'mz', and 'rt' columns.
        mz (Dict[int, float]): Dictionary mapping peak IDs to m/z values.
        rt (Dict[int, float]): Dictionary mapping peak IDs to retention times.
    """

    df: pd.DataFrame

    mz: Dict[int, float] = field(init=False)
    rt: Dict[int, float] = field(init=False)

    def __post_init__(self):
        """Constructs lookup dictionaries after initialization."""
        mz_dict, rt_dict = self._feat_searchable_dicts()
        object.__setattr__(self, 'mz', mz_dict)
        object.__setattr__(self, 'rt', rt_dict)

    def _feat_searchable_dicts(self):
        """Creates lookup dictionaries from the input DataFrame.

        Returns:
            Tuple[Dict[int, float], Dict[int, float]]:
            - Dictionary mapping peak IDs to m/z values.
            - Dictionary mapping peak IDs to retention times.
        """
        peak_ids = self.df['peak_id'].astype(int)

        return (
            dict(zip(peak_ids, self.df['mz'].astype(float))),
            dict(zip(peak_ids, self.df['rt'].astype(float))),
        )

def load_lm_database() -> pd.DataFrame:
    """Load the LIPID MAPS Structural Database (LMSD) from a packaged parquet file.

    The function reads the file `lipid_maps_database.parquet` located in the
    package's `data` directory and returns it as a pandas DataFrame. Any
    missing values in the `SUB_CLASS` column are filled with the
    corresponding entries from the `MAIN_CLASS` column.

    Returns:
        pd.DataFrame: The LIPID MAPS Structural Database (LMSD) loaded into a DataFrame.
    """
    # Import the package where the resource is located
    from . import data

    # Use importlib.resources to access the parquet file
    # with resources.open_text(data, 'lipid_maps_database.parquet') as f:
    #     lm_df = pd.read_parquet(f)
    lm_df = pd.read_parquet(resources.files(data) / 'lipid_maps_database.parquet')

    # Fill NaN in 'SUB_CLASS' with values from 'MAIN_CLASS'
    lm_df['SUB_CLASS'] = lm_df['SUB_CLASS'].fillna(lm_df['MAIN_CLASS'])

    return lm_df


class LoadExampleData:
    """Load example data files for the AddNeuroMed-LPOS lipidomics dataset for use in tutorials. 

    This class provides convenient access to packaged example data files
    included with the library. When instantiated, it loads parquet files needed GLC analyses.
    files:

    - A feature table with metabolite intensities, retention times, and m/z values
    - A Gaussian Graphical Model (GGM) adjacency matrix from the GeneNet R package

    Attributes:
        data (module): Reference to the local data package.
        feat_table (pandas.DataFrame): Loaded feature table.
        ggm (pandas.DataFrame): Loaded example GGM.
    """

    def __init__(self):
        from . import data
        self._data = data
        
        # feature table
        self.feat_table = self._load_parquet("example_feature_table_addneuromed_lpos.parquet")

        # ggm adjacency matrix
        ggm = self._load_parquet("example_ggm_addneuromed_lpos.parquet")
        ggm = ggm.set_index(ggm.columns[0]) # index should be peak IDs
        ggm.index.name = 'peak_id'
        self.ggm = ggm

        # ground truth annotations 
        self.ground_truth = self._load_parquet("example_ground_truth_addneuromed_lpos.parquet")

    def _load_parquet(self, filename):
        """Load a Parquet file stored inside the package resources.

        Args:
            filename (str): Name of the Parquet file inside the package's data directory.
        Returns:
            pandas.DataFrame: The parsed Parquet file as a DataFrame.
        """
        return pd.read_parquet(resources.files(self._data) / filename)



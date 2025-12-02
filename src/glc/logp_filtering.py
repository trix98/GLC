from sklearn.isotonic import IsotonicRegression
import numpy as np
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple, Optional
from . import FeatDicts

def filter_by_logp(
    feat2lmids: Dict[int, List[str]],
    lm_df: pd.DataFrame,
    n_stds: float,
    feat_dicts: FeatDicts,
    return_df: bool = False
) -> Dict[int, List[str]] | Tuple[Dict[int, List[str]], pd.DataFrame]:
    """ Filters the structures mapping to features based on estimating logP vs retention time trend.

    1. Fits a monotonically increasing regression model to logP as a function of retention time (RT).
    3. Computes residuals between observed logP and the isotonic trend.
    4. Filters out LMIDs whose deviation from the trend exceeds a threshold
       defined as ``n_stds`` times the standard deviation of residuals.
    5. Returns an updated mapping of feature to LM IDs.
    

    Args:
        feat2lmids (Dict[int, List[str]]):
            Dictionary mapping feature IDs to lists of LMID identifiers.
        lm_df (pd.DataFrame):
            DataFrame containing at least the columns ``'LM_ID'`` and ``'LOG_P'``.
        n_stds (float):
            Number of standard deviations from the isotonic regression trend
            allowed before filtering an LM ID out.
        feat_dicts (FeatureDicts):
            Object containing retention times, accessed via ``feat_dicts.rt[feature_id]``. Should be from ```glc.FeatureDicts```.
        return_df (bool, optional):
            If ``True``, also returns the full DataFrame full of structures, logp, rt and isotonic regression values. Useful for plotting trend. Defaults to ``False``.

    Returns:
        Dict[int, List[str]]:
            Updated mapping of feature IDs to filtered LM ID lists.

        If ``return_df=True``, returns a tuple:
            (updated_mapping, logp_df)

            where ``logp_df`` is a DataFrame of all LMID/feature/logP/RT entries,
            including isotonic regression values and residuals.
    """

    # Build structure-logP lookup
    struc2logp = {
        struc: logp
        for struc, logp in zip(lm_df["LM_ID"], lm_df["LOG_P"])
    }

    # Build long-form table of feature/structure entries
    results = []
    for feat, id_lst in feat2lmids.items():
        for struc in id_lst:
            results.append({
                "feature_id": feat,
                "struc_id": struc,
                "logP": struc2logp[struc],
                "rt": feat_dicts.rt[feat]
            })

    logp_df = pd.DataFrame(results).dropna(subset=["logP"])

    # Fit isotonic regression (user keeps 2D X, so we respect that)
    X = logp_df["rt"].values.reshape(-1, 1)
    y = logp_df["logP"].values

    ir = IsotonicRegression().fit(X, y)

    logp_df["ir_logp"] = ir.predict(logp_df["rt"].values.reshape(-1, 1))
    logp_df["delta"] = logp_df["ir_logp"] - logp_df["logP"]

    # Filter using STD threshold 
    # TODO: Would be better to use MAD. 
    threshold = n_stds * logp_df["delta"].std()
    logp_df_filt = logp_df[logp_df["delta"].abs() < threshold]

    allowed_id_set = set(logp_df_filt["struc_id"])

    # Build updated mapping
    updated_node_ids = {}
    for feature in feat_dicts.rt.keys():
        updated_node_ids[feature] = [
            struc for struc in feat2lmids[feature]
            if struc in allowed_id_set
        ]

    if return_df:
        return updated_node_ids, logp_df

    return updated_node_ids

from typing import Dict, List, Tuple, Optional
import pandas as pd


def test_annotations(
    predictions: Dict[int, List[Tuple[str, float]]],
    annot_df: pd.DataFrame,
    annot_clf_col: str = "lm_subclass",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate annotations against GLC predicted classes.

    Args:
        predictions: Dictionary mapping feature_id -> list of (class, score) tuples. Intended to be ```glc.GLCModel.predict_all()``` output.
        annot_df: DataFrame with true annotations and must include a 'peak_id' column.
        annot_clf_col: Column name containing the annotated class. (default: 'lm_subclass')
        verbose: If True, prints warnings for missing features. These should be annotations not in the GGM main subgraph, which GLC cannot predict.

    Returns:
        A DataFrame comparing predictions to annotations. 
    """

    results = []

    for feature, true_class in zip(
        annot_df["peak_id"].astype(int), annot_df[annot_clf_col]
    ):
        # Prediction retrieval
        pred_list = predictions.get(feature)

        if not pred_list:
            if verbose:
                print(
                    f"Feature {feature} skipped (missing or not in GGM main subgraph)."
                )
            continue

        # Top prediction
        top_class, top_score = pred_list[0]

        # Count equally highest scores
        n_top = sum(1 for _, s in pred_list if s == top_score)

        results.append(
            {
                "feature": feature,
                "annot_class": true_class,
                "pred_class": top_class,
                "pred_score": top_score,
                "n_top": n_top,
            }
        )

    # Build output DataFrame
    df = pd.DataFrame(results)
    df["correct"] = df["annot_class"] == df["pred_class"]

    return df


from typing import List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class PlotConfusionMatrices:
    """
    Plot confusion matrices and compute precision/recall/F1 metrics of GLC predictions against ground truth annotations. 
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    @staticmethod
    def _extract_codes(code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if len(code) == 6:
            return code[:2], code[2:4], code[4:6]
        elif len(code) == 4:
            return code[:2], code[2:4], "00"
        else:
            return None, None, None

    def _add_misc_labels(self, df: pd.DataFrame, classes: List[str]) -> List[str]:
        preds: List[str] = []
        for _, row in df.iterrows():
            if row["pred_class"] in classes:
                preds.append(row["pred_class"])
            else:
                pred_main = row["pred_class"].split("[")[1][:4]
                annot_main = row["annot_class"].split("[")[1][:4]
                if pred_main == annot_main:
                    preds.append("Other: correct main class")
                else:
                    preds.append("Other: incorrect main class")
        return preds

    def _add_null_labels(self, df: pd.DataFrame, preds: List[str]) -> List[str]:
        updated_pred_axis_labels: List[str] = []
        for pred, pred_axis_label in zip(df["pred_class"], preds):
            if pred == "No prediction [000000]":
                updated_pred_axis_labels.append("No prediction")
            else:
                updated_pred_axis_labels.append(pred_axis_label)
        return updated_pred_axis_labels

    def _get_box_positions(self, class_labels: List[str], plot_all: bool) -> Tuple[List[int], List[int]]:
        count, delta = 0, 0
        running_code = ""
        start_pos: List[int] = []
        stop_pos: List[int] = []

        for lab in class_labels:
            try:
                code = lab.split("[")[1][2:4]
                if code != running_code:
                    running_code = code
                    start_pos.append(count - delta)
                    stop_pos.append(count)
                    delta = 0
            except Exception:
                start_pos.append(count - delta)
                stop_pos.append(count)
                break

            count += 1
            delta += 1

        if plot_all:
            start_pos.append(count - delta)
            stop_pos.append(count)

        return start_pos, stop_pos

    def _color_axes_ticks(self, axs: plt.Axes, annotation_classes: List[str], color: str = "#ec4743") -> None:
        for label in axs.get_xticklabels():
            if label.get_text() not in annotation_classes:
                label.set_color(color)

        for label in axs.get_yticklabels():
            if label.get_text() not in annotation_classes:
                label.set_color(color)

    def _get_cm_labels(self, plot_all: bool = True) -> List[str]:
        _df = self.df.copy()
        _df["pred_class"].fillna("No prediction [000000]", inplace=True)

        if plot_all:
            label_df = pd.DataFrame(pd.unique(_df[["annot_class", "pred_class"]].values.ravel()), columns=["class"])
        else:
            label_df = pd.DataFrame(pd.unique(_df["annot_class"]), columns=["class"])

        label_df["code"] = label_df["class"].str.extract(r"\[(.*?)\]")
        label_df[["category_code", "main_code", "sub_code"]] = label_df["code"].apply(
            lambda x: pd.Series(self._extract_codes(x))
        )
        label_df = label_df.sort_values(by=["category_code", "main_code", "sub_code"])

        class_labels = [i for i in label_df["class"] if i != "No prediction [000000]"]
        return class_labels

    def _plot_confusion_matrix(self, df: Optional[pd.DataFrame] = None, axs: Optional[plt.Axes] = None, vmax: int = 5):
        _df = (df.copy() if df is not None else self.df.copy())

        if axs is None:
            fig, axs = plt.subplots(figsize=(10, 8))

        class_labels = self._get_cm_labels(plot_all=True)

        preds = _df["pred_class"].tolist()
        if "No prediction" in preds:
            preds = self._add_null_labels(_df, preds)
            class_labels.extend(["No prediction"])

        cm = confusion_matrix(_df["annot_class"].tolist(), preds, labels=class_labels)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens", vmax=vmax,
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=3, ax=axs
        )
        axs.set_xlabel("Predicted sub class", fontsize=12)
        axs.set_ylabel("True sub class", fontsize=12)

        annotation_classes = (
            df["annot_class"].unique().tolist()
            if df is not None else self.df["annot_class"].unique().tolist()
        )
        self._color_axes_ticks(axs, annotation_classes)

        start_pos, stop_pos = self._get_box_positions(class_labels, plot_all=True)

        square_color = "#0096FF"
        for start, stop in zip(start_pos, stop_pos):
            plt.plot([start, start], [start, stop], color=square_color, linewidth=2.5)
            plt.plot([start, stop], [start, start], color=square_color, linewidth=2.5)
            plt.plot([start, stop], [stop, stop], color=square_color, linewidth=2.5)
            plt.plot([stop, stop], [start, stop], color=square_color, linewidth=2.5)

        return axs, preds, class_labels

    def _plot_confusion_matrix_short(self, df=None, axs=None, vmax=5):
        _df = (df.copy() if df is not None else self.df.copy())

        if axs is None:
            fig, axs = plt.subplots(figsize=(10, 8))

        class_labels = self._get_cm_labels(plot_all=False)

        preds = self._add_misc_labels(_df, class_labels)
        class_labels.extend(["Other: correct main class", "Other: incorrect main class"])

        if "No prediction" in preds:
            preds = self._add_null_labels(_df, preds)
            class_labels.extend(["No prediction"])

        cm = confusion_matrix(_df["annot_class"].tolist(), preds, labels=class_labels)
        cm = cm[:-2, :]
        if "No prediction" in class_labels:
            cm = cm[:-1, :]

        y_class_labels = [
            i for i in class_labels
            if i not in ["Other: correct main class", "Other: incorrect main class", "No prediction"]
        ]

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens", vmax=vmax,
            xticklabels=class_labels, yticklabels=y_class_labels,
            linewidths=3, ax=axs
        )
        axs.set_xlabel("Predicted sub class", fontsize=12)
        axs.set_ylabel("True sub class", fontsize=12)

        start_pos, stop_pos = self._get_box_positions(class_labels, plot_all=False)
        square_color = "#0096FF"
        for start, stop in zip(start_pos, stop_pos):
            plt.plot([start, start], [start, stop], color=square_color, linewidth=2.5)
            plt.plot([start, stop], [start, start], color=square_color, linewidth=2.5)
            plt.plot([start, stop], [stop, stop], color=square_color, linewidth=2.5)
            plt.plot([stop, stop], [start, stop], color=square_color, linewidth=2.5)

        axs.axvline(x=stop_pos[-1] + 0.1, color="#A9A9A9", linewidth=2.5)

        return axs, preds, class_labels

    def plot_annotation_confusion_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        truncate: bool = True,
        vmax: int = 5,
        barplot: bool = False,
        barplot_title: str = "Number of sub classes that are top-ranking:",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a confusion matrix plot with optional top-k barplot.

        Args:
            df: Optional DataFrame to use. If None, uses evaluator's DataFrame copy.
            truncate: If True, use simplified confusion matrix. Only classes that are in the ground truth are shown.
            vmax: Max heatmap color value.
            barplot: Whether to create a proportional top-k stacked bar above the heatmap.
            barplot_title: Title for the barplot legend.

        Returns:
            Tuple of (figure, heatmap_axes).
        """
        _df = (df.copy() if df is not None else self.df.copy())

        if barplot:
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 6], hspace=0.2)
            ax_heatmap = fig.add_subplot(gs[1])
        else:
            fig, ax_heatmap = plt.subplots(figsize=(10, 8))

        if truncate:
            axs, preds, xtick_labels = self._plot_confusion_matrix_short(df=_df, axs=ax_heatmap, vmax=vmax)
        else:
            axs, preds, xtick_labels = self._plot_confusion_matrix(df=_df, axs=ax_heatmap, vmax=vmax)

        if barplot:
            _df["heatmap_col"] = preds

            grouped = _df.groupby(["heatmap_col", "n_top"]).size().unstack(fill_value=0)
            grouped = grouped.div(grouped.sum(axis=1), axis=0)

            new_idxs = list(set(xtick_labels) - set(grouped.index.tolist()))
            grouped = pd.concat([grouped, pd.DataFrame(index=new_idxs, columns=grouped.columns)])
            grouped = grouped.loc[xtick_labels]

            heatmap_pos = ax_heatmap.get_position()
            bar_pos = [heatmap_pos.x0, heatmap_pos.y1 + 0.01, heatmap_pos.width, 0.075]
            ax_bar = fig.add_axes(bar_pos)

            cmap = plt.get_cmap("Spectral_r")
            sns.set_style("darkgrid")
            grouped.plot(kind="bar", stacked=True, colormap=cmap, ax=ax_bar, width=1, legend=None)

            ax_bar.set_xticks([])
            ax_bar.set_xticklabels([])
            ax_bar.set_ylabel("Proportion")

            legend_patches = [
                mpatches.Patch(color=cmap((col - 1) / (max(max(grouped.columns) - 1, 1))), label=str(col))
                for col in grouped.columns
            ]

            fig.legend(handles=legend_patches, loc="upper left", ncol=len(grouped.columns),
                       bbox_to_anchor=(0.1, 0.85), frameon=False, title=barplot_title)

            plt.tight_layout()
            plt.show()

        return fig, axs

    def peformance_metrics(self, df: Optional[pd.DataFrame] = None, return_df: bool = False, level: str = "sub") -> Union[pd.DataFrame, dict]:
        """Compute classification metrics (precision/recall/f1) for the dataset.

        Args:
            df: Optional DataFrame to use. If None, uses evaluator's DataFrame copy.
            return_df: If True, return a pandas DataFrame; otherwise return a dict.
            level: 'sub' for subclass-level metrics or 'main' for main-class metrics.

        Returns:
            Either a pandas DataFrame (if return_df=True) or a dict with metrics.
        """
        _df = (df.copy() if df is not None else self.df.copy())

        _df["pred_class"].fillna("No prediction [000000]", inplace=True)

        if level == "sub":
            unique_labels = _df["annot_class"].unique()
            y_true = _df["annot_class"].values
            y_preds = _df["pred_class"].values
        elif level == "main":
            _df["main_annot_class"] = _df["annot_class"].apply(lambda x: x.split("[")[1][:4])
            _df["main_pred_class"] = _df["pred_class"].apply(lambda x: x.split("[")[1][:4])
            unique_labels = _df["main_annot_class"].unique()
            y_true = _df["main_annot_class"].values
            y_preds = _df["main_pred_class"].values
        else:
            raise ValueError("level must be 'sub' or 'main'")

        y_pred_filtered = np.where(np.isin(y_preds, unique_labels), y_preds, "other")

        report = classification_report(y_true, y_pred_filtered, labels=unique_labels, output_dict=True, zero_division=0)

        if return_df:
            return pd.DataFrame(report).T
        else:
            return report
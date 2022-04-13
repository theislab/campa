from typing import Any, Tuple, Iterable, Optional

from matplotlib.axes import Axes as MplAxes
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt


def _co_occ_scores(
    adata: ad.AnnData, condition: str, condition_value: Any, cluster1: str, cluster2: str
) -> pd.DataFrame:
    scores = adata[adata.obs[condition] == condition_value].obsm[f"co_occurrence_{cluster1}_{cluster2}"]
    # filter nans from scores (cells in which either cluster1 or cluster2 does not exist)
    scores = scores[~np.isnan(scores).all(axis=1)]
    # rename columns to center of distance interval
    distances = (
        adata.uns["co_occurrence_params"]["interval"][:-1] + adata.uns["co_occurrence_params"]["interval"][1:]
    ) / 2
    scores = scores.rename(columns={str(i): d for i, d in enumerate(distances)})
    # get log2 of co-occ scores
    scores_log = scores.apply(np.log2)
    # return scores ready to plot
    return scores_log.melt(value_name="score", var_name="distance")


def plot_co_occurrence(
    adata: ad.AnnData,
    cluster1: str,
    cluster2: str,
    condition: str,
    condition_values: Optional[Iterable[str]] = None,
    ax: MplAxes = None,
    **kwargs: Any,
) -> None:
    """
    kwargs: additional arguments to sns.lineplot
    """
    adata.obs[condition] = adata.obs[condition].astype("category")
    if condition_values is None:
        condition_values = adata.obs[condition].cat.categories
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    scores = {}
    for v in condition_values:
        scores[v] = _co_occ_scores(adata, condition, v, cluster1, cluster2)
    scores = pd.concat(scores).reset_index(level=0).rename(columns={"level_0": condition}).reset_index(drop=True)
    g = sns.lineplot(data=scores, y="score", x="distance", hue=condition, ax=ax, **kwargs)
    g.set(xscale="log")
    ax.plot()
    ax.set_ylabel("log2(co-occurrence)")
    ax.axhline(y=0, color="black")


def plot_co_occurrence_grid(
    adata: ad.AnnData,
    condition: str,
    condition_values: Optional[Iterable[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    **kwargs: Any,
) -> Any:
    """
    Plot co-occurrence for all cluster-cluster pairs in a grid.

    Parameters
    ----------
    adata
        Adata containing co-occurrence scores in ``adata.obsm['co_occurrence_{cluster1}_{cluster2}']``.
    condition
        Categorical condition to group obs in adata by. Must be a column in ``adata.obs``.
    condition_values

    """
    fig, axes = plt.subplots(
        len(adata.uns["clusters"]),
        len(adata.uns["clusters"]),
        figsize=figsize,
        sharey=True,
    )
    for i, c1 in enumerate(adata.uns["clusters"]):
        for j, c2 in enumerate(adata.uns["clusters"]):
            if i == 0:
                axes[i, j].set_title(c2)
            plot_co_occurrence(adata, c1, c2, condition, condition_values, ax=axes[i][j], **kwargs)
    return fig, axes

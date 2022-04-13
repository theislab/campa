from copy import copy, deepcopy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_object_stats(adata, group_key, features=None, clusters=None, figsize_mult=(2, 2), **kwargs):
    """Compare objects stats of groups in group_key.

    Uses stats in adata.obsm['object_stats_agg'], resulting from calling extr.get_object_stats()
    Args:
        adata: anndata object with 'object_stats_agg' in obsm
        group_key: categorical value in adata.obs to group by
        features: list of features to display. Must be columns of adata.obsm['object_stats_agg'].
            If None, all features are displayed.
        clusters: list of clusters to display. Must be columns of adata.obsm['object_stats_agg'].
            If None, all clusters are displayed.
        kwargs: passed to sns.boxplot
    """
    agg_stats = deepcopy(adata.obsm["object_stats_agg"])
    if not isinstance(agg_stats.columns, pd.MultiIndex):
        # restore multiindex for easier access
        agg_stats.columns = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in agg_stats.columns])
    if features is None:
        features = agg_stats.columns.levels[0]
    if clusters is None:
        clusters = agg_stats.columns.levels[1]
    fig, axes = plt.subplots(
        len(clusters),
        len(features),
        figsize=(len(features) * figsize_mult[0], len(clusters) * figsize_mult[1]),
        sharex=True,
        squeeze=False,
    )
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            cluster = clusters[i]
            feature = features[j]
            df = copy(agg_stats[feature])
            df[group_key] = adata.obs[group_key]
            sns.boxplot(data=df, y=cluster, x=group_key, ax=ax, **kwargs)
            ax.set_title(feature)
    plt.tight_layout()

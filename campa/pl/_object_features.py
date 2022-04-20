from copy import copy, deepcopy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_object_stats(adata, group_key, features=None, clusters=None, figsize_mult=(2, 2), **kwargs):
    """
    Barplot of object statistics.

    Parameters
    ----------
    adata
        Adata containing object statistics in ``adata.obsm['object_stats_agg']``.
        E.g. result of :meth:`FeatureExtractor.get_object_stats`.
    group_key
        Categorical value in ``adata.obs`` to group by.
    features
        List of features to display.
        Must be columns of ``adata.obsm['object_stats_agg']``.
        If None, all features are displayed.
    clusters
        List of clusters to display.
        Must be columns of ``adata.obsm['object_stats_agg']``.
        If None, all clusters are displayed.
    kwargs
        Keyword arguments passed to :meth:`sns.boxplot`.
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

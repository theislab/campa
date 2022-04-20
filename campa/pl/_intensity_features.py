from typing import Any, List, Tuple, Union, Mapping, Iterable, Optional
import warnings

from scipy.stats import zscore
from matplotlib.axes import Axes as MplAxes
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np
import scipy
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _adjust_plotheight(scplot):
    """
    Fix large gap between title and plot for scanpy plots.

    (rather hacky, might not work in all cases)
    """
    # modified code from sc.pl.MatrixPlot.make_figure
    category_height = scplot.DEFAULT_CATEGORY_HEIGHT
    category_width = scplot.DEFAULT_CATEGORY_WIDTH
    mainplot_height = len(scplot.categories) * category_height
    mainplot_width = len(scplot.var_names) * category_width + scplot.group_extra_size
    if scplot.are_axes_swapped:
        mainplot_height, mainplot_width = mainplot_width, mainplot_height

    height = mainplot_height  # + 1  # +1 for labels

    # if the number of categories is small use
    # a larger height, otherwise the legends do not fit
    scplot.height = max([scplot.min_figure_height, height])
    scplot.width = mainplot_width + scplot.legends_width


def _ensure_categorical(adata: ad.AnnData, col: str) -> None:
    if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
        # nothing todo
        return
    adata.obs[col] = adata.obs[col].astype(str).astype("category")
    return


# TODO add group size similar to dotplot here!
def plot_mean_intensity(
    adata: ad.AnnData,
    groupby: str = "cluster",
    marker_dict: Optional[Union[Mapping[str, Iterable[str]], Iterable[str]]] = None,
    save: Optional[str] = None,
    dendrogram: bool = False,
    limit_to_groups: Optional[Mapping[str, Union[str, List[str]]]] = None,
    type: str = "matrixplot",  # noqa: A002
    cmap: str = "viridis",
    adjust_height: bool = True,
    figsize: Tuple[int, int] = (10, 5),
    ax: MplAxes = None,
    **kwargs: Any,
) -> None:
    """
    Show per cluster intensity of each channel.

    Intensity is either shown as mean or z-scored intensity, depending on the ``standard_scale`` kwarg.

    Parameters
    ----------
    adata
        Adata containing aggregated information by clusters.
        E.g. result of :meth:`FeatureExtractor.get_intensity_adata`.
    groupby
        column in ``adata.obs`` containing the groups to compare.
    marker_dict
        Limit/group vars that are shown, either by passing list or dict (adds annotations to plot).
    save
        Path to save figure to.
    dendrogram
        Show dendrogram over columns.
    limit_to_groups
        Dict with obs as keys and groups from obs as values, to subset adata before plotting.
    type
        Type of plot, either `matrixplot` or `violinplot`.
    cmap
        Matplotlib colormap to use.
    adjust_height
        Option to make plots a bit more streamlined.
    figsize
        Size of figure.
    ax
        Axis to plot in.
    kwargs
        Keyword arguments for :func:`sc.pl.stacked_violin`/:func:`sc.pl.matrixplot`.
    """
    if limit_to_groups is None:
        limit_to_groups = {}
    _ensure_categorical(adata, groupby)

    # subset data
    for key, groups in limit_to_groups.items():
        if not isinstance(groups, list):
            groups = [groups]
        adata = adata[adata.obs[key].isin(groups)]
    # group vars together?
    if marker_dict is None:
        marker_dict = np.array(adata.var.index)
    if isinstance(marker_dict, dict):
        marker_list = np.concatenate(list(marker_dict.values()))
    else:
        marker_list = marker_dict

    # calculate values to show
    color_values = pd.DataFrame(index=adata.var.index)
    for g in adata.obs[groupby].cat.categories:
        color = "mean intensity"
        g_expr = adata[adata.obs[groupby] == g].X
        g_size = adata[adata.obs[groupby] == g].obs["size"]
        color_values[g] = (g_expr * g_size[:, np.newaxis]).sum(axis=0) / g_size.sum()
    color_values = color_values.loc[marker_list]

    standard_scale = kwargs.pop("standard_scale", None)
    if standard_scale == "var":
        color_values = color_values.apply(zscore, axis=1)
    elif standard_scale == "obs":
        color_values = color_values.apply(zscore, axis=0)

    # plot
    if dendrogram:
        sc.tl.dendrogram(adata, groupby=groupby)
    title = "mean intensity in " + ", ".join([f"{key}: {val}" for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        title = "mean intensity"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if type == "violinplot":
        scplot = sc.pl.stacked_violin(
            adata,
            var_names=marker_dict,
            groupby=groupby,  # standard_scale='var',
            ax=ax,
            dendrogram=dendrogram,
            return_fig=True,
            title=title,
            **kwargs,
        )
    elif type == "matrixplot":
        scplot = sc.pl.matrixplot(
            adata,
            var_names=marker_dict,
            groupby=groupby,
            cmap=cmap,
            colorbar_title=color,
            ax=ax,
            return_fig=True,
            dendrogram=dendrogram,
            values_df=color_values.T,
            title=title,
            **kwargs,
        )
    else:
        raise NotImplementedError(type)

    if adjust_height:
        _adjust_plotheight(scplot)

    scplot.make_figure()
    # add axis labels
    scplot.ax_dict["mainplot_ax"].set_xlabel("channels")
    scplot.ax_dict["mainplot_ax"].set_ylabel(groupby)

    if save is not None:
        plt.savefig(save, dpi=100)


def plot_mean_size(
    adata: ad.AnnData,
    groupby_row: str = "cluster",
    groupby_col: str = "well_name",
    normby_row: Optional[str] = None,
    normby_col: Optional[str] = None,
    ax: MplAxes = None,
    figsize: Tuple[int, int] = None,
    adjust_height: bool = False,
    save: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Plot mean cluster sizes per cell, grouped by different columns in obs.

    Parameters
    ----------
    adata
        Adata containing aggregated information by clusters.
        E.g. result of :meth:`FeatureExtractor.get_intensity_adata`.
    groupby_row
        Column in ``adata.obs`` containing the row-wise grouping.
    groupby_col
        Column in ``adata.obs`` containing the column-wise grouping.
    normby_row
        Value in ``groupby_row`` to normalise rows by.
    normby_col
        Value in ``groupby_col`` to normalise columns by.
    ax
        Axis to plot in.
    figsize
        Size of figure.
    adjust_height
        Option to make plots a bit more streamlined.
    save
        Path to save figure to.
    kwargs
        Keyword arguments for :func:`sc.pl.matrixplot`.
    """
    _ensure_categorical(adata, groupby_row)
    _ensure_categorical(adata, groupby_col)
    # groupy_col needs to be var
    sizes = {
        c: adata[adata.obs[groupby_col] == c].obs.groupby(groupby_row).mean()["size"]
        for c in adata.obs[groupby_col].cat.categories
    }
    sizes_adata = ad.AnnData(pd.DataFrame(sizes))
    sizes_adata.obs["group"] = sizes_adata.obs.index.astype("category")

    # get values to show
    values_df = pd.DataFrame(sizes)
    title_suffix = ""
    if normby_row is not None:
        values_df = values_df.divide(values_df.loc[normby_row], axis="columns")
        title_suffix += f"\nnormalised by {groupby_row} {normby_row}"
    if normby_col is not None:
        values_df = values_df.divide(values_df.loc[:, normby_col], axis="rows")
        title_suffix += f"\nnormalised by {groupby_col} {normby_col}"
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    scplot = sc.pl.matrixplot(
        sizes_adata,
        var_names=sizes_adata.var_names,
        groupby="group",
        values_df=values_df,
        colorbar_title="mean size\nin group",
        title="mean object size" + title_suffix,
        ax=ax,
        show=False,
        return_fig=True,
        **kwargs,
    )
    if adjust_height:
        _adjust_plotheight(scplot)
    scplot.make_figure()

    if save is not None:
        plt.savefig(save, dpi=100)


def mixed_model(ref_expr, g_expr, ref_well_name, g_well_name):
    # res_data = {'resid'}
    res_data: Mapping[str, List[Any]] = {"df": [], "resid": []}
    # if True:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        # iterate over all channels in the data
        pvals = []
        for i in range(ref_expr.shape[-1]):
            # create dataframe for mixed model
            df = pd.DataFrame(index=range(len(ref_expr) + len(g_expr)))
            df["mean_expr"] = np.log2(np.concatenate([ref_expr[:, i], g_expr[:, i]]))
            df["group"] = [0] * len(df.index)
            df["group"].iloc[len(ref_expr) :] = 1
            df["well"] = np.concatenate([ref_well_name, g_well_name])
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna()
            # display(df)
            # sns.distplot(df[df['group']==0]['mean_expr'])
            # sns.distplot(df[df['group']==1]['mean_expr'])

            model = sm.MixedLM.from_formula("mean_expr ~ group", re_formula="~1", groups="well", data=df)
            result = model.fit()
            pvals.append(result.summary().tables[1].loc["group"]["P>|z|"])
            res_data["df"].append(df)
    return np.array(pvals).astype("float"), res_data


def get_intensity_change(
    adata: ad.AnnData,
    groupby: str,
    marker_dict: Optional[Union[Mapping[str, Iterable[str]], Iterable[str]]] = None,
    limit_to_groups: Optional[Mapping[str, Union[str, List[str]]]] = None,
    reference: Optional[Union[List[str], str]] = None,
    reference_group: Optional[str] = None,
    color: str = "logfoldchange",
    size: str = "mean_reference",
    group_sizes_barplot: Optional[str] = None,
    pval: str = "ttest",
    alpha: float = 0.05,
    norm_by_group: Optional[str] = None,
) -> Mapping[str, Any]:
    """
    Get data for plotting intensity comparison with :func:`plot_intensity_change`.

    Calculate mean intensity differences between perturbations or clusters.
    If no reference is given, use all other groups (except the current one) as reference.
    Colors show log2fc / mean intensity changes / zscore changes, depending on the ``color`` argument.
    Dot size shows mean intensity of reference group that is compared to, or indicates the pvalue,
    depending on the ``size`` argument.

    Parameters
    ----------
    adata
        Adata containing aggregated information by clusters.
        E.g. result of :meth:`FeatureExtractor.get_intensity_adata`.
    groupby
        column in ``adata.obs`` containing the groups to compare.
    marker_dict
        Limit/group vars that are shown, either by passing list or dict (adds annotations to plot).
    limit_to_groups
        Dict with obs as keys and groups from obs as values, to subset adata before plotting.
    reference
        Reference cluster/perturbation to compare to.
        If not defined, will compare each value in groupby against the rest.
    reference_group
        Obs entry that contains reference grouping (by default, ``groupby`` is used).
    color
        Color of dots, either `logfoldchange` or `meanchange`.
    size
        sizes of dots, either `mean_reference` or `pval` (distinguish significant and non-significant dots).
    group_sizes_barplot
        Mean size of groups shown as a bar plot to the right.
        Either None (do not show), `mean` (mean size of groups),
        `meanchange` (mean difference of group size from reference), `foldchange`.
    pval
        Type of test done to determine pvalues. Either `ttest` or `mixed_model`.
        `mixed_model` calculates a mixed model using wells as random effects and should be preferred.
    alpha
        ``pval`` threshold above which dots are not shown
    norm_by_group
        Divide all mean values by the mean values of this group.
        This is done separately for the reference and the values to compare to.

    Returns
    -------
        Mapping[str, Any]:
            data to input to :func:`plot_intensity_change`.
    """
    _ensure_categorical(adata, groupby)
    if limit_to_groups is None:
        limit_to_groups = {}
    # subset data
    for key, groups in limit_to_groups.items():
        if not isinstance(groups, list):
            groups = [groups]
        adata = adata[adata.obs[key].isin(groups)]
    # which vars to show?
    if marker_dict is None:
        marker_dict = np.array(adata.var.index)
    if isinstance(marker_dict, dict):
        marker_list = np.concatenate(list(marker_dict.values()))
    else:
        marker_list = marker_dict
    # subset data to markers that we'd like to show
    adata = adata[:, marker_list].copy()

    # calculate values to show
    color_values = pd.DataFrame(index=adata.var.index)  # intensity values shown as colors
    p_values = pd.DataFrame(index=adata.var.index)  # pvalues (impacting dot sizes)
    size_values = pd.DataFrame(index=adata.var.index)  # dot sizes
    group_size = {}  # mean group sizes (for barplot)
    p_values_data = {}  # additional data returned from mixed model

    # define reference
    adata_ref = None
    if reference_group is None:
        reference_group = groupby
    if reference is not None:
        if not isinstance(reference, list):
            reference = [reference]
        adata_ref = adata[adata.obs[reference_group].isin(reference)]
        # subset adata to not reference
        adata = adata[~adata.obs[reference_group].isin(reference)]
        assert len(adata) > 0, f"no obs in adata that are not one of {reference} in {reference_group}"

    for g in adata.obs[groupby].cat.categories:
        # reference expression
        if reference is not None:
            assert adata_ref is not None
            if reference_group != groupby:
                # reference expression is the current group in the reference group
                # (which is a distinct grouping from the groupby categories)
                # print('reference expression is the current group in the reference group')
                adata_cur_ref = adata_ref[adata_ref.obs[groupby] == g]
            else:
                # reference expression is the reference group
                # print('reference expression is the reference group')
                adata_cur_ref = adata_ref
        else:
            # reference expression is everything except the current group (classic comparison of groupings)
            # print('reference expression is everything except the current group')
            adata_cur_ref = adata[adata.obs[groupby] != g]
        cur_ref_expr = adata_cur_ref.X
        if norm_by_group is not None:
            assert reference is not None, "Need a reference for norm by group"
            assert reference_group != groupby, "Can only norm by group if reference_group is different to groupby"
            assert adata_ref is not None
            cur_ref_expr = cur_ref_expr / adata_ref[adata_ref.obs[groupby] == norm_by_group].X
        cur_ref_size = adata_cur_ref.obs["size"]

        # group expression
        g_expr = adata[adata.obs[groupby] == g].X
        if norm_by_group is not None:
            g_expr = g_expr / adata[adata.obs[groupby] == norm_by_group].X
        g_size = adata[adata.obs[groupby] == g].obs["size"]

        # mean group expression
        mean_g = (g_expr * g_size[:, np.newaxis]).sum(axis=0) / g_size.sum()
        # mean reference expression
        mean_ref = (cur_ref_expr * cur_ref_size[:, np.newaxis]).sum(axis=0) / cur_ref_size.sum()
        # mean group size
        if group_sizes_barplot == "mean":
            group_size[g] = g_size.mean()
        elif group_sizes_barplot == "meanchange":
            group_size[g] = g_size.mean() - cur_ref_size.mean()
        elif group_sizes_barplot == "foldchange":
            group_size[g] = g_size.mean() / cur_ref_size.mean()
        else:
            group_size[g] = 0

        # set p values by testing if distribution of intensities is the same (without adjusting for size!)
        if pval == "mixed_model":
            if norm_by_group == g:
                # not not calc p values, all mean intensities will be 0
                pvals = np.array([1] * len(adata.var.index))
                pvals_data = {}
            else:
                g_well_name = adata[adata.obs[groupby] == g].obs["well_name_x"]
                pvals, pvals_data = mixed_model(
                    cur_ref_expr,
                    g_expr,
                    ref_well_name=adata_cur_ref.obs["well_name_x"],
                    g_well_name=g_well_name,
                )
        elif pval == "ttest":
            pvals_data = {}
            _, pvals = scipy.stats.ttest_ind(cur_ref_expr, g_expr, axis=0)
        else:
            raise NotImplementedError(pval)
        p_values[g] = pvals
        p_values_data[g] = pvals_data

        # set size values to mean intensity of reference group
        if size == "mean_reference":
            size_values[g] = mean_ref
            size_title = "mean intensity\nof reference"
        elif size == "pval":
            # size values are 1 for significant pvals, and 0.5 for non-significant pvals
            sizes = (pvals <= alpha).astype("float")
            sizes[sizes == 0] = 0.5
            size_values[g] = sizes
            size_title = "pvalue"
        else:
            raise NotImplementedError(size)
        # set color values
        if color == "logfoldchange":
            color_values[g] = np.log2(mean_g / mean_ref)
        elif color == "meanchange":
            color_values[g] = mean_g - mean_ref
        else:
            raise NotImplementedError(color)

    # get title for plot
    lmt_str = ", ".join([f'{key}: {",".join(val)}' for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        lmt_str = "all"
    if reference is not None:
        title = f'{color} of {lmt_str} wrt {reference_group}: {",".join([str(r) for r in reference])}'
    else:
        title = f"{color} of {lmt_str} wrt rest"
    cbar_title = f"{color} in group"
    if norm_by_group is not None:
        cbar_title = f"relative {color} in group\nwrt {norm_by_group}"

    return_dict = {
        "adata": adata,
        "color_values": color_values,
        "size_values": size_values,
        "p_values": p_values,
        "p_values_data": p_values_data,
        "group_size": group_size,
        "marker_dict": marker_dict,
        "groupby": groupby,
        "alpha": alpha,
        "plot_data": {
            "title": title,
            "colorbar_title": cbar_title,
            "size_title": size_title,
            "show_unsignificant_dots": size == "pval",
            "group_sizes_barplot": group_sizes_barplot,
        },
    }
    return return_dict


def plot_intensity_change(
    adata: ad.AnnData,
    color_values: pd.DataFrame,
    size_values: pd.DataFrame,
    p_values: pd.DataFrame,
    p_values_data: Mapping[str, Any],
    group_size: Mapping[str, Any],
    marker_dict: Optional[Union[Mapping[str, Iterable[str]], Iterable[str]]],
    groupby: str,
    plot_data: Mapping[str, Any],
    alpha: float,
    adjust_height: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Iterable[int] = (10, 3),
    save: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Plot mean intensity differences between perturbations or clusters.

    Takes returns of :func:`get_intensity_change` as input:
    ``plot_intensity_change(**get_intensity_change(...))``

    Parameters
    ----------
        adjust_height
            Option to make plots a bit more streamlined.
        ax
            Axis to plot in.
        figsize
            Size of figure.
        save
            Path to save figure to.
        kwargs
            Keyword arguments for :func:`sc.pl.dotplot`.
    """
    kwargs["vmin"] = kwargs.get("vmin", -1)
    kwargs["vmax"] = kwargs.get("vmax", 1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    scplot = sc.pl.dotplot(
        adata,
        var_names=marker_dict,
        groupby=groupby,
        dot_color_df=color_values.T,
        cmap="bwr",
        colorbar_title=plot_data["colorbar_title"],
        size_title=plot_data["size_title"],
        title=plot_data["title"],
        show=False,
        return_fig=True,
        ax=ax,
        **kwargs,
    )
    # set dot size
    scplot.dot_size_df = size_values.T
    # do not show unsignificant dots if size does not indicate this
    if not plot_data["show_unsignificant_dots"]:
        scplot.dot_size_df[p_values.T > alpha] = 0

    # add group sizes
    if plot_data["group_sizes_barplot"] is not None:
        group_size: pd.Series = pd.Series(data=group_size)  # type: ignore[no-redef]
        scplot.group_extra_size = 0.8
        scplot.plot_group_extra = {
            "kind": "group_totals",
            "width": 0.8,
            "sort": None,
            "counts_df": group_size,
            "color": None,
        }

    if adjust_height:
        _adjust_plotheight(scplot)

    scplot.make_figure()
    # add axis labels
    scplot.ax_dict["mainplot_ax"].set_xlabel("channel")
    scplot.ax_dict["mainplot_ax"].set_ylabel(groupby)

    # allow negative values in barplot
    if plot_data["group_sizes_barplot"] == "meanchange":
        assert isinstance(group_size, pd.Series)
        scplot.ax_dict["group_extra_ax"].set_xlim(
            (
                group_size.min() - np.abs(group_size.min()) * 0.4,
                group_size.max() + np.abs(group_size.max() * 0.4),
            )
        )

    if save is not None:
        plt.savefig(save, dpi=100)


def plot_size_change(
    adata: ad.AnnData,
    groupby_row: str = "cluster",
    groupby_col: str = "well_name",
    reference_row: Optional[str] = None,
    reference_col: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    adjust_height: bool = True,
    ax: MplAxes = None,
    pval: float = 0.05,
    save: Optional[str] = None,
    size: str = "mean_size",
    limit_to_groups: Optional[Mapping[str, Union[List[str], str]]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot mean intensity differences between perturbations and clusters.

    Parameters
    ----------
    adata
        Adata containing aggregated information by clusters.
        E.g. result of :meth:`FeatureExtractor.get_intensity_adata`.
    groupby_row
        Column in ``adata.obs`` containing the row-wise grouping.
    groupby_col
        Column in ``adata.obs`` containing the column-wise grouping.
    reference_row
        Reference cluster/perturbation to compare to row-wise.
        If not defined, will compare each value in groupby_row against the rest.
    reference_col
        Reference cluster/perturbation to compare to col-wise.
        If not defined, will compare each value in groupby_col against the rest.
    figsize
        Size of figure.
    adjust_height
        Option to make plots a bit more streamlined.
    ax
        Axis to plot in.
    pval
        ``pval`` threshold above which dots are not shown.
    save
        Path to save figure to.
    size
        Sizes of dots, either `mean_size` or `pval` (distinguish significant and non-significant dots).
    limit_to_groups
        Dict with obs as keys and groups from obs as values, to subset adata before plotting.
    kwargs
        Keyword arguments for :func:`sc.pl.dotplot`.
    """
    if limit_to_groups is None:
        limit_to_groups = {}
    assert (reference_col is None) ^ (reference_row is None), "either reference_row or reference_col must be defined"
    _ensure_categorical(adata, groupby_row)
    _ensure_categorical(adata, groupby_col)
    kwargs["vmin"] = kwargs.get("vmin", -1)
    kwargs["vmax"] = kwargs.get("vmax", 1)
    # subset data
    for key, groups in limit_to_groups.items():
        if not isinstance(groups, list):
            groups = [groups]
        adata = adata[adata.obs[key].isin(groups)]
    col_grps = adata.obs[groupby_col].cat.categories
    row_grps = adata.obs[groupby_row].cat.categories

    # calculate mean sizes to plot later
    grp_df = adata.obs.groupby([groupby_row, groupby_col])
    sizes_adata = ad.AnnData(grp_df.mean()["size"].unstack())
    sizes_adata.obs["group"] = sizes_adata.obs.index.astype("category")

    # calculate values to show
    data = grp_df.mean()["size"].unstack()
    color_values = pd.DataFrame(index=row_grps, columns=col_grps)
    p_values = pd.DataFrame(index=row_grps, columns=col_grps)
    size_values = pd.DataFrame(index=row_grps, columns=col_grps)

    # assign color_values and size_values
    if reference_row is not None:
        color_values = np.log2(data.divide(data.loc[reference_row], axis="columns"))
        size_values.loc[:, :] = np.array(data.loc[reference_row])[np.newaxis, :]
    if reference_col is not None:
        color_values = np.log2(data.divide(data.loc[:, reference_col], axis="rows"))
        size_values.loc[:, :] = np.array(data.loc[:, reference_col])[:, np.newaxis]

    # assign p_values
    for r in row_grps:
        for c in col_grps:
            if reference_row is not None:
                ref_dist = grp_df.get_group((reference_row, c))["size"]
            if reference_col is not None:
                ref_dist = grp_df.get_group((r, reference_col))["size"]
            cur_dist = grp_df.get_group((r, c))["size"]
            _, p = scipy.stats.ttest_ind(ref_dist, cur_dist)
            p_values.loc[r, c] = p

    # set size values
    if size == "mean_size":
        # size values is already correct
        size_title = "mean size\nof reference (%)"
    elif size == "pval":
        # size values are 1 for significant pvals, and 0.5 for non-significant pvals
        for r in row_grps:
            for c in col_grps:
                size_values.loc[r, c] = 1 if p_values.loc[r, c] <= pval else 0.5
        size_title = "pvalue"
    else:
        raise NotImplementedError(size)
    size_values = size_values.astype("float")
    size_values = size_values / size_values.max()

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    lmt_str = ", ".join([f'{key}: {",".join(val)}' for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        lmt_str = "all"
    title = f"size logfoldchange of {lmt_str} wrt "
    if reference_col is not None:
        title += f"{groupby_col}: {reference_col}"
    else:
        title += f"{groupby_row}: {reference_row}"
    scplot = sc.pl.dotplot(
        sizes_adata,
        var_names=sizes_adata.var_names,
        groupby="group",
        dot_color_df=color_values,
        dot_size_df=size_values,
        cmap="bwr",
        colorbar_title="logfolchange in group",
        size_title=size_title,
        title=title,
        show=False,
        return_fig=True,
        ax=ax,
        **kwargs,
    )

    # do not show unsignificant dots if size does not indicate this
    if size != "pval":
        scplot.dot_size_df[p_values > pval] = 0

    if adjust_height:
        _adjust_plotheight(scplot)

    scplot.make_figure()

    if save is not None:
        plt.savefig(save, dpi=100)

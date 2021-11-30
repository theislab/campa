import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm

def zscore(adata, key_added='zscored', limit_to_groups={}):
    """
    scale adata based on values in cluster "all".
    can subset adata to perturbation before calculating mean intensities (i.e. reference perturbation)
    adds 
        adata.layers[key_added]
        adata.var['mean_intensity'], adata.var['std_intensity']
    """
    cur_adata = adata[adata.obs['cluster']=='all']
    for key, val in limit_to_groups.items():
        cur_adata = cur_adata[cur_adata.obs[key]==val]
    # mean intensity per channel
    mean_intensity = (cur_adata.X * cur_adata.obs['size'][:, np.newaxis]).sum(axis=0) / cur_adata.obs['size'].sum()
    # std per channel
    std = np.sqrt(((cur_adata.X - mean_intensity)**2).mean(axis=0))
    # scale values
    adata.layers[key_added] = (adata.X - mean_intensity) / std
    adata.var['mean_intensity'] = mean_intensity
    adata.var['std_intensity'] = std

def _adjust_plotheight(scplot):
    """
    fig large gap between title and plot for scanpy plots 
    (rather hacky, might not work in all cases)
    """
    # modified code from sc.pl.MatrixPlot.make_figure
    category_height = scplot.DEFAULT_CATEGORY_HEIGHT
    category_width = scplot.DEFAULT_CATEGORY_WIDTH
    mainplot_height = len(scplot.categories) * category_height
    mainplot_width = (
        len(scplot.var_names) * category_width + scplot.group_extra_size
    )
    if scplot.are_axes_swapped:
        mainplot_height, mainplot_width = mainplot_width, mainplot_height

    height = mainplot_height #+ 1  # +1 for labels

    # if the number of categories is small use
    # a larger height, otherwise the legends do not fit
    scplot.height = max([scplot.min_figure_height, height])
    scplot.width = mainplot_width + scplot.legends_width
    
def _ensure_categorical(adata, col):
    if isinstance(adata.obs[col].dtype, pd.CategoricalDtype):
        # nothing todo
        return
    adata.obs[col] = adata.obs[col].astype(str).astype('category')
    return
    
# TODO add group size similar to dotplot here!
def plot_mean_intensity(adata, groupby='cluster', marker_dict=None, save=None, dendrogram=False, 
                               limit_to_groups={}, layer=None, type='matrixplot',
                               cmap='viridis', adjust_height=True,
                               figsize=(10,5), ax=None, **kwargs):
    """
    show per cluster intensity of each channel.
    intensity is either shown as mean or z-scored intensity
    """
    if layer == 'zscored' and 'zscored' not in adata.layers.keys():
        print('Compute adata.layers[zscored] first!')
        return
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
        if layer == None:
            color = 'mean intensity'
            g_expr = adata[adata.obs[groupby]==g].X
        elif layer == 'zscored':
            color = 'mean zscore'
            g_expr = adata[adata.obs[groupby]==g].layers['zscored']
        else:
            raise NotImplementedError(layer)
        g_size = adata[adata.obs[groupby]==g].obs['size']
        color_values[g] = (g_expr * g_size[:, np.newaxis]).sum(axis=0) / g_size.sum()
    color_values = color_values.loc[marker_list]
    
    # plot
    if dendrogram:
        sc.tl.dendrogram(adata, groupby=groupby)
    title = 'mean intensity in ' + ', '.join([f'{key}: {val}' for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        title = 'mean intensity'
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    
    if type == 'violinplot':
        scplot = sc.pl.stacked_violin(adata, var_names=marker_dict, groupby=groupby, #standard_scale='var', 
                         ax=ax, dendrogram=dendrogram, layer=layer, return_fig=True, title=title, **kwargs)
    elif type == 'matrixplot':
        scplot = sc.pl.matrixplot(adata, var_names=marker_dict, layer=layer, groupby=groupby, 
                         cmap=cmap, 
                        colorbar_title=color, ax=ax, 
                         return_fig=True, dendrogram=dendrogram, values_df=color_values.T, title=title, **kwargs)
    else:
        raise NotImplementedError(type)
    
    if adjust_height:
        _adjust_plotheight(scplot)
  
    scplot.make_figure()
    # add axis labels
    scplot.ax_dict['mainplot_ax'].set_xlabel('channels')
    scplot.ax_dict['mainplot_ax'].set_ylabel(groupby)
    
    if save is not None:
        plt.savefig(save, dpi=100)


def plot_mean_size(adata, groupby_row='cluster', groupby_col='well_name', normby_row=None, normby_col=None, ax=None, figsize=None, 
                      adjust_height=False, save=None, **kwargs):
    """
    plot mean cluster sizes per cell, grouped by different columns in obs
    """
    _ensure_categorical(adata, groupby_row)
    _ensure_categorical(adata, groupby_col)
    # groupy_col needs to be var
    sizes = {c: adata[adata.obs[groupby_col]==c].obs.groupby(groupby_row).mean()['size'] 
             for c in adata.obs[groupby_col].cat.categories}
    sizes_adata = ad.AnnData(pd.DataFrame(sizes))
    sizes_adata.obs['group'] = sizes_adata.obs.index.astype('category')

    # get values to show
    values_df = pd.DataFrame(sizes)
    title_suffix=''
    if normby_row is not None:
        values_df = values_df.divide(values_df.loc[normby_row], axis='columns')
        title_suffix += f'\nnormalised by {groupby_row} {normby_row}'
    if normby_col is not None:
        values_df = values_df.divide(values_df.loc[:, normby_col], axis='rows')
        title_suffix += f'\nnormalised by {groupby_col} {normby_col}'
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    scplot = sc.pl.matrixplot(sizes_adata, var_names=sizes_adata.var_names, groupby='group', 
                     values_df = values_df,
                     colorbar_title='mean size\nin group',
                    title='mean object size'+title_suffix, ax=ax, show=False, return_fig=True, **kwargs)
    if adjust_height:
        _adjust_plotheight(scplot)
    scplot.make_figure()
    
    if save is not None:
        plt.savefig(save, dpi=100)

        


def mixed_model(ref_expr, g_expr, ref_well_name, g_well_name):
    #res_data = {'resid'}
    res_data = {'df':[], 'resid':[]}
    #if True:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        #warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
        # iterate over all channels in the data
        pvals = []
        for i in range(ref_expr.shape[-1]):
            # create dataframe for mixed model
            df = pd.DataFrame(index=range(len(ref_expr)+len(g_expr)))
            df['mean_expr'] = np.log2(np.concatenate([ref_expr[:,i], g_expr[:,i]]))
            df['group'] = [0] * len(df.index)
            df['group'].iloc[len(ref_expr):] = 1
            df['well'] = np.concatenate([ref_well_name, g_well_name])
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna()
            #display(df)
            #sns.distplot(df[df['group']==0]['mean_expr'])
            #sns.distplot(df[df['group']==1]['mean_expr'])

            model = sm.MixedLM.from_formula("mean_expr ~ group",
                                     re_formula="~1",
                                     groups="well",
                                     data=df)
            result = model.fit()
            pvals.append(result.summary().tables[1].loc['group']['P>|z|'])
            res_data['df'].append(df)
    return np.array(pvals).astype('float'), res_data

def get_intensity_change(adata, groupby, marker_dict=None, limit_to_groups={}, reference=None, reference_group=None, color='logfoldchange', size='mean_reference', figsize=(10,3), return_data=False, group_sizes_barplot=None, pval='ttest', alpha=0.05, norm_by_group=None):
    """
    get data needed to call plot_intensity_change.
    
    Calculate mean intensity differences between perturbations or clusters.
    if no reference is given, use all other groups (except the current one) as reference
    colors show log2fc / mean intensity changes / zscore changes
    dot size shows mean intensity of reference group that is compared to, or indicates the pvalue
    
    Args:
        adata: adata containing aggregated information by clusters
        marker_dict: limit/group vars that are shown, either by passing list or dict (adds annotations to plot)
        limit_to_groups: dict with obs as keys and groups from obs as values, to subset adata before plotting
        reference: reference cluster/perturbation to compare to
        reference_group: obs entry that contains reference grouping (if None, groupby is used)
        color: color of dots, either 'logfoldchange' or 'meanchange'
        size: sizes of dots, either 'mean_reference' or 'pval' (distinguish significant and non-significant dots)
        group_sizes_barplot: mean size of groups shown as a bar plot to the right. 
            Either None (do not show), 'mean' (mean size of groups), 
            'meanchange' (mean difference of group size from reference), 'foldchange'
        pval: type of test done to determine pvalues. Either 'ttest' or 'mixed_model'. 
            'mixed_model' calculates a mixed model using wells as random effects.
        alpha: pvalue threshold above which dots are not shown
        norm_by_group: divide all mean values by the mean values of this group. 
            This is done separately for the reference and the values to compare to.
        
    Returns:
        dict with keys 
    """
    _ensure_categorical(adata, groupby)
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
    group_size = {} # mean group sizes (for barplot)
    p_values_data = {} # additional data returned from mixed model
    
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
    
    for g in adata.obs[groupby].cat.categories:
        # reference expression
        if reference is not None:
            if reference_group != groupby:
                # reference expression is the current group in the reference group (which is a distinct grouping from the groupby categories)
                #print('reference expression is the current group in the reference group')
                adata_cur_ref = adata_ref[adata_ref.obs[groupby]==g]
            else:
                # reference expression is the reference group
                #print('reference expression is the reference group')
                adata_cur_ref = adata_ref
        else:
            # reference expression is everything except the current group (classic comparison of groupings)
            #print('reference expression is everything except the current group')
            adata_cur_ref = adata[adata.obs[groupby]!=g]
        cur_ref_expr = adata_cur_ref.X
        if norm_by_group is not None:
            assert reference is not None and reference_group != groupby, "Can only norm by group if reference_group is different to groupby"
            cur_ref_expr = cur_ref_expr / adata_ref[adata_ref.obs[groupby]==norm_by_group].X
        cur_ref_size = adata_cur_ref.obs['size']
        
        # group expression
        g_expr = adata[adata.obs[groupby]==g].X
        if norm_by_group is not None:
            g_expr = g_expr / adata[adata.obs[groupby]==norm_by_group].X
        g_size = adata[adata.obs[groupby]==g].obs['size']
        
        # mean group expression
        mean_g = (g_expr * g_size[:, np.newaxis]).sum(axis=0) / g_size.sum()
        # mean reference expression
        mean_ref = (cur_ref_expr * cur_ref_size[:, np.newaxis]).sum(axis=0) / cur_ref_size.sum()
        # mean group size
        if group_sizes_barplot == 'mean':
            group_size[g] = g_size.mean()
        elif group_sizes_barplot == 'meanchange':
            group_size[g] = g_size.mean() - cur_ref_size.mean()
        elif group_sizes_barplot == 'foldchange':
            group_size[g] = g_size.mean() / cur_ref_size.mean()
        else:
            group_size[g] = 0
        
        # set p values by testing if distribution of intensities is the same (without adjusting for size!)
        if pval == 'mixed_model':
            if norm_by_group == g:
                # not not calc p values, all mean intensities will be 0
                pvals = np.array([1] * len(adata.var.index))
                pvals_data = {}
            else:
                g_well_name = adata[adata.obs[groupby]==g].obs['well_name_x']
                pvals, pvals_data = mixed_model(cur_ref_expr, g_expr, ref_well_name=adata_cur_ref.obs['well_name_x'], g_well_name=g_well_name)
        elif pval == 'ttest':
            pvals_data = {}
            _, pvals = scipy.stats.ttest_ind(cur_ref_expr, g_expr, axis=0)
        else:
            raise NotImplementedError(pval)
        p_values[g] = pvals
        p_values_data[g] = pvals_data
        
        # set size values to mean intensity of reference group
        if size == 'mean_reference':
            size_values[g] = mean_ref
            size_title = 'mean intensity\nof reference'
        elif size == 'pval':
            # size values are 1 for significant pvals, and 0.5 for non-significant pvals
            sizes = (pvals <= alpha).astype('float')
            sizes[sizes==0] = 0.5
            size_values[g] = sizes
            size_title = 'pvalue'
        else:
            raise NotImplementedError(size)
        # set color values
        if color == 'logfoldchange':
            color_values[g] = np.log2(mean_g/mean_ref)
        elif color == 'meanchange':
            color_values[g] = mean_g-mean_ref
        else:
            raise NotImplementedError(color)
            
    # get title for plot
    lmt_str = ', '.join([f'{key}: {",".join(val)}' for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        lmt_str = 'all'
    title = f'{color} of {lmt_str} wrt {reference_group}: {",".join([str(r) for r in reference])}'
    cbar_title = f'{color} in group'
    if norm_by_group is not None:
        cbar_title = f'relative {color} in group\nwrt {norm_by_group}'
        
    return_dict = {
        'adata': adata,
        'color_values': color_values,
        'size_values': size_values,
        'p_values': p_values,
        'p_values_data': p_values_data,
        'group_size': group_size,
        'marker_dict': marker_dict,
        'groupby': groupby,
        'alpha': alpha,
        'plot_data': {
            'title': title,
            'colorbar_title': cbar_title,
            'size_title': size_title,
            'show_unsignificant_dots': size == 'pval',
            'group_sizes_barplot': group_sizes_barplot,
        }
    }
    return return_dict
    

def plot_intensity_change(adata, color_values, size_values, p_values, p_values_data, group_size,
                          marker_dict, groupby, plot_data, alpha, adjust_height=True, ax=None, figsize=(10,3), 
                          save=None, **kwargs):
    """
    plot mean intensity differences between perturbations or clusters.
    Takes returns of get_intensity_change as input.
    
    Args:
        ... results of get_intensity_change
        adjust_height: option to make plots a bit more streamlined
        ax: axis to plot in
        figsize: size of figure
        save: path to save figure to
        kwargs: keyword arguments for sc.pl.dotplot
    """
    kwargs['vmin'] = kwargs.get('vmin', -1)
    kwargs['vmax'] = kwargs.get('vmax', 1)
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
    scplot = sc.pl.dotplot(adata, var_names=marker_dict, groupby=groupby, dot_color_df=color_values.T,
                        cmap='bwr',
                        colorbar_title=plot_data['colorbar_title'], 
                        size_title=plot_data['size_title'], title=plot_data['title'], 
                        show=False, return_fig=True, ax=ax, **kwargs)
    # set dot size
    scplot.dot_size_df = size_values.T
    # do not show unsignificant dots if size does not indicate this
    if not plot_data['show_unsignificant_dots']:
        scplot.dot_size_df[p_values.T > alpha] = 0
    
    # add group sizes
    if plot_data['group_sizes_barplot'] != None:
        group_size = pd.Series(data=group_size)
        scplot.group_extra_size = 0.8
        scplot.plot_group_extra = {'kind': 'group_totals', 'width': 0.8, 'sort': None, 'counts_df':group_size, 'color':None}
    
    if adjust_height:
        _adjust_plotheight(scplot)

    scplot.make_figure()
    # add axis labels
    scplot.ax_dict['mainplot_ax'].set_xlabel('channel')
    scplot.ax_dict['mainplot_ax'].set_ylabel(groupby)
    
    # allow negative values in barplot
    if plot_data['group_sizes_barplot'] == 'meanchange': 
        scplot.ax_dict['group_extra_ax'].set_xlim((group_size.min()-np.abs(group_size.min())*0.4, group_size.max()+np.abs(group_size.max()*0.4)))
    
    if save is not None:
        plt.savefig(save, dpi=100)

        
def plot_size_change(adata, groupby_row='cluster', groupby_col='well_name', reference_row=None, 
                      reference_col=None, figsize=None, adjust_height=True, ax=None, pval=0.05,
                      save=None, size='mean_size',
                      limit_to_groups={}, **kwargs
                        ):
    """
    size: sizes of dots, either 'mean_size' or 'pval' (distinguish significant and non-significant dots)
    
    """
    assert (reference_col==None) ^ (reference_row==None), "either reference_row or reference_col must be defined"
    _ensure_categorical(adata, groupby_row)
    _ensure_categorical(adata, groupby_col)
    kwargs['vmin'] = kwargs.get('vmin', -1)
    kwargs['vmax'] = kwargs.get('vmax', 1)
    # subset data
    for key, groups in limit_to_groups.items():
        if not isinstance(groups, list):
            groups = [groups]
        adata = adata[adata.obs[key].isin(groups)]
    col_grps = adata.obs[groupby_col].cat.categories
    row_grps = adata.obs[groupby_row].cat.categories
    
    # calculate mean sizes to plot later
    grp_df = adata.obs.groupby([groupby_row, groupby_col])
    sizes_adata = ad.AnnData(grp_df.mean()['size'].unstack())
    sizes_adata.obs['group'] = sizes_adata.obs.index.astype('category')
    
    # calculate values to show
    data = grp_df.mean()['size'].unstack()
    color_values = pd.DataFrame(index=row_grps, columns=col_grps)
    p_values = pd.DataFrame(index=row_grps, columns=col_grps)
    size_values = pd.DataFrame(index=row_grps, columns=col_grps)
    
    # assign color_values and size_values
    if reference_row is not None:
        color_values = np.log2(data.divide(data.loc[reference_row], axis='columns'))
        size_values.loc[:,:] = np.array(data.loc[reference_row])[np.newaxis,:]
    if reference_col is not None:
        color_values = np.log2(data.divide(data.loc[:, reference_col], axis='rows'))
        size_values.loc[:,:] = np.array(data.loc[:, reference_col])[:, np.newaxis]
        
    # assign p_values
    for r in row_grps:
        for c in col_grps:
            if reference_row is not None:
                ref_dist = grp_df.get_group((reference_row, c))['size']
            if reference_col is not None:
                ref_dist = grp_df.get_group((r, reference_col))['size']
            cur_dist = grp_df.get_group((r, c))['size']
            _, p = scipy.stats.ttest_ind(ref_dist, cur_dist)
            p_values.loc[r,c] = p

    # set size values
    if size == 'mean_size':
        # size values is already correct
        size_title = 'mean size\nof reference (%)'
    elif size == 'pval':
        # size values are 1 for significant pvals, and 0.5 for non-significant pvals
        for r in row_grps:
            for c in col_grps:
                size_values.loc[r,c] = 1 if p_values.loc[r,c] <= pval else 0.5
        size_title = 'pvalue'
    else:
        raise NotImplementedError(size)
    size_values = size_values.astype('float')
    size_values = size_values / size_values.max()
            
    # plot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        
    lmt_str = ', '.join([f'{key}: {",".join(val)}' for key, val in limit_to_groups.items()])
    if limit_to_groups == {}:
        lmt_str = 'all'
    title = f'size logfoldchange of {lmt_str} wrt '
    if reference_col is not None:
        title += f'{groupby_col}: {reference_col}'
    else:
        title += f'{groupby_row}: {reference_row}'
    scplot = sc.pl.dotplot(sizes_adata, var_names=sizes_adata.var_names, groupby='group', dot_color_df=color_values,
                           dot_size_df=size_values,
                        cmap='bwr', colorbar_title='logfolchange in group', 
                        size_title=size_title, title=title, 
                        show=False, return_fig=True, ax=ax, **kwargs)

    # do not show unsignificant dots if size does not indicate this
    if size != 'pval':
        scplot.dot_size_df[p_values > pval] = 0
    
    if adjust_height:
        _adjust_plotheight(scplot)

    scplot.make_figure()
    
    if save is not None:
        plt.savefig(save, dpi=100)
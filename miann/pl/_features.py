import numpy as np


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
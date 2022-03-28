# -- Parameters determining what features are extracted from a given clustering --
feature_params = {
    # -- set up data to extract features from --
    # path to experiment directory relative to EXPERIMENT_DIR
    'experiment_dir': 'test/CondVAE_pert-CC',
    # name of clustering to use
    'cluster_name': "clustering_res0.2",
    # dir of subsampled clustering to load annotation. Relative to experiment_dir. 
    # Default is taking first of experiment_dir/aggregated/sub-*
    'cluster_dir': None,
    # cluster annotation to use. Defaults to cluster_name
    'cluster_col': None,
    # data dirs to be processed. 
    # Relative to experiment_dir/aggregated/full_data. 
    # If None, all available data_dirs will be processed
    'data_dirs': ["184A1_unperturbed/I09", "184A1_unperturbed/I11", "184A1_meayamycin/I12", "184A1_meayamycin/I20"],
    #'data_dirs': [
    #    "184A1_unperturbed/I09", "184A1_unperturbed/I11", "184A1_unperturbed/J10", "184A1_unperturbed/J12", 
    #    "184A1_DMSO/I14", "184A1_DMSO/J16", "184A1_AZD4573/I13", "184A1_AZD4573/I17", "184A1_AZD4573/J14",
    #    "184A1_AZD4573/J18", "184A1_AZD4573/J21", "184A1_CX5461/I18", "184A1_CX5461/J09", "184A1_CX5461/J22",
    #    "184A1_TSA/I16", "184A1_TSA/J13", "184A1_TSA/J20", "184A1_triptolide/I10", "184A1_triptolide/J15",
    #    "184A1_meayamycin/I12", "184A1_meayamycin/I20"
    #    ],
    # filename to use for saving extracted features.
    'save_name': None,
    # force calculation even when adata exists
    'force': False,
    # -- features to extract --
    # type of features to extract. One or more of intensity, co-occurrence, object-stats
    # Intensity: per-cluster mean and size features. Needs to be calculated first to set up the adata. 
    # Co-occurrence: spatial co-occurrence between pairs of clusters at different distances. 
    # Object stats: number and area of connected components per cluster
    'features': [],
    # parameters for co-occurrence calculation
    'co_occurrence_params': {
        # size of distances interval
        'min': 2.0,
        'max': 80.0,
        'nsteps': 20,
        # use log spacing of co-occurrence intervals
        'logspace': True,
        # number of processes to use to compute co-occurrence scores
        'num_processes': 8
    },
    # parameters for object-stats calculation
    'object_stats_params': {
        # features to extract in mode object-stats
        # possible features: ['area', 'circularity', 'elongation', 'extent']
        'features': ['area', 'circularity', 'elongation', 'extent'],
        # intensity channels to extract mean per cluster from
        'channels': []
    }
}

# use this list to extract several different features at once.
# final feature params for entry i are obtained by `feature_params.update(variable_feature_params[i])`
variable_feature_params = [
    # intensity + co-occurrence + object stats for annotated clustering
    {
        'save_name': 'features_annotation.h5ad',
        'cluster_col': 'annotation',
        'features': ['intensity', 'co-occurrence', 'object-stats'],

    }
]
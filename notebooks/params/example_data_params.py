import os

data_params = {
    # name of the resulting dataset that is defined by these params (relative to DATA_DIR/datasets)
    "dataset_name": "184A1_test_dataset",
    # name of data config (registered in config.ini)
    "data_config": "ExampleData",
    # where to read data from (relative to DATA_DIR defined in data config)
    "data_dirs": [os.path.join("184A1_unperturbed", well) for well in ["I09", "I11"]]
    + [os.path.join("184A1_meayamycin", well) for well in ["I12", "I20"]],
    "channels": [
        "01_CDK9_pT186",
        "01_PABPC1",
        "02_CDK7",
        "03_CDK9",
        "03_RPS6",
        "05_GTF2B",
        "05_Sm",
        "07_POLR2A",
        "07_SETD1A",
        "08_H3K4me3",
        "09_CCNT1",
        "09_SRRM2",
        "10_H3K27ac",
        "10_POL2RA_pS2",
        "11_KPNA2_MAX",
        "11_PML",
        "12_RB1_pS807_S811",
        "12_YAP1",
        "13_PABPN1",
        "13_POL2RA_pS5",
        "14_PCNA",
        "15_SON",
        "15_U2SNRNPB",
        "16_H3",
        "17_HDAC3",
        "17_SRSF2",
        "18_NONO",
        "19_KPNA1_MAX",
        "20_ALYREF",
        "20_SP100",
        "21_COIL",
        "21_NCL",
        "00_DAPI",
        "07_H2B",
    ],
    # list of conditions. Should be defined in data config.
    # The suffix '_one_hot' will convert the condition in a one-hot encoded vector.
    # Conditions are concatenated, except when they are defined as a list of lists.
    # In this case the condition is defined as a pairwise combination of the conditions.
    "condition": ["perturbation_duration_one_hot", "cell_cycle_one_hot"],
    "condition_kwargs": {"cond_params": {}},
    # train/val/test split
    "split_kwargs": {
        "train_frac": 0.7,
        "val_frac": 0.2,
    },
    "test_img_size": 225,
    # subset to objects with certain metadata.
    "subset": True,
    # kwargs to MPPData.subset() defining which object to subset to
    "subset_kwargs": {
        "frac": None,  # special kwarg for random subsetting of objects
        "nona_condition": True,  # special kwarg for removing all objects with NAN condition
        "cell_cycle": "NO_NAN",
    },
    # subsampling of pixels (only for train/val)
    "subsample": True,
    # kwargs for MPPData.subsample() defining the fraction of pixels to be sampled
    "subsample_kwargs": {
        "frac": 0.1,
        "frac_per_obj": None,
        "num": None,
        "num_per_obj": None,
    },
    # neighborhood information
    "neighborhood": True,
    "neighborhood_size": 3,
    # normalisation
    "normalise": True,
    "normalise_kwargs": {
        # background_value is column name in CHANNELS_METADATA, or list of floats per channel
        "background_value": "mean_background",
        "percentile": 98.0,
        "rescale_values": [],
    },
    # make results reproducible
    "seed": 42,
}

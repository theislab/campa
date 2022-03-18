from campa.tl import LossEnum, ModelEnum

base_config = {
    "experiment": {
        "dir": "test",
        "name": None,
        "save_config": True,
    },
    "data": {
        "data_config": "TestData",
        "dataset_name": "184A1_test_dataset",
        "output_channels": None,
    },
    "model": {
        "model_cls": None,
        "model_kwargs": {
            "num_neighbors": 3,
            "num_channels": 34,
            "num_output_channels": 34,
            "latent_dim": 16,
            # encoder definition
            "encoder_conv_layers": [32],
            "encoder_conv_kernel_size": [1],
            "encoder_fc_layers": [32, 16],
            # decoder definition
            "decoder_fc_layers": [],
        },
        # if true, looks for saved weights in experiment_dir
        # if a path, loads these weights
        "init_with_weights": False,
    },
    "training": {
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 128,
        "loss": {"decoder": LossEnum.SIGMA_MSE, "latent": LossEnum.KL},
        "metrics": {"decoder": LossEnum.MSE_metric, "latent": LossEnum.KL},
        # saving models
        "save_model_weights": True,
        "save_history": True,
        "overwrite_history": True,
    },
    "evaluation": {
        "split": "val",
        "predict_reps": ["latent", "decoder"],
        "img_ids": 2,
        "predict_imgs": True,
        "predict_cluster_imgs": True,
    },
    "cluster": {  # cluster config, also used in this format for whole data clustering
        "cluster_name": "clustering",
        "cluster_rep": "latent",
        "cluster_method": "leiden",  # leiden or kmeans
        "leiden_resolution": 0.8,
        "subsample": None,  # 'subsample' or 'som'
        "subsample_kwargs": {},
        "som_kwargs": {},
        "umap": True,
    },
}

variable_config = [
    # unconditional model
    {
        "experiment": {"name": "VAE"},
        "model": {
            "model_cls": ModelEnum.VAEModel,
        },
    },
    # conditional VAE model
    {
        "experiment": {"name": "CondVAE_pert-CC"},
        "model": {
            "model_cls": ModelEnum.VAEModel,
            "model_kwargs": {
                "num_conditions": 14,
                "encode_condition": [10, 10],
            },
        },
    },
    # MPPleiden model (non-trainable)
    {
        "experiment": {"name": "MPPleiden"},
        "model": None,
        "training": None,
        "evaluation": {"predict_reps": [], "predict_imgs": False},
        "cluster": {"cluster_rep": "mpp", "leiden_resolution": 2},
    },
]

# tests saving and loading weights from experiment
import numpy as np
import pandas as pd

from .test_workflow import id_generator, _ensure_test_data  # noqa: <I252>


def test_save_load_model(_ensure_test_data):
    from campa.tl import LossEnum, Estimator, ModelEnum, Experiment

    model_name = id_generator(size=6)
    experiment_config = {
        "experiment": {
            "dir": "test_experiment",
            "name": model_name,
            "save_config": True,
        },
        "data": {
            "data_config": "TestData",
            "dataset_name": "reference_dataset",
            "output_channels": None,
        },
        "model": {
            "model_cls": ModelEnum.VAEModel,
            "model_kwargs": {
                "num_neighbors": 3,
                "num_channels": 5,
                "num_output_channels": 5,
                "latent_dim": 4,
                # encoder definition
                "encoder_conv_layers": [16],
                "encoder_conv_kernel_size": [1],
                "encoder_fc_layers": [8],
                # decoder definition
                "decoder_fc_layers": [],
                "num_conditions": 6,
                "encode_condition": [6],
            },
            # if true, looks for saved weights in experiment_dir
            # if a path, loads these weights
            "init_with_weights": False,
        },
        "training": {
            "learning_rate": 0.001,
            "epochs": 1,
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
            "img_ids": 1,
            "predict_imgs": True,
            "predict_cluster_imgs": True,
        },
        "cluster": {  # cluster config, also used in this format for whole data clustering
            "cluster_name": "clustering",
            "cluster_rep": "latent",
            "cluster_method": "leiden",  # leiden or kmeans
            "leiden_resolution": 0.2,
            "subsample": True,  # 'subsample' or 'som'
            "subsample_kwargs": {"frac": 0.1},
            "som_kwargs": {},
            "umap": True,
        },
    }

    exp = Experiment(experiment_config)
    est = Estimator(exp)
    history = est.train_model()
    scores = est.evaluate_model(est.val_dataset)

    # reload estimator and init with saved weights
    exp2 = Experiment.from_dir(f"{exp.dir}/{exp.name}").set_to_evaluate()
    est2 = Estimator(exp2)
    scores2 = est2.evaluate_model(est2.val_dataset)
    history2 = pd.read_csv(est.history_name, index_col=0)

    print(scores, scores2)
    assert np.isclose(scores, scores2, rtol=0.1).all()

    # check that history was correctly saved
    history2 = pd.read_csv(est.history_name, index_col=0)
    assert np.isclose(history, history2).all()

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from campa.tl import Experiment
    import numpy as np

import os
import logging

import pandas as pd
import tensorflow as tf

from campa.tl import LossEnum, ModelEnum
from campa.data import NNDataset
from campa.tl._layers import UpdateSparsityLevel


# --- Callbacks ---
class LossWarmup(tf.keras.callbacks.Callback):
    def __init__(self, weight_vars, to_weights, to_epochs):
        super().__init__()
        self.to_weights = to_weights
        self.to_epochs = to_epochs
        self.weight_vars = weight_vars

    def on_epoch_begin(self, epoch, logs=None):
        for key in self.to_epochs.keys():
            to_epoch = self.to_epochs[key]
            to_weight = self.to_weights[key]
            if to_epoch == 0 or to_epoch <= epoch:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight)
            else:
                tf.keras.backend.set_value(self.weight_vars[key], to_weight / to_epoch * epoch)
            print(f"set {key} loss weight to {tf.keras.backend.get_value(self.weight_vars[key])}")

        if "latent" in self.weight_vars.keys():
            print("set latent loss weight to {}".format(tf.keras.backend.get_value(self.weight_vars["latent"])))


class AnnealTemperature(tf.keras.callbacks.Callback):
    def __init__(self, temperature, initial_temperature, final_temperature, to_epoch):
        super().__init__()
        self.temperature = temperature
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.to_epoch = to_epoch

    def on_epoch_begin(self, epoch, logs=None):
        """Update temperature"""
        if self.to_epoch == 0 or self.to_epoch <= epoch:
            tf.keras.backend.set_value(self.temperature, self.final_temperature)
        else:
            tf.keras.backend.set_value(
                self.temperature,
                self.initial_temperature + (self.final_temperature - self.initial_temperature) / self.to_epoch * epoch,
            )
        print(f"set temperature to {tf.keras.backend.get_value(self.temperature)}")


# --- Estimator class ---
class Estimator:
    """
    Neural network estimator.

    Handles training and evaluation of models.

    Parameters
    ----------
    exp
        Experiment with model config.
    """

    def __init__(self, exp: Experiment):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = exp
        self.config = exp.estimator_config

        self.config["training"]["loss"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["loss"].items()
        }
        self.config["training"]["metrics"] = {
            key: LossEnum(val).get_fn() for key, val in self.config["training"]["metrics"].items()
        }
        self.callbacks: list[object] = []

        # create model
        self.optimizer = None
        self.epoch = 0
        self.create_model()
        self.compiled_model = False

        # train and val datasets
        # config params impacting y
        self.output_channels = self.config["data"]["output_channels"]
        self.repeat_y = len(self.config["training"]["loss"].keys())
        if self.repeat_y == 1:
            self.repeat_y = False
        self.add_c_to_y = False
        if "adv_head" in self.config["training"]["loss"].keys():
            self.add_c_to_y = True
            self.repeat_y = self.repeat_y - 1
        self.ds = NNDataset(
            self.config["data"]["dataset_name"],
            data_config=self.config["data"]["data_config"],
        )
        self._train_dataset, self._val_dataset, self._test_dataset = None, None, None

        # set up model weights and history paths for saving/loading later
        self.weights_name = os.path.join(self.exp.full_path, "weights_epoch{:03d}")  # noqa: P103
        self.history_name = os.path.join(self.exp.full_path, "history.csv")

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        Shuffled :class:`tf.data.Dataset` of train split.
        """
        if self._train_dataset is None:
            self._train_dataset = self._get_dataset("train", shuffled=True)
        return self._train_dataset

    @property
    def val_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of val split.
        """
        if self._val_dataset is None:
            self._val_dataset = self._get_dataset("val")
        return self._val_dataset

    @property
    def test_dataset(self) -> tf.data.Dataset:
        """
        :class:`tf.data.Dataset` of test split.
        """
        if self._test_dataset is None:
            self._test_dataset = self._get_dataset("test")
        return self._test_dataset

    def _get_dataset(self, split: str, shuffled: bool = False) -> tf.data.Dataset:
        return self.ds.get_tf_dataset(
            split=split,
            output_channels=self.output_channels,
            is_conditional=self.model.is_conditional,
            repeat_y=self.repeat_y,
            add_c_to_y=self.add_c_to_y,
            shuffled=shuffled,
        )

    def create_model(self):
        """
        Initialise neural network model.

        Adds ``self.model``.
        """
        ModelClass = ModelEnum(self.config["model"]["model_cls"]).get_cls()
        self.model = ModelClass(**self.config["model"]["model_kwargs"])
        weights_path = self.config["model"]["init_with_weights"]
        if weights_path is True:
            weights_path = tf.train.latest_checkpoint(self.exp.full_path)
        if isinstance(weights_path, str):
            # first need to compile the model
            self._compile_model()
            self.log.info(f"Initializing model with weights from {weights_path}")
            w1 = self.model.model.layers[5].get_weights()
            self.model.model.load_weights(weights_path).assert_nontrivial_match().assert_existing_objects_matched()
            w2 = self.model.model.layers[5].get_weights()
            assert (w1[0] != w2[0]).any()
            assert (w1[1] != w2[1]).any()
            self.epoch = self.exp.epoch
            # TODO when fine-tuning need to reset self.epoch!

    def _compile_model(self):
        config = self.config["training"]
        # set loss weights
        self.loss_weights = {key: tf.keras.backend.variable(val) for key, val in config["loss_weights"].items()}
        # callback to update weights before each epoch
        self.callbacks.append(
            LossWarmup(
                self.loss_weights,
                config["loss_weights"],
                config["loss_warmup_to_epoch"],
            )
        )
        self.callbacks.append(UpdateSparsityLevel())
        if hasattr(self.model, "temperature"):
            self.callbacks.append(
                AnnealTemperature(
                    self.model.temperature,
                    self.model.config["initial_temperature"],
                    self.model.config["temperature"],
                    self.model.config["anneal_epochs"],
                )
            )
        # create optimizer
        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.model.model.compile(
            optimizer=self.optimizer,
            loss=config["loss"],
            loss_weights=self.loss_weights,
            metrics=config["metrics"],
        )
        self.compiled_model = True

    def train_model(self):
        """
        Train neural network model.

        Needs an initialised model in ``self.model``.
        """
        config = self.config["training"]
        if not self.compiled_model:
            self._compile_model()
        self.log.info("Training model for {} epochs".format(config["epochs"]))
        history = self.model.model.fit(
            # TODO this is only shuffling the first 10000 samples, but as data is shuffled already should be ok
            x=self.train_dataset.shuffle(10000).batch(config["batch_size"]).prefetch(1),
            validation_data=self.val_dataset.batch(config["batch_size"]).prefetch(1),
            epochs=config["epochs"],
            verbose=1,
            callbacks=self.callbacks,
        )
        self.epoch += config["epochs"]
        history = pd.DataFrame.from_dict(history.history)
        history["epoch"] = range(self.epoch - config["epochs"], self.epoch)
        history = history.set_index("epoch")
        if config["save_model_weights"]:
            weights_name = self.weights_name.format(self.epoch)
            self.log.info(f"Saving model to {weights_name}")
            self.model.model.save_weights(weights_name)
        if config["save_history"]:
            if os.path.exists(self.history_name) and not config["overwrite_history"]:
                # if there is a previous history, concatenate to this
                prev_history = pd.read_csv(self.history_name, index_col=0)
                history = pd.concat([prev_history, history])
            history.to_csv(self.history_name)
        return history

    def predict_model(self, data: tf.data.Dataset | np.ndarray, batch_size: int | None = None) -> Any:
        """
        Predict all elements in ``data``.

        Parameters
        ----------
        data
            Data to predict, with first dimension the number of elements.
        batch_size
            Batch size. If None, the training batch size is used.

        Returns
        -------
        Iterable
            prediction
        """
        if isinstance(data, tf.data.Dataset):
            data = data.batch(self.config["training"]["batch_size"])
            batch_size = None
        elif batch_size is None:
            batch_size = self.config["training"]["batch_size"]

        pred = self.model.model.predict(data, batch_size=batch_size)
        if isinstance(pred, list):
            # multiple output model, but only care about first output
            pred = pred[0]
        return pred

    def evaluate_model(self, dataset: tf.data.Dataset | None = None) -> Any:
        """
        Evaluate model using :class:`tf.data.Dataset`.

        Parameters
        ----------
        dataset
            Dataset to evaluate.
            If None, :meth:`Estimator.val_dataset` is used.

        Returns
        -------
        Iterable[float]
            Scores.
        """
        if not self.compiled_model:
            self._compile_model()
        if dataset is None:
            dataset = self.val_dataset
        self.model.model.reset_metrics()
        scores = self.model.model.evaluate(dataset.batch(self.config["training"]["batch_size"]))
        return scores


# TODO Nastassya: convert to unittest
# def test_save_load_model():
#     with tempfile.TemporaryDirectory() as dirpath:
#         dataset_name = '184A1_UNPERTURBED_frac0005_neigh1'
#         dataset_dir = os.path.join(DATA_DIR, 'datasets', dataset_name)
#         config = {
#             'experiment':{
#                 'experiment_dir': dirpath,
#                 'name': 'test',
#                 'save_config': True,
#             },
#             'data': {
#                 'load_dataset': True,
#                 'dataset_dir': dataset_dir,
#                 'output_channels': None,
#             },
#             'model': {
#                 'model_cls': ModelEnum.AEModel,
#                 'model_kwargs': {
#                     'layers': [16,8,16],
#                     'num_neighbors': 1,
#                     'num_channels': 35
#                 },
#                 # if true, looks for saved weights in experiment_dir
#                 # if a path, loads these weights
#                 'init_with_weights': False,
#             },
#             'training': {
#                 'learning_rate': 0.001,
#                 'epochs': 1,
#                 'batch_size': 128,
#                 'loss': {'decoder': LossEnum.MSE},
#                 'metrics': {'decoder': LossEnum.MSE},
#                 # saving models
#                 'save_model_weights': True,
#                 'save_history': True,
#             },
#         }
#         est = Estimator(config)
#         history = est.train_model()
#         scores = est.evaluate_model(est.val_dataset)

#         # reload estimator and init with saved weights
#         est2 = Estimator.for_evaluation(est.experiment_dir)
#         scores2 = est2.evaluate_model(est2.val_dataset)
#         assert np.isclose(scores, scores2).all()

#         # check that history was correctly saved
#         history2 = pd.read_csv(est.history_name, index_col=0)
#         assert np.isclose(history, history2).all()
#     return True

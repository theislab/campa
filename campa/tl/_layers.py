from tensorflow import keras
import numpy as np
import tensorflow as tf


class KSparse(keras.layers.Layer):
    """
    K-sparse Keras layer.

    Based on https://gist.github.com/harryscholes/ed3539ab21ad34dc24b63adc715a97e0 .

    Parameters
    ----------
    sparsity_levels
        np.ndarray, sparsity levels per epoch calculated by `calculate_sparsity_levels`
    """

    def __init__(self, sparsity_levels, **kwargs):
        self.sparsity_levels = sparsity_levels
        self.k = tf.Variable(initial_value=self.sparsity_levels[0], trainable=False, dtype=tf.int32)
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        """Make inputs sparse."""
        kth_largest = tf.expand_dims(tf.sort(inputs, direction="DESCENDING")[..., self.k - 1], -1)
        # mask inputs
        sparse_inputs = inputs * tf.cast(tf.math.greater_equal(inputs, kth_largest), keras.backend.floatx())
        return sparse_inputs

    def get_config(self):
        """Return config."""
        config = {"k": self.k}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        """Return output shape."""
        return input_shape


class UpdateSparsityLevel(keras.callbacks.Callback):
    """Update sparsity level at the beginning of each epoch."""

    def on_epoch_begin(self, epoch, logs=None):
        """
        Update sparsity level.
        """
        try:
            layer = self.model.get_layer("KSparse")
            tf.keras.backend.set_value(layer.k, layer.sparsity_levels[epoch])
            print(f"set k to {tf.keras.backend.get_value(layer.k)}")
        except ValueError:
            pass


def calculate_sparsity_levels(initial_sparsity, final_sparsity, n_epochs):
    """Calculate sparsity levels per epoch.

    # Arguments
        initial_sparsity: int
        final_sparsity: int
        n_epochs: int
    """
    return np.hstack(
        (
            np.linspace(initial_sparsity, final_sparsity, n_epochs // 2, dtype=np.uint),
            np.repeat(final_sparsity, (n_epochs // 2) + 1),
        )
    )[:n_epochs]

"""Loss functions used by the NumPy MNIST network."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """Return mean cross-entropy for integer class labels."""
    if y_pred.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)

    batch_size = y_pred.shape[0]
    probabilities = np.clip(
        y_pred[np.arange(batch_size), y_true],
        a_min=1e-7,
        a_max=None,
    )
    return -np.sum(np.log(probabilities)) / batch_size

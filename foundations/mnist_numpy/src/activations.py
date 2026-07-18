"""Activation functions used by the NumPy MNIST network."""

import numpy as np


class ReLU:
    """Rectified linear unit with a cached mask for backpropagation."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        """Return ``x`` with non-positive entries replaced by zero."""
        self.mask = x > 0
        out = x.copy()
        out[~self.mask] = 0
        return out

    def backward(self, dout):
        """Block gradients at entries that were inactive in ``forward``."""
        dout[~self.mask] = 0
        return dout


class Softmax:
    """Numerically stable row-wise softmax output layer."""

    def forward(self, x):
        """Convert a batch of logits into class probabilities."""
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, dout):
        """Pass through the combined softmax-cross-entropy gradient."""
        return dout

"""Layers used by the NumPy MNIST network."""

import numpy as np


class Affine:
    """Fully connected layer implementing ``x @ W + b``."""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """Cache the input and return the affine projection."""
        self.x = x
        return (x @ self.W) + self.b

    def backward(self, dout):
        """Compute input, weight, and bias gradients."""
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dx


class BatchNorm:
    """Feature-wise batch normalization with running statistics."""

    def __init__(self, gamma, beta, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train=True):
        """Normalize with batch statistics in training and running statistics otherwise."""
        if train:
            mean = np.mean(x, axis=0)
            xc = x - mean
            variance = np.mean(xc**2, axis=0)
            std = np.sqrt(variance + self.eps)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * variance
            )
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)

        return self.gamma * xn + self.beta

    def backward(self, dout):
        """Compute gradients for the input, scale, and shift parameters."""
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(self.xn * dout, axis=0)

        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum(dxn * self.xc / (self.std**2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmean = np.sum(dxc, axis=0)
        return dxc - dmean / self.batch_size


class Dropout:
    """Classic dropout with inference-time output scaling."""

    def __init__(self, drop_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self, x, train=True):
        """Apply a random mask while training and expectation scaling otherwise."""
        if train:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask
        return x * (1 - self.drop_ratio)

    def backward(self, dout):
        """Block gradients at entries removed by the training mask."""
        return dout * self.mask

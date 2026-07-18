"""Optimizers for the NumPy MNIST network."""

import numpy as np


class SGD:
    """In-place stochastic gradient descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        """Update every parameter in the opposite gradient direction."""
        for key in params:
            params[key] -= self.lr * grads[key]


class Adam:
    """Adam with first- and second-moment bias correction."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.m, self.v = {}, {}
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2

    def update(self, params, grads):
        """Update parameters using the Adam optimization rule."""
        if not self.m:
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)

        self.t += 1
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.t)
            / (1.0 - self.beta1**self.t)
        )

        for key in params:
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

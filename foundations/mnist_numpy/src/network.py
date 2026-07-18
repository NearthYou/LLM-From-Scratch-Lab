"""A two-hidden-layer NumPy neural network for MNIST classification."""

from collections import OrderedDict

import numpy as np

from activations import ReLU, Softmax
from layers import Affine, BatchNorm, Dropout
from losses import cross_entropy_loss


class NeuralNetwork:
    """Compose affine, normalization, activation, and dropout layers."""

    def __init__(
        self,
        use_batchnorm=True,
        use_dropout=True,
        dropout_ratio=0.5,
        batchnorm_momentum=0.9,
        weight_decay_lambda=0,
    ):
        weight_init = "relu"
        self.input_size = 784
        self.output_size = 10
        self.hidden_size_list = [512, 256]
        self.hidden_layer_num = len(self.hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}

        self.__init_weight(weight_init)

        activation_layer = {"relu": ReLU}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)],
                self.params["b" + str(idx)],
            )
            if self.use_batchnorm:
                feature_count = self.hidden_size_list[idx - 1]
                self.params["gamma" + str(idx)] = np.ones(feature_count)
                self.params["beta" + str(idx)] = np.zeros(feature_count)
                self.layers["BatchNorm" + str(idx)] = BatchNorm(
                    self.params["gamma" + str(idx)],
                    self.params["beta" + str(idx)],
                    momentum=batchnorm_momentum,
                )

            self.layers["Activation_function" + str(idx)] = activation_layer[
                weight_init
            ]()

            if self.use_dropout:
                self.layers["Dropout" + str(idx)] = Dropout(dropout_ratio)

        output_idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(output_idx)] = Affine(
            self.params["W" + str(output_idx)],
            self.params["b" + str(output_idx)],
        )
        self.last_layer = Softmax()

    def __init_weight(self, weight_init_std):
        """Initialize affine weights with He, Xavier, or a numeric scale."""
        sizes = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(sizes)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / sizes[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / sizes[idx - 1])
            self.params["W" + str(idx)] = scale * np.random.randn(
                sizes[idx - 1], sizes[idx]
            )
            self.params["b" + str(idx)] = np.zeros(sizes[idx])

    def forward(self, x, train=True):
        """Run the ordered layer stack and convert logits to probabilities."""
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train)
            else:
                x = layer.forward(x)
        return self.last_layer.forward(x)

    def backward(self, dout):
        """Backpropagate and collect gradients using parameter-compatible keys."""
        dout = self.last_layer.backward(dout)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            affine = self.layers["Affine" + str(idx)]
            grads["W" + str(idx)] = (
                affine.dW
                + self.weight_decay_lambda * self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = affine.db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                batchnorm = self.layers["BatchNorm" + str(idx)]
                grads["gamma" + str(idx)] = batchnorm.dgamma
                grads["beta" + str(idx)] = batchnorm.dbeta

        self.grads = grads
        return self.grads

    def loss(self, x, y):
        """Return cross-entropy plus configured L2 weight decay."""
        y_pred = self.forward(x, train=True)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            weights = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(weights**2)
        return cross_entropy_loss(y_pred, y) + weight_decay

    def predict(self, x):
        """Run inference with BatchNorm and Dropout in evaluation mode."""
        return self.forward(x, train=False)

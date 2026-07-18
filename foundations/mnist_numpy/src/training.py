"""Training, evaluation, and plotting utilities for the NumPy network."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """Train with shuffled mini-batches and return mean loss per epoch."""
    train_size = x_train.shape[0]
    loss_history = []

    for _ in range(epochs):
        shuffled_idx = np.random.permutation(train_size)
        total_loss = 0

        for start in range(0, train_size, batch_size):
            batch_idx = shuffled_idx[start : start + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            current_batch_size = x_batch.shape[0]

            y_pred = model.forward(x_batch, train=True)
            loss = cross_entropy_loss(y_pred, y_batch)
            total_loss += loss * current_batch_size

            dout = y_pred.copy()
            dout[np.arange(current_batch_size), y_batch] -= 1
            dout /= current_batch_size

            grads = model.backward(dout)
            optimizer.update(model.params, grads)

        loss_history.append(total_loss / train_size)

    return loss_history


def evaluate(model, x, y):
    """Return classification accuracy as a percentage and parameter count."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(parameter.size for parameter in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """Display an epoch-loss curve."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

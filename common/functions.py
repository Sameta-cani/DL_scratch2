import sys, os
sys.path.append(os.pardir)

from common.np import *

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output of the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU (Rectified Linear Unit) activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output of the ReLU activation function.
    """
    return np.maximum(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output of the softmax activation function.
    """
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x

def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> float:
    """
    Compute the cross-entropy between predicted probabilities (y) and true labels (t).

    Args:
        y (numpy.ndarray): Predicted probabilities.
        t (numpy.ndarray): True labels.

    Returns:
        float: Cross-entropy error.
    """
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    batch_size = y.shape[0]

    # Avoid numerical instability by adding a small constant (1e-7) inside the log
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
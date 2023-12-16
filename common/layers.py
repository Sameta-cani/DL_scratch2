import sys, os
sys.path.append(os.pardir)

from common.np import * # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class MatMul:
    def __init__(self, W: np.ndarray):
        """
        Initialize the MatMul layer with weights.

        Args:
            W (numpy.ndarray): Weight matrix.
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the MatMul layer.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the MatMul layer.
        """
        W, = self.params
        self.x = x
        return np.dot(x, W)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the MatMul layer.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the lyaer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
    

class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        """
        Initialize the Affine layer with weights and biases.

        Args:
            W (numpy.ndarray): Wegith matrix.
            b (numpy.ndarray): Bias vector.
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the Affine layer.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the Affine layer.
        """
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the Affine layer.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
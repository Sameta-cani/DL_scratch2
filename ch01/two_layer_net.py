import sys
sys.path.append('..')
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    """
    A two-layer neural network with one hidden layer.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output classes.

    Attributes:
        layers (list): List of neural network layers.
        loss_layer (SoftmaxWithLoss): Softmax with cross-entropy loss layer.
        params (list): List of trainable parameters in the network.
        grads (list): List of gradients corresponding to the parameters.

    Methods:
        predict(x: numpy.ndarray) -> numpy.ndarray:
            Performs forward pass and returns the predicted scores.
        
        forward(x: numpy.ndarray, t: numpy.ndarray) -> float:
            Performs forward pass, calculates the loss, and returns it.

        backward(dout: float = 1) -> numpy.ndarray:
            Performs backward pass, calculates gradients, and returns the gradient with respect to the input.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initializes a two-layer neural network with given sizes.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output classes.
        """
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass and returns the predicted scores.

        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            numpy.ndarray: Predicted scores of shape (batch_size, output_size).
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Performs a forward pass, calculates the loss, and returns it.

        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_size).
            t (numpy.ndarray): True labels in one-hot encoding of shape (batch_size, output_size).

        Returns:
            float: Loss value.
        """
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Performs a backward pass, calculates gradients, and returns the gradient with respect to the input.

        Args:
            dout (float, optional): Gradient of the loss. Defaults to 1.

        Returns:
            numpy.ndarray: Gradient with respect to the input data of shape (batch_size, input_size).
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
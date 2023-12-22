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
    

class Sotfmax:
    def __init__(self):
        """
        Initializes a Softmax object.
        """
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of Softmax.

        Args:
            x (numpy.ndarray): Input data on which Softmax is applied.

        Returns:
            numpy.ndarray: Result of applying Softmax to the input data.
        """
        self.out = softmax(x)
        return self.out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of Softmax.

        Args:
            dout (numpy.ndarray): Gradient propagated from the next layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        """
        Initializes SoftmaxWithLoss object.
        """
        self.parmas, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Performs the forward pass, calculating Softmax and Cross-Entropy loss.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels for the input data.

        Returns:
            float: Cross-Entropy loss.
        """
        self.t = t
        self.y = softmax(x)

        # If the labels are one-hot encoded, convert to class indices
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Performs the backward pass, calculating gradients with respect to the input.

        Args:
            dout (float, optional): Gradient of the loss. Defaults to 1.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        batch_size = len(self.t)

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size

        return dx

class Sigmoid:
    def __init__(self):
        """
        Initializes a Sigmoid object.
        """
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass, applying the Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the Sigmoid function.
        """
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass, calculating the gradient with respect to the input.

        Args:
            dout (numpy.ndarray): Gradient of the loss.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class SigmoidWithLoss:
    def __init__(self):
        """
        Initializes a SigmoidWithLoss object.
        """
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        Performs the forward pass, applying the Sigmoid activation and calculating Cross-Entropy loss.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels for the input data.

        Returns:
            float: Cross-Entropy loss.
        """
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Performs the backward pass, calculating gradients with respect to the input.

        Args:
            dout (float, optional): Gradient of the loss. Defaults to 1.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        batch_size = len(self.t)

        dx = (self.y - self.t) * dout / batch_size
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio: float=0.5):
        """
        Initializes a Dropout layer.

        Args:
            dropout_ratio (float, optional): Probability of dropping out a neuron. Defaults to 0.5.
        """
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x: np.ndarray, train_flg: bool=True) -> np.ndarray:
        """
        Performs the forward pass applying dropout during training.

        Args:
            x (numpy.ndarray): Input data.
            train_flg (bool, optional): Flag indicating whether the model is training mode. Defaults to True.

        Returns:
            np.ndarray: _description_
        """
        if train_flg:
            self.mask = (np.random.rand(*x.shape) > self.dropout_ratio) / (1.0 - self.dropout_ratio)
            return x * self.mask
        else:
            return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass, applying the dropout mask to the gradient.

        Args:
            dout (numpy.ndarray): Gradient of the loss.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        return dout * self.mask
    
class Embedding:
    """
    A class for managing word embeddings in a nerual network.

    This class handles the retrieval of word embeddings based on index and computes gradients for backpropagation in a nerual network.

    Args:
        params (list): A list containing the embedding matrix.
        grads (list): A list containing gradeints of the embedding matrix.
        idx (list, int): Indices of the last retrieved word embeddings.

    Methods:
        forward(idx): Retrieves the embedding vectors for the given indeices.
        backward(dout): Computes and accumulates the gradients for the embedding matrix.
    """
    def __init__(self, W: np.ndarray):
        """
        Initializes the Embedding object with an embedding matrix.

        Args:
            W (numpy.ndarray): The embedding matrix.
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass by retrieving the embedding vectors for the given indices.

        Args:
            idx (numpy.ndarray): An array of indices for which embeddings are to be retrieved.

        Returns:
            numpy.ndarray: The embedding vectors corresponding to the given indices.
        """
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout: np.ndarray) -> None:
        """
        Performs the backward pass by computing and accumulating the gradients for the embedding matrix.

        Args:
            dout (numpy.ndarray): The gradient of the loss with respect to the output of forward pass.

        Returns:
            None
        """
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None
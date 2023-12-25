from common.np import *
from common.layers import *
from common.functions import sigmoid


class RNN:
    """
    A simple implementation of a Recurrent Neural Network (RNN).

    Attributes:
        Wx (numpy.ndarray): The weight matrix for the input x.
        Wh (numpy.ndarray): The weight matrix for the hidden state.
        b (numpy.ndarray): The bias vector.
        grads (list): A list to store the gradients of the weights and bias.
        cache (tuple): A tuple to store intermediate values for backpropagation.
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):
        """
        Initializes the RNN with weight matrices and bias.

        Args:
            Wx (numpy.ndarray): The weight matrix for the input x.
            Wh (numpy.ndarray): The weight matrix for the hidden state.
            b (numpy.ndarray): The bias vector.
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for the RNN.

        Args:
            x (numpy.ndarray): The input vector.
            h_prev (numpy.ndarray): The previous hidden state.

        Returns:
            numpy.ndarray: The next hidden state.
        """
        Wx, Wh, b = self.params
        temp = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(temp)

        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next: np.ndarray) -> tuple:
        """
        Backward pass for the RNN.

        Args:
            dh_next (numpy.ndarray): The gradient of the loss with respect to the next hidden state.

        Returns:
            tuple: Gradients with respect to the input vector, previous hidden state, and the parameters (Wx, Wh, b).
        """
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev
    

class TimeRNN:
    """
    A Recurrent Neural Network (RNN) layer for processing time series data.

    This layer can process a batch of sequences at once and maintains an internal state
    for handling sequences of arbitrary length.

    Attributes:
        params (list): List of parameters (weights and bias).
        grads (list): List of gradients for each parameter.
        layers (list): List of RNN layers.
        h (numpy.ndarray): Hidden state.
        dh (numpy.ndarray): Gradient of the hidden state.
        stateful (bool): Whether to maintain state between forward passes.
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful: bool=False):
        """
        Initializes the TimeRNN Layer.

        Args:
            Wx (numpy.ndarray): Input-to-hidden weight matrix.
            Wh (numpy.ndarray): Hidden-to_hidden weight matrix.
            b (numpy.ndarray): Bias vector.
            stateful (bool, optional): If True, the layer maintains the hidden state between batches. Defaults to False.
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = []
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for processing a batch of sequences.

        Args:
            xs (numpy.ndarray): Input sequences, shaped as (batch_size, sequence_length, feature_size).

        Returns:
            numpy.ndarray: Output sequences.
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        # Reuse RNN layers if they were already created
        if not self.layers:
            self.layers = [RNN(Wx, Wh, b) for _ in range(T)]

        for t in range(T):
            self.h = self.layers[t].forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
        
        return hs
    
def backward(self, dhs: np.ndarray) -> np.ndarray:
    Wx, Wh, _ = self.params
    N, T, H = dhs.shape
    D = Wx.shape[0]

    dxs = np.empty((N, T, D), dtype='f')
    dh = 0
    grads = [0, 0, 0]

    for t in reversed(range(T)):
        dx, dh = self.layers[t].backward(dhs[:, t, :] + dh)
        dxs[:, t, :] = dx

        for i, grad in enumerate(self.layers[t].grads):
            grads[i] += grad

    for i, grad in enumerate(grads):
        self.grads[i][...] = grad
    self.dh = dh

    return dxs

def set_state(self, h: np.ndarray):
    """
    Set the hidden state of the RNN.

    Args:
        h (numpy.ndarray): A hidden state to set.
    """
    self.h = h

def reset_state(self):
    """
    Reset the hidden state of the RNN to None.
    """
    self.h = None
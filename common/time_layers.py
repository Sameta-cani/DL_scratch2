import sys
sys.path.append('..')
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
    
    def backward(self, dhs):
        """
        Backward pass for processing a batch of sequences.

        This method computes the gradient of the loss with respect to the input sequences
        and updates the gradients of the network parameters.

        Args:
            dhs (numpy.ndarray): Gradients w.r.t the output sequences, 
                                 shaped as (batch_size, sequence_length, hidden_size).

        Returns:
            numpy.ndarray: Gradients w.r.t the input sequences, 
                           shaped as (batch_size, sequence_length, feature_size).
        """
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


class LSTM:
    """
    Long Short-Term Memory (LSTM) layer.

    LSTM is a type of recurrent neural network (RNN) layer that is capable of learning long-term dependencies.

    Attributes:
        params (list of numpy.ndarray): Parameters of the LSTM layer, including weight matrices and bias vector.
        grads (list of numpy.ndarray): Gradients of the LSTM layer's parameters.
        cache (tuple): Cached data from the forward pass for use in the backward pass.
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):
        """
        Initializes the LSTM layer.

        Args:
            Wx (numpy.ndarray): Weight matrix for the input x.
            Wh (numpy.ndarray): Weight matrix for the hidden state.
            b (numpy.ndarray): Bias vector.
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple:
        """
        Forward pass for the LSTM layer.

        Args:
            x (numpy.ndarray): Input data.
            h_prev (numpy.ndarray): Previous hidden state.
            c_prev (numpy.ndarray): Previous cell state.

        Returns:
            tuple:
                numpy.ndarray: Next hidden state.
                numpy.ndarray: Next cell state.
        """
        Wx, Wh, b = self.params
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # 분할하여 게이트에 적용
        H = h_prev.shape[1]
        f = sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:])

        # 새로운 셀 상태
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> tuple:
        """
        Backward pass for the LSTM layer.

        Args:
            dh_next (numpy.ndarray): Gradient of loss with respect to the next hidden state.
            dc_next (numpy.ndarray): Gradient of loss with respect to the next cell state.

        Returns:
            tuple: 
                numpy.ndarray: Gradient with respect to the input data.
                numpy.ndarray: Gradient with respect to the previous hidden state.
                numpy.ndarray: Gradient with respect to the previous cell state.
        """
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        # 게이트 그래디언트
        dc_prev = ds * f
        di = ds * g * i * (1 - i)
        df = ds * c_prev * f * (1 - f)
        do = dh_next * tanh_c_next * o * (1 - o)
        dg = ds * i * (1 - g ** 2)

        # 중간 그래디언트
        dA = np.hstack((df, dg, di, do))

        # 파라미터 그래디언트
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # 입력 및 이전 은닉 상태 그래디언트
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev
    

class TimeLSTM:
    """
    Time-distributed LSTM layer for sequence data processing.

    This layer applies an LSTM operation over sequences of time steps, allowing it to capture
    temporal dependencies in sequence data.

    Attributes:
        params (list): Parameters of the LSTM layer.
        grads (list): Gradients of the LSTM layer's parameters.
        layers (list): List of LSTM layers for each time step.
        h (np.ndarray): Hidden state.
        c (np.ndarray): Cell state.
        dh (np.ndarray): Gradient of the hidden state.
        stateful (bool): If True, the layer retains state between batches.
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful: bool=False):
        """
        Initializes the TimeLSTM layer.

        Args:
            Wx (np.ndarray): Input-to-hidden weight matrix.
            Wh (np.ndarray): Hidden-to-hidden weight matrix.
            b (np.ndarray): Bias vector.
            stateful (bool, optional): If True, the layer maintains state between forward passes. Defaults to False.
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = []
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for processing a batch of sequences.

        Args:
            xs (np.ndarray): Input sequences, shaped as (batch_size, sequence_length, feature_size).

        Returns:
            np.ndarray: Output sequences.
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        # Initialize the hidden and cell states if not stateful
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        hs = np.empty((N, T, H), dtype='f')
        if not self.layers:
            self.layers = [LSTM(*self.params) for _ in range(T)]

        for t in range(T):
            self.h, self.c = self.layers[t].forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

        return hs
    
    def backward(self, dhs: np.ndarray) -> np.ndarray:
        """
        Backward pass for processing gradients through the time-distributed LSTM layer.

        Args:
            dhs (np.ndarray): Gradients w.r.t the output of the LSTM layer.

        Returns:
            np.ndarray: Gradients w.r.t the input sequences.
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            dx, dh, dc = self.layers[t].backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(self.layers[t].grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs
    
    def set_state(self, h: np.ndarray, c: np.ndarray=None):
        """
        Sets the hidden and cell states of the LSTM layer.

        Args:
            h (np.ndarray): Hidden state to set.
            c (np.ndarray, optional): Cell state to set. If None, only the hidden state is set. Defaults to None.
        """
        self.h, self.c = h, c

    def reset_state(self):
        """
        Resets the hidden and cell states of the LSTM layer.
        """
        self.h, self.c = None, None


class TimeEmbedding:
    """
    An embedding layer for processing time series data.

    This layer applies an embedding operation to each time step of a batch of sequences.

    Attributes:
        params (list): List of parameters (embedding matrix).
        grads (list): List of gradients for the embedding matrix.
        layers (list): List of Embedding layers for each time step.
        W (numpy.ndarray): Embedding matrix.
    """

    def __init__(self, W: np.ndarray):
        """
        Initializes the TimeEmbedding layer with an embedding matrix.

        Args:
            W (numpy.ndarray): The embedding matrix.
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for processing a batch of sequences through the embedding layer.

        Args:
            xs (numpy.ndarray): Input sequences, shaped as (batch_size, sequence_length).

        Returns:
            numpy.ndarray: Embedded sequences, shaped as (batch_size, sequence_length, embedding_size).
        """
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')

        # Reuse the Embedding layer if it was already created
        if self.layers is None:
            self.layers = [Embedding(self.W) for _ in range(T)]

        for t in range(T):
            out[:, t, :] = self.layers[t].forward(xs[:, t])

        return out
    
    def backward(self, dout: np.ndarray) -> None:
        """
        Backward pass for processing gradients throught the embedding layer.

        Args:
            dout (numpy.ndarray): Gradients w.r.t the output of the embedding layer,
                               shaped as (batch_size, sequence_length, embedding_size).
    
        Returns:
            None: This layer does not pass gradients backward with respect to its input.
        """
        N, T, D = dout.shape
        grad = 0

        for t in range(T):
            self.layers[t].backward(dout[:, t, :])
            grad += self.layers[t].grads[0]

        self.grads[0][...] = grad

        return None

class TimeAffine:
    """
    A time-affine layer for processing time series data in neural networks.

    This layer applies an affine transformation to each time step of a batch of sequences. 
    The affine transformation is a linear operation specified by a weight matrix and a bias vector.

    Attributes:
        params (list): Parameters of the layer, including the weight matrix (W) and bias vector (b).
        grads (list): Gradients of the layer's parameters.
        x (numpy.ndarray): The input data saved during the forward pass.
    """

    def __init__(self, W, b):
        """
        Initializes the TimeAffine layer.

        Args:
            W (numpy.ndarray): Weight matrix for the affine transformation.
            b (numpy.ndarray): Bias vector for the affine transformation.
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """
        Forward pass for the TimeAffine layer.

        Flattens the input data, applies the affine transformation, and then reshapes it 
        back to its original time sequence format.

        Args:
            x (numpy.ndarray): Input data with shape (batch_size, sequence_length, feature_size).

        Returns:
            numpy.ndarray: Transformed data with the same shape as the input.
        """
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        """
        Backward pass for the TimeAffine layer.

        Computes the gradients with respect to the input data and the layer's parameters, 
        reshaping the gradients to match the original time sequence format.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the output 
                                  with shape (batch_size, sequence_length, output_size).

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input data.
        """
        W, b = self.params
        dout = dout.reshape(-1, W.shape[1])
        rx = self.x.reshape(-1, W.shape[0])

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T).reshape(*self.x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class TimeSoftmaxWithLoss:
    """
    A time-distributed softmax-with-loss layer for neural networks.

    This layer computes the softmax loss for each time step of a batch of sequences,
    which is useful in tasks like language modeling. It supports masking for handling variable sequence lengths.

    Attributes:
        params (list): List of parameters of the layer (empty for this layer).
        grads (list): List of gradients of the layer's parameters (empty for this layer).
        cache (tuple): Cache used for storing forward pass values for use in backward pass.
        ignore_label (int): Label that should be ignored in loss calculation.
    """

    def __init__(self):
        """
        Initializes the TimeSoftmaxWithLoss layer.
        """
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Forward pass for the softmax-with-loss layer.

        Computes the loss over a batch of sequences. Supports ignoring specific labels (e.g., padding) in the loss computation.

        Args:
            xs (numpy.ndarray): Input data (scores before softmax), shaped (batch_size, sequence_length, vocab_size).
            ts (numpy.ndarray): Target labels, shaped (batch_size, sequence_length) or (batch_size, sequence_length, vocab_size).

        Returns:
            float: The average loss over the batch.
        """
        N, T, V = xs.shape
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        xs_reshaped = xs.reshape(N * T, V)
        ts_reshaped = ts.reshape(N * T)
        mask_reshaped = mask.reshape(N * T)

        ys = softmax(xs_reshaped)
        ls = np.log(ys[np.arange(N * T), ts_reshaped]) * mask_reshaped
        loss = -np.sum(ls) / mask_reshaped.sum()

        self.cache = (ts_reshaped, ys, mask_reshaped, (N, T, V))
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for the softmax-with-loss layer.

        Computes the gradient of the loss with respect to the input scores.

        Args:
            dout (float, optional): Upstream derivative, usually 1 for loss layers. Defaults to 1.

        Returns:
            numpy.ndarray: Gradient with respect to the input scores, shaped (batch_size, sequence_length, vocab_size)
        """
        ts_reshaped, ys, mask_reshaped, (N, T, V) = self.cache

        dx = ys.copy()
        dx[np.arange(N * T), ts_reshaped] -= 1
        dx *= dout / mask_reshaped.sum()
        dx *= mask_reshaped[:, np.newaxis]

        return dx.reshape(N, T, V)
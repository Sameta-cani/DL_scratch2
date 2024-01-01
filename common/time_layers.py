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
    

class TimeDropout:
    """
    Time-distributed dropout layer.

    This layer applies dropout to the input data along the time axis. Drooput randomly sets
    a fraction of input units to 0 at each update during training time, which helps prevent overfitting.

    Attributes:
        dropout_ratio (float): Dropout ratio.
        mask (np.ndarray): Dropout mask.
        train_flg (bool): Flag indicating whether the layer is in training mode or not.
    """

    def __init__(self, dropout_ratio: float=0.5):
        """
        Initializes the TimeDropout layer.

        Args:
            dropout_ratio (float, optional): Dropout ratio. Defaults to 0.5.
        """
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for the TimeDropout layer.

        Args:
            xs (np.ndarray): Input data.

        Returns:
            np.ndarray: Output data after applying dropout.
        """
        if self.train_flg:
            self.mask = np.random.randn(*xs.shape) > self.dropout_ratio
            return xs * self.mask
        else:
            return xs
        
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for the TimeDropout layer.

        Args:
            dout (np.ndarray): Upstream gradients.

        Returns:
            np.ndarray: Downstream gradients.
        """
        return dout * self.mask
    

class TimeBiLSTM:
    """
    Time Bidirectional LSTM (BiLSTM).

    This layer processes a sequence using two LSTM layers, one for forward pass and 
    one for backward pass, and then concatenates their outputs. This allows the layer
    to capture information from both past and future context.

    Attributes:
        forward_lstm (TimeLSTM): LSTM layer for processing the sequence in forward direction.
        backward_lstm (TimeLSTM): LSTM layer for processing the sequence in backward direction.
        params (list): Parameters of both LSTM layers.
        grads (list): Gradients of both LSTM layers.
    """

    def __init__(self, Wx1, Wh1, b1, 
                Wx2, Wh2, b2, stateful=False):
        """
        Initializes the Time BiLSTM layer.

        Args:
            Wx1, Wh1, b1: Parameters for the forward LSTM layer.
            Wx2, Wh2, b2: Parameters for the backward LSTM layer.
            stateful (bool): If True, the layer maintains state between batches.
        """
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Time BiLSTM layer.

        Args:
            xs (np.ndarray): Input sequence of word IDs (batch_size, time_steps, input_size).

        Returns:
            np.ndarray: Output of the BiLSTM layer (batch_size, time_steps, 2*hidden_size).
        """
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out
    
    def backward(self, dhs: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Time BiLSTM layer.

        Args:
            dhs (np.ndarray): Gradient from the subsequent layer
                              (batch_size, time_steps, 2*hidden_Size).

        Returns:
            np.ndarray: Gradient with respect to the input sequence.
        """
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs


class TimeSigmoidWithLoss:
    """
    Time Sigmoid with Loss layer.

    This layer applies a sigmoid activation followd by a loss computation at each time step of a sequence.
    It is suitable for binary classification tasks in sequence data.

    Attributes:
        params (list): Parameters of the layer (empty as no parameters are needed for sigmoid and loss).
        grads (list): Gradients of the layer's parameters (empty as no gradients are computed for sigmoid and loss).
        xs_shape (tuple): Shape of the input sequence.
        layers (list): List of SoftmaxWithLoss layers for each time step.
    """

    def __init__(self):
        """
        Initializes the Time Sigmoid with Loss layer.
        """
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Forward pass for the Time Sigmoid with Loss layer.

        Args:
            xs (np.ndarray): Input sequence data (batch_size, time_steps).
            ts (np.ndarray): Target labels for the sequence (batch_size, time_steps).

        Returns:
            float: Average loss over all time steps.
        """
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for the Time Sigmoid with Loss layer.

        Args:
            dout (float, optional): Initial gradient for backpropagation. Defaults to 1.

        Returns:
            np.ndarray: Gradient with respect to the input sequence.
        """
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs

class GRU:
    """
    Gated Recurrent Unit (GRU) layer.

    GRU is a type of recurrent neural network layer that uses gating mechanisms to control
    the flow of information.

    Attributes:
        Wx (np.ndarray): Input-to-hidden weight matrix.
        Wh (np.ndarray): Hidden-to-hidden weight matrix.
        dWx (np.ndarray): Gradient of Wx.
        dWh (np.ndarray): Gradient of Wh.
        cache (tuple): Cached values for use in backward pass.
    """

    def __init__(self, Wx, Wh):
        self.Wx, self.Wh = Wx, Wh
        self.dWx, self.dWh = None, None
        self.cache = None

    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for the GRU layer.

        Args:
            x (np.ndarray): Input data.
            h_prev (np.ndarray): Previous hidden state.

        Returns:
            np.ndarray: Next hidden state.
        """
        H = self.Wh.shape[0]
        Wxz, Wxr, Wx = np.split(self.Wx, 3, axis=1)
        Whz, Whr, Wh = np.split(self.Wh, 3, axis=1)

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz))
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr))
        h_hat = np.tanh(np.dot(x, Wx) + np.dot(r * h_prev, Wh))
        h_next = (1 - z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)
        return h_next
    
    def backward(self, dh_next: np.ndarray) -> tuple:
        """
        Backward pass for the GRU layer.

        Args:
            dh_next (np.ndarray): Gradient of the loss with respect to the next hidden state.

        Returns:
            tuple:
                np.ndarray: Gradient with respect to the input data.
                np.ndarray: Gradient with respect to the previous hidden state.
        """
        H = self.W.shape[0]
        Wxz, Wxr, Wx = np.split(self.Wx, 3, axis=1)
        Whz, Whr, Wh = np.split(self.Wh, 3, axis=1)
        x, h_prev, z, r, h_hat = self.cache

        dh_hat = dh_next * z
        dh_prev = dh_next * (1 - z)

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dWh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1 - z)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1 - r)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWx))
        self.dWh = np.hstack((dWhz, dWhr, dWh))

        return dx, dh_prev
    

class TimeGRU:
    """
    Time-distributed GRU layer.

    This layer applies a GRU operation over sequences of time steps, allowing it to capture
    temporal dependencies in sequence data. It can maintain state across batches if stateful.

    Attributes:
        Wx (np.ndarray): Input-to-hidden weight matrix.
        Wh (np.ndarray): Hidden-to-hidden weight matrix.
        stateful (bool): If True, maintains state across batches.
        h (np.ndarray): Hidden state.
        dWx (np.ndarray): Gradient of Wx.
        dWh (np.ndarray): Gradient of Wh.
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, stateful: bool=False):
        """
        Initializes the TimeGRU layer.

        Args:
            Wx (np.ndarray): Input-to-hidden weight matrix.
            Wh (np.ndarray): Hidden-to-hidden weight matrix.
            stateful (bool, optional): If True, the layer maintains state between forward passes. Defaults to False.
        """
        self.Wx, self.Wh = Wx, Wh
        self.stateful = stateful
        self.h = None
        self.dWx, self.dWh = None, None
        self.gru_layer = GRU(Wx, Wh) # Reusing a single GRU layer

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for processing a batch of sequences.

        Args:
            xs (np.ndarray): Input sequences, shaped as (batch_size, sequence_length, feature_size).

        Returns:
            np.ndarray: Output sequences.
        """
        N, T, D = xs.shape
        H = self.Wh.shape[0]

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        hs = np.empty((N, T, H), dtype='f')
        for t in range(T):
            self.h = self.gru_layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h

        return hs
    
    def backward(self, dhs: np.ndarray) -> np.ndarray:
        """
        Backward pass for processing gradients through the time-distributed GRU layer.

        Args:
            dhs (np.ndarray): Gradients w.r.t the output of the GRU layer.

        Returns:
            np.ndarray: Gradients w.r.t the input sequences.
        """
        N, T, H = dhs.shape
        D = self.Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        self.dWx, self.dWh = 0, 0

        for t in reversed(range(T)):
            dx, dh = self.gru_layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            self.dWx += self.gru_layer.dWx
            self.dWh += self.gru_layer.dWh

        return dxs
    
    def set_state(self, h: np.ndarray):
        """
        Sets the hidden state of the GRU layer.

        Args:
            h (np.ndarray): Hidden state to set.
        """
        self.h = h

    def reset_state(self):
        """
        Resets the hidden state of the GRU layer.
        """
        self.h = None


class Simple_TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss
    
    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype='f')

        dout *= 1/T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs
    
class Simple_TimeAffine:
    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype='f')
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out
    
    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype='f')
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs
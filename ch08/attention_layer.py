import sys
sys.path.append('..')
from common.np import *
from common.layers import Softmax


class WeightSum:
    """
    Weighted sum layer.

    This layer computes the weighted sum of a sequence of vectors, given the sequence and
    attention weights.

    Attributes:
        params (list): Parameters of the layer.
        grads (list): Gradients of the layer's parameters.
        cache (tuple): Cached data for use in the backward pass.
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Forward pass for the weighted sum layer.

        Args:
            hs (np.ndarray): Input sequence of vectors (batch_size, time_steps, hidden_size).
            a (np.ndarray): Attention weights (batch_size, time_stpes).

        Returns:
            np.ndarray: Weighted sum vector (batch_size, hidden_size).
        """
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c
    
    def backward(self, dc: np.ndarray) -> tuple:
        """
        Backward pass for the weighted sum layer.

        Args:
            dc (np.ndarray): Gradients w.r.t the output of the weighted sum layer.

        Returns:
            tuple: 
                np.ndarray: Gradients w.r.t the input sequence of vectors.
                np.ndarray: Gradients w.r.t the attention weights.
        """
        hs, ar = self.cache
        N, T, H = hs.shape

        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da
    

class AttentionWeight:
    """
    Attention weight layer.

    This layer computes the attention weights for a given sequence based on a target hidden state.
    
    Attributes:
        params (list): Parameters of the layer.
        grads (list): Gradients of the layer's parameters.
        softmax (Softmax): Softmax layer for calculating attention weights.
        cache (tuple): Cached data for use in the backward pass.
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Forward pass for the attention weight layer.

        Args:
            hs (np.ndarray): Input sequence of vectors (batch_size, time_stpes, hidden_size).
            h (np.ndarray): Target hidden state (batch_size, hidden_size).

        Returns:
            np.ndarray: Attention weights for the input sequence.
        """
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a
    
    def backward(self, da: np.ndarray) -> tuple:
        """
        Backward pass for the attention weight layer.

        Args:
            da (np.ndarray): Gradients w.r.t the attention weights.

        Returns:
            tuple:
                np.ndarray: Gradients w.r.t the input sequence of vectors.
                np.ndarray: Gradients w.r.t the target hidden state.
        """
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh
    

class Attention:
    """
    Attention layer.

    This layer implements the attention mechanism, which focuses on specific parts of
    the input sequence when generating each element of the output sequence.
    
    Attributes:
        params (list): Parameters of the layer.
        grads (list): Gradients of the layer's parameters.
        attention_weight_layer (AttentionWeight): Layer to calculate attention weights.
        weight_sum_layer (WeightSum): Layer to calculate the weighted sum.
        attention_weight (np.ndarray): Calculated attention weights.
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Forward pass for the attention layer.

        Args:
            hs (np.ndarray): Input sequence of vectors (batch_size, time_stpes, hidden_size).
            h (np.ndarray): Target hidden state (batch_size, hidden_size).

        Returns:
            np.ndarray: Output vector after applying attention (batch_size, hidden_size)
        """
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout: np.ndarray) -> tuple:
        """
        Backward pass for the attention layer.

        Args:
            dout (np.ndarray): Gradients w.r.t the output of the attention layer.

        Returns:
            tuple: Gradients w.r.t the input sequence of vectors and the target hidden state.
        """
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh
    

class TimeAttention:
    """
    Time attention layer for sequence-to-sequence models.

    This layer applies an attention mechanism at each time step of the decoder's output.

    Attributes:
        params (list): Parameters of the layer.
        grads (list): Gradients of the layer's parameters.
        layer (list): List of Attention layers for each time step.
        attention_weight (list): List of attention weights for each time step.
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out
    
    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec
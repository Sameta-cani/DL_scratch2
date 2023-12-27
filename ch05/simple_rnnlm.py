import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    """
    A simple Recurrent Neural Network Language Model.

    This model uses a simple RNN layer followed by an affine layer to perform language modeling.

    Attributes:
        layers (list): List of layers in the model.
        loss_layer (TimeSoftmaxWithLoss): The softmax with loss layer.
        rnn_layer (TimeRNN): The RNN layer within the model.
        params (list): List of parameters of all layers.
        grads (list): List of gradients of all layers.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        """
        Initializes the SimpleRnnlm model with given sizes.

        Args:
            vocab_size (int): The size of the vocabulary.
            wordvec_size (int): The size of the word vectors.
            hidden_size (int): The size of the hidden state of the RNN.
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        def init_weight(*shape):
            return (np.random.randn(*shape) / np.sqrt(shape[0])).astype('f')
        
        embed_W = init_weight(V, D)
        rnn_Wx = init_weight(D, H)
        rnn_Wh = init_weight(H, H)
        rnn_b = np.zeros(H).astype('f')
        affine_W = init_weight(H, V)
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Forward pass for the SimpleRnnlm model.

        Args:
            xs (numpy.ndarray): Input data (word indices).
            ts (numpy.ndarray): Target data (word indices).

        Returns:
            float: The loss value for the input batch.
        """
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for the SimpleRnnlm model.

        Args:
            dout (float, optional): Upstream derivative. Defaults to 1.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        """
        Resets the state of the RNN layer.
        """
        self.rnn_layer.reset_state()


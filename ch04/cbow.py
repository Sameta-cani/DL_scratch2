import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    """
    Ths Continous Bag of Words (CBOW) model for word embedding.

    This class implements the CBOW model, a neural network architecture used for learning word embeddings.
    The model predicts a target word from its surrounding context words within a specified window.

    Attributes:
        in_layers (list): List of Embedding layers for context words.
        ns_loss (NegativeSamplingLoss): The negative sampling loss layer.
        params (list): Combined parameters from all layers in the model.
        grads (list): Combined gradients from all layers in the model.
        word_vecs (numpy.ndarray): Word vectors (weights of the input imbedding layers).
    """

    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, corpus: np.ndarray):
        """
        Initializes the CBOW model.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layer.
            window_size (int): The number of words considered as context around the target word.
            corpus (np.ndarray): The corpus of word indices.
        """
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H)
        W_out = 0.01 * np.random.randn(H, V)

        self.in_layers = [Embedding(W_in) for _ in range(2 * window_size)]
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the CBOW model.

        Computes the loss for a batch of context-target pairs.

        Args:
            contexts (numpy.ndarray): The context words as an array of word indices.
            target (numpy.ndarray): The target words as an array of word indices.

        Returns:
            float: The loss for the batch.
        """
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    def backward(self, dout: float=1) -> None:
        """
        Backward pass of the CBOW model.

        Computes the gradients with respect to the loss for backpropagation.

        Args:
            dout (float, optional): The gradient of the loss. Defaults to 1.

        Returns:
            None
        """
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
import sys
sys.path.append('..')
from common.layers import *
from ch04.negative_sampling_layer import NegativeSamplingLoss

class SkipGram:
    """
    The Skip-Gram model for word embedding.

    This class implements the Skip-Gram model, a popular neural network architecture used for learning word embeddings.
    The model aims to predict context words from a target word, which helps in learning word representations.

    Attributes:
        in_layer (Embedding): The embedding layer for the input word.
        loss_layers (list): A list of NegativeSamplingLoss layers, one for each context position.
        params (list): Combined parameters from the embedding and loss layers.
        grads (list): Combined gradeints from the embedding and loss layers.
        word_vecs (numpy.ndarray): The word vectors (weights of the input embedding layer).
    """

    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, corpus: np.ndarray):
        """
        Initializes the SkipGram model.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layer.
            window_size (int): The number of words on each side of the target word to consider as context.
            corpus (numpy.ndarray): The corpus of word indices.
        """
        V, H = vocab_size, hidden_size
        rn = np.random.randn

        W_in = 0.01 * rn(V, H).astype('f')
        W_out = 0.01 * rn(H, V).astype('f')

        self.in_layer = Embedding(W_in)
        self.loss_layers = [NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5) for _ in range(2 * window_size)]

        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the SkipGram model.

        Computes the loss for a batch of target-context pairs.

        Args:
            contexts (numpy.ndarray): The context words as an array of word indices.
            target (numpy.ndarray): The target words as an array of word indices.

        Returns:
            float: Toe total loss for the batch.
        """
        h = self.in_layer.forward(target)

        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss
    
    def backward(self, dout: float=1) -> None:
        """
        Backward pass of the SkipGram model.

        Computes the gradients with respect to the loss for backpropagation.

        Args:
            dout (float, optional): The gradient of the loss. Defaults to 1.
        """
        dh = sum(layer.backward(dout) for layer in self.loss_layers)
        self.in_layer.backward(dh)
        return None
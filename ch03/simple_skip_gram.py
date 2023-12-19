import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram:
    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Simple Skip-Gram model.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Size of the hidden layer.

        Initializes the model with random weights for input and output layers.

        Note:
        The weights are initialized using a normal distribution with a standard deviation of 0.01.

        Attributes:
            in_layer (MatMul): Input layer.
            out_layer (MatMul): Output layer.
            loss_layer1 (SoftmaxWithLoss): Softmax layer with loss for the first context word.
            loss_layer2 (SoftmaxWithLoss): Softmax layer with loss for the second context word.
            params (List): List of model parameters.
            grads (List): List of gradeints of model parameters.
            word_vecs (numpy.ndarray): Word embeddings (word vectors) stored in an instance variable.
        """
        V, H = vocab_size, hidden_size

        # Weight initialization
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # Layer creation
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        # Collect all weights and gradients into lists
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # Store word embeddings in an instance variable
        self.word_vecs = W_in

    def forward(self, contexts: np.ndarray, target: np.ndarray) -> float:
        """
        Perform forward pass of the Skip-Gram model.

        Args:
            contexts (numpy.ndarray): Context words represented as indices.
            target (numpy.ndarray): Target word represented as an index.

        Returns:
            float: Loss value for the given inputs.
        
        Computes the loss for both context words and sums them up.
        """
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss
    
    def backward(self, dout: float=1) -> None:
        """
        Perform backward pass of the Skip-Gram model.

        Args:
            dout (float, optional): Gradient of the loss with respect to the ouput. Defaults to 1.

        Backpropagates the gradient through the model to update the parameters.
        """
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
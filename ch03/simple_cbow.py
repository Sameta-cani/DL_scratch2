import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Simple Continuous Bag of Words (CBOW) model.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Size of the hidden layer.

        Initializes the model with random weights for input and output layers.

        Note:
        The weights are initialized using a normal distribution with a standard deviation of 0.01.

        Attributes:
            in_layer0 (MatMul): Input layer for the first context word.
            in_layer1 (MatMul): Input layer for the second context word.
            out_layer (MatMul): Output layer.
            loss_layer (SoftmaxWithLoss): Softmax layer with loss.
            params (list): List of model parameters.
            grads (list): Lisf of gradients of model parameters.
        """
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, context: np.ndarray, target: np.ndarray) -> float:
        """
        Perform forward pass of the CBOW model.

        Args:
            context (numpy.ndarray): Input context words represented as indices.
            target (numpy.ndarray): Target word represented as an index.

        Returns:
            float: Loss value for the given inputs.

        Computes the average hidden representation of the context words and calculate the loss.
        """
        h0 = self.in_layer0.forward(context[:, 0])
        h1 = self.in_layer1.forward(context[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout: int=1) -> None:
        """
        Perform backward pass of the CBOW model.

        Args:
            dout (int, optional): Gradient of the loss with respect to the output. Defaults to 1.

        Backpropagates the gradient through the model to update the parameters.
        """
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
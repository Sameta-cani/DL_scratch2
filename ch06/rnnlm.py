import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Rnnlm(BaseModel):
    """
    Recurrent Neural Network Langauge Model (RNNLM).

    This model uses a LSTM layer followed by an affine layer for langauge modeling. It is capable of capturing
    temporal dependencies in sequence data, making it suitable for tasks such as text generation and language modeling.

    Attributes:
        layers (list): List of layers in the model.
        loss_layer (TimeSoftmaxWithLoss): Softmax with loss layer.
        lstm_layer (TimeLSTM): LSTM layer within the model.
        params (list): List of parameters of the model.
        grads (list): List of gradients of the model's parameters.
    """

    def __init__(self, vocab_size: int=10000, wordvec_size: int=100, hidden_size: int=100):
        """
        Initializes the Rnnlm model.

        Args:
            vocab_size (int, optional): The size of the vocabulary. Defaults to 10000.
            wordvec_size (int, optional): The size of word vectors. Defaults to 100.
            hidden_size (int, optional): The size of the hidden state of the LSTM. Defaults to 100.
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.layers = [
            TimeEmbedding(self.init_weight(V, D)),
            TimeLSTM(self.init_weight(D, 4 * H), self.init_weight(H, 4 * H), np.zeros(4 * H).astype('f'), stateful=True),
            TimeAffine(self.init_weight(H, V), np.zeros(V).astype('f'))
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def init_weight(self, *shape: tuple) -> np.ndarray:
        """
        Initializes weights with a standard deviation scaled by the square root of the first dimension.

        Args:
            shape (tuple): Shape of the weight matrix.

        Returns:
            np.ndarray: Initialized weight matrix.
        """
        return (np.random.randn(*shape) / np.sqrt(shape[0])).astype('f')
    
    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the model for prediction.

        Args:
            xs (np.ndarray): Input data.

        Returns:
            np.ndarray: The output of the model.
        """
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Forward pass for training the Rnnlm model.

        Args:
            xs (np.ndarray): Input data.
            ts (np.ndarray): Target labels.

        Returns:
            float: Loss value for the input batch.
        """
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for training the Rnnlm model.

        Args:
            dout (float, optional): Upstream derivative. Defaults to 1.

        Returns:
            np.ndarray: Gradient with respect to the input data.
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        """
        Resets the state of the LSTM layer.
        """
        self.lstm_layer.reset_state()
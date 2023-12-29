import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *
from common.base_model import BaseModel


class BetterRnnlm(BaseModel):
    """
    Enhanced Recurrent Neural Network Langauge Model.

    This model uses LSTM layers with dropout for improved langauge modeling. It is desinged to be more effective
    than basic RNNLM at capturing long-term dependencies in sequence data.

    Args:
        vocab_size (int): Size of the vocabulary.
        wordvec_size (int): Size of the word vectors.
        hidden_size (int): Size of the hidden state in LSTM.
        dropout_ratio (float): Dropout ratio.
    """

    def __init__(self, vocab_size: int=10000, wordvec_size: int=650, hidden_size: int=650, dropout_ratio: float=0.5):
        """
        Initializes the BetterRnnlm Model.

        Args:
            vocab_size (int, optional): Size of the vocabulary. Defaults to 10000.
            wordvec_size (int, optional): Size of the word vectors. Defaults to 650.
            hidden_size (int, optional): Size of the hidden state in LSTM. Defaults to 650.
            dropout_ratio (float, optional): Dropout ratio. Defaults to 0.5.
        """
        V, D, H = vocab_size, wordvec_size, hidden_size

        def init_weight(input_size, output_size):
            return (np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)).astype('f')
        
        embed_W = init_weight(V, D)
        lstm_Wx1, lstm_Wh1, lstm_b1 = init_weight(D, 4 * H), init_weight(H, 4 * H), np.zeros(4 * H).astype('f')
        lstm_Wx2, lstm_Wh2, lstm_b2 = init_weight(H, 4 * H), init_weight(H, 4 * H), np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b) # Embedding layer weights are shared here.
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs: np.ndarray, train_flg: bool=False) -> np.ndarray:
        """
        Performs the forward pass of the model for prediction.

        Args:
            xs (np.ndarray): Input data.
            train_flg (bool, optional): Flag indicating whether the layer is in training mode or not. Defaults to False.

        Returns:
            np.ndarray: The output of the model.
        """
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs: np.ndarray, ts: np.ndarray, train_flg: bool=True) -> float:
        """
        Forward pass for training the BetterRnnlm model.

        Args:
            xs (np.ndarray): Input data.
            ts (np.ndarray): Target labels.
            train_flg (bool, optional): Flag indicating whether the layer is in training mode or not. Defaults to True.

        Returns:
            float: Loss value for the input batch.
        """
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for training the BetterRnnlm model.

        Args:
            dout (float, optional): Upstreamderivative. Defaults to 1.

        Returns:
            np.ndarray: Gradient with respect to the input data.
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        """
        Resets the state of the LSTM layers.
        """
        for layer in self.lstm_layers:
            layer.reset_state()

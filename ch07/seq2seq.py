import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Encoder:
    """
    Encoder for sequence processing using an embedding layer followed by LSTM.

    Attributes:
        embed (TimeEmbedding): Embedding layer.
        lstm (TimeLSTM): LSTM layer.
        params (list): Parameters of the encoder.
        grads (list): Gradients of the encoder's parameters.
        hs (np.ndarray): Hidden states of the LSTM layer.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        V, D, H = vocab_size, wordvec_size, hidden_size

        def init_weight(input_size, output_size):
            return (np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)).astype('f')
        

        embed_W = init_weight(V, D)
        lstm_Wx = init_weight(D, 4 * H)
        lstm_Wh = init_weight(H, 4 * H)
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for the encoder.

        Args:
            xs (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the last time step of LSTM.
        """
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]
    
    def backward(self, dh: np.ndarray) -> np.ndarray:
        """
        Backward pass for the encoder.

        Args:
            dh (np.ndarray): Gradient from the subsequent layer.

        Returns:
            np.ndarray: Gradient with respect to the input.
        """
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dhs)
        return dout
    

class Decoder:
    """
    Decoder for sequence generation using embedding, LSTM, and affine layers.

    Attributes:
        embed (TimeEmbedding): Embedding layer.
        lstm (TimeLSTM): LSTM layer.
        affine (TimeAffine): Affine layer.
        params (list): Parameters of the decoder.
        grads (list): Gradients of the decoder's parameters.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        V, D, H = vocab_size, wordvec_size, hidden_size

        def init_weight(input_size, output_size):
            return (np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)).astype('f')
        
        embed_W = init_weight(V, D)
        lstm_Wx = init_weight(D, 4 * H)
        lstm_Wh = init_weight(H, 4 * H)
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = init_weight(H, V)
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Forward pass for the decoder.

        Args:
            xs (np.ndarray): Input data.
            h (np.ndarray): Initial hidden state.

        Returns:
            np.ndarray: The score (output before softmax) of each time step.
        """
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score
    
    def backward(self, dscore: np.ndarray) -> np.ndarray:
        """
        Backward pass for the decoder.

        Args:
            dscore (np.ndarray): Gradient from the subsequent layer.

        Returns:
            np.ndarray: Gradient with respect to the initial hidden state.
        """
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
    
    def generate(self, h: np.ndarray, start_id: int, sample_size: int) -> list:
        """
        Generate a sequence of word IDs from a given start ID.

        Args:
            h (np.ndarray): Initial hidden state.
            start_id (int): Starting word ID.
            sample_size (int): Length of the sequence to generate.

        Returns:
            list: Generated sequence of word IDs.
        """
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id].reshape((1, 1)))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled
    

class Seq2seq(BaseModel):
    """
    Sequence-to-Sequence model for transforming one sequence to another.

    This model uses an encoder-decoder architecture to transform an input sequence into a target sequence.

    Attributes:
        encoder (Encoder): The encoder model.
        decoder (Decoder): The decoder model.
        softmax (TimeSoftmaxWithLoss): Softmax layer with loss computation.
        params (list): Parameters of the Seq2seq model.
        grads (list): Gradients of the Seq2seq model's parameters.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs: np.ndarray, ts: np.ndarray) -> float:
        """
        Forward pass for the Seq2seq model.

        Args:
            xs (np.ndarray): Input sequence.
            ts (np.ndarray): Target sequence.

        Returns:
            float: Loss for the given input and target sequence.
        """
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass for the Seq2seq model.

        Args:
            dout (float, optional): Upstream derivative. Defaults to 1.

        Returns:
            np.ndarray: Gradient with respect to the input sequence.
        """
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self, xs: np.ndarray, start_id: int, sample_size: int) -> list:
        """
        Generate a sequence using a Seq2seq model.

        Args:
            xs (np.ndarray): Input sequence to condition the generation.
            start_id (int): Starting ID for the sequence generation.
            sample_size (int): Length of the sequence to generate.

        Returns:
            list: Generated sequence of word IDs.
        """
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
import sys
sys.path.append('..')
from common.time_layers import *
from seq2seq import Seq2seq, Encoder


class PeekyDecoder:
    """
    Peeky Decoder for sequence generation.

    This decoder uses the 'peeky' approach, where the encoder's last hidden state is concatenated
    with the input at each time step of the decoder. It includes an embedding layer, ans LSTM layer,
    and an affine layer for the generation process.

    Attributes:
        embed (TimeEmbedding): Embedding layer for input sequences.
        lstm (TimeLSTM): LSTM layer for sequence processing.
        affine (TimeAffine): Affine layer for output generation.
        params (list): Combined parameters of all layers.
        grads (list): Combined gradients of all layers.
        cache (int): Size of the hidden layer, chached for backward pass.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        V, D, H = vocab_size, wordvec_size, hidden_size

        def init_weight(input_size, output_size):
            return (np.random.randn(input_size, output_size) / np.sqrt(input_size / 2)).astype('f')
        
        embed_W = init_weight(V, D)
        lstm_Wx = init_weight(H + D, 4 * H)
        lstm_Wh = init_weight(H, 4 * H)
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = init_weight(H + H, V)
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Forward pass for the PeekyDecoder.

        Args:
            xs (np.ndarray): Input sequence of wrod IDs.
            h (np.ndarray): Initial hidden state from the encoder.

        Returns:
            np.ndarray: Output score (before softmax) for each time step.
        """
        N, T = xs.shape
        H = h.shape[1]

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score
    
    def backward(self, dscore: np.ndarray) -> np.ndarray:
        """
        Backward pass for the PeekyDecoder.

        Args:
            dscore (np.ndarray): Gradient from the subsequent layer.

        Returns:
            np.ndarray: Gradient with respect to the initial hidden state.
        """
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dh0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dh1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dh0 + dh1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh
    
    def generate(self, h: np.ndarray, start_id: int, sample_size: int) -> list:
        """
        Generate a sequence using the PeekyDecoder.

        Args:
            h (np.ndarray): Initial hidden state from the encoder.
            start_id (int): Starting ID for the sequence generation.
            sample_size (int): Length of the sequence to generate.

        Returns:
            list: Generated sequence of word IDs.
        """
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled
    

class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
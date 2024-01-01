import sys
sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention

class AttentionEncoder(Encoder):
    """
    Attention Encoder.

    This encoder extends the basic Encoder for use in attention mechanisms. It processes the input
    sequence through an embedding layer followd by an LSTM layer.

    Inherits from:
        Encoder: The basic encoder class.
    """

    def forward(self, xs: np.ndarray) -> np.ndarray:
        """
        Forward pass for the attention encoder.

        Args:
            xs (np.ndarray): Input sequence of word IDs (batch_size, time_steps).

        Returns:
            np.ndarray: Output of the LSTM layer (batch_size, time_steps, hidden_size).
        """
        # Processing through embedding and LSTM layers
        return self.lstm.forward(self.embed.forward(xs))
    
    def backward(self, dhs: np.ndarray) -> np.ndarray:
        """
        Backward pass for the attention encoder.

        Args:
            dhs (np.ndarray): Gradients from the subsequent layer (batch_size, time_steps, hidden_size).

        Returns:
            np.ndarray: Gradients with respect to the input sequence.
        """
        # Propagating gradients through LSTM and embedding layers
        return self.embed.backward(self.lstm.backward(dhs))


class AttentionDecoder:
    """
    Attention Decoder:

    This decoder implements an attention mechanism in a sequence-to-sequence model. If takes a sequence
    of word IDs and the encoder's output, computes attention-based context vectors, and generates an
    output sequence.

    Attributes:
        embed (TimeEmbedding): Embedding layer for input sequences.
        lstm (TimeLSTM): LSTM layer for processing the embedded input.
        attention (TimeAttention): Attention layer for computing context vectors.
        affine (TimeAffine): Affine layer for generating the final output sequence.
        params (list): Parameters of the decoder.
        grads (list): Gradients of the decoder's parameters.
    """

    def __init__(self, vocab_size: int, wordvec_size: int, hidden_size: int):
        """
        Initializes the Attention Decoder

        Args:
            vocab_size (int): Size of the vocabulary.
            wordvec_size (int): Dimensionality of word embeddings.
            hidden_size (int): Size of the hidden state in the LSTM layer.
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2 * H, V)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs: np.ndarray, enc_hs: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Attention Decoder.

        Args:
            xs (np.ndarray): Input sequence of word IDs (batch_size, time_steps).
            enc_hs (np.ndarray): Output of the encoder (batch_size, time_steps, hidden_size).

        Returns:
            np.ndarray: Score for each word in the vocabulary at each time step.
        """
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore: np.ndarray) -> np.ndarray:
        """
        Backward pass the Attention Decoder.

        Args:
            dscore (np.ndarray): Gradient from the subsequent layer (batch_size, time_steps, vocab_size).

        Returns:
            np.ndarray: Gradient with respect to the encoder's output.
        """
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs
    
    def generate(self, enc_hs: np.ndarray, start_id: int, sample_size: int) -> list:
        """
        Generate a sequence using the Attention Decoder.

        Args:
            enc_hs (np.ndarray): Output of the encoder (batch_size, time_steps, hidden_size).
            start_id (int): ID of the start token to initiate the generation.
            sample_size (int): Length of the sequence to generate.

        Returns:
            list: Generated sequence of word IDs.
        """
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled
    

class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
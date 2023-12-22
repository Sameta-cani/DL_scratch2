import sys
sys.path.append('..')
from common.np import *
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    """
    A class representing an embedding dot product layer.

    This layer takes an embedding matrix and performs a dot product operation
    with an input vector, useful for tasks like word embedding in natural langauge processing.

    Attributes:
        embed (Embedding): An embedding layer.
        params (list): Parameters of the embedding layer.
        grads (list): Gradients of the embedding layer.
        cache (tuple): Cached data for use in backward pass.
    """

    def __init__(self, W: np.ndarray):
        """
        Initializes the EmbeddingDot layer.

        Args:
            W (numpy.ndarray): The weights of the embedding layer, typically the word embedding matrix.
        """
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Forward pass of the EmbeddingDot layer.

        Computes the dot product of the embedding vectors and the input vector.

        Args:
            h (numpy.ndarray): The input vector.
            idx (numpy.ndarray): Indices of the words (or tokens) in the embedding matrix.

        Returns:
            numpy.ndarray: The result of the dot product operation.
        """
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass of the EmbeddingDot layer.

        Computes the gradients with respect to the input vector and the embedding matrix.
        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the output of the layer.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input vector h.
        """
        h, target_W = self.cache
        dout = dout.reshape(dout.reshape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    

class UnigramSampler:
    """
    A sampler for generating negative samples based on unigram distribution.

    This class is used for negative sampling in natural language processing tasks,
    where it randomly samples words from a given corpus based on a unigram distribution,
    raised to a specified power. This technique is often used in word embedding models.

    Attributes:
        sample_size (int): The number of negative smaples to generate for each target word.
        vocab_size (int): The size of the vocabulary in the corpus.
        word_p (numpy.ndarray): The probability distribution of words in the corpus.
    """

    def __init__(self, corpus: np.ndarray, power: float, sample_size: int):
        """
        Initializes the UnigramSampler with a given corpus, power, and sample size.

        Args:
            corpus (numpy.ndarray): The corpus of word IDs.
            power (float): The power to which the unigram distribution is raised.
            sample_size (int): The number of negative samples to be generated for each target word.
        """
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target: np.ndarray) -> np.ndarray:
        """
        Generates negative samples for a given target.

        Args:
            target (numpy.ndarray): An array of target word indices.

        Returns:
            numpy.ndarray: An array of negative samples.
        """
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                target_idx = target[i]
                p = self.word_p.copy()
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size), 
                                               replace=True, p=self.word_p)
        
        return negative_sample
    

class NegativeSamplingLoss:
    """
    A loss layer for negative sampling in word embedding models.

    This layer implements negative sampling, a technique commonly used in training word embedding models.
    It uses a unigram sampler to generate negative samples and calculates the loss for both positive  and
    negative samples.

    Attributes:
        sample_size (int): The number of negative samples to generate for each positive sample.
        sampler (UnigramSampler): The sampler used to generate negative samples.
        loss_layer (SigmoidWithLoss): The sigmoid loss layer and used for calculating the loss.
        embed_dot_layer (EmbeddingDot): The embedding dot layer used for calculating the dot product.
        params (list): Parameters of the embed dot layer.
        grads (list): Gradients of the embed dot layer.
    """

    def __init__(self, W: np.ndarray, corpus: np.ndarray, power: float=0.75, sample_size: int=5):
        """
        Initializes the NegativeSamplingLoss layer.

        Args:
            W (numpy.ndarray): The weight matrix for the embedding layer.
            corpus (numpy.ndarray): The corpus of word IDs.
            power (float, optional): The power to which the unigram distribution is raised. Defaults to 0.75.
            sample_size (int, optional): The number of negative samples to generate for each positive sample. Defaults to 5.
        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h: np.ndarray, target: np.ndarray) -> float:
        """
        Forward pass of the NegativeSamplingLoss layer.

        Computes the loss for a batch of positive samples and their corresponding negative samples.

        Args:
            h (numpy.ndarray): The input hidden layer vectors.
            target (numpy.ndarray): An array of target word indices.

        Returns:
            float: The total loss for the batch.
        """
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """
        Backward pass of the NegativeSamplingLoss layer.

        Computes the gradient of the loss with respect to the input vector.

        Args:
            dout (float, optional): The gradient of the loss with respect to the output. Defaults to 1.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input hidden layer vector.
        """
        dh = 0
        for l0, l1 in zip(self.loss_layer, self.embed_dot_layer):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh
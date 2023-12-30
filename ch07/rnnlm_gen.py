import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    """
    RNN Language Model for Text Generation.

    This model extends Rnnlm to generate text sequences based on a given start word.

    """

    def generate(self, start_id: int, skip_ids: list=None, sample_size: int=100) -> list:
        """
        Generate a sequences of word IDs from a given start ID.

        Args:
            start_id (int): Starting word ID.
            skip_ids (list, optional): List of word IDs to skip. Defaults to None.
            sample_size (int, optional): Length of the sequence to generate. Defaults to 100.

        Returns:
            list: Generated sequence of word IDs.
        """
        word_ids = [start_id]
        x = np.array([start_id]).reshape(1, 1)

        while len(word_ids) < sample_size:
            score = self.predict(x)
            p = softmax(score.flatten())
            sampled = np.random.choice(len(p), size=1, p=p)
            if skip_ids is None or sampled[0] not in skip_ids:
                x[...] = sampled
                word_ids.append(int(sampled[0]))

        return word_ids
    
    def get_state(self) -> tuple:
        """
        Get the current state of the model.

        Returns:
            tuple: Current hidden state and cell state of the LSTM layer.
        """
        return self.lstm_layer.h, self.lstm_layer.c
    
    def set_state(self, state: tuple):
        """
        Set the state of the model.

        Args:
            state (tuple): State to set (hidden state, cell state).
        """
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):
    """
    Better RNN Language Model for Text Generation.

    This model extends BetterRnnlm to generate text sequences based on a given start word.

    """
    def generate(self, start_id: int, skip_ids: list=None, sample_size: int=100) -> list:
        """
        Generate a sequence of word IDs from a given start ID.

        Args:
            start_id (int): Starting word ID.
            skip_ids (list, optional): List of word IDs to skip. Defaults to None.
            sample_size (int, optional): Length of the sequence to generate. Defaults to 100.

        Returns:
            list: _description_
        """
        word_ids = [start_id]
        x = np.array([start_id]).reshape(1, 1)

        while len(word_ids) < sample_size:
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if skip_ids is None or sampled[0] not in skip_ids:
                x[...] = sampled
                word_ids.append(int(sampled[0]))

        return word_ids
    
    def get_state(self) -> list:
        """
        Get the current states of LSTM layers.

        Returns:
            list of tuples: Current hidden states and cell states of all LSTM layers.
        """
        return [(layer.h, layer.c) for layer in self.lstm_layers]
    
    def set_state(self, states: list):
        """
        Set the states of LSTM layers.

        Args:
            states (list of tuples): States to set (hidden states, cell states) for each LSTM layer.
        """
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)
import sys
sys.path.append('..')
import os
from common.np import *

def preprocess(text: str) -> tuple:
    """
    Preprocesses the input text, assigns unique IDs to words, and generates an array based on the word mappings.

    Args:
        text (str): Text data to be preprocessed

    Returns:
        tuple: 
            - corpus (numpy.ndarray): Array where each word is converted to a unique ID
            - word_to_id (dict): Dictionary mapping each word to its unique ID
            - id_to_word (dict): Dictionary mapping each unique ID to its corresponding word
    """
    # Convert to lowercase and add spaces before and after periods
    text = text.lower().replace('.', ' .')

    # Split words based on whitespace
    words = text.split(' ')

    # Remove duplicate words and assign unique IDs to each word
    word_to_id = {word: idx for idx, word in enumerate(set(words))}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Convert each word to its corresponding ID and store in an array
    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word

def clip_grads(grads: list, max_norm: float) -> None:
    """
    Clip gradients to prevent the exploding gradient problem.

    Args:
        grads (list): List of gradient arrays.
        max_norm (float): Maximum allowed gradient norm.

    Returns:
        None: The function modifies the input gradients in-place.
    """
    total_norm = np.sqrt(np.sum(np.square(grads)))
    rate = max_norm / (total_norm + 1e-6)

    if rate < 1:
        for grad in grads:
            grad *= rate
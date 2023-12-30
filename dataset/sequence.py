import sys
sys.path.append('..')
import os
import numpy as np

id_to_char = {}
char_to_id = {}


def _update_vocab(txt: str):
    """
    Update the vocabulary with new characters from the given text.

    This function assigns a unique ID to each new character found in the text and updates
    the char_to_id and id_to_char dictionaries.

    Args:
        txt (str): The text to process.
    """

    # Iterate through each character in the text
    for char in txt:
        # If the character is not in the char_to_id dictionary, add it
        if char not in char_to_id:
            new_id = len(char_to_id)
            char_to_id[char] = new_id
            id_to_char[new_id] = char


def load_data(file_name: str='addition.txt', seed: int=1984) -> tuple:
    """
    Load data from a file and split it into training and test sets.

    Args:
        file_name (str, optional): Name of the file containing the data. Defaults to 'addition.txt'.
        seed (int, optional): Seed for random shuffling of the data. Defaults to 1984.

    Returns:
        tuple: Tuple containing training and test data.
    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)

    if not os.path.exists(file_path):
        print(f'No file: {file_name}')
        return None
    
    questions, answers = [], []
    for line in open(file_path, 'r'):
        idx = line.find('_')
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # Update vocabulary based on the data
    for question, answer in zip(questions, answers):
        _update_vocab(question)
        _update_vocab(answer)

    # Convert questions and answers to integer arrays
    x = np.array([[char_to_id[c] for c in question] for question in questions], dtype=np.int32)
    t = np.array([[char_to_id[c] for c in answer] for answer in answers], dtype=np.int32)

    # Shuffle the data
    indices = np.arange(len(x))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    x, t = x[indices], t[indices]

    # Split the data into training and test sets
    split_at = len(x) - len(x) // 10
    x_train, x_test = x[:split_at], x[split_at:]
    t_train, t_test = t[:split_at], t[split_at:]


    return (x_train, t_train), (x_test, t_test)

def get_vocab():
    return char_to_id, id_to_char
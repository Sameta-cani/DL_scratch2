import sys
sys.path.append('..')
import os
from common.np import *

def preprocess(text):
    """
    Preprocesses the input text, assigns unique IDs to words in the order of their first appearance,
    and generates an array based on the word mappings.

    Args:
        text (str): Text data to be preprocessed

    Returns:
        corpus (numpy.ndarray): Array where each word is converted to a unique ID
        word_to_id (dict): Dictionary mapping each word to its unique ID in the order of first appearance
        id_to_word (dict): Dictionary mapping each unique ID to its corresponding word
    """
    # Convert to lowercase and add spaces before and after periods
    text = text.lower().replace('.', ' .')

    # Split words based on whitespace
    words = text.split(' ')

    # Remove duplicate words and assign unique IDs to each word in the order of first appearance
    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    id_to_word = {idx: word for word, idx in word_to_id.items()}

    # Convert each word to its corresponding ID and store in an array
    corpus = np.array([word_to_id[word] for word in words])

    return corpus, word_to_id, id_to_word

def cos_similarity(x: np.ndarray, y: np.ndarray, eps: float=1e-8) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        x (numpy.ndarray): First vector.
        y (numpy.ndarray): Second vector.
        eps (float, optional): Small value to avlid division by zero. Defaults to 1e-8.

    Returns:
        float: Cosine similarity between vectors x and y.
    """
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x) + eps
    norm_y = np.linalg.norm(y) + eps
    similarity = dot_product / (norm_x * norm_y)
    return similarity
    
def most_similar(query: str, word_to_id: dict, id_to_word: dict, word_matrix: np.ndarray, top: int=5):
    """
    Find and print words most similar to the given query word.

    Args:
        query (str): Query word.
        word_to_id (dict): Dictionary mapping words to their unique IDs.
        id_to_word (dict): Dictionary mapping unique IDs to their corresponding words.
        word_matrix (numpy.ndarray): Matrix where each row represents a word vector.
        top (int, optional): Number of top similar words to print. Defaults to 5.
    """
    if query not in word_to_id:
        print(f'{query}(을)를 찾을 수 없습니다.')
        return
    
    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 유사도가 높은 순으로 정렬하여 상위 단어 출력
    most_similar_indices = np.argsort(similarity)[::-1]

    count = 0
    for i in most_similar_indices:
        if id_to_word[i] == query:
            continue
        print(f' {id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return



def create_co_matrix(corpus: np.ndarray, vocab_size: int, window_size: int=1) -> np.ndarray:
    """
    Create a co-occurrence matrix based on the given corpus.

    Args:
        corpus (numpy.ndarray): Array where each element represents the unique ID of a word.
        vocab_size (int): Size of the vocabulary, indicating the number of unique words.
        window_size (int, optional): Size of the context window for co-occurrence. Defaults to 1.

    Returns:
        numpy.ndarray: Co-occurrence matrix where each element (i, j) represents the 
        co-occurrence count of word with ID i and word with ID j within the specified window.
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        start = max(0, idx - window_size)
        end = min(corpus_size, idx + window_size + 1)

        context_ids = corpus[start:end]
        context_ids = np.delete(context_ids, np.where(context_ids == word_id))

        for context_id in context_ids:
            co_matrix[word_id, context_id] += 1

    return co_matrix


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


def ppmi(C: np.ndarray, verbose: bool=False, eps: float=1e-8) -> np.ndarray:
    """
    Calculate Positive Pointwise Mutual Information (PPMI) matrix from the given co-occurrence matrix.

    Args:
        C (numpy.ndarray): Co-occurrence matrix.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        numpy.ndarray: PPMI matrix.
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100 + 1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    return M
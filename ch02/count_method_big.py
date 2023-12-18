import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

def calculate_word_vectors(corpus: list, window_size: int=2, wordvec_size: int=100) -> np.ndarray:
    """
    Calculate word vectors using the given corpus.

    Args:
        corpus (list): Word indices in the corpus.
        window_size (int, optional): Size of the context window for co-occurrence. Defaults to 2.
        wordvec_size (int, optional): Size of the resulting wor vectors. Defaults to 100.

    Returns:
        numpy.ndarray: Matrix contraining word vectors.
    """
    vocab_size = len(set(corpus))
    C = create_co_matrix(corpus, vocab_size, window_size)
    W = ppmi(C, verbose=True)

    print('Calculating SVD...')
    try:
        from sklearn.utils.extmath import randomized_svd
        U, _, _ = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        U, _, _ = np.linalg.svd(W)

    return U[:, :wordvec_size]

def main():
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    word_vecs = calculate_word_vectors(corpus)

    query_words = ['you', 'year', 'car', 'toyota']
    for query in query_words:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    
if __name__ == '__main__':
    main()
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

def print_matrix_info(matrix, title):
    print(title)
    print(matrix)
    print('-' * 50)

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print_matrix_info(C, '동시발생 행렬')
print_matrix_info(W, 'PPMI')
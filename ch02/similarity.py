import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

word1, word2 = 'you', 'i'
id1, id2 = word_to_id[word1], word_to_id[word2]

# "you"와 "i"의 단어 벡터 추출
vec1, vec2 = C[id1], C[id2]

# 코사인 유사도 계산 및 출력
similarity = cos_similarity(vec1, vec2)
print(similarity)
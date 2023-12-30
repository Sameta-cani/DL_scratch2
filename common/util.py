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


def convert_one_hot(corpus: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Convert a given corpus into one-hot encoded representation.

    Parameters:
    - corpus (numpy.ndarray): Input corpus to be converted.
        If `corpus` is 1-dimensional, it is treated as a collection of word indices.
        If `corpus` is 2-dimensional, each row represents a sequence of word indices.
    - vocab_size (int): Size of the vocabulary, determining the dimensionality of the one-hot encoding.

    Returns:
    - numpy.ndarray: One-hot encoded representation of the input corpus.
        If the input is 1-dimensional, the output will be a 2-dimensional array.
        If the input is 2-dimensional, the output will be a 3-dimensional array.

    Note:
    The function uses NumPy's eye function for vectorized one-hot encoding without explicit loops.

    Example:
    ```python
    corpus_1d = np.array([0, 2, 1])
    result_1d = convert_one_hot(corpus_1d, vocab_size=4)
    print(result_1d)
    # Output: [[1 0 0 0]
    #          [0 0 1 0]
    #          [0 1 0 0]]

    corpus_2d = np.array([[0, 2], [1, 3]])
    result_2d = convert_one_hot(corpus_2d, vocab_size=4)
    print(result_2d)
    # Output: [[[1 0 0 0] [0 0 1 0]]
    #          [[0 1 0 0] [0 0 0 1]]]
    """
    one_hot = np.eye(vocab_size, dtype=np.int32)[corpus]

    return one_hot


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


def to_cpu(x):
    """
    Transfers a given array to the CPU memory.

    This function is designed to handle arrays that are either already in NumPy format
    or in a different format (like those used in GPU computations). If the array is not
    in NumPy format, it converts it to a NumPy array.

    Args:
        x: The array to be transferred to CPU memory. Can be a NumPy array or an array in another format.

    Returns:
        numpy.ndarray: The array in NumPy format, ensuring it is in the CPU memory.
    """
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    """
    Transfers a given array to GPU memory.

    This function is designed to handle arrays that are either already in CuPy format
    (a library for array operations equivalent to NumPy but on NVIDIA GPUs) or in NumPy format.
    If the array is in NumPy format, it converts it to CuPy format, facilitating GPU-accelerated operations.

    Args:
        x: The array to be transferred to GPU memory. Can be a CuPy array or a NumPy array.

    Returns:
        cupy.ndarray: The array in CuPy format, ensuring it is in GPU memory.
    """
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)



def clip_grads(grads: list, max_norm: float) -> None:
    """
    Clip gradients to prevent the exploding gradient problem.

    Args:
        grads (list): List of gradient arrays.
        max_norm (float): Maximum allowed gradient norm.

    Returns:
        None: The function modifies the input gradients in-place.
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def eval_perplexity(model, corpus: np.ndarray, batch_size: int=10, time_size: int=35) -> float:
    """
    Evaluates the perplexity of the model on the given corpus.

    Args:
        model (Model): The model to evaluate.
        corpus (np.ndarray): The corpus as an array of word indices.
        batch_size (int, optional): The batch size. Defaults to 10.
        time_size (int, optional): The size of each time slice. Defaults to 35.

    Returns:
        float: The evaluated perplexity.
    """
    print('퍼플렉서티 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl

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


import numpy as np

def create_contexts_target(corpus, window_size=1):
    """
    Create context-target pairs from a given corpus for word embeddings.

    Parameters:
    - corpus (numpy.ndarray): Array representing the input corpus as a sequence of word indices.
    - window_size (int): Size of the context window on each side of the target word. Default is 1.

    Returns:
    - contexts (numpy.ndarray): Array where each row represents the context words for a target word.
    - target (numpy.ndarray): Array representing the target words corresponding to each row in the contexts array.
    """
    target = corpus[window_size:-window_size]
    contexts = np.array([list(corpus[i - window_size:i]) + list(corpus[i + 1:i + window_size + 1])
                         for i in range(window_size, len(corpus) - window_size)])

    return contexts, np.array(target)

def analogy(a: str, b: str, c: str, word_to_id: dict, id_to_word: dict, word_matrix: np.ndarray, top: int=5, answer: str=None):
    """
    Solves word analogies using word vectors.

    Given three words, a, b, and c, the function finds words that complete the analogy 
    'a is to b as c is to ?'. It computes this by finding the word whose vector representation 
    is closest to the vector 'b - a + c' in terms of cosine similarity.

    Args:
        a (str): First word in the analogy (word a).
        b (str): Second word in the analogy (word b).
        c (str): Third word in the analogy (word c).
        word_to_id (dict): A dictionary mapping words to their respective indices in the word matrix.
        id_to_word (dict): A dictionary mapping indices in the word matrix to their respective words.
        word_matrix (numpy.ndarray): A matrix where each row represents a word vector.
        top (int, optional): The number of top similar words to return. Defaults to 5.
        answer (str, optional): The expected answer word, if known. Defaults to None.

    Prints:
        The function prints the analogy, the expected answer (if provided), and the top similar words
        from the word matrix that complete the analogy.
        
    Note:
        If any of the words a, b, or c are not in the word_to_id dictionary, the function prints an
        error message and returns without performing the analogy.
    """
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(을)를 찾을 수 없습니다.' % word)
            return
        
    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return

    
def eval_seq2seq(model, question: np.ndarray, correct: np.ndarray, id_to_char: dict,
                 verbos: bool=False, is_reverse: bool=False):
    """
    Evaluate a Seq2seq model.

    This function generates a response to a given question using the Seq2seq model,
    and compares it with the correct answer. It optionally prints both the question and
    the generated answer, along with the correct answer for visualization.

    Args:
        model (Seq2seq): The Seq2seq model to be evaluated.
        question (np.ndarray): The input question represented as a sequence of IDs.
        correct (np.ndarray): The correct answer represented as a sequence of IDs.
        id_to_char (dict): A dictionary mapping ID to character.
        verbos (bool, optional): If True, prints the question, the correct answer, and the model's answer.
        is_reverse (bool, optional): If True, reverses the order of the question's characters before printing.

    Returns:
        int: 1 if the model's answer matches the correct answer, 0 otherwise.
    """
    correct = correct.flatten()
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbos:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' '+ guess)
        print('---')

    return 1 if guess == correct else 0


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalizes a given NumPy array.

    This function normalizes a NumPy array either in one-dimensional (1D) or two-dimensional (2D) form.
    For a 2D array, each row is treated as a separate vector and is normalized independently.
    For a 1D array, it is treated as a single vector for normalization.
    Normalization is performed by dividing each element by the Euclidean norm (sqrt of the sum of squares) of the vector.

    Args:
        x (numpy.ndarray): A NumPy array to be normalized. Can be either 1D or 2D.

    Returns:
        numpy.ndarray: The normalized array with the same shape as the input.
    """
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(axis=1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x

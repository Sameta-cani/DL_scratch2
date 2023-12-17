import numpy as np

def load_data(seed: int=1984) -> tuple:
    """
    Generate synthetic data for a classification task with three classes.

    Args:
        seed (int, optional): Seed for random number generation. Defaults to 1984.
    
    Returns:
        tuple: A tuple containing two numpy arrays:
            - x (numpy.ndarray): Input data of shape (300, 2) representing 2D coordinates.
            - t (numpy.ndarray): One-hot encoded labels of shape (300, 3) for three classes.
    """
    np.random.seed(seed)
    N = 100
    DIM = 2
    CLS_NUM = 3

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int16)

    for j in range(CLS_NUM):
        rates = np.arange(N) / N
        radii = 1.0 * rates
        thetas = j * 4.0 + 4.0 * rates + np.random.randn(N) * 0.2

        indices = range(N * j, N * (j + 1))
        x[indices] = np.column_stack([radii * np.sin(thetas), radii * np.cos(thetas)])
        t[indices, j] = 1

    return x, t
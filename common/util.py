import sys
sys.path.append('..')
import os
from common.np import *

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
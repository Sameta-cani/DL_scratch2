import numpy as np

# 그래디언트 클리핑 함수
def clip_grads(grads: list, max_norm: float):
    """
    Clips gradients to a specified maximum norm.

    Args:
        grads (list): List of gradients to be clipped.
        max_norm (float): The maximum norm value.
    """
    # 그래디언트의 총 노름 계산
    total_norm = np.sqrt(sum(np.sum(grad ** 2) for grad in grads))

    # 클리핑 비율 계산 및 적용
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

# 예시 그래디언트
dW1 = np.random.randn(3, 3) * 10
dW2 = np.random.randn(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

print('before:', dW1.flatten())
clip_grads(grads, max_norm)
print('after:', dW1.flatten())
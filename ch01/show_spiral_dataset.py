import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print(f'x, {x.shape}')
print(f't, {t.shape}')

# 데이터점 플롯
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']

for i, marker in enumerate(markers):
    plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=marker, label=f'Class {i + 1}')

plt.legend()
plt.show()
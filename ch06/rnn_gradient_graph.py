import numpy as np
import matplotlib.pyplot as plt

# 초기 설정
N = 2 # 배치 크기
H = 3 # 은닉 상태 벡터의 크기
T = 20 # 시계열 데이터의 길이

dh = np.ones((N, H)) # 초기 그래디언트
np.random.seed(3) # 재현할 수 있도록 난수의 시드 고정
# Wh = np.random.randn(H, H) # 그래디언트 폭발
Wh = np.random.randn(H, H) * 0.5 # 그래디언트 소실

# 그래디언트 폭발/소실을 확인하기 위한 리스트
norm_list = []

# RNN 그래디언트 폭발/소실 시연
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

# 결과 출력
print(norm_list)

# 결과 시각화
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('Time step')
plt.ylabel('Norm of gradients')
plt.title('Exploding/Vanishing Gradients in RNNs')
plt.show()
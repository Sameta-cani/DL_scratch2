# 게이트가 추가된 RNN

```
📦ch06
 ┣ 📜better_rnnlm.py
 ┣ 📜clip_grads.py
 ┣ 📜rnnlm.py
 ┣ 📜rnn_gradient_graph.py
 ┣ 📜train_better_rnnlm.py
 ┗ 📜train_rnnlm.py
```

## 정리
- 단순한 RNN의 학습에서는 기울기 소실과 기울기 폭발이 문제가 된다.
- 기울기 폭발에는 기울기 클리핑, 기울기 소실에는 게이트가 추가된 RNN(LSTM과 GRU 등)이 효과적이다.
- LSTM에는 input 게이트, forget 게이트, output 게이트 등 3개의 게이트가 있다.
- 게이트에는 전용 가중치가 있으며, 시그모이드 함수를 사용하여 0.0~1.0 사이의 실수를 출력한다.
- 언어 모델 개선에는 LSTM 계층 다층화, 드롭아웃, 가중치 공유 등의 기법이 효과적이다.
- RNN의 정규화는 중요한 주제이며, 드롭아웃 기반의 다양한 기법이 제안되고 있다.

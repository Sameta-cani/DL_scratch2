# RNN을 사용한 문장 생성

```
📦ch07
 ┣ 📜generate_better_text.py
 ┣ 📜generate_text.py
 ┣ 📜peeky_seq2seq.py
 ┣ 📜rnnlm_gen.py
 ┣ 📜seq2seq.py
 ┣ 📜show_addition_dataset.py
 ┗ 📜train_seq2seq.py
``

## 정리

- RNN을 이용한 언어 모델은 새로운 문장을 생성할 수 있다.
- 문장을 생성할 때는 하나의 단어(혹은 문자)를 주고 모델의 출력(확률분포)에서 샘플링하는 과정을 반복한다.
- RNN을 2개 조합함으로써 시계열 데이터를 다른 시계열 데이터로 변환할 수 있다.
- seq2seq는 Encoder가 출발어 입력문을 인코딩하고, 인코딩된 정보를 Decoder가 받아 이코딩하여 도착어 출력문을 얻는다.
- 입력문을 반전시키는 기법(Reverse), 또는 인코딩된 정보를 Decoder의 여러 계층에 전달하는 기법(Peeky)는 seq2seq의 정확도 향상에 효과적이다.
- 기계 번역, 챗봇, 이미지 캡셔닝 등 seq2seq는 다양한 애플리케이션에 이용할 수 있다.

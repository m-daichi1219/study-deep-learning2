import numpy as np
from common.layers import Softmax

# Encoderで変換した情報をすべて保持し、
# 各時系列のDecoderで対応する単語を選び出す計算手順。
# エンコード情報の配列hsに対して、各単語の重要度aの重み付きの和を求める
T, H = 5, 4
hs = np.random.randn(T, H)
a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])

ar = a.reshape(5, 1).repeat(4, axis=1)
print(ar.shape)

t = hs * ar
print(t.shape)

c = np.sum(t, axis=0)
print(c.shape)

# バッチ処理版の重み付きの和
N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
a = np.random.randn(N, T)
ar = a.reshape(N, T, 1).repeat(H, axis=2)

t = hs * ar
print(t.shape)

c = np.sum(t, axis=1)
print(c.shape)

print("----------")
# Encoderから出力される隠れ層hsとLSTMの出力hの
# ベクトル間の内積から類似度のスコアを算出
N, T, H = 10, 5, 4
hs = np.random.randn(N, T, H)
h = np.random.randn(N, H)
hr = h.reshape(N, 1, H).repeat(T, axis=1)

t = hs * hr
print(t.shape)

s = np.sum(t, axis=2)
print(s.shape)

softmax = Softmax()
a = softmax.forward(s)
print(a.shape)


import numpy as np


W = np.arange(21).reshape(7, 3)
print("▼W")
print(W)

# 特定の行の抜き出し
print("▼W[2]")
print(W[2])
print("▼W[5]")
print(W[5])

idx = np.array([1, 0, 3, 0])
print("▼複数行をまとめて抜き出す")
print(W[idx])

# 次は4.2 word2vecの改良②から進める
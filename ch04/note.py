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

# Negative Sampling
# np.random.choice()メソッドの使用例
print("▼0から9の数字の中からひとつの数字をランダムにサンプリング")
print(np.random.choice(10))

print("▼wordsからひとつだけランダムにサンプリング")
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
print(np.random.choice(words))

print("▼5つだけランダムサンプリング（重複あり）")
print(np.random.choice(words, size=5))

print("▼5つだけランダムサンプリング（重複なし）")
print(np.random.choice(words, size=5, replace=False))

print("▼確率分布に従ってサンプリング")
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p=p))



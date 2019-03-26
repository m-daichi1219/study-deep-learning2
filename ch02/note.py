import numpy as np
import matplotlib.pyplot as plt
from ch02 import functions


text = 'You say goodbye and I say hello.'

# text = text.lower()
# text = text.replace('.', ' .')
# print("▼text")
# print(text)
#
# words = text.split(' ')
# print("▼words")
# print(words)
#
# word_to_id = {}
# id_to_word ={}
#
# for word in words:
#     if word not in word_to_id:
#         new_id = len(word_to_id)
#         word_to_id[word] = new_id
#         id_to_word[new_id] = word
#
# print("▼id_to_word")
# print(id_to_word)
# print("▼word_to_id")
# print(word_to_id)
#
# corpus = [word_to_id[w] for w in words]
# corpus = np.array(corpus)
# print("▼corpus")
# print(corpus)

corpus, word_to_id, id_to_word = functions.preprocess(text);
print("▼id_to_word")
print(id_to_word)
print("▼word_to_id")
print(word_to_id)
print("▼corpus")
print(corpus)

# C = np.array([
#     [0, 1, 0, 0, 0, 0, 0],
#     [1, 0, 1, 0, 1, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 1, 0],
# ], dtype=np.int32)
#
# print("▼co-occurence matrix")
# print(C[0])
# print(C[2])
# print(C[word_to_id["say"]])

vocab_size = len(word_to_id)
C = functions.create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print("▼youとiのコサイン類似度")
print(functions.cos_similarity(c0, c1))

functions.most_similar('you', word_to_id, id_to_word, C, top=5)

W = functions.ppmi(C)

np.set_printoptions(precision=3)
print("covariance matrix")
print(C)
print('-'*50)
print('PPMI')
print(W)

# Singular Value Decomposition
U, S, V = np.linalg.svd(W)

print("▼共起行列")
print(C[0])
print("▼PPMI行列")
print(W[0])
print("▼SVD")
print(U[0])
print("▼二次元ベクトルに削減")
print(U[0, :2])

# グラフにプロット
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id,  1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()

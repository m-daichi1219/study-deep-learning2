import numpy as np
from ch02 import preprocess


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

corpus, word_to_id, id_to_word = preprocess.preprocess(text);
print("▼id_to_word")
print(id_to_word)
print("▼word_to_id")
print(word_to_id)
print("▼corpus")
print(corpus)

C = np.array([
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
], dtype=np.int32)

print("▼co-occurence matrix")
print(C[0])
print(C[2])
print(C[word_to_id["say"]])

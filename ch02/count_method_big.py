import numpy as np
from ch02 import functions, ptb
import time

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = functions.create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = functions.ppmi(C, verbose=True)

print('calculating SVD ...')

# start time
start = time.time()

# truncated SVD(fast)
from sklearn.utils.extmath import randomized_svd
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

# SVD(slow)  elapsed_time:628.2649700641632[sec]
#U, S, V = np.linalg.svd(W)

# elapsed time
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    functions.most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

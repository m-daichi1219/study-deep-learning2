from common.util import most_similar, analogy
import pickle

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_ward = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_ward, word_vecs, top=5)

# 類推テスト
#   単語ベクトルの空間上において、類推を行う単語のベクトルができるだけ近くなる単語を探す
#   (「man → woman」ベクトルと「king → ?」ベクトルができるだけ近くなる単語を探して類推を行う）
analogy('man', 'king', 'woman',  word_to_id, id_to_ward, word_vecs, top=5)
analogy('king', 'man', 'queen',  word_to_id, id_to_ward, word_vecs, top=5)
analogy('take', 'took', 'go',  word_to_id, id_to_ward, word_vecs, top=5)
analogy('car', 'cars', 'child',  word_to_id, id_to_ward, word_vecs, top=5)
analogy('good', 'better', 'bad',  word_to_id, id_to_ward, word_vecs, top=5)
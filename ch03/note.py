import numpy as np
from ch03 import layers
from ch03 import functions
from ch03.simple_cbow import SimpleCBOW
from common.trainer import Trainer
from common.optimizer import Adam


print("▼one-hot表現の単語を全結合層によって変換")
c = np.array([[1, 0, 0, 0, 0, 0, 0]])   # 入力
W = np.random.randn(7, 3)               # 重み

#h = np.dot(c, W)                        # 中間ノード

layer = layers.MatMul(W)                # MatMulレイヤに重みを設定
h = layer.forward(c)                    # 順伝播の処理を行う
print(h)

# CBOWモデルの推論処理
# サンプルのコンテキストデータ
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 重みの初期化
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# レイヤの生成
in_layer0 = layers.MatMul(W_in)
in_layer1 = layers.MatMul(W_in)
out_layer = layers.MatMul(W_out)

# 順伝播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h1 + h0)
s = out_layer.forward(h)

print("▼CBOWモデル推論結果（スコア）")
print(s)

# コーパスからコンテキストとターゲットを作成
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = functions.preprocess(text)

print("▼コーパス")
print(corpus)

contexts, target = functions.create_contexts_target(corpus, window_size=1)

print("▼コンテキスト")
print(contexts)
print("▼ターゲット")
print(target)

# one-hot表現に変換
vocab_size = len(word_to_id)
target = functions.convert_one_hot(target, vocab_size)
contexts = functions.convert_one_hot(contexts, vocab_size)

print("▼ターゲット（one-hot)")
print(target)   # 形状:(6, 7)
print("▼コンテキスト（one-hot)")
print(contexts) # 形状:(6, 2, 7)

# 3.4.1 学習コードの実装
windows_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = functions.preprocess(text)
vocab_size = len(word_to_id)
contexts, target = functions.create_contexts_target(corpus, windows_size)
target = functions.convert_one_hot(target, vocab_size)
contexts = functions.convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 重みを確認
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

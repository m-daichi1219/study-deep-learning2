import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq


# データセットの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 改良①:入力データの反転
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# モデル / オプティマイザ / トレーナーの生成
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# 改良②:PeekySeq2seq(エンコードされた入力データのベクトルhをすべてのレイヤで使用する）
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_train[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# 改良前：少しずつだが、着実に学習している
# | epoch 1 |  iter 1 / 351 | time 0[s] | loss 2.56
# | epoch 1 |  iter 21 / 351 | time 1[s] | loss 2.53
# | epoch 1 |  iter 41 / 351 | time 2[s] | loss 2.17
# ...
# val acc 0.040%
# | epoch 5 |  iter 1 / 351 | time 0[s] | loss 1.28
# | epoch 5 |  iter 21 / 351 | time 0[s] | loss 1.29
# | epoch 5 |  iter 41 / 351 | time 1[s] | loss 1.28

# 改良①:学習の進みが早くなる
#           変換前と変換後の単語の位置（時系列）が近くなるため、
#           学習の進みが早くなると考えられている（理論的なことはわかっていない。。）
# | epoch 1 |  iter 1 / 351 | time 0[s] | loss 2.56
# | epoch 1 |  iter 21 / 351 | time 0[s] | loss 2.52
# | epoch 1 |  iter 41 / 351 | time 1[s] | loss 2.17
# ...
# val acc 0.080%
# | epoch 5 |  iter 1 / 351 | time 0[s] | loss 1.01
# | epoch 5 |  iter 21 / 351 | time 0[s] | loss 1.01
# | epoch 5 |  iter 41 / 351 | time 1[s] | loss 1.00

# 改良②:Peekyを実装
# val acc 0.020%
# | epoch 25 |  iter 1 / 351 | time 0[s] | loss 0.02
# | epoch 25 |  iter 21 / 351 | time 0[s] | loss 0.01
# | epoch 25 |  iter 41 / 351 | time 1[s] | loss 0.01
# →まったく学習が進んでいない、実装が間違えてる。。

# 2019/04/22
# eval_seq2seqの引数に改良①（リバース）の引数が漏れてた
# そのため、学習は進んでいたが、評価時に正しいデータとマッチングできていなかった
# val acc 96.960%
# | epoch 25 |  iter 1 / 351 | time 0[s] | loss 0.02
# | epoch 25 |  iter 21 / 351 | time 0[s] | loss 0.01
# | epoch 25 |  iter 41 / 351 | time 1[s] | loss 0.01
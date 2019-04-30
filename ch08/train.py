import numpy as np
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch08.attention_seq2seq import AttentionSeq2Seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq

# データの読み込み
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 入力文を反転
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# ハイパーパラメータの設定
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size= batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val avv %.3f%%' % (acc * 100))

model.save_params()

# epoch3終了時には精度99.9%に到達する！
#
# | epoch 3 |  iter 1 / 351 | time 0[s] | loss 0.35
# | epoch 3 |  iter 21 / 351 | time 13[s] | loss 0.30
# | epoch 3 |  iter 41 / 351 | time 25[s] | loss 0.21
# | epoch 3 |  iter 61 / 351 | time 37[s] | loss 0.14
# | epoch 3 |  iter 81 / 351 | time 50[s] | loss 0.09
# | epoch 3 |  iter 101 / 351 | time 62[s] | loss 0.07
# | epoch 3 |  iter 121 / 351 | time 74[s] | loss 0.05
# | epoch 3 |  iter 141 / 351 | time 87[s] | loss 0.04
# | epoch 3 |  iter 161 / 351 | time 99[s] | loss 0.03
# | epoch 3 |  iter 181 / 351 | time 112[s] | loss 0.03
# | epoch 3 |  iter 201 / 351 | time 124[s] | loss 0.02
# | epoch 3 |  iter 221 / 351 | time 136[s] | loss 0.02
# | epoch 3 |  iter 241 / 351 | time 149[s] | loss 0.02
# | epoch 3 |  iter 261 / 351 | time 161[s] | loss 0.01
# | epoch 3 |  iter 281 / 351 | time 173[s] | loss 0.01
# | epoch 3 |  iter 301 / 351 | time 186[s] | loss 0.01
# | epoch 3 |  iter 321 / 351 | time 198[s] | loss 0.01
# | epoch 3 |  iter 341 / 351 | time 211[s] | loss 0.01
# Q 10/15/94
# T 1994-10-15
# O 1994-10-15
# ---
# Q thursday, november 13, 2008
# T 2008-11-13
# O 2008-11-13
# ---
# Q Mar 25, 2003
# T 2003-03-25
# O 2003-03-25
# ---
# Q Tuesday, November 22, 2016
# T 2016-11-22
# O 2016-11-22
# ---
# Q Saturday, July 18, 1970
# T 1970-07-18
# O 1970-07-18
# ---
# Q october 6, 1992
# T 1992-10-06
# O 1992-10-06
# ---
# Q 8/23/08
# T 2008-08-23
# O 2008-08-23
# ---
# Q 8/30/07
# T 2007-08-30
# O 2007-08-30
# ---
# Q 10/28/13
# T 2013-10-28
# O 2013-10-28
# ---
# Q sunday, november 6, 2016
# T 2016-11-06
# O 2016-11-06
# ---
# val avv 99.900%
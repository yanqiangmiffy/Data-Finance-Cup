# #! -*- coding:utf-8 -*-
#
# import json
# import numpy as np
# import pandas as pd
# from random import choice
# from keras_bert import load_trained_model_from_checkpoint, Tokenizer
# import re, os
# import codecs
#
#
# class OurTokenizer(Tokenizer):
#     def _tokenize(self, text):
#         R = []
#         for c in text:
#             if c in self._token_dict:
#                 R.append(c)
#             elif self._is_space(c):
#                 R.append('[unused1]')  # space类用未经训练的[unused1]表示
#             else:
#                 R.append('[UNK]')  # 剩余的字符是[UNK]
#         return R
#
#
# maxlen = 100
# config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'
#
# token_dict = {}
#
# with codecs.open(dict_path, 'r', 'utf8') as reader:
#     for line in reader:
#         token = line.strip()
#         token_dict[token] = len(token_dict)
# tokenizer = OurTokenizer(token_dict)
#
# neg = pd.read_excel('neg.xls', header=None)
# pos = pd.read_excel('pos.xls', header=None)
#
# data = []
#
# for d in neg[0]:
#     data.append((d, 0))
#
# for d in pos[0]:
#     data.append((d, 1))
#
# # 按照9:1的比例划分训练集和验证集
# random_order = range(len(data))
# np.random.shuffle(random_order)
# train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
# valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
#
#
# def seq_padding(X, padding=0):
#     L = [len(x) for x in X]
#     ML = max(L)
#     return np.array([
#         np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
#     ])
#
#
# def load_data(text_list):
#     """
#
#     :return:
#     """
#     X1 = []
#     X2 = []
#
#     for text in text_list:
#         text = text[:maxlen]
#         x1, x2 = tokenizer.encode(first=text)
#         X1.append(x1)
#         X2.append(x2)
#     X1 = seq_padding(X1)
#     X2 = seq_padding(X2)
#     return [X1, X2]
#
#
# class data_generator:
#     def __init__(self, data, batch_size=32):
#         self.data = data
#         self.batch_size = batch_size
#         self.steps = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1
#
#     def __len__(self):
#         return self.steps
#
#     def __iter__(self):
#         while True:
#             idxs = range(len(self.data))
#             np.random.shuffle(idxs)
#             X1, X2, Y = [], [], []
#             for i in idxs:
#                 d = self.data[i]
#                 text = d[0][:maxlen]
#                 x1, x2 = tokenizer.encode(first=text)
#                 y = d[1]
#                 X1.append(x1)
#                 X2.append(x2)
#                 Y.append([y])
#                 if len(X1) == self.batch_size or i == idxs[-1]:
#                     X1 = seq_padding(X1)
#                     X2 = seq_padding(X2)
#                     Y = seq_padding(Y)
#                     yield [X1, X2], Y
#                     [X1, X2, Y] = [], [], []
#
#
# from keras.layers import *
# from keras.models import Model
# import keras.backend as K
# from keras.optimizers import Adam
#
# bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
#
# for l in bert_model.layers:
#     l.trainable = True
#
# x1_in = Input(shape=(None,))
# x2_in = Input(shape=(None,))
#
# x = bert_model([x1_in, x2_in])
# x = Lambda(lambda x: x[:, 0])(x)
# p = Dense(1, activation='sigmoid')(x)
#
# model = Model([x1_in, x2_in], p)
# model.compile(
#     loss='binary_crossentropy',
#     optimizer=Adam(1e-5),  # 用足够小的学习率
#     metrics=['accuracy']
# )
# model.summary()
#
# train_D = data_generator(train_data)
# valid_D = data_generator(valid_data)
#
# model.fit_generator(
#     train_D.__iter__(),
#     steps_per_epoch=len(train_D),
#     epochs=5,
#     validation_data=valid_D.__iter__(),
#     validation_steps=len(valid_D)
# )

import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs
import ipykernel

# A toy input example
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

#
# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=10,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)
#

# Use the trained model
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,      # The input layers and output layer will be returned if `training` is `False`
    trainable=False,     # Whether the model is trainable. The default value is the same with `training`
    output_layer_num=4,  # The number of layers whose outputs will be concatenated as a single output.
                         # Only available when `training` is `False`.
)
print(inputs,output_layer)


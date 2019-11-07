#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/16 21:46
@Author:  yanqiang
@File: 10.entity_embedding_neural_net.py

"""
# https://www.kaggle.com/aquatic/entity-embedding-neural-net

import numpy as np
import pandas as pd
from keras.callbacks import *
from keras.layers import *

# random seeds for stochastic parts of neural network
np.random.seed(10)
from tensorflow import set_random_seed
from sklearn.metrics import *

set_random_seed(15)
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding
from sklearn.model_selection import StratifiedKFold
import ipykernel


class roc_auc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = (roc_auc_score(self.y, y_pred) * 2) - 1

        y_pred_val = self.model.predict(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = (roc_auc_score(self.y_val, y_pred_val) * 2) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (
            str(round(roc, 5)), str(round(roc_val, 5)), str(round((roc * 2 - 1), 5)), str(round((roc_val * 2 - 1), 5))),
              end=10 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# Data loading & preprocessing

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
df_train = train.merge(train_target, on='id')
df_test = pd.read_csv("new_data/test.csv")
df = pd.concat([df_train, df_test], sort=False, axis=0)
df.fillna(value=999999999, inplace=True)  # bankCard存在空值
# 删除重复列
# duplicated_features = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6',
#                        'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
#                        'x_15', 'x_17', 'x_18', 'x_19', 'x_21',
#                        'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
#                        'x_59', 'x_60', 'x_77', 'x_78'] + \
#                       ['x_40', 'x_70'] + \
#                       ['x_41'] + \
#                       ['x_43'] + \
#                       ['x_45'] + \
#                       ['x_61']
# df = df.drop(columns=duplicated_features)
no_features = ['id', 'target']

numerical_features = ['certValidBegin', 'certValidStop', 'lmt']
# certId
df['certId_first2'] = df['certId'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['certId_middle2'] = df['certId'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['certId_last2'] = df['certId'].apply(lambda x: int(str(x)[4:6]))  # 最后两位
# dist
df['dist_first2'] = df['dist'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['dist_middle2'] = df['dist'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['dist_last2'] = df['dist'].apply(lambda x: int(str(x)[4:6]))  # 最后两位

# residentAddr
df['residentAddr_first2'] = df['residentAddr'].apply(lambda x: int(str(x)[:2]) if x != -999 else -999)  # 前两位
df['residentAddr_middle2'] = df['residentAddr'].apply(lambda x: int(str(x)[2:4]) if x != -999 else -999)  # 中间两位
df['residentAddr_last2'] = df['residentAddr'].apply(lambda x: int(str(x)[4:6]) if x != -999 else -999)  # 最后两位

# bankCard
df['bankCard'] = df['bankCard'].astype(int)
df['bankCard_first6'] = df['bankCard'].apply(lambda x: int(str(x)[:6]) if x != -999 else -999)
df['bankCard_last3'] = df['bankCard'].apply(lambda x: int(str(x)[6:].strip()) if x != -999 else -999)

categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
for feat in numerical_features + ['certValidPeriod']:
    df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化

X_train, y_train = df[:len(train)], df[:len(train)].target
X_test = df[len(train):]

cols_use = [c for c in X_train.columns if (c not in no_features)]
X_train = X_train[cols_use]
X_test = X_test[cols_use]

col_vals_dict = {c: list(X_train[c].unique()) for c in categorical_features}
print(col_vals_dict)
embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c]) >= 2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c]))  # look at value counts to know the embedding dimensions
print('\n')
print(len(embed_cols))


def build_embedding_network():
    inputs = []
    embeddings = []
    for i in range(len(embed_cols)):
        cate_input = Input(shape=(1,))
        input_dim = len(col_vals_dict[embed_cols[i]])
        if input_dim > 1000:
            output_dim = 50
        else:
            output_dim = (len(col_vals_dict[embed_cols[i]]) // 2) + 1

        embedding = Embedding(input_dim, output_dim, input_length=1)(cate_input)
        embedding = Reshape(target_shape=(output_dim,))(embedding)
        inputs.append(cate_input)
        embeddings.append(embedding)

    input_numeric = Input(shape=(4,))
    embedding_numeric = Dense(5)(input_numeric)
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)
    x = Dense(300, activation='relu')(x)
    x = Dropout(.35)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.15)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    return model


# converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    print(other_cols, len(other_cols))
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


# gini scoring function from kernel at:
# https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n


def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


def train():
    # network training
    K = 5
    runs_per_fold = 1
    n_epochs = 10

    cv_ginis = []
    full_val_preds = np.zeros(np.shape(X_train)[0])
    y_preds = np.zeros((np.shape(X_test)[0], K))

    kfold = StratifiedKFold(n_splits=K,
                            random_state=231,
                            shuffle=True)

    for i, (f_ind, outf_ind) in enumerate(kfold.split(X_train, y_train)):

        X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
        y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]

        X_test_f = X_test.copy()

        # upsampling adapted from kernel:
        # https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
        pos = (pd.Series(y_train_f == 1))
        # Add positive examples
        X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
        y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)

        # Shuffle data
        idx = np.arange(len(X_train_f))
        np.random.shuffle(idx)
        X_train_f = X_train_f.iloc[idx]
        y_train_f = y_train_f.iloc[idx]

        # preprocessing
        proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)

        # track oof prediction for cv scores
        val_preds = 0
        auc_callback = roc_auc_callback(training_data=(proc_X_train_f, y_train_f),
                                        validation_data=(proc_X_val_f, y_val_f))
        NN = build_embedding_network()

        NN.summary()
        NN.fit(proc_X_train_f,
               y_train_f.values,
               epochs=n_epochs,
               batch_size=128,
               verbose=1,
               # callbacks=[auc_callback]
               )

        val_preds += NN.predict(proc_X_val_f)[:, 0] / runs_per_fold
        y_preds[:, i] += NN.predict(proc_X_test_f)[:, 0] / runs_per_fold
        NN.save('models/entity{}.hd5'.format(i + 1))

        full_val_preds[outf_ind] += val_preds
        cv_gini = gini_normalizedc(y_val_f.values, val_preds)
        cv_auc = roc_auc_score(y_val_f.values, val_preds)
        cv_ginis.append(cv_gini)
        print('\nFold %i prediction cv gini: %.5f\n' % (i, cv_gini))
        print('\nFold %i prediction cv auc: %.5f\n' % (i, cv_auc))

    print('Mean out of fold gini: %.5f' % np.mean(cv_ginis))
    print('Full validation gini: %.5f' % gini_normalizedc(y_train.values, full_val_preds))

    y_pred_final = np.mean(y_preds, axis=1)
    print(len(df_test.id))
    print(len(y_pred_final))
    df_sub = pd.DataFrame({'id': df_test.id,
                           'target': y_pred_final},
                          columns=['id', 'target'])
    df_sub.to_csv('result/NN_EntityEmbed_10fold-sub.csv', index=False)

    pd.DataFrame(full_val_preds).to_csv('result/NN_EntityEmbed_10fold-val_preds.csv', index=False)


train()


def extract_embedding():
    NN = build_embedding_network()
    NN.load_weights('models/entity1.hd5')
    weight = NN.get_weights()

    print(weight)
    print(type(weight))
    print(weight[0])
    print(type(weight[0]))
    print(weight[0].shape)
    print(weight[1].shape)
    print(weight[2].shape)
    print(weight[3].shape)
    print(len(weight))
    # for w in weight:
    #     print(w)
# extract_embedding()

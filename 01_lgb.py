#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/8 21:33
@Author:  yanqiang
@File: 01_lgb.py
"""
# lgb模型
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
import operator
import time
from gensim.models import Word2Vec
import multiprocessing

# 1.读取文件
train = pd.read_csv("data/train.csv")
train_target = pd.read_csv('data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("data/test.csv")

print(train.shape)
print(test.shape)
# 2.合并数据
test['target'] = -1
data = pd.concat([train, test], sort=False, axis=0)
print(train.shape)
print(test.shape)
print(data.shape)

L = 50


def w2v_feat(data_frame, feat, mode):
    for i in feat:
        if data_frame[i].dtype != 'object':
            data_frame[i] = data_frame[i].astype(str)
    data_frame.fillna('nan', inplace=True)

    print(f'Start {mode} word2vec ...')
    model = Word2Vec(data_frame[feat].values.tolist(), size=L, window=2, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    stat_list = ['min', 'max', 'mean', 'std']
    new_all = pd.DataFrame()
    for m, t in enumerate(feat):
        print(f'Start gen feat of {t} ...')
        tmp = []
        for i in data_frame[t].unique():
            tmp_v = [i]
            tmp_v.extend(model[i])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = [f'w2c_{t}_{n}' for n in range(L)]
        tmp_df.columns = [t] + w2c_list
        tmp_df = data_frame[['id', t]].merge(tmp_df, on=t)
        print(tmp_df)
        # tmp_df = tmp_df.drop_duplicates().groupby('UID').agg(stat_list).reset_index()
        # tmp_df.columns = ['UID'] + [f'{p}_{q}' for p in w2c_list for q in stat_list]
        if m == 0:
            new_all = pd.concat([new_all, tmp_df], axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='id')
    new_all.to_csv('all.csv', index=None)
    return new_all


no_feas = ['id', 'target'] + ['certId', 'bankCard', 'dist', 'residentAddr']
data['certPeriod'] = data['certBalidStop'] - data['certValidBegin']
numerical_features = ['certBalidStop', 'certValidBegin', 'lmt', 'age', 'certPeriod']
categorical_features = [fea for fea in data.columns if fea not in numerical_features + no_feas]

# %%

features = [fea for fea in data.columns if fea not in no_feas]
w2v_feat(data, features, 'all')

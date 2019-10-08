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

# 1.读取文件
train = pd.read_csv("data/train.csv")
train_target = pd.read_csv('data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("data/test.csv")

print(train.shape)
print(test.shape)
# 2.合并数据
test['y'] = -1
data = pd.concat([train, test], axis=0)
print(train.shape)
print(test.shape)
print(data.shape)

# 3.对特征进行分析，分为数值型 、 类别型
numerical_features = []
categorical_features = []

stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]

stats = []
for col in test.columns:
    stats.append((col, test[col].nunique(), test[col].isnull().sum() * 100 / test.shape[0],
                  test[col].value_counts(normalize=True, dropna=False).values[0] * 100, test[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Percentage of missing values', ascending=False)[:10]
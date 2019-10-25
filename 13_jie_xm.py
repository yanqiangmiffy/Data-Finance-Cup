#!/usr/local/Cellar python
# _*_coding:utf-8_*_

"""
@Author: 姜小帅
@file: xm.py
@Time: 2019-10-17 09:29
@Say something:  
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
"""
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv("data/xm/train.csv")
train_target = pd.read_csv('data/xm/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("data/xm/test.csv")
test['target'] = -1
df = pd.concat([train, test], sort=False, axis=0)


duplicated_features = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6',
                       'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
                       'x_15', 'x_17', 'x_18', 'x_19', 'x_21',
                       'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
                       'x_59', 'x_60', 'x_77', 'x_78'] + \
                      ['x_22', 'x_40', 'x_70'] + \
                      ['x_41'] + \
                      ['x_43'] + \
                      ['x_45'] + \
                      ['x_61']

df = df.drop(columns=duplicated_features)
print(df.shape)


no_features = ['id', 'target', 'isNew']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

df['bankCard6'] = df['bankCard'].apply(lambda x: x // 1000)
df['bankCard-3'] = df['bankCard'].apply(lambda x: x % 1000)

df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']

cols = ['bankCard', 'residentAddr', 'certId', 'dist', 'certValidPeriod', 'age', 'job', 'ethnic', 'basicLevel',
        'linkRela']
for col in cols:
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

df['is_edu_equal'] = (df['edu'] == df['highestEdu']).astype(int)
df['is_dist_certId_equal'] = (df['dist'] == df['certId']).astype(int)

feature = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold

n_fold = 5
y_scores = 0
y_pred_l1 = np.zeros([n_fold, test.shape[0]])
y_pred_all_l1 = np.zeros(test.shape[0])

label = ['target']

kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1314)
for i, (train_index, valid_index) in enumerate(kfold.split(train[feature], train[label])):

    print(i)
    X_train, y_train, X_valid, y_valid = train.loc[train_index][feature], train[label].loc[train_index], \
                                         train.loc[valid_index][feature], train[label].loc[valid_index]

    bst = xgb.XGBClassifier(max_depth=3, n_estimators=10000, learning_rate=0.01, tree_method='gpu_hist')
    bst.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', verbose=500, early_stopping_rounds=500)

    if i != 1:
        y_pred_l1[i] = bst.predict_proba(test[feature])[:, 1]
        y_pred_all_l1 += y_pred_l1[i]
        y_scores += bst.best_score

test['target'] = y_pred_all_l1 / 4
print('average score is {}'.format(y_scores / 4))

test[['id', 'target']].to_csv('xgb_submit2.csv', index=False) 
test['target'].plot()
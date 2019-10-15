# %%

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
import operator
import time

# %%

# 1.读取文件
train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")
print(train.shape)
print(test.shape)
# 2.合并数据
test['target'] = -1
data = pd.concat([train, test], sort=False, axis=0)
print(train.shape)
print(test.shape)
print(data.shape)

# %%

# 简单数据描述
stats = []
for col in train.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False)[:30]

# %%

stats = []
for col in test.columns:
    stats.append((col, test[col].nunique(), test[col].isnull().sum() * 100 / test.shape[0],
                  test[col].value_counts(normalize=True, dropna=False).values[0] * 100, test[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False)[:30]

# %%

# 特征工程
# 根据 unique values确定
dup_feature = ['x_2', 'x_3', 'x_4', 'x_5', 'x_6',
               'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13', 'x_15', 'x_17',
               'x_18', 'x_19', 'x_21', 'x_23', 'x_24',
               'x_36', 'x_37', 'x_38', 'x_40', 'x_57', 'x_58', 'x_59', 'x_60',
               'x_70', 'x_77', 'x_78']
# + \
# ['x_61', 'x_22', 'x_40', 'x_41', 'x_45', 'x_43']
no_feas = ['id', 'target'] + ['certId', 'bankCard', 'dist', 'residentAddr', 'certValidStop',
                              'certValidBegin'] + dup_feature
data['certPeriod'] = data['certValidStop'] - data['certValidBegin']
numerical_features = ['certValidStop', 'certValidBegin', 'lmt', 'age', 'certPeriod']
# data['certBalidStop_certValidBegin_ratio']=data ['certBalidStop']/data['certValidBegin']
# data['lmt_age_ratio']=data ['lmt']/data['age']
# data['certPeriod_age_ratio']=data ['certPeriod']/data['age']
#
# data['lmt_age_mul']=data ['lmt']*data['age']
# data['certPeriod_age_mul']=data ['certPeriod']*data['age']

categorical_features = [fea for fea in data.columns if fea not in numerical_features + no_feas]
# cols = [col for col in (set(numerical_features))]
# for col in cols:
#     data[col + '_Rank'] = data[col].rank()

# from tqdm import tqdm
# for cate in tqdm(['certId', 'bankCard', 'dist', 'residentAddr']):
#     for fea in numerical_features:
#         grouped_df = data.groupby(cate).agg({fea: ['mean','skew',pd.DataFrame.kurt]})
#         grouped_df.columns = [cate+'_' + '_'.join(col).strip() for col in grouped_df.columns.values]
#         grouped_df = grouped_df.reset_index()
#         data = pd.merge(data, grouped_df, on=cate, how='left')
# %%

features = [fea for fea in data.columns if fea not in no_feas]

# %%

train = data.loc[data['target'] != -1, :]  # train set
test = data.loc[data['target'] == -1, :]  # test set
y = train['target'].values.astype(int)
X = train[features].values
print("X shape:", X.shape)
print("y shape:", y.shape)
test_data = test[features].values
print("test shape", test_data.shape)

print(len(features))

# 保存特征数据
np.save("tmp/fea_train.npy", X)
np.save("tmp/fea_test.npy", test_data)

# %%

# 训练
# 采取分层采样
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("start：********************************")
start = time.time()

N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)

auc_cv = []
pred_cv = []
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], \
                                       y[train_in], y[test_in]

    # 数据结构
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 4,
        'min_child_weight': 6,
        'num_leaves': 16,
        'learning_rate': 0.02,  # 0.05
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        # 'lambda_l1':0.25,
        # 'lambda_l2':0.5,
        # 'scale_pos_weight':10.0/1.0, #14309.0 / 691.0, #不设置
        # 'num_threads':4,
    }
    print('................Start training..........................')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100,
                    verbose_eval=100)

    print('................Start predict .........................')
    # 预测
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # 评估
    tmp_auc = roc_auc_score(y_test, y_pred)
    auc_cv.append(tmp_auc)
    print("valid auc:", tmp_auc)
    # test
    pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    pred_cv.append(pred)
# K交叉验证的平均分数
print('the cv information:')
print(auc_cv)
print('cv mean score', np.mean(auc_cv))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 10.5折交叉验证结果均值融合，保存文件
mean_auc = np.mean(auc_cv)
print("mean auc:", mean_auc)
filepath = 'result/lgb_' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(pred_cv)
print("总的结果：", res.shape)
# 最后结果平均，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = DataFrame()
result['id'] = test['id']
result['target'] = r
result.to_csv(filepath, index=False, sep=",")

# %%


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

print("start：********************************")
start = time.time()

auc_list = []
pred_list = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 参数设置
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eta': 0.02,
              'max_depth': 4,
              'min_child_weight': 6,
              'colsample_bytree': 0.7,
              'subsample': 0.7,
              # 'gamma':0,
              # 'lambda':1,
              # 'alpha ':0，
              'silent': 1
              }
    params['eval_metric'] = ['auc']
    # 数据结构
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtrain, 'train'), (dvali, 'valid')]  # 'valid-auc' will be used for early stopping
    # 模型train
    model = xgb.train(params, dtrain,
                      num_boost_round=2000,
                      evals=evallist,
                      early_stopping_rounds=100,
                      verbose_eval=100)
    # 预测验证
    pred = model.predict(dvali, ntree_limit=model.best_ntree_limit)
    # 评估
    auc = roc_auc_score(y_test, pred)
    print('...........................auc value:', auc)
    auc_list.append(auc)
    # 预测
    dtest = xgb.DMatrix(test_data)
    pre = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    pred_list.append(pre)
print('......................validate result mean :', np.mean(auc_list))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)

print("over:*********************************")

# 11.5折结果均值融合，并保存文件
mean_auc = np.mean(auc_list)
print("mean auc:", mean_auc)
filepath = 'result/xgb_' + str(mean_auc) + '.csv'  # 线下平均分数
# 转为array
res = np.array(pred_list)
print("5折结果：", res.shape)

# 最后结果，mean
r = res.mean(axis=0)
print('result shape:', r.shape)
result = DataFrame()
result['id'] = test['id']
result['target'] = r
result.to_csv(filepath, index=False, sep=",")

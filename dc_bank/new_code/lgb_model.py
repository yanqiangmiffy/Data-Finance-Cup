# -*- coding:utf-8 _*-
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import copy
import lightgbm as lgb
from tqdm import tqdm
import os
from datetime import timedelta
from sklearn.feature_selection import chi2, SelectPercentile
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

scaler = StandardScaler()


def get_fea(train, test):
    test['target'] = 'test'

    df = pd.concat((train, test))

    no_features = ['id', 'target', 'isNew', 'x_61', 'x_22', 'x_40', 'x_41', 'x_45', 'x_43']
    no_features.extend(['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
                        'x_15', 'x_17', 'x_18', 'x_19', 'x_21', 'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
                        'x_59', 'x_60', 'x_77', 'x_78', 'x_65', 'x_31', 'x_16', 'x_56', 'x_44', 'x_32', 'x_39'])

    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']

    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features


def get_result(train, test, label, my_model, splits_nums=5):
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    # train = train.values
    # test = test.values

    label = label.values.astype(int)

    score_list = []
    pred_cv = []
    important = []

    k_fold = StratifiedKFold(n_splits=splits_nums, shuffle=True, random_state=1314)
    for index, (train_index, test_index) in enumerate(k_fold.split(train, label)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        # importance = pd.DataFrame({
        #     'column': features,
        #     'importance': model.feature_importance(),
        # }).sort_values(by='importance')
        important.append(model.feature_importance())

        # plt.figure(figsize=(12,6))
        # lgb.plot_importance(model, max_num_features=300)
        # plt.title("Featurertances_%s" % index)
        # plt.show()

        vali_pre = model.predict(X_test, num_iteration=model.best_iteration)
        print(np.array(y_test))
        print(np.array(vali_pre))

        score = roc_auc_score(list(y_test), list(vali_pre))

        score_list.append(score)
        print(score)

        pred_result = model.predict(test, num_iteration=model.best_iteration)
        pred_cv.append(pred_result)

        res = np.array(pred_cv)

        print("总的结果：", res.shape)
        print(score_list)
        print(np.mean(score_list))
        r = res.mean(axis=0)

    importance_all = pd.DataFrame({
        'column': features,
        'importance': np.array(important).mean(axis=0),
    }).sort_values(by='importance')

    print(importance_all)
    return r


def lgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 16,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'random_state': 1024,
        'n_jobs': -1,
    }

    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    num_round = 50000
    model = lgb.train(params,
                      trn_data,
                      num_round,
                      valid_sets=[trn_data, val_data],
                      verbose_eval=10,
                      early_stopping_rounds=100,
                      feature_name=features)
    return model


def xgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 5,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.8,  # 随机采样训练样本
        'colsample_bytree': 0.8,  # 生成树时进行的列采样
        'min_child_weight': 18,
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.03,  # 如同学习率
        'eval_metric': 'auc',
    }
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
    model = xgb.train(params, xgb_train, num_boost_round=500, evals=watchlist)
    return model


label = pd.read_csv('../data/train_target.csv')
train = pd.read_csv('../data/train.csv')
train = pd.merge(train, label, on='id', how='left')
test = pd.read_csv('../data/test.csv')

train, test, no_features = get_fea(train, test)
print('get_fea ok')

label = train['target']
sub = test[['id']]

features = [fea for fea in train.columns if fea not in no_features]

# features = [fea for fea in features if 'x' not in fea]

train_df = train[features]

test_df = test[features]

print(train_df.head())

r = get_result(train_df, test_df, label, lgb_para_binary_model, splits_nums=5)

test['target'] = r

test[['id', 'target']].to_csv('../result/submission_lgb.csv', index=None)




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

scaler = StandardScaler()

def get_fea(train, test):
    test['target'] = 'test'

    df = pd.concat((train, test))

    # df['same_addr'] = df.apply(lambda x: 1 if x['residentAddr'] == x['dist'] else 0, axis=1)

    # 相同 ['x_0', 'x_1', 'x_10', 'x_11', 'x_13', 'x_15', 'x_17', 'x_18', 'x_19', 'x_2', 'x_21', 'x_22', 'x_23', 'x_24',
    # 'x_3', 'x_36', 'x_37', 'x_38', 'x_4', 'x_40', 'x_5', 'x_57', 'x_58', 'x_59', 'x_6', 'x_60', 'x_7', 'x_70', 'x_77',
    # 'x_78', 'x_8', 'x_9']

    no_features = ['id', 'target']

    # no_features.extend(['x_1', 'x_10', 'x_11', 'x_13', 'x_15', 'x_17', 'x_18', 'x_19', 'x_2', 'x_21', 'x_22', 'x_23',
    #                     'x_24','x_3', 'x_36', 'x_37', 'x_38', 'x_4', 'x_40', 'x_5', 'x_57', 'x_58', 'x_59', 'x_6',
    #                     'x_60', 'x_7', 'x_70', 'x_77', 'x_78', 'x_8', 'x_9'])

    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']

    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features

def get_result(train, test, label, my_model, splits_nums=5):

    print(list(set(label)))

    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    print(label.unique())
    # label = np.array(label)

    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)
    print(np.sum(label))

    score_list = []
    # test = xgb.DMatrix(test)
    # k_fold = KFold(n_splits=splits_nums, shuffle=True, random_state=1024)
    k_fold = StratifiedKFold(n_splits=splits_nums, shuffle=True, random_state=2048)
    for index, (train_index, test_index) in enumerate(k_fold.split(train, label)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        importance = pd.DataFrame({
            'column': feature_names,
            'importance': model.feature_importance(),
        }).sort_values(by='importance')

        plt.figure(figsize=(12, 6))
        lgb.plot_importance(model, max_num_features=300)
        plt.title("Featurertances_%s" % index)
        plt.show()

        print(np.sum(y_train), np.sum(y_test), np.sum(y_train) + np.sum(y_test))

        # X_test = xgb.DMatrix(X_test)
        vali_pre = model.predict(X_test)
        print(np.array(y_test))
        print(np.array(vali_pre))

        score = roc_auc_score(list(y_test), list(vali_pre))

        score_list.append(score)
        print(score)


        pred_result = model.predict(test)
        sub['target'] = pred_result

        if index == 0:
            re_sub = copy.deepcopy(sub)
        else:
            re_sub['target'] = re_sub['target'] + sub['target']

    re_sub['target'] = re_sub['target'] / splits_nums

    print('score list:', score_list)
    try:
        print(np.mean(score_list))
    except:
        pass
    return re_sub

def lgb_para_binary_model(X_train, y_train, X_test, y_test):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'max_depth': 3,
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 36,
        'learning_rate': 0.01,
        'feature_fraction': 0.4,
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
                      feature_name=feature_names)
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

train_df = train[features]

test_df = test[features]
feature_names = list(train_df.columns)
print(train_df.head())

result = get_result(train_df, test_df, label, lgb_para_binary_model, splits_nums=10)

# result[['id', 'target']].to_csv('../result/submission_lgb.csv', index=None)

test['target'] = result['target']
test['target'] = test.apply(lambda x: x['target'] if x['lmt'] <= 40 else 0, axis=1)

test[['id', 'target']].to_csv('../result/test_lgb.csv', index=None)

# # score list: [0.6680937941929648, 0.6793155292798533, 0.6932546761123287, 0.6998713113500841, 0.6710042811273635]
# # 0.6823079184125189

# score list: [0.6698666089105757, 0.682489424596096, 0.70734162377045, 0.6975421742011111, 0.6916154885072117]
# 0.6897710639970889


# score list: [0.6033886213661493, 0.7229397074562969, 0.7080678864481932, 0.6933795423270985, 0.7148106620457673, 0.7166148514346873, 0.6721064165944652, 0.6793945262728709, 0.7229142245553235, 0.679343560470924]
# 0.6912959998971775








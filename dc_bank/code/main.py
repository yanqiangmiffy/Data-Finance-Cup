# -*- coding:utf-8 _*-
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import copy
import lightgbm as lgb
from sklearn.linear_model import *
from sklearn.svm import SVR, SVC



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

scaler = StandardScaler()

def get_fea(train, test):
    test['target'] = 'test'

    df = pd.concat((train, test))

    # for i in ['certId']:
    # # for i in ['loanProduct', 'bankCard', 'x_45', 'ethnic', 'certId', 'x_33', 'residentAddr', 'x_68']:
    #     for j in ['lmt']:
    #         if i != j:
    #             df['%s_mean_%s' % (i, j)] = df.groupby('%s' % i)['%s' % j].transform('mean')
    #             df['%s_std_%s' % (i, j)] = df.groupby('%s' % i)['%s' % j].transform('std')

    # df['certPeriod'] = df['certValidStop'] - df['certValidBegin']

    # n = 0
    # fea = ['lmt', 'bankCard', 'loanProduct', 'certValidStop', 'certValidBegin', 'certId', 'setupHour', 'age',
    #        'residentAddr', 'job', 'x_45', 'basicLevel', 'dist', 'linkRela', 'weekday', 'x_33', 'x_68', 'x_46', 'ethnic', 'edu']
    # for i in fea:
    #     n = n+1
    #
    #     n2 = n
    #     for j in fea[n:]:
    #         n2 = n2+1
    #         for k in fea[n2:]:
    #             print(i, j, k)
    #             df['%s--%s--%s' % (i, j, k)] = df.apply(lambda x: str(x[i]) + str(x[j]) + str(x[k]), axis=1)
    #             lb = LabelEncoder()
    #             df['%s--%s--%s' % (i, j, k)] = lb.fit_transform(df['%s--%s--%s' % (i, j, k)])

    # for i in fea:
    #     n = n+1
    #     if n == len(fea):
    #         break
    #     for j in fea[n:]:
    #         if i != j:
    #             print(i, j)
    #             df['%s--%s'%(i, j)] = df.apply(lambda x: str(x[i])+str(x[j]), axis=1)
    #             lb = LabelEncoder()
    #             df['%s--%s' % (i, j)] = lb.fit_transform(df['%s--%s'%(i, j)])

    # n = 0
    # fea = ['lmt', 'bankCard', 'loanProduct', 'certValidStop', 'certValidBegin', 'certId', 'setupHour', 'age',
    #        'residentAddr', 'job', 'x_45', 'basicLevel', 'dist', 'linkRela', 'weekday', 'x_33', 'x_68', 'x_46', 'ethnic', 'edu']
    # for i in fea:
    #     n = n+1
    #     if n == len(fea):
    #         break
    #     for j in fea[n:]:
    #         if i != j:
    #             print(i, j)
    #             df['%s--%s'%(i, j)] = df.apply(lambda x: str(x[i])+str(x[j]), axis=1)
    #             lb = LabelEncoder()
    #             df['%s--%s' % (i, j)] = lb.fit_transform(df['%s--%s'%(i, j)])


    # , ['x_68', 'x_46'] , ['loanProduct', 'x_45'] ,['x_45', 'dist'], ['job', 'dist'], ['x_45', 'weekday']
    # , ['loanProduct', 'age'], ['x_45', 'basicLevel'], ['job', 'weekday'], ['basicLevel', 'dist'],
    #                ['linkRela', 'weekday'], ['loanProduct', 'setupHour']


    # for fea in[['bankCard', 'age'], ['job', 'weekday']]:
    #     print(fea)
    #     i = fea[0]
    #     j = fea[1]
    #     df['%s--%s' % (i, j)] = df.apply(lambda x: str(x[i]) + str(x[j]), axis=1)
    #     lb = LabelEncoder()
    #     df['%s--%s' % (i, j)] = lb.fit_transform(df['%s--%s' % (i, j)])


    no_features = ['id', 'target', 'isNew']

    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']

    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features

def get_result(train, test, label, my_model, splits_nums=5, random_state=1314):
    oof = np.zeros(train.shape[0])

    print(str(my_model))

    proba = False

    for mo in ['logistic_re', 'bayes_gauss_model_re', 'svc_model', 'forest', 'proba']:
        if mo in str(my_model):
            proba = True

    # scaler.fit(train)
    # train = scaler.transform(train)
    # test = scaler.transform(test)
    try:
        train.fillna(99, inplace=True)
        test.fillna(99, inplace=True)
        train = train.values
        test = test.values
    except:
        pass



    if 'xgb_para_binary_model' in str(my_model):
        test = xgb.DMatrix(test)

    label = label.values.astype(int)

    score_list = []
    pred_cv = []
    important = []

    k_fold = StratifiedKFold(n_splits=splits_nums, shuffle=True, random_state=random_state)
    for index, (train_index, test_index) in enumerate(k_fold.split(train, label)):
        X_train, X_test = train[train_index], train[test_index]
        y_train, y_test = label[train_index], label[test_index]
        model = my_model(X_train, y_train, X_test, y_test)

        if proba:
            vali_pre = model.predict_proba(X_test)
            vali_pre = vali_pre[:, 1]

            print(np.array(y_test))
            print(np.array(vali_pre))

            pred_result = model.predict_proba(test)
            pred_result = pred_result[:, 1]
        else:
            if 'xgb_para_binary_model' in str(my_model):
                X_test = xgb.DMatrix(X_test)
            vali_pre = model.predict(X_test)

            print(np.array(y_test))
            print(np.array(vali_pre))
            pred_result = model.predict(test)

        oof[test_index] = vali_pre

        score = roc_auc_score(list(y_test), list(vali_pre))
        score_list.append(score)
        print(score)

        pred_cv.append(pred_result)
        res = np.array(pred_cv)
        r = res.mean(axis=0)

    print(score_list)
    print(np.mean(score_list))

    return r, oof

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

def liner_re(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def ridge_re(X_train, y_train, X_test, y_test):
    model = Ridge(alpha=0.2)
    model.fit(X_train, y_train)
    return model


def bayes_re(X_train, y_train, X_test, y_test):
    model = BayesianRidge()
    model.fit(X_train, y_train)
    return model

def svm_model_re(X_train, y_train, X_test, y_test):
    model = SVR(kernel='rbf', degree=3, coef0=0.0, tol=0.01,
               C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
               verbose=False, max_iter=-1).fit(X_train,y_train)
    return model

def xgb_model_re(X_train, y_train, X_test, y_test):

    model = xgb.XGBRegressor(colsample_bytree=0.7,
                             eval_metric='rmse',
                             gamma=0.0,
                             learning_rate=0.01,
                             max_depth=4,
                             min_child_weight=1.5,
                             n_estimators=1000000,
                             reg_alpha=1,
                             reg_lambda=0.6,
                             subsample=0.2,
                             seed=42,
                             silent=1)

    model = model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],early_stopping_rounds=100, verbose=50)
    return model

def lgb_model_re(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(n_estimators=9000, max_depth=5, random_state=5, n_jobs=4, metric='mse', learning_rate=0.05)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=100, verbose=50)
    return model

def xgbt1_proba(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(n_estimators=150, max_depth=3, max_features='sqrt', random_state=321, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def xgbt2_proba(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, max_features='sqrt', random_state=789, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def xgbt3_proba(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, max_features='sqrt', random_state=456, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def gbdt1(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, max_features='log2', random_state=123,learning_rate=0.08)
    model.fit(X_train, y_train)
    return model

def gbdt2(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, max_features='log2', random_state=456,learning_rate=0.08)
    model.fit(X_train, y_train)
    return model

def gbdt3(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=300, max_depth=5, max_features='log2', random_state=789,learning_rate=0.08)
    model.fit(X_train, y_train)
    return model

def forest1(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=7, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def forest2(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=200, max_features='log2', random_state=9, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def forest3(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=11, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def lgb1_proba(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=5, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def lgb2_proba(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(n_estimators=200, max_depth=4, random_state=7, n_jobs=8)
    model.fit(X_train, y_train)
    return model

def lgb3_proba(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(n_estimators=300, max_depth=4, random_state=9, n_jobs=8)
    model.fit(X_train, y_train)
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
    model = xgb.train(params, xgb_train, num_boost_round=5000, evals=watchlist, early_stopping_rounds=100, verbose_eval=5)
    return model

def rf_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=90, max_depth=4,random_state=10)
    model.fit(X_train, y_train)
    return model

def gbdt_model(X_train, y_train, X_test, y_test):
    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, max_features='log2', random_state=2,
                                      learning_rate=0.08)
    model.fit(X_train, y_train)
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

pred_train = []
pred_test = []
# for index, my_model in enumerate([lgb_para_binary_model, liner_re, ridge_re, bayes_re, xgb_model_re, lgb_model_re,
#                                   xgb_para_binary_model, rf_model, gbdt_model]):
# for index, my_model in enumerate([lgb_para_binary_model, xgb_model_re, lgb_model_re, xgb_para_binary_model, rf_model, gbdt_model]):
for index, my_model in enumerate([lgb_para_binary_model, xgb_model_re, lgb_model_re, xgb_para_binary_model, xgbt1_proba,
                                  xgbt2_proba, xgbt3_proba, gbdt1, gbdt2, gbdt3, forest1, forest2, forest3, lgb1_proba,
                                  lgb2_proba, lgb3_proba]):
# for index, my_model in enumerate([liner_re, ridge_re, bayes_re]):
    r, oof = get_result(train_df, test_df, label, my_model, splits_nums=5, random_state=index)
    pred_train.append(oof)
    pred_test.append(r)

def bayes_re(X_train, y_train, X_test, y_test):
    model = BayesianRidge()
    model.fit(X_train, y_train)
    return model

pred_train = np.vstack(np.array(pred_train)).transpose()
pred_test = np.vstack(np.array(pred_test)).transpose()

print('vstack ok -----------------------------------------------')
np.save("../data/pred_test_boost.npy", np.array(pred_test))
print('pred_test ok --------------------------------------------')
np.save("../data/pred_train_boost.npy", np.array(pred_train))
print('pred_train ok -------------------------------------------')
r, oof = get_result(pred_train, pred_test, label, bayes_re, splits_nums=5)

test['target'] = r
test['target'] = test['target'].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)

test[['id', 'target']].to_csv('../result/submission_stack_boost.csv', index=None)

# r, oof = get_result(train_df, test_df, label, lgb1_proba, splits_nums=5)

#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 18_lgb_bayesopt.py 
@time: 2019-11-09 23:16
@description:https://www.kaggle.com/qwe1398775315/eda-lgbm-bayesianoptimization/comments#397789
"""
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import json
import matplotlib.pyplot as plt
import lightgbm as lgb
import datetime
import seaborn as sns
from gen_feas import load_data
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

train, test, no_features, features = load_data()

X_train=train[features].values
y_train=train['target']
target=y_train
X_test=test[features].values

def lgb_eval(num_leaves, max_depth, lambda_l2, lambda_l1, min_child_samples, bagging_fraction, feature_fraction):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        "metric": "auc",
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "num_threads": 4,
        "min_child_samples": int(min_child_samples),
        "learning_rate": 0.03,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "subsample_freq": 5,
        "bagging_seed": 42,
        "verbosity": -1
    }


    lgtrain = lgb.Dataset(X_train, label=y_train,)
    cv_result = lgb.cv(params,
                       lgtrain,
                       10000,
                       early_stopping_rounds=100,
                       stratified=False,
                       nfold=5)
    return cv_result['auc-mean'][-1]

def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (16, 50),
                                                'max_depth': (3, 15),
                                                'lambda_l2': (0.0, 0.05),
                                                'lambda_l1': (0.0, 0.05),
                                                'bagging_fraction': (0.5, 0.8),
                                                'feature_fraction': (0.5, 0.8),
                                                'min_child_samples': (20, 50),
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO


result = param_tuning(15,20)
print(result.max['params'])


def lgb_train(num_leaves, max_depth, lambda_l2, lambda_l1, min_child_samples, bagging_fraction, feature_fraction):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        "metric": "auc",
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "lambda_l2": lambda_l2,
        "lambda_l1": lambda_l1,
        "num_threads": 4,
        "min_child_samples": int(min_child_samples),
        "learning_rate": 0.01,
        "bagging_fraction": bagging_fraction,
        "feature_fraction": feature_fraction,
        "subsample_freq": 5,
        "bagging_seed": 42,
        "verbosity": -1
    }
    t_x, v_x, t_y, v_y = train_test_split(X_train, y_train, test_size=0.2)
    lgtrain = lgb.Dataset(t_x, label=t_y)
    lgvalid = lgb.Dataset(v_x, label=v_y)
    model = lgb.train(params, lgtrain, 2000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
    pred_test_y = model.predict(X_test, num_iteration=model.best_iteration)
    return pred_test_y, model

prediction1,model1 = lgb_train(**result.max['params'])
prediction2,model2 = lgb_train(**result.max['params'])
prediction3,model3 = lgb_train(**result.max['params'])

from pandas import DataFrame
result = DataFrame()
result['id'] = test['id']
result['target'] = (prediction1+prediction2+prediction3)/3
result.to_csv('result/bayes_opt.csv', index=False, sep=",",float_format='%.8f')


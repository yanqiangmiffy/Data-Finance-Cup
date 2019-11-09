#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: 18_lgb_bayesopt.py 
@time: 2019-11-09 23:16
@description:https://www.kaggle.com/qwe1398775315/eda-lgbm-bayesianoptimization/comments#397789
"""
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
        "objective": "regression",
        "metric": "rmse",
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
    return -cv_result['rmse-mean'][-1]

def param_tuning(init_points,num_iter,**args):
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 50),
                                                'max_depth': (5, 15),
                                                'lambda_l2': (0.0, 0.05),
                                                'lambda_l1': (0.0, 0.05),
                                                'bagging_fraction': (0.5, 0.8),
                                                'feature_fraction': (0.5, 0.8),
                                                'min_child_samples': (20, 50),
                                                })

    lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
    return lgbBO


result = param_tuning(5,15)

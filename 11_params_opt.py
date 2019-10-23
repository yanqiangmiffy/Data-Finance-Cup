#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2019/10/22 10:35
@Author:  yanqiang
@File: 11_params_opt.py
参数优化
"""

# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib

# Suppressing warnings because of skopt verbosity
import warnings

warnings.filterwarnings("ignore")

# Our example dataset
from sklearn.datasets import load_boston

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Hyperparameters distributions
from scipy.stats import randint
from scipy.stats import uniform

# Model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Metrics
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt import gp_minimize  # Bayesian optimization using Gaussian Processes
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args  # decorator to convert a list of parameters to named arguments
from skopt.callbacks import DeadlineStopper  # Stop the optimization before running out of a fixed budget of time.
from skopt.callbacks import VerboseCallback  # Callback to control the verbosity
# Stop the optimization If the last two positions at which the objective has been evaluated are less than delta
from skopt.callbacks import DeltaXStopper
from gen_feas import load_data

# Uploading the Boston dataset
# X, y = load_boston(return_X_y=True)

train, test, no_features, features = load_data()
X = train[features].values
y = train['target'].astype('int32')
# Transforming the problem into a classification (unbalanced)
y_bin = (y > np.percentile(y, 90)).astype(int)
clf = lgb.LGBMClassifier(boosting_type='gbdt',
                         class_weight='balanced',
                         objective='binary',
                         n_jobs=1,
                         verbose=0)
search_spaces = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'num_leaves': Integer(2, 500),
    'max_depth': Integer(0, 500),
    'min_child_samples': Integer(0, 200),
    'max_bin': Integer(100, 100000),
    'subsample': Real(0.01, 1.0, 'uniform'),
    'subsample_freq': Integer(0, 10),
    'colsample_bytree': Real(0.01, 1.0, 'uniform'),
    'min_child_weight': Integer(0, 10),
    'subsample_for_bin': Integer(100000, 500000),
    'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
    'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
    'scale_pos_weight': Real(1e-6, 500, 'log-uniform'),
    'n_estimators': Integer(10, 10000)
}


# Reporting util for different optimizers
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers

    optimizer = a sklearn or a skopt optimizer
    X = the training set
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    best_score = optimizer.best_score_
    best_score_std = optimizer.cv_results_['std_test_score'][optimizer.best_index_]
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1" + " %.3f") % (time() - start,
                                     len(optimizer.cv_results_['params']),
                                     best_score,
                                     best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params


# Converting average precision score into a scorer suitable for model selection
avg_prec = make_scorer(average_precision_score, greater_is_better=True, needs_proba=True)
# Setting a 5-fold stratified cross-validation (note: shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=avg_prec,
                    cv=skf,
                    n_iter=40,
                    n_jobs=-1,
                    refit=True,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=22,
                    return_train_score=False,
                    )

best_params = report_perf(opt, X, y_bin, 'LightGBM',
                          callbacks=[DeltaXStopper(0.001),
                                     DeadlineStopper(60 * 5)])

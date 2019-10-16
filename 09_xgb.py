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
from gen_feas import load_data

train, test, no_features, features = load_data()
X = train[features].values
y = train['target'].astype('int32')
test_data = test[features].values
print(X.shape)

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
    params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'eta': 0.02, 'max_depth': 4, 'min_child_weight': 6,
              'colsample_bytree': 0.7, 'subsample': 0.7, 'silent': 1, 'eval_metric': ['auc']}
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

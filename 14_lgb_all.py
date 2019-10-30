# -*- coding:utf-8 _*-
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import time
from gen_feas import load_data
import matplotlib.pyplot as plt

train, test, no_features, features = load_data()
X = train[features].values
y = train['target'].astype('int32')
test_data = test[features].values
print(X.shape)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
test_data = scaler.transform(test_data)

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

trn_data = lgb.Dataset(X, y)
val_data = lgb.Dataset(X, y)
num_round = 550
model = lgb.train(params,
                  trn_data,
                  num_round,
                  valid_sets=[trn_data, val_data],
                  verbose_eval=10,
                  early_stopping_rounds=100,
                  feature_name=features)

r = model.predict(test_data, num_iteration=model.best_iteration)
lgb.plot_importance(model, max_num_features=20)
plt.show()
test['target'] = r

test[['id', 'target']].to_csv('result/lgb_all.csv', index=None)




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


def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True


train, test, no_features, features = load_data()

X = train[features].values
y = train['target'].astype('int32')
test_data = test[features].values
print(X.shape)
# %%

# 训练
# 采取分层采样
from sklearn.model_selection import StratifiedKFold, RepeatedKFold
from sklearn.metrics import roc_auc_score

print("start：********************************")
start = time.time()

N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=2018)

gini_cv = []
auc_cv = []
pred_cv = []
for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    X_train, X_valid, y_train, y_valid = X[train_in], X[test_in], \
                                         y[train_in], y[test_in]

    # 数据结构
    lgb_train = lgb.Dataset(X_train, y_train,params={'verbose': -1})
    lgb_eval = lgb.Dataset(X_valid, y_valid, params={'verbose': -1},reference=lgb_train)

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
        'verbose': -1
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
                    valid_sets=(lgb_train, lgb_eval),
                    early_stopping_rounds=100,
                    verbose_eval=100,
                    feval=evalerror,
                    feature_name=features,
                    )

    print('................Start predict .........................')
    # 预测
    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    # 评估
    tmp_auc = roc_auc_score(y_valid, y_pred)
    tmp_gini = Gini(y_valid, y_pred)
    auc_cv.append(tmp_auc)
    gini_cv.append(tmp_gini)
    print("valid auc:", tmp_auc)
    print("valid gini:", tmp_gini)
    # test
    pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    pred_cv.append(pred)

lgb.plot_importance(gbm, max_num_features=20)
plt.show()

# 特征选择
df = pd.DataFrame(train[features].columns.tolist(), columns=['feature'])
df['importance'] = list(gbm.feature_importance())  # 特征分数
df = df.sort_values(by='importance', ascending=False)
print(list(df['feature'].values)[:50])
# 特征排序
df.to_excel("tmp/feature_score.xlsx", index=None)  # 保存分数

# K交叉验证的平均分数
print('the cv information:')
print(auc_cv)
print('cv auc mean score', np.mean(auc_cv))
print('cv gini mean score', np.mean(gini_cv))

end = time.time()
print("......................run with time: ", (end - start) / 60.0)
print("over:*********************************")

# 10.5折交叉验证结果均值融合，保存文件
mean_auc = np.mean(auc_cv)
print("mean auc:", mean_auc)
filepath = 'result/' + str(mean_auc) + '.csv'  # 线下平均分数

# 转为array
res = np.array(pred_cv)
print("总的结果：", res.shape)
# 最后结果平均，mean
# r = res.mean(axis=0)
r = np.median(res, axis=0)
print('result shape:', r.shape)
result = DataFrame()
result['id'] = test['id']
result['target'] = r
result.to_csv(filepath, index=False, sep=",")
result.to_csv('result/result.csv', index=False, sep=",")

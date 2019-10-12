from scipy.sparse.construct import hstack
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing.data import OneHotEncoder
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')
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
no_feas = ['id', 'target'] + ['certId', 'bankCard', 'dist', 'residentAddr', 'certValidStop', 'certValidBegin']
data['certPeriod'] = data['certValidStop'] - data['certValidBegin']
numerical_features = ['certValidStop', 'certValidBegin', 'lmt', 'age', 'certPeriod']
categorical_features = [fea for fea in data.columns if fea not in numerical_features + no_feas]
data = pd.get_dummies(data, columns=categorical_features)
print(data.shape)

features = [fea for fea in data.columns if fea not in no_feas]
print(features)
train = data.loc[data['target'] != -1, :]  # train set
test = data.loc[data['target'] == -1, :]  # test set
y = train['target'].values.astype(int)
X = train[features].values
test_data = test[features].values


def gbdt_lr_train():
    cv_lr_scores = []
    cv_lr_trans_scores = []
    cv_lr_trans_raw_scores = []
    cv_gbdt_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    for train_index, valid_index in skf.split(X, y):
        X_train = X[train_index]
        X_valid = X[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        # 定义GBDT模型
        gbdt = GradientBoostingClassifier(n_estimators=60, max_depth=3, verbose=0, max_features=0.5)
        # 训练学习
        gbdt.fit(X_train, y_train)
        y_pred_gbdt = gbdt.predict_proba(X_valid)[:, 1]
        gbdt_auc = roc_auc_score(y_valid, y_pred_gbdt)
        print('基于原有特征的gbdt auc: %.5f' % gbdt_auc)
        cv_gbdt_scores.append(gbdt_auc)

        # lr对原始特征样本模型训练
        lr = LogisticRegression()
        lr.fit(X_train, y_train)  # 预测及AUC评测
        y_pred_test = lr.predict_proba(X_valid)[:, 1]
        lr_valid_auc = roc_auc_score(y_valid, y_pred_test)
        print('基于原有特征的LR AUC: %.5f' % lr_valid_auc)
        cv_lr_scores.append(lr_valid_auc)

        # GBDT编码原有特征
        X_train_leaves = gbdt.apply(X_train)[:, :, 0]
        X_valid_leaves = gbdt.apply(X_valid)[:, :, 0]

        # 对所有特征进行ont-hot编码
        (train_rows, cols) = X_train_leaves.shape

        gbdtenc = OneHotEncoder()
        X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_valid_leaves), axis=0))

        # 定义LR模型
        lr = LogisticRegression()
        # lr对gbdt特征编码后的样本模型训练
        lr.fit(X_trans[:train_rows, :], y_train)
        # 预测及AUC评测
        y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
        gbdt_lr_auc1 = roc_auc_score(y_valid, y_pred_gbdtlr1)
        print('基于GBDT特征编码后的LR AUC: %.5f' % gbdt_lr_auc1)
        cv_lr_trans_scores.append(gbdt_lr_auc1)

        # 定义LR模型
        lr = LogisticRegression(n_jobs=-1)
        # 组合特征
        X_train_ext = hstack([X_trans[:train_rows, :], X_train])
        X_valid_ext = hstack([X_trans[train_rows:, :], X_valid])

        print(X_train_ext.shape)
        # lr对组合特征的样本模型训练
        lr.fit(X_train_ext, y_train)

        # 预测及AUC评测
        y_pred_gbdtlr2 = lr.predict_proba(X_valid_ext)[:, 1]
        gbdt_lr_auc2 = roc_auc_score(y_valid, y_pred_gbdtlr2)
        print('基于组合特征的LR AUC: %.5f' % gbdt_lr_auc2)
        cv_lr_trans_raw_scores.append(gbdt_lr_auc2)

    cv_lr = np.mean(cv_lr_scores)
    cv_lr_trans = np.mean(cv_lr_trans_scores)
    cv_lr_trans_raw = np.mean(cv_lr_trans_raw_scores)
    cv_gbdt = np.mean(cv_gbdt_scores)
    print("==" * 20)
    print("gbdt原始特征cv_gbdt：", cv_gbdt)
    print("lr原始特征cv_lr：", cv_lr)
    print("lr基于gbdt的特征cv_lr_trans：", cv_lr_trans)
    print("lr基于gbdt特征个原始特征cv_lr_trans_raw：", cv_lr_trans_raw)


# gbdt_lr_train()


def xgb_lr_train():
    cv_lr_scores = []
    cv_lr_trans_scores = []
    cv_lr_trans_raw_scores = []
    cv_xgb_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    for train_index, valid_index in skf.split(X, y):
        X_train = X[train_index]
        X_valid = X[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        # 定义xgb模型
        xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
                                    n_estimators=100, max_depth=4,
                                    gamma=0, subsample=0.7, colsample_bytree=0.7,
                                    verbosity=1)
        # 训练学习
        xgboost.fit(X_train, y_train)
        y_pred_valid = xgboost.predict_proba(X_valid)[:, 1]
        xgb_valid_auc = roc_auc_score(y_valid, y_pred_valid)
        print('基于原有特征的xgb auc: %.5f' % xgb_valid_auc)
        cv_xgb_scores.append(xgb_valid_auc)

        # xgboost编码原有特征
        X_train_leaves = xgboost.apply(X_train)
        X_valid_leaves = xgboost.apply(X_valid)
        # 合并编码后的训练数据和测试数据
        All_leaves = np.concatenate((X_train_leaves, X_valid_leaves), axis=0)
        All_leaves = All_leaves.astype(np.int32)
        # 对所有特征进行ont-hot编码
        xgbenc = OneHotEncoder()
        X_trans = xgbenc.fit_transform(All_leaves)
        (train_rows, cols) = X_train_leaves.shape

        # 定义LR模型
        lr = LogisticRegression()
        # lr对xgboost特征编码后的样本模型训练
        lr.fit(X_trans[:train_rows, :], y_train)
        # 预测及AUC评测
        y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
        xgb_lr_auc1 = roc_auc_score(y_valid, y_pred_xgblr1)
        print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)
        cv_lr_trans_scores.append(xgb_lr_auc1)

        # 定义LR模型
        lr = LogisticRegression(n_jobs=-1)
        # 组合特征
        X_train_ext = hstack([X_trans[:train_rows, :], X_train])
        X_test_ext = hstack([X_trans[train_rows:, :], X_valid])

        # lr对组合特征的样本模型训练
        lr.fit(X_train_ext, y_train)

        # 预测及AUC评测
        y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
        xgb_lr_auc2 = roc_auc_score(y_valid, y_pred_xgblr2)
        print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)
        cv_lr_trans_raw_scores.append(xgb_lr_auc2)
    cv_lr_trans = np.mean(cv_lr_trans_scores)
    cv_lr_trans_raw = np.mean(cv_lr_trans_raw_scores)
    cv_xgb = np.mean(cv_xgb_scores)

    print("==" * 20)
    print("xgb原始特征cv_gbdt：", cv_xgb)
    print("lr基于xgb的特征cv_lr_trans：", cv_lr_trans)
    print("lr基于xgb特征个原始特征cv_lr_trans_raw：", cv_lr_trans_raw)


# xgb_lr_train()
if __name__ == '__main__':
    gbdt_lr_train()
    xgb_lr_train()

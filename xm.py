#!/usr/local/Cellar python
# _*_coding:utf-8_*_

"""
@Author: 姜小帅
@file: xm.py
@Time: 2019/11/6 10:44 下午
@Say something:  
# 良好的阶段性收获是坚持的重要动力之一
# 用心做事情，一定会有回报
"""
# !pip install xgboost --user
# !pip install tqdm --user
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")
test['target'] = -1
df = pd.concat([train, test], sort=False, axis=0)

# 删除重复列
duplicated_features = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6',
                       'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
                       'x_15', 'x_17', 'x_18', 'x_19', 'x_21',
                       'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
                       'x_59', 'x_60', 'x_77', 'x_78'] + \
                      ['x_22', 'x_40', 'x_70'] + \
                      ['x_41'] + \
                      ['x_43'] + \
                      ['x_45'] + \
                      ['x_61']

# df = df.drop(columns=duplicated_features)
###############
###############

x_feature = []
for i in range(79):
    x_feature.append('x_{}'.format(i))

no_features = ['id', 'target', 'isNew']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

###########
import time

df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
df['begin'] = df['certValidBegin'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['end'] = df['certValidStop'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
df['begin'] = df['begin'].apply(lambda x: int(x.split('-')[0]))
df['end'] = df['end'].apply(lambda x: int(x.split('-')[0]))
df['val_period'] = df['end'] - df['begin']

###########
# 0.76新加
###########
cols = []

df['certId12'] = df['certId'].apply(lambda x: int(str(x)[:2]) if x != -999 else -999)
df['certId34'] = df['certId'].apply(lambda x: int(str(x)[2:4]) if x != -999 else -999)
df['certId56'] = df['certId'].apply(lambda x: int(str(x)[4:]) if x != -999 else -999)

df['certId12_basicLevel'] = df['certId12'].astype(str) + df['basicLevel'].astype(str)
df['certId34_basicLevel'] = df['certId34'].astype(str) + df['basicLevel'].astype(str)
df['certId56_basicLevel'] = df['certId56'].astype(str) + df['basicLevel'].astype(str)

df['certId12_loanProduct'] = df['certId12'].astype(str) + df['loanProduct'].astype(str)
df['certId34_loanProduct'] = df['certId34'].astype(str) + df['loanProduct'].astype(str)
df['certId56_loanProduct'] = df['certId56'].astype(str) + df['loanProduct'].astype(str)

cols += ['certId12_basicLevel', 'certId34_basicLevel', 'certId56_basicLevel',
         'certId12_loanProduct', 'certId34_loanProduct', 'certId56_loanProduct']

df['dist56'] = df['dist'].apply(lambda x: int(str(x)[4:]) if x != -999 else -999)
df['dist56_basicLevel'] = df['dist56'].astype(str) + df['basicLevel'].astype(str)
df['dist56_loanProduct'] = df['dist56'].astype(str) + df['loanProduct'].astype(str)
cols += ['dist56_basicLevel', 'dist56_loanProduct']

df['loanProduct_lmt'] = df.groupby('loanProduct')['lmt'].transform('mean')
df['loanProduct_lmt'] = df.groupby('loanProduct')['lmt'].transform('median')


####
# df['loanProduct_bankCard'] = df['loanProduct'].astype(str) + df['bankCard'].astype(str)
# cols += ['loanProduct_bankCard']

# df['certId12_lmt'] = df.groupby('certId12')['lmt'].transform('mean')
# df['certId12_lmt'] = df.groupby('certId12')['lmt'].transform('median')

# df['certId34_lmt'] = df.groupby('certId34')['lmt'].transform('mean')
# df['certId34_lmt'] = df.groupby('certId34')['lmt'].transform('median')

# df['certId56_lmt'] = df.groupby('certId56')['lmt'].transform('mean')
# df['certId56_lmt'] = df.groupby('certId56')['lmt'].transform('median')

################
# things not work
# df['certId12_edu'] = df['certId12'].astype(str) + df['edu'].astype(str)
# df['certId34_edu'] = df['certId34'].astype(str) + df['edu'].astype(str)
# df['certId56_edu'] = df['certId56'].astype(str) + df['edu'].astype(str)
# 'certId12_edu', 'certId34_edu','certId56_edu'

# useless = ['x_59','x_22','x_23','x_24','x_30','x_31','x_32','x_35','x_36','x_37','x_38','x_39','x_40','x_42',
#  'x_57','x_58','x_60','x_69','x_70','x_77','x_78','ncloseCreditCard','unpayIndvLoan','unpayOtherLoan',
#  'unpayNormalLoan','5yearBadloan','x_21','x_19','is_edu_equal','x_9','x_7','x_8','x_4','x_10','x_11',
#  'x_1','x_6','x_13','x_3','x_18','x_15','x_17','x_2','x_5']

# df['bankCard_nan'] = df['bankCard'].apply(lambda x: 1 if x== np.nan else 0)
# df['bankCard69'] = df['bankCard'].apply(lambda x: x%100 if x != -999 else -999)
# df['bankCard06_loanProduct'] = df['bankCard'].astype(str) + df['loanProduct'].astype(str)
# df['bankCard69_loanProduct'] = df['bankCard69'].astype(str) + df['loanProduct'].astype(str)

# cols += ['bankCard1']
################

# 待尝试
# 四个一起降分
# df['certId12_job'] = df['certId12'].astype(str) + df['job'].astype(str)
# df['certId12_ethnic'] = df['certId12'].astype(str) + df['ethnic'].astype(str)
# cols += ['certId12_job', 'certId12_ethnic']

# df['certId12_edu'] = df['certId12'].astype(str) + df['edu'].astype(str)
# df['certId12_highestEdu'] = df['certId12'].astype(str) + df['highestEdu'].astype(str)
# cols += ['certId12_edu', 'certId12_highestEdu']

# df['edu_basicLevel'] = df['edu'].astype(str) + df['basicLevel'].astype(str)
# df['edu_loanProduct'] = df['edu'].astype(str) + df['loanProduct'].astype(str)
# df['edu_lmt'] = df.groupby('edu')['lmt'].transform('mean')
# df['edu_lmt'] = df.groupby('edu')['lmt'].transform('median')
# cols += ['edu_basicLevel', 'edu_loanProduct']

# df['highestEdu_basicLevel'] = df['highestEdu'].astype(str) + df['basicLevel'].astype(str)
# df['highestEdu_loanProduct'] = df['highestEdu'].astype(str) + df['loanProduct'].astype(str)
# df['highestEdu_lmt'] = df.groupby('highestEdu')['lmt'].transform('mean')
# df['highestEdu_lmt'] = df.groupby('highestEdu')['lmt'].transform('median')
# cols += ['highestEdu_basicLevel', 'highestEdu_loanProduct']

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


for col in cols:
    lab = LabelEncoder()
    df[col] = lab.fit_transform(df[col])

cols += ['certId12', 'certId34', 'certId56', 'dist56']

cols += ['bankCard', 'residentAddr', 'certId', 'dist', 'age', 'job', 'basicLevel', 'loanProduct', 'val_period']
# count
for col in cols:
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

df['is_edu_equal'] = (df['edu'] == df['highestEdu']).astype(int)

print(df.shape)

feature = [fea for fea in df.columns if fea not in no_features + ['begin', 'end']]
train, test = df[:len(train)], df[len(train):]

n_fold = 5
y_scores = 0
y_pred_l1 = np.zeros([n_fold, test.shape[0]])
y_pred_all_l1 = np.zeros(test.shape[0])
fea_importances = np.zeros(len(feature))
auc_cv=[]
gini_cv=[]
label = ['target']
kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=1314)
for i, (train_index, valid_index) in enumerate(kfold.split(train[feature], train[label])):
    print(i)
    X_train, y_train, X_valid, y_valid = train.loc[train_index][feature].values, train[label].values[train_index], \
                                         train.loc[valid_index][feature].values, train[label].values[valid_index]

    bst = xgb.XGBClassifier(max_depth=3, n_estimators=10000, learning_rate=0.01,
                            tree_method='gpu_hist'
                            )
    bst.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='auc', verbose=500,
            early_stopping_rounds=500)

    y_pred_l1[i] = bst.predict_proba(test[feature].values)[:, 1]
    y_pred_all_l1 += y_pred_l1[i]
    y_scores += bst.best_score

    y_pred = bst.predict_proba(X_valid)[:, 1]

    # 评估
    tmp_auc = roc_auc_score(y_valid, y_pred)
    print("valid auc:", tmp_auc)
    y_valid=y_valid.reshape(-1,)
    tmp_gini = Gini(y_valid, y_pred)
    print("valid gini:", tmp_gini)
    auc_cv.append(tmp_auc)
    gini_cv.append(tmp_gini)
    fea_importances += bst.feature_importances_

print("auc_cv",auc_cv)
print("gini_cv",gini_cv)
test['target'] = y_pred_all_l1 / 4
print('average score is {}'.format(y_scores / 4))

test[['id', 'target']].to_csv('xgb_seed1314.csv', index=False)

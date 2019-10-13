# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

path = '../data/'

train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')
quality_map = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
train['label'] = train['Quality_label'].map(quality_map)

# 将标签onehot编码，方便管理和统计。
# idea1：可以考虑做4个二分类，或者尝试使用mae 做loss 或者mse 做loss，并观察结果的分布以及线下得分。
train = pd.get_dummies(train, columns=['Quality_label'])
bin_label = ['Quality_label_Excellent', 'Quality_label_Good', 'Quality_label_Pass', 'Quality_label_Fail']

data = pd.concat([train, test], ignore_index=True)
data['id'] = data.index

# 因为1，属性特征目测是连续特征，2，数值大小分布差异过大，所以将属性做log变换之后做处理，会更合适一些。也可以考虑分桶处理
# 为什么做log变换更合理呢？试想一下，假如统计一个人某个数值属性，发现是如下的一个列表[1,2,1.1,1.5,20],
# 这种场景分布偏差较大的情况下，如果取均值作为特征，是否合适？
# 再比如，如果预测一组数值，一条极端数据对结果的影响超过了N多的数据，这样的模型是否是一个好的模型？

# 参数特征这里使用5-10
para_feat = ['Parameter{0}'.format(i) for i in range(5, 11)]
# 属性特征
attr_feat = ['Attribute{0}'.format(i) for i in range(1, 11)]

data[attr_feat] = np.log1p(data[attr_feat])
# or data[attr_feat] = np.log10(data[attr_feat] + 1)
# 此时绘图观察分布显得合理很多
for i in attr_feat:
    data[i].hist()
    plt.show()

# 使用预测属性的model去预测属性，
# 1、测试集没有属性，该怎么用？当然是预测它了。预测的方法，可以由模型获得
# 2、训练集的属性怎么用？既然测试集的属性是预测出来的，训练集也应该用同等性质的属性，也就是5折交叉预测出来的属性。
# 3、第一次想到该方法于2018年的光伏预测赛中，首见成效，之后教与ration，后在icme中，也用到类似方法。
# 4、该方法带来的提升目前不太稳定。6750 -- 6800。

def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)
    train_data = data[~test_index].reset_index(drop=True)
    test_data = data[test_index]

    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100
                          )
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          # categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
        elif model_type == 'ctb':
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                      eval_metric='mae',
                      cat_features=cate_feature,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=100)
        train_data.loc[val_idx, predict_label] = model.predict(test_x)
        if len(test_data) != 0:
            test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
    test_data[predict_label] = test_data[predict_label] / n_splits
    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


lgb_attr_model = lgb.LGBMRegressor(
    boosting_type="gbdt", num_leaves=31, reg_alpha=10, reg_lambda=5,
    max_depth=7, n_estimators=500,
    subsample=0.7, colsample_bytree=0.4, subsample_freq=2, min_child_samples=10,
    learning_rate=0.05, random_state=2019,
)

features = para_feat
for i in attr_feat:
    data, predict_label = get_predict_w(lgb_attr_model, data, label=i,
                                        feature=features, random_state=2019, n_splits=5)
    print(predict_label, 'done!!')

# 该方案共获得10个属性特征。
pred_attr_feat = ['predict_Attribute{0}'.format(i) for i in range(1, 11)]

lgb_mc_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=23, reg_alpha=10, reg_lambda=5,
    max_depth=5, n_estimators=300, objective='multiclass',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1, min_child_samples=5,
    learning_rate=0.05, random_state=42,
)

features = para_feat + pred_attr_feat

X = data[~data.label.isnull()][features]
y = data[~data.label.isnull()]['label']

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
y_test = 0
# best_score = []
best_iter = []
for index, (trn_idx, test_idx) in enumerate(skf.split(X, y)):
    print(index)
    train_x, test_x, train_y, test_y = X.loc[trn_idx], X.loc[test_idx], y.loc[trn_idx], y.loc[test_idx]
    eval_set = [(test_x, test_y)]
    lgb_mc_model.fit(train_x, train_y, eval_set=eval_set,
                     # early_stopping_rounds=100,
                     verbose=100
                     )

    y_pred = lgb_mc_model.predict_proba(data.loc[test_idx][features])
    for i in range(4):
        data.loc[test_idx, 'pred_{0}'.format(i)] = y_pred[:, i]

    y_test = lgb_mc_model.predict_proba(data[data.label.isnull()][features]) / 5 + y_test
    # best_score.append(lgb_mc_model.best_score_['valid_0']['multi_logloss'])
    best_iter.append(lgb_mc_model.best_iteration_)

# ***************线下验证****************
pred_label = ['pred_{0}'.format(i) for i in range(4)]

# 概率的最小值为0，最大为1，如果使用回归模型预测，有可能出现越界情况。（分类模型，无需该处理）
for i in pred_label:
    data[i] = data[i].apply(lambda x: max(min(x, 1), 0))

# 同一条数据的4种情况的概率总和应该为1。（多分类模型，无需该处理）
data['pred_sum'] = data[pred_label].sum(axis=1)
for i in pred_label:
    data[i] = data[i] / data['pred_sum']


# 思路：线下随机构造groupid，然后获得线下验证策略。
def gen_sample(data, group_values, seed=0):
    group_values = shuffle(group_values, random_state=seed)
    # Group id 由group的值，做打乱获得，这里保存seed 信息，方便复现，因为有seed的话，就可以复现指定group
    data['Group'] = seed * 1000 + group_values
    return data


group_values = test.Group.values.copy()
data_2 = data.copy()
# 随机10组验证，道理上组数越多方差越小，越稳定
for i in range(1, 10):
    data_2 = pd.concat([
        gen_sample(data[~data.label.isnull()], group_values, i),
        data_2], ignore_index=True
    )
    print(i, data_2.shape)

data_3 = data_2.groupby(['Group'], as_index=False)[bin_label + pred_label].mean()
print(
    'score:',
    1 / (1 + 10 * abs(
        data_3[data_3.Group >= 120][pred_label].values - data_3[data_3.Group >= 120][bin_label].values).mean())
)

# ***************生成提交****************

sub = data[data.label.isnull()][['Group']].astype(int)
for i in range(4):
    sub['pred_{0}'.format(i)] = y_test[:, i]
labels = ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']
sub.columns = ['Group'] + labels
sub = sub.groupby('Group')[labels].mean().reset_index()
sub[['Group'] + labels].to_csv(path + 'sub/sub.csv', index=False)

# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
# 这种玩意看着真难受，谁有法子弄掉 ，盼复

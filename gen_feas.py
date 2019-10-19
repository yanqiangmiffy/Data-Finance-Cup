import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")
df = pd.concat([train, test], sort=False, axis=0)
stats = []
for col in df.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False, inplace=True)
stats_df.to_excel('tmp/stats_df.xlsx', index=None)

# 特征工程
df.fillna(value=-999, inplace=True)  # bankCard存在空值
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
df = df.drop(columns=duplicated_features)
print(df.shape)

no_features = ['id', 'target'] + ['bankCard', 'residentAddr', 'certId', 'dist']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

# 数值特征处理
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
for feat in numerical_features + ['certValidPeriod']:
    df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化
# 类别特征处理

# 特殊处理
# bankCard 5991
# residentAddr 5288
# certId 4033
# dist 3738
cols = ['bankCard', 'residentAddr', 'certId', 'dist']
# 计数
for col in cols:
    col_nums = dict(df[col].value_counts())
    df[col + '_nums'] = df[col].apply(lambda x: col_nums[x])
# 对lmt进行mean encoding
for fea in tqdm(cols):
    grouped_df = df.groupby(fea).agg({'lmt': ['min', 'max', 'mean', 'sum', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')
df = df.drop(columns=cols)  # 删除四列

import time

combs = [
    ('ethnic', 'highestEdu'),
    ('basicLevel', 'linkRela'),
]

col_vals_dict = {c: list(train[c].unique()) for c in categorical_features}
print(col_vals_dict)
embed_cols = []
for c in col_vals_dict:
    if 6<=len(col_vals_dict[c]) <=50:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c]))  # look at value counts to know the embedding dimensions
print('\n')
print(len(embed_cols))
start = time.time()
from itertools import combinations

# for n_c, (f1, f2) in enumerate(combs):
for n_c, (f1, f2) in tqdm(enumerate(combinations(embed_cols, 2))):
    print(f1, f2)
    # f1 = rv[0]
    # f2 = rv[1]
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    df[name1] = df[f1].apply(lambda x: str(x)) + "_" + df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(df[name1].values))
    df[name1] = lbl.transform(list(df[name1].values))

    categorical_features.append(name1)

# dummies
# df = pd.get_dummies(df, columns=categorical_features)
# df.to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

for f in categorical_features:
    df[f + "_avg"], df[f + "_avg"] = target_encode(trn_series=df[f],
                                                   tst_series=df[f],
                                                   target=df['target'],
                                                   min_samples_leaf=200,
                                                   smoothing=10,
                                                   noise_level=0)
# df = pd.get_dummies(df, columns=categorical_features)
train, test = df[:len(train)], df[len(train):]

features = [fea for fea in df.columns if fea not in no_features]


def load_data():
    return train, test, no_features, features

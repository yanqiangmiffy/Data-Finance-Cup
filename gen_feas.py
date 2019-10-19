import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from utils import *
from itertools import combinations

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")
test['target'] = [0] * len(test)
df = pd.concat([train, test], sort=False, axis=0)

# =========  简单数据描述 ============
stats = []
for col in df.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False, inplace=True)
stats_df.to_excel('tmp/stats_df.xlsx', index=None)

# ========= 数据预处理 ============
df.fillna(value=-999, inplace=True)  # bankCard存在空值

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
df = df.drop(columns=duplicated_features)  # 删除重复列

# ================ 特征工程 ==========

no_features = ['id', 'target'] + ['bankCard', 'residentAddr', 'certId', 'dist']  # 需要删除的特征
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']  # 类别特征

# ------ 数值特征 ------

df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
numerical_features.append('certValidPeriod')

# 计数
cols = ['bankCard', 'residentAddr', 'certId', 'dist',
        'lmt', 'ethnic', 'age', 'setupHour', 'job',
        'linkRela', 'highestEdu', 'edu', 'x_33', 'weekday',
        'x_34', 'basicLevel']
for col in cols:
    col_nums = dict(df[col].value_counts())
    df[col + '_cnt'] = df[col].apply(lambda x: col_nums[x])
    numerical_features.append(col + '_cnt')

# 对lmt进行mean encoding
for fea in tqdm(cols):
    grouped_df = df.groupby(fea).agg(
        {'lmt': ['min', 'max', 'mean', 'sum', 'median', pd.DataFrame.skew, pd.DataFrame.kurtosis]})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]

    numerical_features.extend(list(grouped_df.columns))

    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')

scale = MinMaxScaler()
for feat in numerical_features:
    # df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化
    df[feat] = scale.fit_transform(df[feat].values.reshape(-1, 1))  # 排序，并且进行归一化

for n_c, (f1, f2) in tqdm(enumerate(combinations(cols, 2))):
    name1 = f1 + "_plus_" + f2
    df[name1 + '_add'] = df[f1] + df[f2]
    df[name1 + '_diff'] = df[f1] - df[f2]
    df[name1 + '_multiply'] = df[f1] * df[f2]
    df[name1 + '_division'] = df[f1] / (df[f2] + 1)
    numerical_features.append(name1 + '_add')
    numerical_features.append(name1 + '_diff')
    numerical_features.append(name1 + '_multiply')
    numerical_features.append(name1 + '_division')
# lmt cut
df['lmt_bin'] = pd.cut(df['lmt'], 7, labels=range(7)).astype(int)

print(df['lmt_bin'])
# ------- 类别特征 -------
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

# 组合特征
col_vals_dict = {c: list(df[c].unique()) for c in categorical_features}
embed_cols = ['lmt']
for c in col_vals_dict:
    if 6 <= len(col_vals_dict[c]) <= 50:
        embed_cols.append(c)
for n_c, (f1, f2) in tqdm(enumerate(combinations(embed_cols, 2))):
    # f1 = rv[0]
    # f2 = rv[1]
    name1 = f1 + "_plus_" + f2
    df[name1] = df[f1].apply(lambda x: str(x)) + "_" + df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(df[name1].values))
    df[name1] = lbl.transform(list(df[name1].values))
    categorical_features.append(name1)

# # dummies
# # df = pd.get_dummies(df, columns=categorical_features)

#
# for f in cols:
#     df[f + "_avg"], df[f + "_avg"] = target_encode(trn_series=df[f],
#                                                    tst_series=df[f],
#                                                    target=df['target'],
#                                                    min_samples_leaf=200,
#                                                    smoothing=10,
#                                                    noise_level=0)
# df = pd.get_dummies(df, columns=categorical_features)
train, test = df[:len(train)], df[len(train):]
df.head().to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)
features = [fea for fea in df.columns if fea not in no_features]


def load_data():
    return train, test, no_features, features

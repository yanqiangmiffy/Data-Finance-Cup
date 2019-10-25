import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

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

# 缺失值个数统计
train.fillna(value=-999, inplace=True)  # bankCard存在空值
test.fillna(value=-999, inplace=True)  # bankCard存在空值
train['missing'] = (train == -999).sum(axis=1).astype(float)
test['missing'] = (test == -999).sum(axis=1).astype(float)

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
ind_features = duplicated_features
count = 0
for c in ind_features:
    if count == 0:
        df['new_ind0'] = df[c].astype(str) + '_'
        count += 1
    else:
        df['new_ind0'] += df[c].astype(str) + '_'
for c in ['new_ind0']:
    d0 = df[c].value_counts().to_dict()
    df['%s_count' % c] = df[c].apply(lambda x: d0.get(x, 0))

df = df.drop(columns=duplicated_features)
print(df.shape)

no_features = ['id', 'target'] + ['bankCard', 'residentAddr', 'certId', 'dist', 'new_ind0', 'new_ind1', 'new_ind2',
                                  'new_ind3']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop', 'missing']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

ind_features = [c for c in categorical_features if 'x_' in c]
count = 0
for c in ind_features:
    if count == 0:
        df['new_ind1'] = df[c].astype(str) + '_'
        count += 1
    else:
        df['new_ind1'] += df[c].astype(str) + '_'
for c in ['new_ind1']:
    d = df[c].value_counts().to_dict()
    df['%s_count' % c] = df[c].apply(lambda x: d.get(x, 0))

ind_features = ['bankCard', 'residentAddr', 'certId', 'dist']
count = 0
for c in ind_features:
    if count == 0:
        df['new_ind2'] = df[c].astype(str) + '_'
        count += 1
    else:
        df['new_ind2'] += df[c].astype(str) + '_'
for c in ['new_ind2']:
    d1 = df[c].value_counts().to_dict()
    df['%s_count' % c] = df[c].apply(lambda x: d1.get(x, 0))

ind_features = ['unpayOtherLoan', 'job', 'setupHour', 'linkRela', 'basicLevel', 'ethnic']

count = 0
for c in ind_features:
    if count == 0:
        df['new_ind3'] = df[c].astype(str) + '_'
        count += 1
    else:
        df['new_ind3'] += df[c].astype(str) + '_'
for c in ['new_ind3']:
    d3 = df[c].value_counts().to_dict()
    df['%s_count' % c] = df[c].apply(lambda x: d3.get(x, 0))
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
    grouped_df = df.groupby(fea).agg({'lmt': ['min', 'max', 'mean', 'sum', 'median', 'size']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')
df = df.drop(columns=cols)  # 删除四列

# dummies
df = pd.get_dummies(df, columns=categorical_features)
df.to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]


def load_data():
    return train, test, no_features, features

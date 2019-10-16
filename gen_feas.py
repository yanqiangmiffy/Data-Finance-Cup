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

# ============ 特征工程 begin===============
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
no_features = ['id', 'target']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']
large_num_cates = ['bankCard', 'residentAddr', 'certId', 'dist']  # 类别种类很多
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features + large_num_cates]

# =============== 数值特征处理 ===========
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
for feat in numerical_features + ['certValidPeriod']:
    df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化

# =============== 类别特征处理 =============
# 对类别种类很多的类别特殊处理
# bankCard 5991
# residentAddr 5288
# certId 4033
# dist 3738
# 计数
for col in large_num_cates:
    col_nums = dict(df[col].value_counts())
    df[col + '_nums'] = df[col].apply(lambda x: col_nums[x])
# 对lmt进行mean encoding
for fea in tqdm(large_num_cates):
    for num_col in numerical_features:
        grouped_df = df.groupby(fea).agg({num_col: ['min', 'max', 'mean', 'sum', 'median']})
        grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        # print(grouped_df)
        df = pd.merge(df, grouped_df, on=fea, how='left')
df = df.drop(columns=large_num_cates)  # 删除四列

# dummies
df = pd.get_dummies(df, columns=categorical_features)
df.to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

# ============ 特征工程 end===============

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]


def load_data():
    return train, test, no_features, features

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from numpy import random
from sklearn.preprocessing import LabelEncoder


def simple_statics():
    print("生成excel数据")
    train['train'] = 'train'
    test['train'] = 'test'
    # df = pd.concat([train, test], sort=False, axis=0)
    # df.to_excel('df.xlsx', index=None)
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0],
                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_df.xlsx', index=None)

    stats = []
    for col in train.columns:
        stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                      train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_train.xlsx', index=None)

    stats = []
    for col in test.columns:
        stats.append((col, test[col].nunique(), test[col].isnull().sum() * 100 / test.shape[0],
                      test[col].value_counts(normalize=True, dropna=False).values[0] * 100, test[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_test.xlsx', index=None)


random.seed(2019)
train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")

df = pd.concat([train, test], sort=False, axis=0)
# 特征工程
df['missing'] = (df == -1).sum(axis=1).astype(float)  # 统计每行中为-999的个数
df['bankCard'] = df['bankCard'].fillna(value=999999999)  # bankCard存在空值
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
# 类别组合特征
count = 0
for c in duplicated_features:
    if count == 0:
        df['new_ind' + str(-1)] = df[c].astype(str) + '_'
        count += 1
    else:
        df['new_ind' + str(-1)] += df[c].astype(str) + '_'
for c in ['new_ind' + str(-1)]:
    d = df[c].value_counts().to_dict()
    df['%s_count' % c] = df[c].apply(lambda x: d.get(x, 0))
df.drop(columns=['new_ind' + str(-1)], inplace=True)
df = df.drop(columns=duplicated_features)
print(df.shape)
simple_statics()

no_features = ['id', 'target'] + ['bankCard', 'residentAddr', 'certId', 'dist', 'new_ind1', 'new_ind2']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop', 'missing']  # 不是严格意义的数值特征，可以当做类别特征
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

group_features1 = [c for c in categorical_features if 'x_' in c]  # 匿名
group_features2 = ['bankCard', 'residentAddr', 'certId', 'dist']  # 地区特征
group_features3 = ['lmt', 'certValidBegin', 'certValidStop']  # 征信1

group_features4 = ['age', 'job', 'ethnic', 'basicLevel', 'linkRela']  # 基本属性
group_features5 = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']

group_features = [
    group_features1, group_features2, group_features3, group_features4,
    group_features5,
]

for index, ind_features in enumerate(group_features):
    index += 1
    count = 0
    for c in ind_features:
        if count == 0:
            df['new_ind' + str(index)] = df[c].astype(str) + '_'
            count += 1
        else:
            df['new_ind' + str(index)] += df[c].astype(str) + '_'
    for c in ['new_ind' + str(index)]:
        d = df[c].value_counts().to_dict()
        df['%s_count' % c] = df[c].apply(lambda x: d.get(x, 0))
    df.drop(columns=['new_ind' + str(index)], inplace=True)


def create_group_fea(df_, groups_fea, group_name):
    count = 0
    for c in groups_fea:
        if count == 0:
            df_[group_name] = df_[c].astype(str) + '_'
            count += 1
        else:
            df_[group_name] += df_[c].astype(str) + '_'
    for c in [group_name]:
        tmp_d = df_[c].value_counts().to_dict()
        df_['%s_count' % c] = df_[c].apply(lambda x: tmp_d.get(x, 0))
    lb = LabelEncoder()
    df_[group_name] = lb.fit_transform(df_[group_name])
    # df_.drop(columns=[group_name], inplace=True)
    return df_


# certId
df['certId_first2'] = df['certId'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['certId_middle2'] = df['certId'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['certId_last2'] = df['certId'].apply(lambda x: int(str(x)[4:6]))  # 最后两位

# certId 贷款
certId_first2_loanProduct = ['certId_first2', 'loanProduct']
df = create_group_fea(df, certId_first2_loanProduct, 'certId_first2_loanProduct')
certId_middle2_loanProduct = ['certId_middle2', 'loanProduct']
df = create_group_fea(df, certId_middle2_loanProduct, 'certId_middle2_loanProduct')
certId_last2_loanProduct = ['certId_last2', 'loanProduct']
df = create_group_fea(df, certId_last2_loanProduct, 'certId_last2_loanProduct')

df['certValidBegin_bin'] = pd.qcut(df['certValidBegin'], 20, labels=[i for i in range(20)])
certId_first2_cvb = ['certId_first2', 'certValidBegin_bin']
df = create_group_fea(df, certId_first2_cvb, 'certId_first2_cvb')
certId_middle2_cvb = ['certId_middle2', 'certValidBegin_bin']
df = create_group_fea(df, certId_middle2_cvb, 'certId_middle2_cvb')
certId_last2_cvb = ['certId_last2', 'certValidBegin_bin']
df = create_group_fea(df, certId_last2_cvb, 'certId_last2_cvb')

certId_first2_basicLevel = ['certId_first2', 'basicLevel']
df = create_group_fea(df, certId_first2_basicLevel, 'certId_first2_basicLevel')
certId_middle2_basicLevel = ['certId_middle2', 'basicLevel']
df = create_group_fea(df, certId_middle2_basicLevel, 'certId_middle2_basicLevel')
certId_last2_basicLevel = ['certId_last2', 'basicLevel']
df = create_group_fea(df, certId_last2_basicLevel, 'certId_last2_basicLevel')

certId_first2_edu = ['certId_first2', 'edu']
df = create_group_fea(df, certId_first2_edu, 'certId_first2_edu')
certId_middle2_edu = ['certId_middle2', 'edu']
df = create_group_fea(df, certId_middle2_edu, 'certId_middle2_edu')
certId_last2_edu = ['certId_last2', 'edu']
df = create_group_fea(df, certId_last2_edu, 'certId_last2_edu')

# certId_first2_job = ['certId_first2', 'job']
# df = create_group_fea(df, certId_first2_job, 'certId_first2_job')
# certId_middle2_job = ['certId_middle2', 'job']
# df = create_group_fea(df, certId_middle2_job, 'certId_middle2_job')
# certId_last2_job = ['certId_last2', 'job']
# df = create_group_fea(df, certId_last2_job, 'certId_last2_job')


# dist
df['dist_first2'] = df['dist'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['dist_middle2'] = df['dist'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['dist_last2'] = df['dist'].apply(lambda x: int(str(x)[4:6]))  # 最后两位

dist_first2_loanProduct = ['dist_first2', 'loanProduct']
df = create_group_fea(df, dist_first2_loanProduct, 'dist_first2_loanProduct')
dist_middle2_loanProduct = ['dist_middle2', 'loanProduct']
df = create_group_fea(df, dist_middle2_loanProduct, 'dist_middle2_loanProduct')
dist_last2_loanProduct = ['dist_last2', 'loanProduct']
df = create_group_fea(df, dist_last2_loanProduct, 'dist_last2_loanProduct')

dist_first2_basicLevel = ['dist_first2', 'basicLevel']
df = create_group_fea(df, dist_first2_basicLevel, 'dist_first2_basicLevel')
dist_middle2_basicLevel = ['dist_middle2', 'basicLevel']
df = create_group_fea(df, dist_middle2_basicLevel, 'dist_middle2_basicLevel')
dist_last2_basicLevel = ['dist_last2', 'basicLevel']
df = create_group_fea(df, dist_last2_basicLevel, 'dist_last2_basicLevel')

dist_first2_edu = ['dist_first2', 'edu']
df = create_group_fea(df, dist_first2_edu, 'dist_first2_edu')
dist_middle2_edu = ['dist_middle2', 'edu']
df = create_group_fea(df, dist_middle2_edu, 'dist_middle2_edu')
dist_last2_edu = ['dist_last2', 'edu']
df = create_group_fea(df, dist_last2_edu, 'dist_last2_edu')

# residentAddr
df['residentAddr_first2'] = df['residentAddr'].apply(lambda x: int(str(x)[:2]) if x != -999 else -999)  # 前两位
df['residentAddr_middle2'] = df['residentAddr'].apply(lambda x: int(str(x)[2:4]) if x != -999 else -999)  # 中间两位
df['residentAddr_last2'] = df['residentAddr'].apply(lambda x: int(str(x)[4:6]) if x != -999 else -999)  # 最后两位

residentAddr_first2_loanProduct = ['residentAddr_first2', 'loanProduct']
df = create_group_fea(df, residentAddr_first2_loanProduct, 'residentAddr_first2_loanProduct')
residentAddr_middle2_loanProduct = ['residentAddr_middle2', 'loanProduct']
df = create_group_fea(df, residentAddr_middle2_loanProduct, 'residentAddr_middle2_loanProduct')
residentAddr_last2_loanProduct = ['residentAddr_last2', 'loanProduct']
df = create_group_fea(df, residentAddr_last2_loanProduct, 'residentAddr_last2_loanProduct')

residentAddr_first2_basicLevel = ['residentAddr_first2', 'basicLevel']
df = create_group_fea(df, residentAddr_first2_basicLevel, 'residentAddr_first2_basicLevel')
residentAddr_middle2_basicLevel = ['residentAddr_middle2', 'basicLevel']
df = create_group_fea(df, residentAddr_middle2_basicLevel, 'residentAddr_middle2_basicLevel')
residentAddr_last2_basicLevel = ['residentAddr_last2', 'basicLevel']
df = create_group_fea(df, residentAddr_last2_basicLevel, 'residentAddr_last2_basicLevel')

residentAddr_first2_edu = ['residentAddr_first2', 'edu']
df = create_group_fea(df, residentAddr_first2_edu, 'residentAddr_first2_edu')
residentAddr_middle2_edu = ['residentAddr_middle2', 'edu']
df = create_group_fea(df, residentAddr_middle2_edu, 'residentAddr_middle2_edu')
residentAddr_last2_edu = ['residentAddr_last2', 'edu']
df = create_group_fea(df, residentAddr_last2_edu, 'residentAddr_last2_edu')

# bankCard
df['bankCard'] = df['bankCard'].astype(int)
df['bankCard_first6'] = df['bankCard'].apply(lambda x: int(str(x)[:6]) if x != -999 else -999)
df['bankCard_last3'] = df['bankCard'].apply(lambda x: int(str(x)[6:].strip()) if x != -999 else -999)

bankCard_first6_loanProduct = ['bankCard_first6', 'loanProduct']
df = create_group_fea(df, bankCard_first6_loanProduct, 'bankCard_first6_loanProduct')
bankCard_last3_loanProduct = ['bankCard_last3', 'loanProduct']
df = create_group_fea(df, bankCard_last3_loanProduct, 'bankCard_last3_loanProduct')

bankCard_first6_basicLevel = ['bankCard_first6', 'basicLevel']
df = create_group_fea(df, bankCard_first6_basicLevel, 'bankCard_first6_basicLevel')
bankCard_last3_basicLevel = ['bankCard_last3', 'basicLevel']
df = create_group_fea(df, bankCard_last3_basicLevel, 'bankCard_last3_basicLevel')

bankCard_first6_edu = ['bankCard_first6', 'edu']
df = create_group_fea(df, bankCard_first6_edu, 'bankCard_first6_edu')
bankCard_last3_edu = ['bankCard_last3', 'edu']
df = create_group_fea(df, bankCard_last3_edu, 'bankCard_last3_edu')

# 数值特征处理
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
# 类别特征处理

# 特殊处理
# bankCard 5991
# residentAddr 5288
# certId 4033
# dist 3738
cols = ['bankCard', 'residentAddr', 'certId', 'dist']
# 计数
for col in cols:
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

# 对lmt进行mean encoding
for fea in tqdm(cols):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')
df = df.drop(columns=cols)  # 删除四列

for fea in tqdm(['certId_first2', 'certId_middle2', 'certId_last2']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['certId_first2_loanProduct', 'certId_middle2_loanProduct', 'certId_last2_loanProduct',
                 'certId_first2_basicLevel', 'certId_middle2_basicLevel', 'certId_last2_basicLevel',
                 'certId_first2_edu', 'certId_middle2_edu', 'certId_last2_edu']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['dist_first2', 'dist_middle2', 'dist_last2']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm([
    'dist_first2_loanProduct', 'dist_middle2_loanProduct', 'dist_last2_loanProduct',
    'dist_first2_basicLevel', 'dist_middle2_basicLevel', 'dist_last2_basicLevel',
    'dist_first2_edu', 'dist_middle2_edu', 'dist_last2_edu']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['residentAddr_first2', 'residentAddr_middle2', 'residentAddr_last2']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['bankCard_first6', 'bankCard_last3']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

# dummies
df = pd.get_dummies(df, columns=categorical_features)
df.head().to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]


def load_data():
    return train, test, no_features, features

import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from numpy import random

random.seed(1314)
train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")

df = pd.concat([train, test], sort=False, axis=0)


def simple_statics():
    print("生成excel数据")
    train['train'] = 'train'
    test['train'] = 'test'
    df = pd.concat([train, test], sort=False, axis=0)
    df.to_excel('df.xlsx', index=None)
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0],
                      df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                            'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('Unique_values', ascending=False, inplace=True)
    stats_df.to_excel('tmp/stats_df.xlsx', index=None)


# simple_statics()

# ========================== 数据预处理 =========================
df.fillna(value=-999, inplace=True)  # bankCard存在空值
df['missing'] = (df == -999).sum(axis=1).astype(float)

# ========================== 删除重复列 =========================
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
numerical_features = ['lmt', 'certValidBegin', 'certValidStop', 'missing']+ ['bankCard', 'residentAddr', 'certId', 'dist']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

# ========================== 类别count特征 =========================
# cols=['bankCard', 'residentAddr', 'certId', 'dist', 'certValidPeriod', 'age', 'job', 'ethnic', 'basicLevel',
#         'linkRela']
# for col in cols:
#     df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

large_cols1 = ['bankCard', 'residentAddr', 'certId', 'dist']
# 计数
for col in large_cols1:
    col_nums = dict(df[col].value_counts())
    df[col + '_count'] = df[col].apply(lambda x: col_nums[x])

large_cols2 = ['basicLevel', 'ethnic', 'age', 'setupHour', 'job', 'edu',
        'linkRela', 'highestEdu', 'weekday', 'x_34', 'x_33']
for col in large_cols2:
    col_nums = dict(df[col].value_counts())
    df[col + '_count'] = df[col].apply(lambda x: col_nums[x])


# ========================== 类别特征组合 =========================
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
    df_.drop(columns=[group_name], inplace=True)
    return df_


group_features1 = [c for c in categorical_features if 'x_' in c]  # 匿名
group_features2 = ['bankCard', 'residentAddr', 'certId', 'dist']  # 地区特征
group_features3 = ['lmt', 'certValidBegin', 'certValidStop']  # 征信1

group_features4 = ['age', 'job', 'ethnic', 'basicLevel', 'linkRela']  # 基本属性
group_features5 = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan',
                   'unpayNormalLoan', '5yearBadloan']  #
group_features = [
    group_features1, group_features2,
    group_features3,
    group_features4,
    group_features5,
]

for index, groups in enumerate(group_features):
    index += 1
    name = 'group_features' + str(index)
    df = create_group_fea(df, groups, name)

# ========================== 特殊处理 =========================
# certValidBegin
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
for feat in numerical_features + ['certValidPeriod']:
    df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化

# 用户基本信息
group_features6 = ['gender', 'age', 'edu', 'job', 'ethnic', 'highestEdu']
df = create_group_fea(df, group_features6, 'group_features6')


def cert_val_transform(x):
    """
    target==1的时候 certValidBegin的范围为：
    # 3778531200  3293136000
    # 3776198400  3326313600
    :param x:
    :return:
    """

    if 3326313600 <= x <= 3776198400:
        return 1
    elif x > 3776198400:
        return 2
    else:
        return 3


group_features7 = []
df['certValidBegin_flag'] = df['certValidBegin'].apply(lambda x: cert_val_transform(x))
group_features7.append('certValidBegin_flag')


def id_num_transform(df_, col_name, base=1000):
    df_[col_name + '_prefix'] = df_[col_name].apply(lambda x: x // base)
    df_[col_name + '_suffix'] = df_[col_name].apply(lambda x: x % base)
    return df_


df = id_num_transform(df, 'certId')
group_features7.append('certId_prefix')
group_features7.append('certId_suffix')
df = id_num_transform(df, 'dist')
group_features7.append('dist_prefix')
group_features7.append('dist_suffix')
df = create_group_fea(df, group_features7, 'group_features7')

# 借贷信息
df['lmt_bin'] = pd.cut(df['lmt'], 5, labels=[1, 2, 3, 4, 5])
group_features8 = ['loanProduct', 'lmt_bin', 'basicLevel', 'linkRela', 'setupHour', 'weekday']
df = create_group_fea(df, group_features8, 'group_features8')

group_features9 = []
df = id_num_transform(df, 'bankCard')
df = id_num_transform(df, 'residentAddr')
group_features9.append('bankCard_prefix')
group_features9.append('bankCard_suffix')
group_features9.append('residentAddr_prefix')
group_features9.append('residentAddr_suffix')
df = create_group_fea(df, group_features9, 'group_features9')

# ========================== mean encoding 特殊处理 =========================
# 对lmt进行mean encoding
for fea in tqdm(['bankCard', 'residentAddr', 'certId', 'dist']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')


# dummies
df = pd.get_dummies(df, columns=categorical_features)

df.head(100).to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]


def load_data():
    return train, test, no_features, features

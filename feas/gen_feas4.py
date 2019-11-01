import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import *
from numpy import random


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


# simple_statics()


random.seed(1314)
train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")

df = pd.concat([train, test], sort=False, axis=0)

# ========================== 数据预处理 =========================
no_features = ['id', 'target']
features = []
df.fillna(value=-999, inplace=True)  # bankCard存在空值
df['missing'] = (df == -999).sum(axis=1).astype(float)

df['0_count'] = (df[[fea for fea in df.columns if fea not in no_features]].isin([0]) ).sum(axis=1).astype(float)
df['1_count'] = (df[[fea for fea in df.columns if fea not in no_features]].isin([1]) ).sum(axis=1).astype(float)

print(df.shape)



# ===================== 用户基本属性信息 =====================
# certId, gender, age, dist, edu, job, ethnic, highestEdu, certValidBegin, certValidStop,

# certId
df['certId_first2'] = df['certId'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['certId_middle2'] = df['certId'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['certId_last2'] = df['certId'].apply(lambda x: int(str(x)[4:6]))  # 最后两位

# print(len(df['certId_first2'].value_counts()))  # 31
# print(len(df['certId_middle2'].value_counts()))  # 48
# print(len(df['certId_last2'].value_counts()))  # 52

# dist
df['dist_first2'] = df['dist'].apply(lambda x: int(str(x)[:2]))  # 前两位
df['dist_middle2'] = df['dist'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
df['dist_last2'] = df['dist'].apply(lambda x: int(str(x)[4:6]))  # 最后两位

# print(len(df['dist_first2'].value_counts()))  # 32
# print(len(df['dist_middle2'].value_counts()))  # 48
# print(len(df['dist_last2'].value_counts()))  # 52


# certValidBegin
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
# df['certValidBegin_flag'] = df['certValidBegin'].apply(lambda x: cert_val_transform(x))
df['certValidBegin_bin'] = pd.cut(df['certValidBegin'], 100, labels=[i for i in range(100)])
df['certValidStop_bin'] = pd.qcut(df['certValidStop'], 10, labels=[i for i in range(10)])

# print(df['certValidBegin_bin'].value_counts())  #
# print(df['certValidStop_bin'].value_counts())  #

# 用户基本信息组合
user_information_combination1 = ['gender', 'age', 'edu', 'job', 'highestEdu']
df = create_group_fea(df, user_information_combination1, 'user_information_combination1')
# print(df['user_information_combination1'].value_counts()) #3k+

user_information_combination2 = ['gender', 'ethnic', 'edu', 'job', 'highestEdu']
df = create_group_fea(df, user_information_combination2, 'user_information_combination2')
# print(df['user_information_combination2'].value_counts()) # 1367

user_information_combination3 = ['gender', 'edu', 'job', 'highestEdu']
df = create_group_fea(df, user_information_combination3, 'user_information_combination3')
# print(df['user_information_combination3'].value_counts()) # 335

# ===================== 借贷相关信息 =====================
# loanProduct, lmt, basicLevel, bankCard, residentAddr, linkRela,setupHour, weekday

# bankCard 正常9位
df['bankCard'] = df['bankCard'].astype(int)
df['bankCard_first6'] = df['bankCard'].apply(lambda x: int(str(x)[:6]) if x != -999 else -999)
df['bankCard_last3'] = df['bankCard'].apply(lambda x: int(str(x)[6:].strip()) if x != -999 else -999)
# print(df['bankCard_first6'].value_counts()) # 174
df['bankCard_first3'] = df['bankCard'].apply(lambda x: int(str(x)[:3]) if x != -999 else -999)
df['bankCard_middle3'] = df['bankCard'].apply(lambda x: int(str(x)[3:6]) if x != -999 else -999)
df['bankCard_last3'] = df['bankCard'].apply(lambda x: int(str(x)[6:]) if x != -999 else -999)
# print(df['bankCard_first3'].value_counts()) # 20

# residentAddr
df['residentAddr_first2'] = df['residentAddr'].apply(lambda x: int(str(x)[:2]) if x != -999 else -999)  # 前两位
df['residentAddr_middle2'] = df['residentAddr'].apply(lambda x: int(str(x)[2:4]) if x != -999 else -999)  # 中间两位
df['residentAddr_last2'] = df['residentAddr'].apply(lambda x: int(str(x)[4:6]) if x != -999 else -999)  # 最后两位

# 借贷组合特征
loan_combination1 = ['loanProduct', 'basicLevel', 'linkRela', 'setupHour', 'weekday']  # 4978
df = create_group_fea(df, loan_combination1, 'loan_combination1')
# print(df['loan_combination1'].value_counts()) # 4k+

loan_combination2 = ['loanProduct', 'basicLevel', 'linkRela']  # 4978
df = create_group_fea(df, loan_combination2, 'loan_combination2')
# print(df['loan_combination2'].value_counts())  # 335

# ===================== 用户征信相关信息 =====================
# x_0至x_78以及ncloseCreditCard, unpayIndvLoan, unpayOtherLoan, unpayNormalLoan, 5yearBadloan
# x_ 组合特征
x_combination1 = ['x_' + str(i) for i in range(23)]
df = create_group_fea(df, x_combination1, 'x_combination1')
print(df['x_combination1'].value_counts())  # 335
x_combination2 = ['x_' + str(i) for i in range(23, 46)]
df = create_group_fea(df, x_combination2, 'x_combination2')
print(df['x_combination2'].value_counts())  # 335
x_combination3 = ['x_' + str(i) for i in range(46, 79)]
df = create_group_fea(df, x_combination3, 'x_combination3')
print(df['x_combination3'].value_counts())  # 335

unpayloan__combination = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']
df = create_group_fea(df, unpayloan__combination, 'unpayloan__combination')
print(df['unpayloan__combination'].value_counts())  # 335
# ========================== 删除重复列 =========================
duplicated_features = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6',
                       'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
                       'x_15', 'x_17', 'x_18', 'x_19', 'x_21',
                       'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
                       'x_59', 'x_60', 'x_77', 'x_78'] + \
                      ['x_40', 'x_70'] + \
                      ['x_41'] + \
                      ['x_43'] + \
                      ['x_45'] + \
                      ['x_61']
df = df.drop(columns=duplicated_features)

# ===================== 构造其他特征 begin =====================
# 组合特征
certId_first2_loanProduct = ['certId_first2', 'loanProduct']
df = create_group_fea(df, certId_first2_loanProduct, 'certId_first2_loanProduct')

certId_middle2_loanProduct = ['certId_middle2', 'loanProduct']
df = create_group_fea(df, certId_middle2_loanProduct, 'certId_middle2_loanProduct')

certId_last2_loanProduct = ['certId_last2', 'loanProduct']
df = create_group_fea(df, certId_last2_loanProduct, 'certId_last2_loanProduct')

df['lmt_bin'] = pd.qcut(df['certValidBegin'], 20, labels=[i for i in range(20)])
certId_first2_lmt = ['certId_first2', 'lmt_bin']
df = create_group_fea(df, certId_first2_lmt, 'certId_first2_lmt')
certId_middle2_lmt = ['certId_middle2', 'lmt_bin']
df = create_group_fea(df, certId_middle2_lmt, 'certId_middle2_lmt')
certId_last2_lmt = ['certId_last2', 'lmt_bin']
df = create_group_fea(df, certId_last2_lmt, 'certId_last2_lmt')
# count特征
count_latge_feas = ['bankCard', 'residentAddr', 'certId', 'dist', 'certValidPeriod',
                    'age', 'job', 'ethnic', 'basicLevel', 'linkRela']
for col in count_latge_feas:
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

count_cert_feas = ['certId_first2', 'certId_middle2', 'certId_last2']
for col in count_cert_feas:
    df['{}_count'.format(col)] = df.groupby(col)['id'].transform('count')

# ===================== 聚合特征 =====================
for fea in tqdm(['bankCard', 'residentAddr', 'dist',
                 'certId_first2', 'certId_middle2', 'certId_last2',
                 'unpayloan__combination']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['x_combination1', 'x_combination2', 'x_combination3']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['bankCard_first3', 'bankCard_middle3', 'bankCard_last3']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

for fea in tqdm(['dist_first2', 'dist_middle2', 'dist_last2']):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    df = pd.merge(df, grouped_df, on=fea, how='left')

features = [fea for fea in df.columns if fea not in no_features]
df.head(100).to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)
train, test = df[:len(train)], df[len(train):]


# simple_statics()

def load_data():
    return train, test, no_features, features

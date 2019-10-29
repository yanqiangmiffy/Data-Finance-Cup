import pandas as pd
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")

train['missing'] = (train == -999).sum(axis=1).astype(float)
test['missing'] = (test == -999).sum(axis=1).astype(float)

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
df.fillna(value=0, inplace=True)  # bankCard存在空值
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

no_features = ['id', 'target'] + ['bankCard', 'residentAddr', 'certId', 'dist', 'new_ind1', 'new_ind2']
features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop', 'missing']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]

group_features1 = [c for c in categorical_features if 'x_' in c]  # 匿名
group_features2 = ['bankCard', 'residentAddr', 'certId', 'dist']  # 地区特征
group_features3 = ['lmt', 'certValidBegin', 'certValidStop']  # 征信1

group_features4 = ['age', 'job', 'ethnic', 'basicLevel', 'linkRela']  # 基本属性
group_features5 = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']

group_features = [
    group_features1, group_features2, group_features3, group_features4,
    group_features5,
    # group_features6, group_features7, group_features8,
    # group_features9, group_features10, group_features11, group_features12,
    # group_features13
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

# 数值特征处理
df['certValidPeriod'] = df['certValidStop'] - df['certValidBegin']
for feat in numerical_features + ['certValidPeriod']:
    df[feat] = df[feat].rank() / float(df.shape[0])  # 排序，并且进行归一化
# 类别特征处理

# 特殊处理
cols = ['bankCard', 'residentAddr', 'certId', 'dist']
# 计数
for col in cols:
    col_nums = dict(df[col].value_counts())
    df[col + '_nums'] = df[col].apply(lambda x: col_nums[x])
# 对lmt进行mean encoding
for fea in tqdm(cols):
    grouped_df = df.groupby(fea).agg({'lmt': ['mean', 'median']})
    grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
    grouped_df = grouped_df.reset_index()
    # print(grouped_df)
    df = pd.merge(df, grouped_df, on=fea, how='left')
df = df.drop(columns=cols)  # 删除四列


# x的比例特征：
def get_features_x(data):
    print("x特征")
    model_sample_strong_feature = data.copy()

    x_strong_features = ['x_46', 'x_34', 'x_33', 'x_67', 'x_72', 'x_68', 'x_20', 'x_75', 'x_62', 'x_52', 'x_65']
    res = 0
    for i in range(len(x_strong_features)):
        res += 2 ** i * data[x_strong_features[i]]
    model_sample_strong_feature['x_1_strong'] = res

    group_features5 = ['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']
    res = 0
    for i in range(len(group_features5)):
        res += 2 ** i * data[x_strong_features[i]]
    model_sample_strong_feature['load_1_strong'] = res

    for c1, c2 in combinations(group_features1, 2):
        model_sample_strong_feature[c1 + '/' + c2] = data[c1] / (data[c2] + 1e-10)

    return model_sample_strong_feature


df = get_features_x(df)

fea_score = pd.read_excel('tmp/feature_score.xlsx')
cols = fea_score[fea_score['importance'] == 0]['feature'].values.tolist()
df=df.drop(columns=cols)

# dummies
# df = pd.get_dummies(df, columns=categorical_features)
df.head().to_csv('tmp/df.csv', index=None)
print("df.shape:", df.shape)

features = [fea for fea in df.columns if fea not in no_features]
train, test = df[:len(train)], df[len(train):]


def load_data():
    return train, test, no_features, features

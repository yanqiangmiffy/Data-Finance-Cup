import pandas as pd

train = pd.read_csv("new_data/train.csv")
train_target = pd.read_csv('new_data/train_target.csv')
train = train.merge(train_target, on='id')
test = pd.read_csv("new_data/test.csv")
df = pd.concat([train, test], sort=False, axis=0)
print(df.shape)
stats = []
for col in df.columns:
    stats.append((col, train[col].nunique(), train[col].isnull().sum() * 100 / train.shape[0],
                  train[col].value_counts(normalize=True, dropna=False).values[0] * 100, train[col].dtype))

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                        'Percentage of values in the biggest category', 'type'])
stats_df.sort_values('Unique_values', ascending=False, inplace=True)
stats_df.to_excel('tmp/stats_df.xlsx', index=None)

df.fillna(value=-999, inplace=True)  # bankCard存在空值

# 特征工程
no_features = ['id', 'target']

features = []
numerical_features = ['lmt', 'certValidBegin', 'certValidStop']
categorical_features = [fea for fea in df.columns if fea not in numerical_features + no_features]
# bankCard 5991
# residentAddr 5288
# certId 4033
# dist 3738
print(categorical_features)
for i in range(len(categorical_features)):
    duplicated_features = []
    for j in range(i + 1, len(categorical_features)):
        if categorical_features[i] in df.columns and categorical_features[j] in df.columns:
            if categorical_features[i] not in duplicated_features:
                duplicated_features.append(categorical_features[i])
            if (df[categorical_features[i]] == df[categorical_features[j]]).all():
                if categorical_features[j] not in duplicated_features:
                    duplicated_features.append(categorical_features[j])
    if len(duplicated_features) > 1:
        print(len(duplicated_features), duplicated_features)
    df = df.drop(columns=duplicated_features)

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

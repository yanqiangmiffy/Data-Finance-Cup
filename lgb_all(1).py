# -*- coding:utf-8 _*-
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import time


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

scaler = StandardScaler()


def get_fea(train, test):
    test['target'] = 'test'

    df = pd.concat((train, test))

    no_features = ['id', 'target', 'isNew', 'x_61', 'x_22', 'x_40', 'x_41', 'x_45', 'x_43', 'begin', 'stop']
    no_features.extend(['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_13',
                        'x_15', 'x_17', 'x_18', 'x_19', 'x_21', 'x_23', 'x_24', 'x_36', 'x_37', 'x_38', 'x_57', 'x_58',
                        'x_59', 'x_60', 'x_77', 'x_78', 'x_65', 'x_31', 'x_16', 'x_56', 'x_44', 'x_32', 'x_39'])

    df['begin'] = df['certValidBegin'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    df['begin_year'] = df['begin'].apply(lambda x: int(x[2:4]))
    df['begin_month'] = df['begin'].apply(lambda x: int(x[5:7]))
    df['begin_day'] = df['begin'].apply(lambda x: int(x[8:10]))

    df['certValidStop'] = df['certValidStop'].replace(256000000000, 7858771200)
    df['stop'] = df['certValidStop'].apply(lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x)))
    df['stop_year'] = df['stop'].apply(lambda x: int(x[1:4]))
    df['stop_month'] = df['stop'].apply(lambda x: int(x[5:7]))
    df['stop_day'] = df['stop'].apply(lambda x: int(x[8:10]))

    df['certId_first2'] = df['certId'].apply(lambda x: int(str(x)[:2]))  # 前两位
    df['certId_middle2'] = df['certId'].apply(lambda x: int(str(x)[2:4]))  # 中间两位
    df['certId_last2'] = df['certId'].apply(lambda x: int(str(x)[4:6]))  # 最后两位
    from tqdm import tqdm
    for fea in tqdm(['bankCard', 'certId_first2', 'certId_middle2', 'certId_last2']):
        grouped_df = df.groupby(fea).agg({'lmt': ['mean']})
        grouped_df.columns = [fea + '_' + '_'.join(col).strip() for col in grouped_df.columns.values]
        grouped_df = grouped_df.reset_index()
        df = pd.merge(df, grouped_df, on=fea, how='left')
    for fea in[['bankCard', 'age'], ['job', 'weekday']]:
        print(fea)
        i = fea[0]
        j = fea[1]
        df['%s--%s' % (i, j)] = df.apply(lambda x: str(x[i]) + str(x[j]), axis=1)
        lb = LabelEncoder()
        df['%s--%s' % (i, j)] = lb.fit_transform(df['%s--%s' % (i, j)])

    train = df[df['target'] != 'test']
    test = df[df['target'] == 'test']

    train = train[train['target'].notnull()].reset_index(drop=True)
    print(train.shape)
    return train, test, no_features

label = pd.read_csv('new_data/train_target.csv')
train = pd.read_csv('new_data/train.csv')
train = pd.merge(train, label, on='id', how='left')
test = pd.read_csv('new_data/test.csv')

train, test, no_features = get_fea(train, test)
print('get_fea ok')

label = train['target']
label = label.values.astype(int)

sub = test[['id']]

features = [fea for fea in train.columns if fea not in no_features]
train_df = train[features]
test_df = test[features]

scaler.fit(train_df)
train_df = scaler.transform(train_df)
test_df = scaler.transform(test_df)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'max_depth': 4,
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 16,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'random_state': 1024,
    'n_jobs': -1,
}

trn_data = lgb.Dataset(train_df, label)
val_data = lgb.Dataset(train_df, label)
num_round = 550
model = lgb.train(params,
                  trn_data,
                  num_round,
                  valid_sets=[trn_data, val_data],
                  verbose_eval=10,
                  early_stopping_rounds=100,
                  feature_name=features)

r = model.predict(test_df, num_iteration=model.best_iteration)

test['target'] = r

test[['id', 'target']].to_csv('result/submission_lgb_all.csv', index=None)

# 0.823104
# 0.829545



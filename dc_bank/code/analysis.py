import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
label = pd.read_csv('../data/train_target.csv')
train = pd.merge(train, label, on='id', how='left')


# 'pearson', 'kendall', 'spearman'
# print(train.corr())

# print(train.describe()[['id', 'certId']])
# print(test.describe()[['id', 'certId']])



# for i in train.columns:
#     print(i, len(train), len(train[train[i] != -999]), train[i].nunique())
#
# for i in test.columns:
#     print(i, len(test), len(test[test[i] != -999]), test[i].nunique())

data = pd.concat((train, test))
corr = train.corr()
print(corr)
print(corr[corr['x_1'] == 1][['x_1']].index)

list_151 = []
for i in data.columns:
    if data[i].nunique() < 100:
        print(i)
        print(data[i].value_counts())

    # print(len(data[data[i] == -999]))

    # if len(data[data[i] == -999]) == 151:
    #     # list_151.append(i)
    #     print(i, data[i].unique())
#         if data[i].nunique() == 2:
#             list_151.append(i)
#
# print(list_151)

# print(label.shape)
# print(label['target'].value_counts())
#
# tar = list(label['target'])
# print(sorted(tar, reverse=True)[int(len(tar)*0.0062)])

# print(train)

# print(len(['x_0', 'x_1', 'x_10', 'x_11', 'x_13', 'x_15', 'x_17', 'x_18', 'x_19', 'x_2', 'x_21', 'x_22', 'x_23', 'x_24', 'x_3', 'x_36', 'x_37', 'x_38', 'x_4', 'x_40', 'x_5', 'x_57', 'x_58', 'x_59', 'x_6', 'x_60', 'x_7', 'x_70', 'x_77', 'x_78', 'x_8', 'x_9']))
# x_0  x_1  x_10  x_11  x_13  x_15  x_17  x_18  x_19  x_2  x_21  x_22  x_23  x_24  x_3

# ['ncloseCreditCard', 'unpayIndvLoan', 'unpayNormalLoan', 'unpayOtherLoan', 'x_0', 'x_1', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19', 'x_2', 'x_20', 'x_21', 'x_22', 'x_23', 'x_24', 'x_25', 'x_26', 'x_27', 'x_28', 'x_29', 'x_3', 'x_30', 'x_31', 'x_32', 'x_33', 'x_34', 'x_35', 'x_36', 'x_37', 'x_38', 'x_39', 'x_4', 'x_40', 'x_41', 'x_42', 'x_43', 'x_44', 'x_45', 'x_46', 'x_47', 'x_48', 'x_49', 'x_5', 'x_50', 'x_51', 'x_52', 'x_53', 'x_54', 'x_55', 'x_56', 'x_57', 'x_58', 'x_59', 'x_6', 'x_60', 'x_61', 'x_62', 'x_63', 'x_64', 'x_65', 'x_66', 'x_67', 'x_68', 'x_69', 'x_7', 'x_70', 'x_71', 'x_72', 'x_73', 'x_74', 'x_75', 'x_76', 'x_77', 'x_78', 'x_8', 'x_9']


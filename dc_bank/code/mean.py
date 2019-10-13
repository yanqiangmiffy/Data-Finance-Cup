# -*- coding:utf-8 _*-
import pandas as pd
import numpy as np
test = pd.read_csv('../data/test.csv')
sub = test[['id']]

test_pred = np.load('../data/pred_test_boost.npy')
print(test_pred.shape)
pred_mean = np.mean(test_pred, axis=1)
print(pred_mean.shape)
print(pred_mean)
sub['target'] = pred_mean
sub['target'] = sub['target'].apply(lambda x: 0 if x < 0 else 1 if x > 1 else x)

sub[['id', 'target']].to_csv('../result/submission_stack_mean.csv', index=None)




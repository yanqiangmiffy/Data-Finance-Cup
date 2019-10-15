import pandas as pd



def rh1():
    lgb = pd.read_csv('../result/submission_lgb.csv')
    xgb = pd.read_csv('../result/submission_xgb.csv')
    lgb['target'] = lgb['target']*0.7 + xgb['target']*0.3

    lgb[['id', 'target']].to_csv('../result/submission_l7_x3_gb.csv', index=None)

rh1()





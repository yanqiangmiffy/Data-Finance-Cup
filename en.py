import pandas as pd

lgb = pd.read_csv('result/lgb_0.6999063429983563.csv')
xgb = pd.read_csv('result/w300_cnn.csv')

res=lgb.copy()
res['target'] = lgb['target'] * 0.3 + xgb['target'] * 0.7
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None)
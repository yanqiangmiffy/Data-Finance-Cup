import pandas as pd

lgb = pd.read_csv('result/lgb_0.6999.csv')
xgb = pd.read_csv('result/xgb_0.6916075534459247.csv')

res=lgb.copy()
res['target'] = lgb['target'] * 0.9 + xgb['target'] * 0.1
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None)
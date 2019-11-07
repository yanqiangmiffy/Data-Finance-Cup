import pandas as pd

lgb = pd.read_csv('result/0.7087926129573728.csv')
xgb = pd.read_csv('result/xgb.csv')
# cnn = pd.read_csv('result/w3000.6495996632995624_cnn.csv')

res = lgb.copy()
res['target'] = lgb['target'] * 0.5 + xgb['target'] * 0.5
# res=lgb*0.8+xgb*0.2   # 0.877785  xgb 原先
res.to_csv('result/en73.csv', index=None)
